#!/usr/bin/env python3
"""
Kushinada音声感情認識API - OpenSMILE互換版
日本語音声の感情認識を行うAPIサービス
産総研のKushinadaモデル（HuBERT-large）を使用
"""

import os
import gc
import time
import tempfile
import torch
import librosa
import numpy as np
import warnings
from transformers import HubertModel
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from supabase import create_client, Client

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List, Optional

from models import (
    HealthResponse,
    ErrorResponse,
    EmotionFeaturesRequest,
    EmotionFeaturesResponse,
    ChunkResult,
    EmotionScore
)
from supabase_service import SupabaseService

warnings.filterwarnings('ignore')

# 環境変数の読み込み
load_dotenv()

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="Kushinada Emotion Recognition API - OpenSMILE Compatible",
    description="Kushinadaモデルを使用したfile_pathsベースの感情分析サービス",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabaseクライアントの初期化
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if supabase_url and supabase_key:
    supabase_client: Client = create_client(supabase_url, supabase_key)
    supabase_service = SupabaseService(supabase_client)
    print(f"✅ Supabase接続設定完了: {supabase_url}")
else:
    supabase_service = None
    print("⚠️ Supabase環境変数が設定されていません")

# AWS S3クライアントの初期化
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket_name = os.getenv('S3_BUCKET_NAME', 'watchme-vault')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

if not aws_access_key_id or not aws_secret_access_key:
    raise ValueError("AWS_ACCESS_KEY_IDおよびAWS_SECRET_ACCESS_KEYが設定されていません")

s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)
print(f"✅ AWS S3接続設定完了: バケット={s3_bucket_name}, リージョン={aws_region}")

# セグメント設定
SEGMENT_DURATION = 10.0  # 10秒固定（最適バランス確認済み）

# Kushinada感情ラベルの詳細情報（4感情）
LABELS_INFO = {
    "neutral": {"ja": "中立", "en": "Neutral", "group": "neutral"},
    "joy": {"ja": "喜び", "en": "Joy", "group": "positive_active"},
    "anger": {"ja": "怒り", "en": "Anger", "group": "negative_active"},
    "sadness": {"ja": "悲しみ", "en": "Sadness", "group": "negative_passive"}
}

LABEL_MAP = {
    0: "neutral",
    1: "joy",
    2: "anger",
    3: "sadness"
}


class KushinadaAnalyzer:
    """Kushinadaモデルを使用した感情分析クラス"""

    def __init__(self):
        self.upstream = None
        self.featurizer_weights = None
        self.projector = None
        self.post_net = None
        self.loaded = False

    def load_models(self):
        """モデルをロード"""
        if self.loaded:
            return

        print("🔧 Kushinadaモデルをロード中...")

        # HuggingFaceトークンの設定
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token

        # HuBERT モデルのロード
        self.upstream = HubertModel.from_pretrained("imprt/kushinada-hubert-large")
        self.upstream.eval()

        # チェックポイントのロード（Hugging Faceから自動ダウンロード）
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(
            repo_id="imprt/kushinada-hubert-large-jtes-er",
            filename="s3prl/result/downstream/kushinada-hubert-large-jtes-er_fold1/dev-best.ckpt",
            token=os.getenv("HF_TOKEN")
        )

        print(f"✅ チェックポイントをダウンロード: {checkpoint_path}")

        downstream_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Featurizer weights（全25層の重み）
        if 'Featurizer' in downstream_ckpt:
            self.featurizer_weights = downstream_ckpt['Featurizer']['weights']

        downstream_dict = downstream_ckpt["Downstream"]

        # Projector（1024次元 → 256次元）
        projector_weight = downstream_dict["projector.weight"]
        self.projector = torch.nn.Linear(projector_weight.size(1), projector_weight.size(0))
        self.projector.load_state_dict({
            "weight": projector_weight,
            "bias": downstream_dict["projector.bias"]
        })
        self.projector.eval()

        # Post-net（Classifier: 256次元 → 4次元）
        post_net_weight = downstream_dict["model.post_net.linear.weight"]
        self.post_net = torch.nn.Linear(post_net_weight.size(1), post_net_weight.size(0))
        self.post_net.load_state_dict({
            "weight": post_net_weight,
            "bias": downstream_dict["model.post_net.linear.bias"]
        })
        self.post_net.eval()

        self.loaded = True
        print("✅ Kushinadaモデルのロード完了！\n")

    def weighted_sum_layers(self, all_hidden_states):
        """全25層の重み付き和を計算（Featurizer）"""
        norm_weights = torch.softmax(self.featurizer_weights, dim=0)
        stacked = torch.stack(all_hidden_states, dim=0)
        weighted = (stacked * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted

    def predict_segment(self, waveform_segment):
        """
        単一セグメントの感情分析

        Args:
            waveform_segment: torch.Tensor [samples]

        Returns:
            dict: 感情分析結果
        """
        if len(waveform_segment) < 1600:  # 0.1秒未満は処理しない
            return None

        waveform = waveform_segment.unsqueeze(0)  # [1, samples]

        with torch.no_grad():
            # HuBERT: 全25層を取得
            outputs = self.upstream(waveform, output_hidden_states=True)
            all_hidden_states = outputs.hidden_states

            # Featurizer: 全層の重み付き和
            if self.featurizer_weights is not None:
                features = self.weighted_sum_layers(all_hidden_states)
            else:
                features = outputs.last_hidden_state

            # MeanPooling
            pooled = features.mean(dim=1)

            # Projector → Classifier
            projected = self.projector(pooled)
            logits = self.post_net(projected)

        # logitsをそのまま使用（情報劣化を防ぐため、softmaxは適用しない）
        logits_np = logits[0].numpy()
        predicted_class = logits_np.argmax()

        # 4感情すべてのlogitsを取得（生スコア: -∞～+∞の範囲）
        emotion_scores = {LABEL_MAP[i]: float(logits_np[i]) for i in range(4)}
        dominant_emotion = LABEL_MAP[predicted_class]
        confidence = float(logits_np[predicted_class])

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "all_emotions": emotion_scores
        }

    def analyze_audio_file(self, audio_path: str) -> tuple:
        """
        音声ファイルを10秒セグメントに分割して感情分析

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            tuple: (セグメント結果リスト, 総時間)
        """
        if not self.loaded:
            self.load_models()

        # 音声読み込み（16kHz、モノラル）
        waveform_np, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(waveform_np) / 16000

        # セグメントに分割
        segment_samples = int(SEGMENT_DURATION * 16000)
        num_segments = int(np.ceil(len(waveform_np) / segment_samples))

        chunks_results = []

        for i in range(num_segments):
            chunk_id = i + 1
            start_sample = i * segment_samples
            end_sample = min((i + 1) * segment_samples, len(waveform_np))

            start_time = start_sample / 16000
            end_time = end_sample / 16000
            duration = end_time - start_time

            # セグメントを抽出
            segment_waveform = torch.from_numpy(waveform_np[start_sample:end_sample]).float()

            # 感情分析実行
            result = self.predict_segment(segment_waveform)

            if result:
                # 4感情すべてをemotions配列に整形
                emotions = []
                for label_id in range(4):
                    label = LABEL_MAP[label_id]
                    score = result["all_emotions"][label]
                    info = LABELS_INFO[label]

                    emotions.append({
                        "label": label,
                        "score": round(score, 6),  # logits生スコア（-∞～+∞）
                        "name_ja": info["ja"],
                        "name_en": info["en"],
                        "group": info["group"]
                    })

                # スコア順にソート
                emotions.sort(key=lambda x: x["score"], reverse=True)

                # チャンク結果を作成
                chunk_result = {
                    "chunk_id": chunk_id,
                    "start_time": round(start_time, 1),
                    "end_time": round(end_time, 1),
                    "duration": round(duration, 1),
                    "emotions": emotions,
                    "primary_emotion": emotions[0] if emotions else None
                }

                chunks_results.append(chunk_result)

            # メモリ解放
            del segment_waveform
            gc.collect()

        # メモリ解放
        del waveform_np
        gc.collect()

        return chunks_results, int(total_duration)


# グローバル変数でアナライザーを保持
kushinada_analyzer = None




# 起動時にモデルをロード
@app.on_event("startup")
async def startup_event():
    global kushinada_analyzer
    kushinada_analyzer = KushinadaAnalyzer()
    kushinada_analyzer.load_models()


@app.get("/", response_model=dict)
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Kushinada Emotion Recognition API - OpenSMILE Compatible",
        "version": "2.0.0",
        "model": "kushinada-hubert-large-jtes-er",
        "segment_duration": f"{SEGMENT_DURATION}秒",
        "emotions": list(LABELS_INFO.keys()),
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェックエンドポイント"""
    try:
        return HealthResponse(
            status="healthy",
            service="Kushinada API - OpenSMILE Compatible",
            version="2.0.0",
            model_loaded=kushinada_analyzer is not None and kushinada_analyzer.loaded
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/process/emotion-features", response_model=EmotionFeaturesResponse)
async def process_emotion_features(request: EmotionFeaturesRequest):
    """file_paths-based emotion analysis (spot_features table with UTC timestamp)"""
    start_time = time.time()

    try:
        print(f"\n=== Kushinada Emotion Analysis Start (UTC-based architecture) ===")
        print(f"file_paths: {len(request.file_paths)} files to process")
        print(f"Segment duration: {SEGMENT_DURATION} seconds")
        print(f"=" * 50)

        if not supabase_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Supabase service unavailable. Check environment variables."
            )

        if not kushinada_analyzer or not kushinada_analyzer.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kushinada model not loaded."
            )

        processed_files = 0
        error_files = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for file_path in request.file_paths:
                try:
                    print(f"\n📥 Fetching file from S3: {file_path}")

                    # Get device_id and recorded_at from audio_files table
                    audio_file_response = supabase_client.table('audio_files') \
                        .select('device_id, recorded_at') \
                        .eq('file_path', file_path) \
                        .single() \
                        .execute()

                    if not audio_file_response.data:
                        print(f"⚠️ Audio file record not found: {file_path}")
                        error_files.append(file_path)
                        continue

                    device_id = audio_file_response.data['device_id']
                    recorded_at = audio_file_response.data['recorded_at']

                    # Update status to processing
                    await supabase_service.update_audio_files_status(file_path, 'processing')

                    # Download from S3
                    temp_file_path = os.path.join(temp_dir, f"{device_id}_{recorded_at}.wav")

                    try:
                        s3_client.download_file(s3_bucket_name, file_path, temp_file_path)
                        print(f"✅ S3 download success: {file_path}")
                    except ClientError as e:
                        error_code = e.response['Error']['Code']
                        if error_code == 'NoSuchKey':
                            print(f"⚠️ File not found: {file_path}")
                            await supabase_service.update_audio_files_status(file_path, 'error')
                            error_files.append(file_path)
                            continue
                        else:
                            raise e

                    print(f"🎵 Kushinada emotion analysis start: {file_path}")

                    # Run emotion analysis
                    analysis_start = time.time()
                    chunks_results, duration_seconds = kushinada_analyzer.analyze_audio_file(temp_file_path)
                    processing_time = time.time() - analysis_start

                    # Save to spot_features table
                    save_success = await supabase_service.save_to_spot_features(
                        device_id,
                        recorded_at,
                        chunks_results
                    )

                    if save_success:
                        # Update status to completed
                        await supabase_service.update_audio_files_status(file_path, 'completed')
                        processed_files += 1

                        # Display primary emotions
                        if chunks_results:
                            for chunk in chunks_results:
                                primary = chunk["primary_emotion"]
                                print(f"  Segment {chunk['chunk_id']}: {primary['name_ja']} (score: {primary['score']:.2f})")

                        print(f"✅ Completed: {file_path} → {len(chunks_results)} segments analyzed")
                    else:
                        await supabase_service.update_audio_files_status(file_path, 'error')
                        error_files.append(file_path)

                except Exception as e:
                    error_files.append(file_path)
                    print(f"❌ Error: {file_path} - {str(e)}")

                    # Save error to spot_features
                    try:
                        audio_file_response = supabase_client.table('audio_files') \
                            .select('device_id, recorded_at') \
                            .eq('file_path', file_path) \
                            .single() \
                            .execute()

                        if audio_file_response.data:
                            await supabase_service.save_to_spot_features(
                                audio_file_response.data['device_id'],
                                audio_file_response.data['recorded_at'],
                                [],
                                error=str(e)
                            )
                            await supabase_service.update_audio_files_status(file_path, 'error')
                    except:
                        pass

        # Response
        total_time = time.time() - start_time

        print(f"\n=== Kushinada Emotion Analysis Complete ===")
        print(f"📥 S3 processing: {processed_files} files")
        print(f"❌ Errors: {len(error_files)} files")
        print(f"⏱️ Total processing time: {total_time:.2f} seconds")
        print(f"=" * 50)

        return EmotionFeaturesResponse(
            success=True,
            processed_files=processed_files,
            saved_count=processed_files,
            error_files=error_files,
            total_processing_time=total_time,
            message=f"Processed {processed_files} files from S3 and saved to spot_features table"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during emotion analysis: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """グローバル例外ハンドラー"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    # ポート8018で起動（v3と同じポート）
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8018,
        reload=True,
        log_level="info"
    )
