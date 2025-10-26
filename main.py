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
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs[0].numpy()
        predicted_class = probs_np.argmax()

        # 4感情すべてのスコアを取得
        emotion_scores = {LABEL_MAP[i]: float(probs_np[i]) for i in range(4)}
        dominant_emotion = LABEL_MAP[predicted_class]
        confidence = float(probs_np[predicted_class])

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
                        "score": round(score, 6),
                        "percentage": round(score * 100, 3),
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


def extract_info_from_file_path(file_path: str) -> dict:
    """ファイルパスからデバイス情報を抽出

    Args:
        file_path: 'files/device_id/date/time/audio.wav' 形式

    Returns:
        dict: {'device_id': str, 'date': str, 'time_block': str}
    """
    parts = file_path.split('/')
    if len(parts) >= 5:
        return {
            'device_id': parts[1],
            'date': parts[2],
            'time_block': parts[3]
        }
    else:
        raise ValueError(f"不正なファイルパス形式: {file_path}")


async def update_audio_files_status(file_path: str) -> bool:
    """audio_filesテーブルのemotion_features_statusを更新

    Args:
        file_path: 処理完了したファイルのパス

    Returns:
        bool: 更新成功可否
    """
    try:
        update_response = supabase_client.table('audio_files') \
            .update({'emotion_features_status': 'completed'}) \
            .eq('file_path', file_path) \
            .execute()

        if update_response.data:
            print(f"✅ ステータス更新成功: {file_path}")
            return True
        else:
            print(f"⚠️ 対象レコードが見つかりません: {file_path}")
            return False

    except Exception as e:
        print(f"❌ ステータス更新エラー: {str(e)}")
        return False


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
    """file_pathsベースの感情分析（OpenSMILE互換）"""
    start_time = time.time()

    try:
        print(f"\n=== Kushinada file_pathsベースによる感情分析開始 ===")
        print(f"file_pathsパラメータ: {len(request.file_paths)}件のファイルを処理")
        print(f"セグメント長: {SEGMENT_DURATION}秒")
        print(f"=" * 50)

        if not supabase_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Supabaseサービスが利用できません。環境変数を確認してください。"
            )

        if not kushinada_analyzer or not kushinada_analyzer.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kushinadaモデルがロードされていません。"
            )

        processed_files = 0
        error_files = []
        supabase_records = []

        # 一時ディレクトリを作成してWAVファイルを処理
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_path in request.file_paths:
                try:
                    print(f"\n📥 S3からファイル取得開始: {file_path}")

                    # ファイルパスから情報を抽出
                    path_info = extract_info_from_file_path(file_path)
                    device_id = path_info['device_id']
                    date = path_info['date']
                    time_block = path_info['time_block']

                    # S3から一時ファイルにダウンロード
                    temp_file_path = os.path.join(temp_dir, f"{time_block}.wav")

                    try:
                        s3_client.download_file(s3_bucket_name, file_path, temp_file_path)
                        print(f"✅ S3ダウンロード成功: {file_path}")
                    except ClientError as e:
                        error_code = e.response['Error']['Code']
                        if error_code == 'NoSuchKey':
                            print(f"⚠️ ファイルが見つかりません: {file_path}")
                            error_files.append(file_path)
                            continue
                        else:
                            raise e

                    print(f"🎵 Kushinada感情分析開始: {file_path}")

                    # 感情分析を実行
                    analysis_start = time.time()
                    chunks_results, duration_seconds = kushinada_analyzer.analyze_audio_file(temp_file_path)
                    processing_time = time.time() - analysis_start

                    processed_files += 1

                    # Supabase用のレコードを準備
                    supabase_record = {
                        "device_id": device_id,
                        "date": date,
                        "time_block": time_block,
                        "filename": os.path.basename(file_path),
                        "duration_seconds": duration_seconds,
                        "features_timeline": chunks_results,  # Kushinadaの4感情結果
                        "selected_features_timeline": [],  # 空配列を設定
                        "processing_time": processing_time,
                        "error": None
                    }
                    supabase_records.append(supabase_record)

                    # audio_filesテーブルのステータスを更新
                    await update_audio_files_status(file_path)

                    # 主要感情を表示
                    if chunks_results:
                        for chunk in chunks_results:
                            primary = chunk["primary_emotion"]
                            print(f"  セグメント{chunk['chunk_id']}: {primary['name_ja']} ({primary['percentage']:.1f}%)")

                    print(f"✅ 完了: {file_path} → {len(chunks_results)}セグメントの感情分析完了")

                except Exception as e:
                    error_files.append(file_path)
                    print(f"❌ エラー: {file_path} - {str(e)}")

                    # エラーレコードもSupabaseに保存
                    try:
                        path_info = extract_info_from_file_path(file_path)
                        supabase_record = {
                            "device_id": path_info['device_id'],
                            "date": path_info['date'],
                            "time_block": path_info['time_block'],
                            "filename": os.path.basename(file_path),
                            "duration_seconds": 0,
                            "features_timeline": [],
                            "selected_features_timeline": [],
                            "processing_time": 0,
                            "error": str(e)
                        }
                        supabase_records.append(supabase_record)
                    except:
                        pass

        # Supabaseにバッチで保存
        print(f"\n=== Supabase保存開始 ===")
        print(f"保存対象: {len(supabase_records)} レコード")
        print(f"=" * 50)

        saved_count = 0
        save_errors = []

        if supabase_records:
            try:
                # バッチでUPSERT実行
                await supabase_service.batch_upsert_emotion_data(supabase_records)
                saved_count = len(supabase_records)
                print(f"✅ Supabase保存成功: {saved_count} レコード")
            except Exception as e:
                print(f"❌ Supabaseバッチ保存エラー: {str(e)}")
                # 個別に保存を試みる
                for record in supabase_records:
                    try:
                        await supabase_service.upsert_emotion_data(
                            device_id=record["device_id"],
                            date=record["date"],
                            time_block=record["time_block"],
                            filename=record["filename"],
                            duration_seconds=record["duration_seconds"],
                            features_timeline=record["features_timeline"],
                            processing_time=record["processing_time"],
                            error=record.get("error"),
                            selected_features_timeline=record.get("selected_features_timeline", [])
                        )
                        saved_count += 1
                    except Exception as individual_error:
                        save_errors.append(f"{record['time_block']}: {str(individual_error)}")
                        print(f"❌ 個別保存エラー: {record['time_block']} - {str(individual_error)}")

        # レスポンス作成
        total_time = time.time() - start_time

        print(f"\n=== Kushinada感情分析完了 ===")
        print(f"📥 S3処理: {processed_files} ファイル")
        print(f"💾 Supabase保存: {saved_count} レコード")
        print(f"❌ エラー: {len(error_files)} ファイル")
        print(f"⏱️ 総処理時間: {total_time:.2f}秒")
        print(f"=" * 50)

        return EmotionFeaturesResponse(
            success=True,
            processed_files=processed_files,
            saved_count=saved_count,
            error_files=error_files,
            total_processing_time=total_time,
            message=f"S3から{processed_files}個のファイルを処理し、{saved_count}個のレコードをSupabaseに保存しました"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"感情分析処理中にエラーが発生しました: {str(e)}"
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
