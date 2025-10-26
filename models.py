"""
Pydanticモデル定義
SUPERB感情分析API用のリクエスト・レスポンススキーマ
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    status: str = Field(description="サービスステータス")
    service: str = Field(description="サービス名")
    version: str = Field(description="バージョン")
    model_loaded: bool = Field(description="モデルロード状態")


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: str = Field(description="エラーメッセージ")
    detail: Optional[str] = Field(None, description="詳細なエラー情報")
    error_code: Optional[str] = Field(None, description="エラーコード")


class EmotionScore(BaseModel):
    """感情スコア"""
    label: str = Field(description="感情ラベル（ang, hap, sad, neu, exc, fru, sur, dis）")
    score: float = Field(description="スコア（0.0-1.0）")
    percentage: float = Field(description="パーセンテージ（0.0-100.0）")
    name_ja: str = Field(description="日本語名")
    name_en: str = Field(description="英語名")
    group: str = Field(description="感情グループ")


class ChunkResult(BaseModel):
    """30秒チャンクの分析結果"""
    chunk_id: int = Field(description="チャンクID")
    start_time: float = Field(description="開始時間（秒）")
    end_time: float = Field(description="終了時間（秒）")
    duration: float = Field(description="チャンクの長さ（秒）")
    emotions: List[EmotionScore] = Field(description="8感情のスコア")
    primary_emotion: EmotionScore = Field(description="最も強い感情")


class EmotionFeaturesRequest(BaseModel):
    """感情特徴量抽出リクエスト（OpenSMILE互換）"""
    file_paths: List[str] = Field(
        description="処理対象のファイルパス一覧 (例: ['files/device_id/date/time/audio.wav'])"
    )


class EmotionFeaturesResponse(BaseModel):
    """感情特徴量抽出レスポンス"""
    success: bool = Field(description="処理成功フラグ")
    processed_files: int = Field(description="処理されたファイル数")
    saved_count: int = Field(description="Supabaseに保存されたレコード数")
    error_files: List[str] = Field(default=[], description="エラーが発生したファイル")
    total_processing_time: float = Field(description="総処理時間（秒）")
    message: str = Field(description="処理結果メッセージ")