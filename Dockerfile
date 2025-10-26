FROM python:3.12-slim

# 作業ディレクトリの設定
WORKDIR /app

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY main.py .
COPY models.py .
COPY supabase_service.py .

# HuggingFaceトークンを引数から受け取る（ビルド時のみ使用）
ARG HF_TOKEN
RUN test -n "$HF_TOKEN" || (echo "Error: HF_TOKEN build arg is required" && exit 1)

# Kushinadaモデルのプリロード（Hugging Faceから自動ダウンロード）
# 注意: チェックポイントは実行時に自動ダウンロードされます
RUN HF_TOKEN=${HF_TOKEN} python3 -c "from transformers import HubertModel; HubertModel.from_pretrained('imprt/kushinada-hubert-large', token='${HF_TOKEN}')" || echo "モデルプリロードスキップ（実行時にダウンロード）"

# ポート8018を公開
EXPOSE 8018

# 環境変数の設定
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8018/health || exit 1

# アプリケーションの起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8018", "--workers", "1"]
