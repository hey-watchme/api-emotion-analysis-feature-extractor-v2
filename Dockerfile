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

# Kushinadaモデルとチェックポイントのプリロード（ビルド時に完全ダウンロード）
# これにより、実行時のダウンロード時間（3-5分）を完全に排除
RUN HF_TOKEN=${HF_TOKEN} python3 -c "\
from transformers import HubertModel; \
from huggingface_hub import hf_hub_download; \
import os; \
os.environ['HF_TOKEN'] = '${HF_TOKEN}'; \
print('🔧 HuBERTモデルをダウンロード中...'); \
HubertModel.from_pretrained('imprt/kushinada-hubert-large', token='${HF_TOKEN}'); \
print('✅ HuBERTモデルダウンロード完了'); \
print('🔧 Kushinadaチェックポイントをダウンロード中...'); \
checkpoint_path = hf_hub_download( \
    repo_id='imprt/kushinada-hubert-large-jtes-er', \
    filename='s3prl/result/downstream/kushinada-hubert-large-jtes-er_fold1/dev-best.ckpt', \
    token='${HF_TOKEN}' \
); \
print(f'✅ チェックポイントダウンロード完了: {checkpoint_path}'); \
"

# ポート8018を公開
EXPOSE 8018

# 環境変数の設定
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache

# ヘルスチェック
# start-period: モデルが既にイメージに含まれているため30秒で十分
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8018/health || exit 1

# アプリケーションの起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8018", "--workers", "1"]
