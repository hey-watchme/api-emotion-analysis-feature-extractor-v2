#!/bin/bash

# Kushinada Emotion Recognition API - 本番環境デプロイスクリプト
# 標準仕様: /Users/kaya.matsumoto/projects/watchme/server-configs/CICD_STANDARD_SPECIFICATION.md

set -e  # エラー時に即座に終了

echo "🚀 Starting Kushinada API deployment..."

# ディレクトリ確認
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📍 Current directory: $(pwd)"

# 環境変数ファイルの確認
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with required environment variables."
    exit 1
fi

echo "✅ .env file found"

# ECR認証とイメージの取得
echo "🔐 Logging into Amazon ECR..."
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 754724220380.dkr.ecr.ap-southeast-2.amazonaws.com

echo "📥 Pulling latest image from ECR (forced)..."
# docker-compose pull ではなく、docker pull で強制的に最新を取得
docker pull --platform linux/arm64 754724220380.dkr.ecr.ap-southeast-2.amazonaws.com/watchme-emotion-analysis-feature-extractor:latest

# プルしたイメージの確認
echo "✅ Image pulled successfully:"
docker images | grep watchme-emotion-analysis-feature-extractor | head -1

# コンテナの完全削除（複数パターン対応）
echo "🗑️ Removing existing containers..."

# 1. 新コンテナ名の削除
RUNNING_CONTAINERS=$(docker ps -q --filter "name=emotion-analysis-feature-extractor")
if [ ! -z "$RUNNING_CONTAINERS" ]; then
    echo "  Stopping running containers (emotion-analysis-feature-extractor)..."
    docker stop $RUNNING_CONTAINERS
fi

ALL_CONTAINERS=$(docker ps -aq --filter "name=emotion-analysis-feature-extractor")
if [ ! -z "$ALL_CONTAINERS" ]; then
    echo "  Removing all containers (emotion-analysis-feature-extractor)..."
    docker rm -f $ALL_CONTAINERS
fi

# 2. docker-compose管理コンテナの削除
echo "  Running docker-compose down..."
docker-compose -f docker-compose.prod.yml down || true

# 3. 旧コンテナ名の削除（v3表記、superb-api）
for OLD_NAME in "emotion-analysis-feature-extractor-v3" "superb-api"; do
    OLD_CONTAINERS=$(docker ps -aq --filter "name=$OLD_NAME")
    if [ ! -z "$OLD_CONTAINERS" ]; then
        echo "  Removing old containers ($OLD_NAME)..."
        docker rm -f $OLD_CONTAINERS
    fi
done

echo "✅ Container cleanup completed"

# 新規コンテナの起動
echo "🚀 Starting new container..."
docker-compose -f docker-compose.prod.yml up -d

# 起動確認
echo "⏳ Waiting for container to start..."
sleep 10

# コンテナステータス確認
if docker ps | grep -q emotion-analysis-feature-extractor; then
    echo "✅ Container is running"
    docker ps | grep emotion-analysis-feature-extractor
else
    echo "❌ Container failed to start"
    echo "Recent logs:"
    docker logs emotion-analysis-feature-extractor --tail 50 || true
    exit 1
fi

# ヘルスチェック（初回デプロイはモデルロードに約40秒かかる）
echo "🏥 Running health check..."
echo "⏳ モデルロード中（初回は約40秒かかります）..."
for i in {1..12}; do
    if curl -f http://localhost:8018/health > /dev/null 2>&1; then
        echo "✅ Health check passed"
        echo "🎉 Deployment completed successfully!"
        exit 0
    fi
    echo "  Attempt $i/12 failed, retrying in 5 seconds..."
    sleep 5
done

echo "⚠️ Health check failed after 12 attempts (60 seconds)"
echo "Container logs:"
docker logs emotion-analysis-feature-extractor --tail 50
exit 1
