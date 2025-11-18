# Kushinada音声感情認識API

日本語音声の感情認識を行うAPIです。産総研（AIST）が開発したHuBERT-largeベースの**Kushinada**モデルを使用し、JTES（Japanese Twitter-based Emotional Speech）データセットで学習された感情分類を実行します。

---

## 📋 API仕様

### 基本情報

| 項目 | 値 |
|------|-----|
| **モデル** | Kushinada HuBERT-large（産総研AIST開発） |
| **精度** | JTES評価セットで平均84.77% |
| **感情カテゴリ** | 4種類（neutral, joy, anger, sadness） |
| **処理方式** | 10秒セグメント分析 |
| **フレームワーク** | PyTorch + Transformers |

### エンドポイント

| エンドポイント | メソッド | 説明 |
|-------------|---------|------|
| `https://api.hey-watch.me/emotion-analysis/features/health` | GET | ヘルスチェック |
| `https://api.hey-watch.me/emotion-analysis/features/` | GET | API情報 |
| `https://api.hey-watch.me/emotion-analysis/features/process/emotion-features` | POST | 感情分析実行（Lambda専用） |

### インフラ構成

| 項目 | 値 |
|------|-----|
| **デプロイ先** | EC2 (3.24.16.82) ap-southeast-2 |
| **コンテナ名** | `emotion-analysis-feature-extractor` |
| **ポート** | 8018（内部のみ） |
| **ECRリポジトリ** | `watchme-emotion-analysis-feature-extractor` |
| **ECR URI** | `754724220380.dkr.ecr.ap-southeast-2.amazonaws.com/watchme-emotion-analysis-feature-extractor:latest` |
| **EC2ディレクトリ** | `/home/ubuntu/emotion-analysis-feature-extractor` |

---

## 🚀 デプロイ

### 自動デプロイ（CI/CD）

```bash
# mainブランチにpushすると自動デプロイ
git add .
git commit -m "fix: update feature"
git push origin main
```

**デプロイフロー:**
1. Dockerイメージビルド（HF_TOKEN付き、ARM64対応）
2. ECRにプッシュ
3. EC2に設定ファイル転送
4. 環境変数ファイル作成（`.env`）
5. 既存コンテナ削除
6. 新規コンテナ起動
7. ヘルスチェック

**進捗確認:**
```bash
# GitHub Actions確認
gh run list --limit 3

# リアルタイム監視
gh run watch <run-id> --exit-status
```

### デプロイ検証

```bash
# デプロイが正しく完了したか検証
./verify-deployment.sh
```

**検証内容:**
- コンテナが正しく起動しているか
- ヘルスエンドポイントが応答するか
- Kushinadaモデルコードが含まれているか
- `percentage`フィールドが実装されているか
- 正しいECRイメージが使用されているか

---

## 🔧 本番環境操作

### 基本コマンド

```bash
# EC2接続
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82

# コンテナ状態確認
docker ps | grep emotion-analysis-feature-extractor

# ログ確認
docker logs emotion-analysis-feature-extractor --tail 100 -f

# ヘルスチェック
curl http://localhost:8018/health
```

### トラブルシューティング

#### コンテナが起動しない場合

```bash
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82
cd /home/ubuntu/emotion-analysis-feature-extractor

# コンテナを完全削除して再起動
./run-prod.sh
```

#### 古いコードが稼働している場合

```bash
# 検証スクリプトを実行
./verify-deployment.sh

# 失敗した場合、ECRイメージを確認
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82
docker inspect emotion-analysis-feature-extractor --format='{{.Config.Image}}'

# 期待されるイメージ
# 754724220380.dkr.ecr.ap-southeast-2.amazonaws.com/watchme-emotion-analysis-feature-extractor:latest
```

---

## 🛠️ ローカル開発

### 前提条件

- Python 3.12以上
- Hugging Faceアカウント
- Hugging Faceトークン（[取得方法](https://huggingface.co/settings/tokens)）
- Kushinadaモデルへのアクセス許可（[モデルページ](https://huggingface.co/imprt/kushinada-hubert-large)で同意）

### 環境構築

```bash
cd /Users/kaya.matsumoto/projects/watchme/api/emotion-analysis/feature-extractor-v2

# Python仮想環境を作成
python3 -m venv venv
source venv/bin/activate

# 依存関係をインストール
pip3 install -r requirements.txt
```

### 環境変数設定（`.env`）

```env
# Hugging Face設定
HF_TOKEN=your-hugging-face-token-here

# AWS S3設定
AWS_REGION=ap-southeast-2
S3_BUCKET_NAME=watchme-vault
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Supabase設定
SUPABASE_URL=https://qvtlwotzuzbavrzqhyvt.supabase.co
SUPABASE_KEY=your-supabase-key

# API設定
API_PORT=8018
SEGMENT_DURATION=10
```

### テスト実行

```bash
# セグメント分析（推奨）
python3 test_segment_analysis.py /path/to/audio.wav --segment-duration 10

# 全体分析
python3 test_final_weighted_sum.py

# カスタム音声ファイルの分析
python3 test_custom_audio.py /path/to/audio.wav
```

---

## 📊 技術詳細

### Kushinadaモデルの実装

**重要: S3PRL公式アーキテクチャへの準拠**

Kushinadaモデルを正しく動作させるには、以下のアーキテクチャを完全に再現する必要があります:

1. **Featurizer（全25層の重み付き和）**
   - HuBERTの全25層の出力を取得
   - 学習済みの重み（checkpoint内の`Featurizer.weights`）でsoftmax正規化
   - 重み付き和を計算

2. **MeanPooling**
   - 時間方向に平均を取る（パディング除外）

3. **Projector + Classifier**
   - Projector: 1024次元 → 256次元
   - Classifier: 256次元 → 4次元（感情カテゴリ）

**正しい実装では logits範囲 > 5.0 になります。**

### セグメント分析（10秒推奨）

| セグメント長 | 評価 |
|------------|------|
| 5秒 | 細かい変化を捉えるが、joy誤検出が多い |
| **10秒** ✅ | **バランス最良・推奨** |
| 20秒 | anger比率高いがsadness出現 |

---

## 📝 環境変数（本番）

GitHub Secretsで管理:

| 変数名 | 説明 |
|--------|------|
| `AWS_ACCESS_KEY_ID` | AWS認証 |
| `AWS_SECRET_ACCESS_KEY` | AWS認証 |
| `SUPABASE_URL` | Supabaseプロジェクト URL |
| `SUPABASE_KEY` | Supabaseサービスロールキー |
| `HF_TOKEN` | Hugging Faceトークン |

---

## 🔗 参考リンク

- [Kushinadaモデル (Hugging Face)](https://huggingface.co/imprt/kushinada-hubert-large-jtes-er)
- [S3PRL Framework (GitHub)](https://github.com/s3prl/s3prl)
- [JTES Dataset](https://github.com/Emika-Takeishi/JTES)
- [産総研 いざなみ・くしなだ解説記事](https://note.com/kazyamada/n/n50a66bbd6917)

---

**最終更新**: 2025-11-18
**バージョン**: 2.3.0（命名統一・デプロイ問題修正完了）
