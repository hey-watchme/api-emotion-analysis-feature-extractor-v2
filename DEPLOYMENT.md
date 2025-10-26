# Kushinada API デプロイメントガイド

このドキュメントは、Kushinada APIをEC2本番環境にデプロイするための完全な手順書です。

---

## 📋 前提条件

### GitHub Secretsの確認

以下のSecretsがGitHubリポジトリに設定されていることを確認してください：

```
AWS_ACCESS_KEY_ID       # AWS認証
AWS_SECRET_ACCESS_KEY   # AWS認証
EC2_HOST                # EC2のIPアドレス（例: 3.24.16.82）
EC2_SSH_PRIVATE_KEY     # SSH接続用秘密鍵
EC2_USER                # SSHユーザー名（通常はubuntu）
SUPABASE_URL            # Supabase プロジェクトURL
SUPABASE_KEY            # Supabase サービスロールキー
HF_TOKEN                # Hugging Face トークン（Kushinada用）
```

**確認方法**:
1. GitHubリポジトリ > Settings > Secrets and variables > Actions
2. 上記の全てのSecretsが登録されていることを確認

---

## 🚀 初回デプロイ手順

### ステップ1: ECRリポジトリの確認

**重要**: v3と同じECRリポジトリを使用して完全に置き換えます。

```bash
# ECRリポジトリの存在確認
aws ecr describe-repositories \
  --repository-names watchme-emotion-analysis-feature-extractor-v3 \
  --region ap-southeast-2

# ✅ リポジトリが既に存在するため、新規作成は不要
```

### ステップ2: EC2サーバーへの初回セットアップ

**注意**: EC2上のディレクトリは既に存在しているため、このステップは **スキップ可能** です。
ただし、念のため確認しておくことを推奨します。

```bash
# 1. EC2に接続
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82

# 2. ディレクトリの存在確認
ls -la /home/ubuntu/emotion-analysis-feature-extractor-v3

# 3. Dockerネットワークの確認
docker network inspect watchme-network

# 4. ログアウト
exit
```

**期待される結果**:
- ✅ `/home/ubuntu/emotion-analysis-feature-extractor-v3` ディレクトリが存在する
- ✅ `watchme-network` Dockerネットワークが存在する

### ステップ3: GitHub Secretsの最終確認

Kushinadaには`HF_TOKEN`が必須です。v3には無かった環境変数なので確認してください。

```bash
# ローカルの.envファイルからHF_TOKENを確認
grep HF_TOKEN /Users/kaya.matsumoto/projects/watchme/api/emotion-analysis/feature-extractor-v2/.env
```

**GitHub Secretsに`HF_TOKEN`を追加**:
1. GitHubリポジトリ > Settings > Secrets and variables > Actions
2. "New repository secret" をクリック
3. Name: `HF_TOKEN`
4. Value: `<your-hugging-face-token>`
5. "Add secret" をクリック

### ステップ4: デプロイ実行

```bash
# 1. 最新コードをコミット＆プッシュ
cd /Users/kaya.matsumoto/projects/watchme/api/emotion-analysis/feature-extractor-v2
git add .
git commit -m "Add CI/CD configuration for Kushinada API"
git push origin main

# 2. GitHub Actionsの実行を確認
# https://github.com/{organization}/{repository}/actions
```

**GitHub Actionsが自動的に以下を実行します**:
1. ✅ Dockerイメージをビルド
2. ✅ ECRにプッシュ（v3のイメージを上書き）
3. ✅ EC2に設定ファイル（docker-compose.prod.yml、run-prod.sh）をコピー
4. ✅ EC2に.envファイルを作成/更新
5. ✅ 既存コンテナ（v3）を完全削除
6. ✅ 新規コンテナ（v2/Kushinada）を起動
7. ✅ ヘルスチェック実行

### ステップ5: デプロイ確認

```bash
# 1. EC2に接続
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82

# 2. コンテナが起動しているか確認
docker ps | grep emotion-analysis-feature-extractor-v3

# 3. コンテナログ確認
docker logs emotion-analysis-feature-extractor-v3 --tail 100

# 4. ヘルスチェック
curl http://localhost:8018/health

# 5. API情報確認
curl http://localhost:8018/

# 期待されるレスポンス:
# {
#   "message": "Kushinada音声感情認識API",
#   "version": "2.0",
#   "model": "Kushinada (HuBERT-large JTES)",
#   ...
# }
```

### ステップ6: 外部からの動作確認

```bash
# ローカルマシンから外部URLでアクセス
curl https://api.hey-watch.me/emotion-analysis/features/

# Swagger UIで確認
open https://api.hey-watch.me/emotion-analysis/features/docs
```

---

## 🔄 2回目以降のデプロイ

2回目以降は、コードを変更して `git push` するだけで自動デプロイされます。

```bash
# 1. コード修正
vim main.py

# 2. コミット＆プッシュ
git add .
git commit -m "Update emotion analysis logic"
git push origin main

# 3. GitHub Actionsが自動的にデプロイ
# https://github.com/{organization}/{repository}/actions
```

---

## 🛠️ トラブルシューティング

### GitHub Actionsが失敗する場合

#### 1. HF_TOKEN が設定されていない

**症状**:
```
Error: 401 Client Error: Unauthorized for url
```

**解決**:
- GitHub Secrets に `HF_TOKEN` を追加（ステップ3参照）

#### 2. コンテナが起動しない

**診断**:
```bash
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82
docker logs emotion-analysis-feature-extractor-v3 --tail 100
```

**よくあるエラー**:
- **環境変数不足**: `.env`ファイルに必要な変数が含まれているか確認
- **モデルダウンロード失敗**: `HF_TOKEN` が正しいか確認
- **メモリ不足**: EC2のメモリ使用量を確認（`free -h`）

#### 3. ヘルスチェックが失敗する

**症状**:
```
⚠️ Health check failed after 5 attempts
```

**解決**:
```bash
# コンテナログで詳細なエラーを確認
docker logs emotion-analysis-feature-extractor-v3 --tail 200

# モデルの読み込みに時間がかかる場合は待機
# 初回起動時はcheckpointのダウンロードに5-10分かかる可能性あり
```

### v3への切り戻し

万が一、v2で問題が発生した場合、v3に切り戻すことができます。

```bash
# 1. v3のリポジトリに移動
cd /Users/kaya.matsumoto/projects/watchme/api/emotion-analysis/feature-extractor-v3

# 2. 空コミットでCI/CDを再トリガー
git commit --allow-empty -m "Rollback to v3"
git push origin main

# 3. GitHub Actionsが自動的にv3を再デプロイ
```

---

## 📊 デプロイ設定詳細

### ECRリポジトリ

| 項目 | 値 |
|-----|-----|
| リポジトリ名 | `watchme-emotion-analysis-feature-extractor-v3` |
| リージョン | `ap-southeast-2` |
| フルパス | `754724220380.dkr.ecr.ap-southeast-2.amazonaws.com/watchme-emotion-analysis-feature-extractor-v3` |

### EC2デプロイ先

| 項目 | 値 |
|-----|-----|
| ディレクトリ | `/home/ubuntu/emotion-analysis-feature-extractor-v3` |
| コンテナ名 | `emotion-analysis-feature-extractor-v3` |
| ポート | `8018` |
| ネットワーク | `watchme-network` |

### 環境変数

| 変数名 | 説明 | デフォルト値 |
|-------|------|------------|
| `AWS_ACCESS_KEY_ID` | AWS認証 | - |
| `AWS_SECRET_ACCESS_KEY` | AWS認証 | - |
| `AWS_REGION` | S3リージョン | `us-east-1` |
| `S3_BUCKET_NAME` | S3バケット名 | `watchme-vault` |
| `SUPABASE_URL` | Supabase URL | - |
| `SUPABASE_KEY` | Supabase キー | - |
| `HF_TOKEN` | Hugging Face トークン | - |

---

## 🎯 重要な違い: v3 → v2 移行

| 項目 | v3 (SUPERB) | v2 (Kushinada) |
|-----|-------------|----------------|
| モデル | wav2vec2-base-superb-er | Kushinada (HuBERT-large) |
| 感情数 | 8種類 | 4種類 |
| チャンク長 | 30秒 | 10秒 |
| 精度 | 英語データで訓練 | 日本語データで訓練（JTES） |
| 追加環境変数 | なし | `HF_TOKEN` |
| checkpoint | 不要 | 必要（自動ダウンロード） |

---

## ✅ デプロイチェックリスト

デプロイ前に以下を確認してください：

- [ ] GitHub Secretsが全て設定されている
- [ ] `HF_TOKEN` がGitHub Secretsに追加されている
- [ ] EC2ディレクトリ `/home/ubuntu/emotion-analysis-feature-extractor-v3` が存在する
- [ ] Dockerネットワーク `watchme-network` が存在する
- [ ] ECRリポジトリ `watchme-emotion-analysis-feature-extractor-v3` が存在する
- [ ] ローカルで`docker build`テストが成功している（オプション）

デプロイ後に確認：

- [ ] GitHub Actionsが成功している
- [ ] EC2でコンテナが起動している（`docker ps`）
- [ ] ヘルスチェックが通る（`curl http://localhost:8018/health`）
- [ ] 外部URLでアクセスできる（`https://api.hey-watch.me/emotion-analysis/features/`）
- [ ] Swagger UIが表示される

---

**最終更新**: 2025-10-26
**バージョン**: v2.0（Kushinada）
**デプロイ先**: EC2 3.24.16.82:/home/ubuntu/emotion-analysis-feature-extractor-v3
