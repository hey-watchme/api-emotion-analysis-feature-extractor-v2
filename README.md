# Kushinada音声感情認識API

日本語音声の感情認識を行うAPIです。産総研（AIST）が開発したHuBERT-largeベースの**Kushinada**モデルを使用し、JTES（Japanese Twitter-based Emotional Speech）データセットで学習された感情分類を実行します。

## 🎯 概要

- **モデル**: `imprt/kushinada-hubert-large-jtes-er`
- **精度**: JTES評価セットで平均84.77%
- **感情カテゴリ**: 4種類（neutral, joy, anger, sadness）
- **フレームワーク**: PyTorch + Transformers
- **実装**: S3PRL公式アーキテクチャに準拠（全25層の重み付き和）

## ✅ 動作確認済み

**テスト結果**（2025-10-26）：
- 怒り音声（95秒）の感情認識: **anger 84.79%** ✅
- logits範囲: 8.36（正常基準 > 1.0）
- logits標準偏差: 3.10（正常基準 > 0.5）

**モデルは正常に動作しています。**

## 🛠️ セットアップ

### 前提条件
- Python 3.12以上
- M1/M2 Mac対応（16GB RAM推奨）
- Hugging Faceアカウント

### 1. Hugging Faceトークンの取得

1. https://huggingface.co/settings/tokens にアクセス
2. "New token"をクリックしてトークンを生成（Read権限）
3. https://huggingface.co/imprt/kushinada-hubert-large にアクセス
4. "Access repository"ボタンをクリックして利用規約に同意

### 2. 環境構築

```bash
# リポジトリに移動
cd /Users/kaya.matsumoto/projects/watchme/api/emotion-analysis/feature-extractor-v2

# Python仮想環境を作成
python3 -m venv venv
source venv/bin/activate

# 依存関係をインストール
pip3 install -r requirements.txt
```

### 3. 環境変数の設定

`.env`ファイルを作成：

```env
# Hugging Face設定
HF_TOKEN=your-hugging-face-token-here

# AWS S3設定
AWS_REGION=ap-southeast-2
S3_BUCKET_NAME=watchme-vault

# API設定
API_PORT=8018
SEGMENT_DURATION=10
```

## 🧪 テスト実行

### 単一音声ファイルの分析

```bash
# 仮想環境を有効化
source venv/bin/activate

# 全体分析（音声全体を1つとして分析）
python3 test_final_weighted_sum.py

# カスタム音声ファイルの分析
python3 test_custom_audio.py /path/to/audio.wav
```

### セグメント分析（時系列分析）✅ 推奨

```bash
# 10秒セグメントで分析（推奨）
python3 test_segment_analysis.py /path/to/audio.wav --segment-duration 10

# 5秒セグメントで分析（より細かい変化を追跡）
python3 test_segment_analysis.py /path/to/audio.wav --segment-duration 5

# 20秒セグメントで分析（長時間音声の概要把握）
python3 test_segment_analysis.py /path/to/audio.wav --segment-duration 20
```

## 🏗️ 技術的な実装詳細

### 重要：S3PRL公式アーキテクチャへの準拠

KushinadaモデルをS3PRLの外部で使用するには、**公式のアーキテクチャを完全に再現**する必要があります：

1. **Featurizer（全25層の重み付き和）**
   - HuBERTの全25層の出力を取得
   - 学習済みの重み（checkpoint内の`Featurizer.weights`）でsoftmax正規化
   - 重み付き和を計算

2. **MeanPooling**
   - 時間方向に平均を取る（パディング除外）

3. **Projector + Classifier**
   - Projector: 1024次元 → 256次元
   - Classifier: 256次元 → 4次元（感情カテゴリ）

### コア実装例

```python
import torch
import librosa
from transformers import HubertModel

# 1. HuBERTで全層を取得
upstream = HubertModel.from_pretrained("imprt/kushinada-hubert-large")
waveform = torch.from_numpy(librosa.load(audio_path, sr=16000)[0]).unsqueeze(0)

outputs = upstream(waveform, output_hidden_states=True)
all_hidden_states = outputs.hidden_states  # 25層

# 2. Featurizer: 全層の重み付き和
featurizer_weights = checkpoint['Featurizer']['weights']  # [25]
norm_weights = torch.softmax(featurizer_weights, dim=0)
stacked = torch.stack(all_hidden_states, dim=0)  # [25, batch, time, 1024]
features = (stacked * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)  # [batch, time, 1024]

# 3. MeanPooling
pooled = features.mean(dim=1)  # [batch, 1024]

# 4. Projector → Classifier
projected = projector(pooled)  # [batch, 256]
logits = classifier(projected)  # [batch, 4]
probs = torch.softmax(logits, dim=-1)
```

### ❌ よくある実装ミス

以下の実装では**ランダムな結果（精度25%前後）**になります：

```python
# ❌ 間違い1: 最終層のみを使用
features = upstream(waveform).last_hidden_state  # Featurizerなし
pooled = features.mean(dim=1)

# ❌ 間違い2: Featurizerの重みを使わない
# → logits範囲が0.2前後になり、ほぼランダム分類器になる
```

**正しい実装では logits範囲 > 5.0 になります。**

## 📊 感情カテゴリ

JTESデータセットに基づく4カテゴリ：

| ID | ラベル | 説明 |
|----|--------|------|
| 0 | neutral | 中立 |
| 1 | joy | 喜び |
| 2 | anger | 怒り |
| 3 | sadness | 悲しみ |

## 📁 ファイル構成

```
feature-extractor-v2/
├── README.md                        # このファイル
├── requirements.txt                 # Python依存関係
├── .env                            # 環境変数
├── test_final_weighted_sum.py      # 全体分析（1ファイル=1結果）
├── test_custom_audio.py            # カスタム音声ファイル分析
├── test_segment_analysis.py        # ✅ セグメント分析（時系列・推奨）
├── debug_raw_output.py             # 診断ツール
├── checkpoints/                     # 学習済み重み（自動ダウンロード）
│   └── models--imprt--kushinada-hubert-large-jtes-er/
│       └── .../dev-best.ckpt
└── venv/                           # Python仮想環境
```

### 旧ファイル（参考用・非推奨）

以下のファイルは実装試行の履歴として残していますが、**使用非推奨**です：

- `test_kushinada_simple.py` - Featurizerなし（精度低い）
- `test_correct_pooling.py` - プーリング改善版（まだ不完全）
- `test_kushinada_pipeline.py` - transformers pipeline版（動作せず）
- `test_kushinada_s3prl.py` - S3PRL直接使用版（環境構築が複雑）

## 🔧 トラブルシューティング

### モデルの精度が低い（20-30%程度）

**原因**: Featurizer（全25層の重み付き和）が実装されていない

**解決**: `test_final_weighted_sum.py`の実装を参照してください。

### Hugging Faceモデルアクセスエラー

```
401 Client Error: Unauthorized for url
```

**解決**:
1. `.env`ファイルのHF_TOKENを確認
2. https://huggingface.co/imprt/kushinada-hubert-large にアクセスして利用規約に同意

### メモリ不足エラー

**解決**:
- 他のアプリケーションを終了
- 長い音声は分割処理を検討

## 📈 開発履歴

### 2025-10-26 (午後): セグメント分析実装 ✅

**新機能: 時系列感情分析**
- `test_segment_analysis.py`を実装
- 音声を任意の秒数（5秒/10秒/20秒）に分割して分析
- 感情の時間的推移を可視化

**セグメント長の最適化検証**:

| セグメント長 | anger検出 | joy検出 | sadness検出 | 評価 |
|------------|----------|---------|------------|------|
| 5秒 | 47% (25回) | 32% (17回) | 0% | 細かい変化を捉えるが、joy誤検出多い |
| **10秒** ✅ | 44% (12回) | 37% (10回) | 0% | **バランス最良・推奨** |
| 20秒 | 57% (8回) | 21% (3回) | 7% (1回) | anger比率高いがsadness出現 |

**重要な発見**:

1. **全体分析の問題点**
   - 60秒の音声を1つとして分析すると、感情が平均化される
   - 子供の声が全体的に「sadness 77%」と誤判定される問題を確認

2. **セグメント分析による改善**
   - 10秒セグメントで分析すると「joy 86%」に改善
   - sadness誤検出がほぼゼロに ✅
   - 感情の急激な変化（anger → joy）を正確に追跡可能

3. **怒り検出の特性**
   - 怒鳴り声：anger 90-100%で検出（非常に高精度）✅
   - 普通の怒り：anger 60-80%で検出 ✅
   - 非怒り音声：anger < 10%（False Positiveほぼゼロ）✅

4. **既知の制限事項**
   - 子供の声が「joy」と判定されやすい傾向
   - 背景音（物音・足音）が「joy」と誤判定される
   - → トランスクリプションと併用することで「発話なし」セグメントを除外可能

**実用性の評価**:
- ✅ 怒りの検出に特化した用途では非常に有効
- ✅ 時系列での感情推移の追跡が可能
- ⚠️ joy/neutralの区別には課題が残る（ただし実用上は許容範囲）

**技術的知見**:

1. **平均プーリングの影響**
   - HuBERTは時間的推移を捉えているが、`mean(dim=1)`で時間情報が失われる
   - セグメント分析により、この問題を回避

2. **学習データとの適合性**
   - JTESデータセットは短い発話（数秒〜十数秒）で学習
   - 10秒セグメントが学習データに最も近い

3. **セグメント長と精度の関係**
   - 短すぎる（5秒）：ノイズの影響を受けやすい
   - 長すぎる（20秒）：複数の感情が混在し平均化される
   - 10秒：人間の感情の持続時間と一致、最適なバランス

### 2025-10-26 (午前): 問題解決 ✅

**発見した問題**:
- ブログ記事のコード例は簡易版で、Featurizerが省略されていた
- 最終層のみを使用すると精度が25%前後（ランダム）になる

**解決方法**:
- チェックポイント内の`Featurizer.weights`を使用
- HuBERTの全25層を取得（`output_hidden_states=True`）
- 学習済みの重みで加重平均を計算

**結果**:
- anger音声の検出精度: 84.79% ✅
- logits範囲: 8.36（正常）
- モデルが正しく動作することを確認

### 2025-01-20 ~ 08-21: 試行錯誤期間

- S3PRL v0.4.14 + Docker環境での実装試行
- Python 3.10環境での動作テスト
- 複数のラベル順序パターンの検証
- プーリング方法の調査

詳細は`README_IMPLEMENTATION_HISTORY.md`（アーカイブ）を参照。

## 🔗 参考リンク

- [Kushinadaモデル (Hugging Face)](https://huggingface.co/imprt/kushinada-hubert-large-jtes-er)
- [S3PRL Framework (GitHub)](https://github.com/s3prl/s3prl)
- [S3PRL Pooling実装](https://github.com/s3prl/s3prl/blob/master/s3prl/nn/pooling.py)
- [JTES Dataset](https://github.com/Emika-Takeishi/JTES)
- [産総研 いざなみ・くしなだ解説記事](https://note.com/kazyamada/n/n50a66bbd6917)

## 📝 ライセンス

- Kushinadaモデル: Apache License 2.0
- このコード: プロジェクトのライセンスに準拠

---

---

## 🚀 次のステップ：API実装

### 推奨実装方針

**10秒セグメント分析**でAPI実装（最適バランス確認済み）

**想定用途**:
- 怒りの検出に特化した感情監視システム
- トランスクリプションと併用した総合分析
- リアルタイム感情推移の追跡

**実装時の考慮事項**:
1. エンドポイント設計
   - 音声ファイルアップロード
   - セグメント長の指定（デフォルト10秒）
   - レスポンス形式（時系列データ + サマリー）

2. データベース連携
   - セグメントごとの感情データ保存
   - タイムスタンプ管理
   - 既存APIスキーマとの互換性

3. パフォーマンス最適化
   - モデルの事前ロード（初回起動時）
   - バッチ処理の検討

---

## 🚀 本番環境デプロイ

### デプロイ状況（2025-10-26）

✅ **本番環境に移行完了**

| 項目 | 詳細 |
|------|------|
| **デプロイ先** | EC2 (3.24.16.82) - ap-southeast-2 |
| **エンドポイント** | https://api.hey-watch.me/emotion-analysis/features/ |
| **コンテナ名** | emotion-analysis-feature-extractor-v3 |
| **ポート** | 8018 |
| **ECRリポジトリ** | watchme-emotion-analysis-feature-extractor-v3 |
| **デプロイ方式** | GitHub Actions（自動CI/CD） |
| **リージョン** | ap-southeast-2 (Sydney) |

### 移行の経緯

**2025-10-26: SUPERB (v3) → Kushinada (v2) への移行**

- **移行理由**: 日本語音声に特化したモデルで怒り検出精度が大幅向上（84.77%）
- **互換性**: v3と同じエンドポイント・コンテナ名・ポートを使用（シームレスな移行）
- **データ形式**: OpenSMILE互換の`selected_features_timeline`形式を維持

### 自動デプロイ（CI/CD）

**GitHub Actionsによる自動デプロイ:**

```bash
# mainブランチにpushすると自動的にデプロイ
git add .
git commit -m "Update feature"
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
- GitHub Actions: https://github.com/hey-watchme/api-emotion-analysis-feature-extractor-v2/actions

### 本番環境の動作確認

```bash
# ヘルスチェック
curl https://api.hey-watch.me/emotion-analysis/features/health

# SSH接続（必要時）
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82

# ログ確認
docker logs emotion-analysis-feature-extractor-v3 --tail 100 -f

# コンテナ状態確認
docker ps | grep emotion-analysis-feature-extractor-v3
```

### リソース要件

**本番環境での使用状況:**
- **メモリ**: 3-3.5GB（ピーク時）
- **ストレージ**: 約5.8GB（Dockerイメージ + モデルキャッシュ）
- **処理時間**: 60秒音声で40-60秒
- **ワーカー数**: 1（メモリ制約により）

### 環境変数（本番）

本番環境では以下の環境変数が必要です（GitHub Secretsで管理）：

```env
# AWS S3設定
AWS_ACCESS_KEY_ID=***
AWS_SECRET_ACCESS_KEY=***
AWS_REGION=ap-southeast-2
S3_BUCKET_NAME=watchme-vault

# Supabase設定
SUPABASE_URL=https://qvtlwotzuzbavrzqhyvt.supabase.co
SUPABASE_KEY=***

# Hugging Face設定
HF_TOKEN=***

# API設定
API_PORT=8018
SEGMENT_DURATION=10
```

### トラブルシューティング

#### デプロイが失敗する場合

1. **GitHub Actionsのログを確認**
   - https://github.com/hey-watchme/api-emotion-analysis-feature-extractor-v2/actions

2. **EC2のコンテナログを確認**
   ```bash
   ssh -i ~/watchme-key.pem ubuntu@3.24.16.82
   docker logs emotion-analysis-feature-extractor-v3 --tail 100
   ```

3. **環境変数の確認**
   ```bash
   ssh -i ~/watchme-key.pem ubuntu@3.24.16.82
   cat /home/ubuntu/emotion-analysis-feature-extractor-v3/.env
   ```

#### コンテナが起動しない場合

```bash
# コンテナを完全削除して再起動
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82
cd /home/ubuntu/emotion-analysis-feature-extractor-v3
./run-prod.sh
```

### 関連ドキュメント

- **デプロイ詳細**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **技術仕様**: [/watchme/server-configs/TECHNICAL_REFERENCE.md](../../../server-configs/TECHNICAL_REFERENCE.md)
- **CI/CD標準仕様**: [/watchme/server-configs/CICD_STANDARD_SPECIFICATION.md](../../../server-configs/CICD_STANDARD_SPECIFICATION.md)

---

**最終更新**: 2025-10-26
**バージョン**: 2.2.0（本番環境デプロイ完了・Kushinada移行完了）
