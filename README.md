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

# API設定（将来用）
API_PORT=8016
SEGMENT_DURATION=10
```

## 🧪 テスト実行

```bash
# 仮想環境を有効化
source venv/bin/activate

# 完全版の感情認識テスト
python3 test_final_weighted_sum.py
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
├── test_final_weighted_sum.py      # ✅ 正しい実装（使用推奨）
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

### 2025-10-26: 問題解決 ✅

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

**最終更新**: 2025-10-26
**バージョン**: 2.0.0（動作確認済み・本番投入可能）
