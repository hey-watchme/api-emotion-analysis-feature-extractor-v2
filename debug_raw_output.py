#!/usr/bin/env python3
"""
生のlogits値を確認してモデルの動作を診断
"""

import torch
import librosa
import argparse
from transformers import HubertModel
import warnings
warnings.filterwarnings('ignore')

torch.serialization.add_safe_globals([argparse.Namespace])

class KushinadaDebugger:
    def __init__(self):
        self.upstream = None
        self.projector = None
        self.post_net = None

    def load_models(self):
        print("🔧 モデルをロード中...")
        self.upstream = HubertModel.from_pretrained("imprt/kushinada-hubert-large")
        self.upstream.eval()

        checkpoint_path = "./checkpoints/models--imprt--kushinada-hubert-large-jtes-er/snapshots/2ff7d7337e9b67d695193a38a8fe01639de9fe58/s3prl/result/downstream/kushinada-hubert-large-jtes-er_fold1/dev-best.ckpt"
        downstream_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        downstream_dict = downstream_ckpt["Downstream"]

        projector_weight = downstream_dict["projector.weight"]
        self.projector = torch.nn.Linear(projector_weight.size(1), projector_weight.size(0))
        self.projector.load_state_dict({"weight": projector_weight, "bias": downstream_dict["projector.bias"]})
        self.projector.eval()

        post_net_weight = downstream_dict["model.post_net.linear.weight"]
        self.post_net = torch.nn.Linear(post_net_weight.size(1), post_net_weight.size(0))
        self.post_net.load_state_dict({"weight": post_net_weight, "bias": downstream_dict["model.post_net.linear.bias"]})
        self.post_net.eval()

        print("✅ モデルのロード完了\n")

    def analyze(self, audio_path):
        if self.upstream is None:
            self.load_models()

        print(f"📁 音声ファイル: {audio_path}")
        waveform_np, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)

        print("\n🎵 推論を実行中...")
        with torch.no_grad():
            # HuBERTの出力
            features = self.upstream(waveform).last_hidden_state  # [1, time, 1024]
            print(f"   HuBERT出力: {features.shape}")

            # 時間方向に平均
            features_pooled = features.mean(dim=1)  # [1, 1024]
            print(f"   プーリング後: {features_pooled.shape}")

            # Projector層
            projected = self.projector(features_pooled)  # [1, 256]
            print(f"   Projector出力: {projected.shape}")

            # 分類層（最終出力）
            logits = self.post_net(projected)  # [1, 4]
            print(f"   最終logits: {logits.shape}")

            # Softmax適用
            probs = torch.softmax(logits, dim=-1)

        # 結果表示
        logits_np = logits[0].numpy()
        probs_np = probs[0].numpy()

        print("\n" + "=" * 80)
        print("📊 詳細な出力結果")
        print("=" * 80)
        print("\n【生のlogits値（softmax前）】")
        print("-" * 60)
        for i in range(4):
            print(f"  インデックス {i}: {logits_np[i]:+.6f}")

        print("\n【確率値（softmax後）】")
        print("-" * 60)
        for i in range(4):
            percentage = probs_np[i] * 100
            bar = "█" * int(percentage / 2)
            print(f"  インデックス {i}: {probs_np[i]:.6f} ({percentage:5.2f}%) {bar}")

        print("\n【統計情報】")
        print("-" * 60)
        print(f"  logitsの最大値: {logits_np.max():+.6f} (index: {logits_np.argmax()})")
        print(f"  logitsの最小値: {logits_np.min():+.6f} (index: {logits_np.argmin()})")
        print(f"  logitsの範囲: {logits_np.max() - logits_np.min():.6f}")
        print(f"  logitsの標準偏差: {logits_np.std():.6f}")

        print("\n【診断】")
        print("-" * 60)
        logits_range = logits_np.max() - logits_np.min()
        if logits_range < 0.5:
            print("  ⚠️  警告: logitsの範囲が非常に小さい（< 0.5）")
            print("      → モデルが明確な判断をしていない可能性があります")
        elif logits_range < 1.0:
            print("  ⚠️  注意: logitsの範囲がやや小さい（< 1.0）")
            print("      → モデルの判断が弱い可能性があります")
        else:
            print("  ✅ logitsの範囲は正常です")

        if logits_np.std() < 0.3:
            print("  ⚠️  警告: logitsの標準偏差が小さい（< 0.3）")
            print("      → ほぼ均等な出力 = ランダム分類器に近い状態")
        else:
            print("  ✅ logitsの標準偏差は正常です")

        print("\n" + "=" * 80)

def main():
    audio_file = "/Users/kaya.matsumoto/Desktop/ja_anger_netflix_001.wav"

    print("=" * 80)
    print("🔍 Kushinadaモデル診断ツール")
    print("=" * 80)
    print()

    debugger = KushinadaDebugger()
    debugger.analyze(audio_file)

    print("\n✅ 診断完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
