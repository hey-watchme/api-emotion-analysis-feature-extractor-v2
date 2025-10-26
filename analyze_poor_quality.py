#!/usr/bin/env python3
"""
音質が悪い音声データでの感情分析テスト
"""

import torch
import librosa
import argparse
import json
from datetime import datetime
from transformers import HubertModel
import warnings
warnings.filterwarnings('ignore')

torch.serialization.add_safe_globals([argparse.Namespace])

class KushinadaEmotionRecognizer:
    LABEL_MAP = {
        0: "neutral",
        1: "joy",
        2: "anger",
        3: "sadness"
    }

    def __init__(self):
        self.upstream = None
        self.featurizer_weights = None
        self.projector = None
        self.post_net = None
        self.loaded = False

    def load_models(self):
        print("🔧 モデルをロード中...")
        self.upstream = HubertModel.from_pretrained("imprt/kushinada-hubert-large")
        self.upstream.eval()

        checkpoint_path = "./checkpoints/models--imprt--kushinada-hubert-large-jtes-er/snapshots/2ff7d7337e9b67d695193a38a8fe01639de9fe58/s3prl/result/downstream/kushinada-hubert-large-jtes-er_fold1/dev-best.ckpt"
        downstream_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Featurizer
        if 'Featurizer' in downstream_ckpt:
            self.featurizer_weights = downstream_ckpt['Featurizer']['weights']

        downstream_dict = downstream_ckpt["Downstream"]

        projector_weight = downstream_dict["projector.weight"]
        self.projector = torch.nn.Linear(projector_weight.size(1), projector_weight.size(0))
        self.projector.load_state_dict({"weight": projector_weight, "bias": downstream_dict["projector.bias"]})
        self.projector.eval()

        post_net_weight = downstream_dict["model.post_net.linear.weight"]
        self.post_net = torch.nn.Linear(post_net_weight.size(1), post_net_weight.size(0))
        self.post_net.load_state_dict({"weight": post_net_weight, "bias": downstream_dict["model.post_net.linear.bias"]})
        self.post_net.eval()

        self.loaded = True
        print("✅ モデルロード完了\n")

    def weighted_sum_layers(self, all_hidden_states):
        norm_weights = torch.softmax(self.featurizer_weights, dim=0)
        stacked = torch.stack(all_hidden_states, dim=0)
        weighted = (stacked * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted

    def predict_emotion(self, audio_path):
        if not self.loaded:
            self.load_models()

        print(f"📁 音声ファイル: {audio_path}")

        # 音声読み込み
        waveform_np, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)

        duration = len(waveform_np) / 16000
        print(f"   長さ: {duration:.2f}秒")
        print(f"   サンプルレート: 16000Hz")

        # 音質情報
        rms = (waveform_np ** 2).mean() ** 0.5
        print(f"   RMS音量: {rms:.6f}")

        print("\n🎵 感情分析を実行中...")
        with torch.no_grad():
            outputs = self.upstream(waveform, output_hidden_states=True)
            all_hidden_states = outputs.hidden_states

            if self.featurizer_weights is not None:
                features = self.weighted_sum_layers(all_hidden_states)
            else:
                features = outputs.last_hidden_state

            pooled = features.mean(dim=1)
            projected = self.projector(pooled)
            logits = self.post_net(projected)
            probs = torch.softmax(logits, dim=-1)

        logits_np = logits[0].numpy()
        probs_np = probs[0].numpy()
        predicted_class = probs_np.argmax()

        emotion_scores = {self.LABEL_MAP[i]: float(probs_np[i]) for i in range(4)}
        dominant_emotion = self.LABEL_MAP[predicted_class]
        confidence = float(probs_np[predicted_class])

        return {
            "audio_file": audio_path,
            "duration_seconds": duration,
            "rms_volume": float(rms),
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "all_emotions": emotion_scores,
            "logits_range": float(logits_np.max() - logits_np.min()),
            "logits_std": float(logits_np.std()),
            "quality_note": "poor_quality_test"
        }

def main():
    print("=" * 80)
    print("🎯 音質が悪い音声データでの感情分析テスト")
    print("=" * 80)
    print()

    audio_file = "./test_poor_quality.wav"
    recognizer = KushinadaEmotionRecognizer()
    result = recognizer.predict_emotion(audio_file)

    print("\n" + "=" * 80)
    print("📊 分析結果")
    print("=" * 80)

    sorted_emotions = sorted(result['all_emotions'].items(), key=lambda x: x[1], reverse=True)

    print("\n感情スコア（降順）:")
    print("-" * 60)

    for i, (emotion, score) in enumerate(sorted_emotions, 1):
        bar_length = int(score * 50)
        bar = "█" * bar_length
        marker = "👑" if i == 1 else "  "
        print(f"{marker} {i}. {emotion:10} : {score:6.2%} {bar}")

    print("-" * 60)
    print(f"\n🎭 主要感情: {result['dominant_emotion']} (信頼度: {result['confidence']:.2%})")

    print(f"\n📈 診断指標:")
    print(f"   RMS音量: {result['rms_volume']:.6f}")
    print(f"   logits範囲: {result['logits_range']:.4f} (正常: > 1.0)")
    print(f"   logits標準偏差: {result['logits_std']:.4f} (正常: > 0.5)")

    if result['logits_range'] > 1.0 and result['logits_std'] > 0.5:
        print("\n✅ モデルは明確な判断をしています")
    else:
        print("\n⚠️  判断が弱い可能性があります（音質の影響？）")

    output = {
        "model": "kushinada-hubert-large-jtes-er",
        "timestamp": datetime.now().isoformat(),
        **result
    }

    output_file = "poor_quality_test_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 結果を保存: {output_file}")
    print("\n" + "=" * 80)
    print("✅ テスト完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
