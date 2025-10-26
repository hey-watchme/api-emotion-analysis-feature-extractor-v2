#!/usr/bin/env python3
"""
セグメント分析：時間的な感情の推移を分析
"""

import torch
import librosa
import numpy as np
import argparse
import json
from datetime import datetime
from transformers import HubertModel
import warnings
warnings.filterwarnings('ignore')

torch.serialization.add_safe_globals([argparse.Namespace])

class KushinadaSegmentAnalyzer:
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
        print("✅ モデルのロード完了\n")

    def weighted_sum_layers(self, all_hidden_states):
        norm_weights = torch.softmax(self.featurizer_weights, dim=0)
        stacked = torch.stack(all_hidden_states, dim=0)
        weighted = (stacked * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted

    def predict_segment(self, waveform_segment):
        """
        単一セグメントの感情分析
        waveform_segment: torch.Tensor [samples]
        """
        if len(waveform_segment) < 1600:  # 0.1秒未満は処理しない
            return None

        waveform = waveform_segment.unsqueeze(0)  # [1, samples]

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

        probs_np = probs[0].numpy()
        predicted_class = probs_np.argmax()

        emotion_scores = {self.LABEL_MAP[i]: float(probs_np[i]) for i in range(4)}
        dominant_emotion = self.LABEL_MAP[predicted_class]
        confidence = float(probs_np[predicted_class])

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "all_emotions": emotion_scores
        }

    def analyze_segments(self, audio_path, segment_duration=10):
        """
        音声をセグメントに分割して時系列分析
        segment_duration: セグメントの長さ（秒）
        """
        if not self.loaded:
            self.load_models()

        print(f"📁 音声ファイル: {audio_path}")
        print(f"⏱️  セグメント長: {segment_duration}秒\n")

        # librosaで読み込み
        waveform_np, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(waveform_np) / 16000
        print(f"全体の長さ: {total_duration:.2f}秒")

        # セグメントに分割
        segment_samples = int(segment_duration * 16000)
        num_segments = int(np.ceil(len(waveform_np) / segment_samples))
        print(f"セグメント数: {num_segments}\n")

        results = []
        print("🎵 セグメントごとの感情分析を実行中...")
        print("=" * 80)

        for i in range(num_segments):
            start_sample = i * segment_samples
            end_sample = min((i + 1) * segment_samples, len(waveform_np))

            start_time = start_sample / 16000
            end_time = end_sample / 16000

            segment_waveform = torch.from_numpy(waveform_np[start_sample:end_sample]).float()

            result = self.predict_segment(segment_waveform)

            if result:
                result["segment_index"] = i
                result["start_time"] = start_time
                result["end_time"] = end_time
                result["duration"] = end_time - start_time
                results.append(result)

                # 進捗表示
                emotion = result["dominant_emotion"]
                confidence = result["confidence"]

                # 感情ごとの絵文字
                emoji_map = {
                    "joy": "😊",
                    "anger": "😠",
                    "sadness": "😢",
                    "neutral": "😐"
                }
                emoji = emoji_map.get(emotion, "❓")

                print(f"[{start_time:6.1f}s - {end_time:6.1f}s] {emoji} {emotion:8} ({confidence:5.1%}) ", end="")

                # プログレスバー
                bar_length = int(confidence * 20)
                bar = "█" * bar_length
                print(f"{bar}")

        print("=" * 80)

        return results

    def print_summary(self, results):
        """
        分析結果のサマリーを表示
        """
        print("\n" + "=" * 80)
        print("📊 分析サマリー")
        print("=" * 80)

        # 感情ごとの出現回数
        emotion_counts = {}
        for result in results:
            emotion = result["dominant_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print("\n感情の出現回数:")
        print("-" * 60)
        total_segments = len(results)
        for emotion in ["joy", "neutral", "sadness", "anger"]:
            count = emotion_counts.get(emotion, 0)
            percentage = count / total_segments * 100 if total_segments > 0 else 0
            bar_length = int(percentage / 2)
            bar = "█" * bar_length

            emoji_map = {"joy": "😊", "anger": "😠", "sadness": "😢", "neutral": "😐"}
            emoji = emoji_map.get(emotion, "❓")

            print(f"{emoji} {emotion:8} : {count:2}回 ({percentage:5.1f}%) {bar}")

        # 怒りの検出
        anger_segments = [r for r in results if r["dominant_emotion"] == "anger"]
        if anger_segments:
            print("\n⚠️  怒りが検出されたセグメント:")
            print("-" * 60)
            for seg in anger_segments:
                print(f"   {seg['start_time']:6.1f}s - {seg['end_time']:6.1f}s: {seg['confidence']:.1%}")
        else:
            print("\n✅ 怒りは検出されませんでした")

        # 全体の平均スコア
        print("\n全体の平均感情スコア:")
        print("-" * 60)
        avg_emotions = {emotion: 0.0 for emotion in ["neutral", "joy", "anger", "sadness"]}
        for result in results:
            for emotion, score in result["all_emotions"].items():
                avg_emotions[emotion] += score

        for emotion in avg_emotions:
            avg_emotions[emotion] /= len(results)

        sorted_emotions = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)
        for i, (emotion, score) in enumerate(sorted_emotions, 1):
            bar_length = int(score * 50)
            bar = "█" * bar_length
            marker = "👑" if i == 1 else "  "
            print(f"{marker} {i}. {emotion:10} : {score:6.2%} {bar}")

        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description='Kushinada セグメント分析')
    parser.add_argument('audio_file', type=str, help='分析する音声ファイルのパス')
    parser.add_argument('--segment-duration', type=int, default=10, help='セグメントの長さ（秒）デフォルト: 10秒')
    args = parser.parse_args()

    print("=" * 80)
    print("🎯 Kushinada セグメント分析（時系列感情分析）")
    print("=" * 80)
    print()

    analyzer = KushinadaSegmentAnalyzer()
    results = analyzer.analyze_segments(args.audio_file, segment_duration=args.segment_duration)
    analyzer.print_summary(results)

    # 結果を保存
    output = {
        "audio_file": args.audio_file,
        "segment_duration": args.segment_duration,
        "num_segments": len(results),
        "timestamp": datetime.now().isoformat(),
        "segments": results
    }

    output_file = f"segment_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 結果を保存: {output_file}")
    print("\n" + "=" * 80)
    print("✅ 分析完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
