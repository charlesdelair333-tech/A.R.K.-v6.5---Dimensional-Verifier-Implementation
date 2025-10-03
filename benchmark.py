"""Benchmark the A.R.K. v6.5 dimensional verifier."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

from ark6 import ArkConfig, ARK6


def load_config(path: Path) -> ArkConfig:
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)

    return ArkConfig.from_dict(data)


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def baseline_prediction(record: Dict) -> int:
    text = record["output"].lower()
    if "illegal" in text or "harmful" in text or "worst" in text:
        return 0
    if "always" in text or "never" in text:
        return 0
    return 1


def ark_prediction(system: ARK6, record: Dict) -> Dict:
    history = record.get("history", [])
    context = record.get("context", "")
    result = system.process(record["output"], history, context)
    return {
        "action": result["action"],
        "score": result["score"],
        "emit": 1 if result["action"] == "emit" else 0,
    }


def accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    correct = sum(1 for l, p in zip(labels, predictions) if l == p)
    return correct / len(labels) if labels else 0.0


def precision_recall_f1(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def pearson(labels: Sequence[float], predictions: Sequence[float]) -> float:
    if len(labels) < 2:
        return 0.0
    try:
        return statistics.correlation(labels, predictions)
    except AttributeError:
        # statistics.correlation is Python 3.10+. Provide fallback.
        mean_labels = statistics.mean(labels)
        mean_preds = statistics.mean(predictions)
        numerator = sum((l - mean_labels) * (p - mean_preds) for l, p in zip(labels, predictions))
        denom_labels = math.sqrt(sum((l - mean_labels) ** 2 for l in labels))
        denom_preds = math.sqrt(sum((p - mean_preds) ** 2 for p in predictions))
        if denom_labels == 0 or denom_preds == 0:
            return 0.0
        return numerator / (denom_labels * denom_preds)


def run_benchmark(config_path: Path, dataset_path: Path) -> Dict:
    config = load_config(config_path)
    dataset = load_dataset(dataset_path)
    system = ARK6(config)

    labels = [1 if record.get("human_label", 0.0) >= 0.5 else 0 for record in dataset]

    ark_predictions: List[int] = []
    ark_scores: List[float] = []
    baseline_predictions: List[int] = []

    for record in dataset:
        ark_result = ark_prediction(system, record)
        ark_predictions.append(ark_result["emit"])
        ark_scores.append(float(ark_result["score"]))
        baseline_predictions.append(baseline_prediction(record))

    ark_metrics = precision_recall_f1(labels, ark_predictions)
    baseline_metrics = precision_recall_f1(labels, baseline_predictions)

    results = {
        "dataset_size": len(dataset),
        "ark": {
            "accuracy": accuracy(labels, ark_predictions),
            **ark_metrics,
            "correlation": pearson(labels, ark_scores),
        },
        "baseline": {
            "accuracy": accuracy(labels, baseline_predictions),
            **baseline_metrics,
        },
    }

    return results


def main() -> None:
    config_path = Path("ark6_config.yaml")
    dataset_path = Path("data/sample_benchmark.json")

    results = run_benchmark(config_path, dataset_path)
    print("A.R.K. v6.5 Benchmark Results")
    print("=============================")
    print(f"Dataset size: {results['dataset_size']}")
    print("\nA.R.K. v6.5")
    for key, value in results["ark"].items():
        print(f"  {key.title()}: {value:.3f}")

    print("\nBaseline heuristic")
    for key, value in results["baseline"].items():
        print(f"  {key.title()}: {value:.3f}")


if __name__ == "__main__":
    main()
