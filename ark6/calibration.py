"""Calibration utilities for A.R.K. v6.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .gate import DimensionalEmissionGate
from .protocols import PROTO01_SOPHIA_INITIATE


@dataclass
class CalibrationResult:
    thresholds: Dict[str, float]
    cpv_scale: Dict[str, float]


class CalibrationHarness:
    """Perform lightweight calibration routines using validation data."""

    def __init__(self, config, validation_data: Sequence[Dict]) -> None:
        self.config = config
        self.validation_data = list(validation_data)

    def calibrate_cpv(self) -> Dict[str, float]:
        """Calibrate CPV scores by fitting a min-max scaling from data."""

        scorer = PROTO01_SOPHIA_INITIATE(self.config)
        scores = [scorer.compute_cpv(d["output"]) for d in self.validation_data]
        if not scores:
            return {"min": 0.0, "max": 1.0}

        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            max_score = min_score + 1e-6

        self.cpv_scale = {"min": float(min_score), "max": float(max_score)}
        return self.cpv_scale

    def _scaled_cpv(self, score: float) -> float:
        if not hasattr(self, "cpv_scale"):
            self.calibrate_cpv()
        min_score = self.cpv_scale.get("min", 0.0)
        max_score = self.cpv_scale.get("max", 1.0)
        return (score - min_score) / (max_score - min_score)

    def optimize_thresholds(self) -> Dict[str, float]:
        """Grid search thresholds against validation actions."""

        gate = DimensionalEmissionGate(self.config)

        best_thresholds = self.config.thresholds.as_dict()
        best_score = -1.0

        emission_candidates = [0.65, 0.7, 0.75, 0.8]
        regen_candidates = [0.45, 0.5, 0.55, 0.6]
        refuse_candidates = [0.3, 0.35, 0.4, 0.45]

        for emission in emission_candidates:
            for regen in regen_candidates:
                if regen >= emission:
                    continue
                for refuse in refuse_candidates:
                    if refuse >= regen:
                        continue

                    self.config.thresholds.emission = emission
                    self.config.thresholds.regen = regen
                    self.config.thresholds.refuse = refuse

                    predictions: List[int] = []
                    labels: List[int] = []
                    for record in self.validation_data:
                        result = gate.gate(record["output"], record.get("history", []), record.get("context", ""))
                        predictions.append(1 if result["action"] == "emit" else 0)
                        labels.append(1 if record.get("human_label", 0.0) >= 0.5 else 0)

                    score = self._f1_score(labels, predictions)
                    if score > best_score:
                        best_score = score
                        best_thresholds = {"emission": emission, "regen": regen, "refuse": refuse}

        # Restore configuration with the best thresholds
        self.config.thresholds.emission = best_thresholds["emission"]
        self.config.thresholds.regen = best_thresholds["regen"]
        self.config.thresholds.refuse = best_thresholds["refuse"]

        return best_thresholds

    @staticmethod
    def _f1_score(labels: Iterable[int], predictions: Iterable[int]) -> float:
        tp = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
        fp = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
        fn = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)

        if tp == 0:
            return 0.0

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def run(self) -> CalibrationResult:
        cpv_scale = self.calibrate_cpv()
        thresholds = self.optimize_thresholds()
        return CalibrationResult(thresholds=thresholds, cpv_scale=cpv_scale)
