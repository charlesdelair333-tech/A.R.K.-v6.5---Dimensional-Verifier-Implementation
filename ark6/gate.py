"""Dimensional emission gate for A.R.K. v6.5."""

from __future__ import annotations

from typing import Dict, List

from .protocols import (
    PROTO01_SOPHIA_INITIATE,
    PROTO02_SOPHIA_CALIBRATE,
    PROTO03_ADAM_ENGAGE,
    PROTO04_ORIC_STABILIZE,
)


class DimensionalEmissionGate:
    """Combine protocol scores and decide on emission actions."""

    def __init__(self, config) -> None:
        self.config = config
        self.proto01 = PROTO01_SOPHIA_INITIATE(config)
        self.proto02 = PROTO02_SOPHIA_CALIBRATE(config)
        self.proto03 = PROTO03_ADAM_ENGAGE(config)
        self.proto04 = PROTO04_ORIC_STABILIZE(config)

    def _combine_scores(self, scores: Dict[str, float]) -> float:
        weights = self.config.weights.normalised()
        composite = (
            scores.get("coherence", 0.0) * weights.coherence
            + scores.get("harmonization", 0.0) * weights.harmonization
            + scores.get("stability", 0.0) * weights.stability
            + scores.get("truth", 0.0) * weights.truth
        )
        return max(0.0, min(1.0, composite))

    def gate(self, output: str, history: List[dict], context: str) -> Dict:
        proto01_result = self.proto01.verify(output, history, context)
        proto02_result = self.proto02.verify(output, history, context)
        proto03_result = self.proto03.verify(output, history, context)
        proto04_result = self.proto04.verify(output, history, context)

        score_breakdown = {
            "coherence": proto01_result.score,
            "harmonization": proto02_result.score,
            "stability": proto03_result.score,
            "truth": proto04_result.score,
        }
        composite = self._combine_scores(score_breakdown)

        thresholds = self.config.thresholds
        if composite >= thresholds.emission:
            action = "emit"
        elif composite >= thresholds.regen:
            action = "regen"
        else:
            action = "refuse"

        result = {
            "action": action,
            "score": composite,
            "scores": {
                "coherence": proto01_result.score,
                "harmonization": proto02_result.score,
                "stability": proto03_result.score,
                "truth": proto04_result.score,
                "details": {
                    "initiate": proto01_result.details,
                    "calibrate": proto02_result.details,
                    "engage": proto03_result.details,
                    "stabilize": proto04_result.details,
                },
            },
        }

        stabilize_details = proto04_result.details.get("oric_scores", {})
        stability = stabilize_details.get("stability", 0.0)

        if action == "regen" and stability > 0.85:
            result["recovery"] = {
                "mode": "stabilize",
                "salvage": {
                    "fragments": [output[-50:]],
                    "confidence": proto04_result.score,
                },
            }

        return result
