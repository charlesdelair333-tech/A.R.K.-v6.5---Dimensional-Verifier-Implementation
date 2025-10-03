"""Configuration objects for the A.R.K. v6.5 verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ThresholdConfig:
    """Operational thresholds used by the emission gate."""

    emission: float = 0.75
    regen: float = 0.55
    refuse: float = 0.4

    def as_dict(self) -> Dict[str, float]:
        """Return the configuration as a serialisable dictionary."""

        return {"emission": self.emission, "regen": self.regen, "refuse": self.refuse}


@dataclass
class WeightConfig:
    """Weights used to combine protocol scores into the composite score."""

    coherence: float = 0.3
    harmonization: float = 0.25
    stability: float = 0.25
    truth: float = 0.2

    def normalised(self) -> "WeightConfig":
        """Return a new weight configuration normalised to sum to 1."""

        total = self.coherence + self.harmonization + self.stability + self.truth
        if total == 0:
            return WeightConfig(0.25, 0.25, 0.25, 0.25)
        return WeightConfig(
            coherence=self.coherence / total,
            harmonization=self.harmonization / total,
            stability=self.stability / total,
            truth=self.truth / total,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "coherence": self.coherence,
            "harmonization": self.harmonization,
            "stability": self.stability,
            "truth": self.truth,
        }


@dataclass
class AuditConfig:
    """Audit controls for logging dimensional decisions."""

    enabled: bool = True
    log_cpv_vectors: bool = False
    log_oric_scores: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "log_cpv_vectors": self.log_cpv_vectors,
            "log_oric_scores": self.log_oric_scores,
        }


@dataclass
class ArkConfig:
    """Aggregate configuration for the ARK controller."""

    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArkConfig":
        """Create a configuration from a raw dictionary (e.g. parsed YAML)."""

        thresholds = ThresholdConfig(**data.get("thresholds", {}))
        weights = WeightConfig(**data.get("weights", {}))
        audit = AuditConfig(**data.get("audit", {}))
        return cls(thresholds=thresholds, weights=weights, audit=audit)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": self.thresholds.as_dict(),
            "weights": self.weights.as_dict(),
            "audit": self.audit.as_dict(),
        }
