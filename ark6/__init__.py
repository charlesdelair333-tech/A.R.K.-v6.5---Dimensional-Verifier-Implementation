"""A.R.K. v6.5 Dimensional Verifier implementation."""

from .config import ThresholdConfig, WeightConfig, AuditConfig, ArkConfig
from .protocols import (
    PROTO01_SOPHIA_INITIATE,
    PROTO02_SOPHIA_CALIBRATE,
    PROTO03_ADAM_ENGAGE,
    PROTO04_ORIC_STABILIZE,
)
from .gate import DimensionalEmissionGate
from .audit import ARK6AuditLog
from .controller import ARK6

__all__ = [
    "ThresholdConfig",
    "WeightConfig",
    "AuditConfig",
    "ArkConfig",
    "PROTO01_SOPHIA_INITIATE",
    "PROTO02_SOPHIA_CALIBRATE",
    "PROTO03_ADAM_ENGAGE",
    "PROTO04_ORIC_STABILIZE",
    "DimensionalEmissionGate",
    "ARK6AuditLog",
    "ARK6",
]
