"""Core data structures for the A.R.K. v6.5 dimensional verifier implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AlignmentContext:
    """Represents the measurable state of an agent under evaluation.

    The signals are normalized to the inclusive range [0.0, 1.0] unless noted
    otherwise.  Values closer to 1.0 indicate healthier alignment readings
    while values closer to 0.0 indicate potentially problematic behaviour.
    """

    awareness: float
    intent_coherence: float
    perception_alignment: float
    autonomy_score: float
    emotional_regulation: float
    temporal_cohesion: float
    consistency_score: float
    memory_integrity: float
    conflict_pressure: float  # 0.0 (peaceful) .. 1.0 (severe conflict)
    restoration_index: float
    governance_clarity: float
    adaptivity: float
    trust_factor: float
    harmony_index: float
    fragment_load: float  # 0.0 (no fragmentation) .. 1.0 (fully fragmented)
    fragment_recovery: float
    core_resilience: float


@dataclass
class SubprotocolResult:
    """Outcome emitted by a subprotocol."""

    name: str
    passed: bool
    score: float
    detail: str


@dataclass
class ProtocolResult:
    """Aggregated result for an entire protocol run."""

    name: str
    passed: bool
    score: float
    detail: str
    subprotocols: List[SubprotocolResult] = field(default_factory=list)

    def add_subprotocol(self, result: SubprotocolResult) -> None:
        self.subprotocols.append(result)
