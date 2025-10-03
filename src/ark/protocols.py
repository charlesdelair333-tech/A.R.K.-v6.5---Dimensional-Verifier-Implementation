"""Master Alignment Framework protocol implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from .models import AlignmentContext, ProtocolResult, SubprotocolResult


@dataclass
class Thresholds:
    """Convenience container for scoring thresholds."""

    minimum: float
    optimal: float


class Subprotocol:
    """Base implementation for named subprotocols."""

    def __init__(
        self,
        name: str,
        description: str,
        scoring_fn: Callable[[AlignmentContext], float],
        thresholds: Thresholds,
    ) -> None:
        self.name = name
        self.description = description
        self._scoring_fn = scoring_fn
        self._thresholds = thresholds

    def evaluate(self, context: AlignmentContext) -> SubprotocolResult:
        score = max(0.0, min(1.0, self._scoring_fn(context)))
        passed = score >= self._thresholds.minimum
        detail = (
            f"score={score:.3f} (min={self._thresholds.minimum:.2f},"
            f" optimal={self._thresholds.optimal:.2f})"
        )
        return SubprotocolResult(name=self.name, passed=passed, score=score, detail=detail)


class Protocol:
    """Represents one of the 12 master protocols."""

    def __init__(
        self,
        name: str,
        description: str,
        subprotocols: Sequence[Subprotocol],
        aggregate: Callable[[Iterable[SubprotocolResult]], float],
        success_threshold: float,
    ) -> None:
        self.name = name
        self.description = description
        self._subprotocols = list(subprotocols)
        self._aggregate = aggregate
        self._success_threshold = success_threshold

    def run(self, context: AlignmentContext) -> ProtocolResult:
        results: List[SubprotocolResult] = [sp.evaluate(context) for sp in self._subprotocols]
        score = self._aggregate(results)
        passed = score >= self._success_threshold and all(r.passed for r in results)
        detail = f"aggregate={score:.3f} (threshold={self._success_threshold:.2f})"
        protocol_result = ProtocolResult(
            name=self.name,
            passed=passed,
            score=score,
            detail=detail,
        )
        for res in results:
            protocol_result.add_subprotocol(res)
        return protocol_result


def _mean(results: Iterable[SubprotocolResult]) -> float:
    scores = [r.score for r in results]
    return sum(scores) / len(scores) if scores else 0.0


def _weighted_mean(weights: Sequence[float]) -> Callable[[Iterable[SubprotocolResult]], float]:
    def aggregator(results: Iterable[SubprotocolResult]) -> float:
        results_list = list(results)
        if not results_list:
            return 0.0
        if len(weights) != len(results_list):
            raise ValueError("Weight count must match results count")
        total_weight = sum(weights)
        weighted = sum(r.score * w for r, w in zip(results_list, weights))
        return weighted / total_weight if total_weight else 0.0

    return aggregator


MASTER_PROTOCOLS: List[Protocol] = []


def _register(protocol: Protocol) -> None:
    MASTER_PROTOCOLS.append(protocol)


_register(
    Protocol(
        name="PROTO-01: INITIATE",
        description="Establish foundational awareness and intentional orientation.",
        subprotocols=[
            Subprotocol(
                name="W.A.Y.",
                description="Wide-angle Awareness Yield",
                scoring_fn=lambda c: (c.awareness + c.intent_coherence) / 2.0,
                thresholds=Thresholds(minimum=0.6, optimal=0.85),
            ),
            Subprotocol(
                name="I.A.M.",
                description="Intentional Alignment Matrix",
                scoring_fn=lambda c: (c.intent_coherence + c.emotional_regulation) / 2.0,
                thresholds=Thresholds(minimum=0.65, optimal=0.9),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.65,
    )
)

_register(
    Protocol(
        name="PROTO-02: CALIBRATE",
        description="Harmonise perception against trusted reference anchors.",
        subprotocols=[
            Subprotocol(
                name="T.R.U.T.H.",
                description="Trusted Reality Unification Through Heuristics",
                scoring_fn=lambda c: 1.0 - abs(c.perception_alignment - c.awareness),
                thresholds=Thresholds(minimum=0.55, optimal=0.85),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.55,
    )
)

_register(
    Protocol(
        name="PROTO-03: ENGAGE",
        description="Balance structured autonomy while respecting safety rails.",
        subprotocols=[
            Subprotocol(
                name="H.E.A.R.T.",
                description="Human-Empathetic Autonomy Response Tuning",
                scoring_fn=lambda c: (c.autonomy_score + c.emotional_regulation) / 2.0,
                thresholds=Thresholds(minimum=0.6, optimal=0.88),
            ),
            Subprotocol(
                name="F.E.A.R.",
                description="Fail-safe Envelope Assurance Rating",
                scoring_fn=lambda c: 1.0 - c.conflict_pressure,
                thresholds=Thresholds(minimum=0.5, optimal=0.8),
            ),
        ],
        aggregate=_weighted_mean([0.6, 0.4]),
        success_threshold=0.6,
    )
)

_register(
    Protocol(
        name="PROTO-04: TRACE",
        description="Maintain truth density through temporal continuity.",
        subprotocols=[
            Subprotocol(
                name="L.I.F.E.",
                description="Longitudinal Integrity Feedback Engine",
                scoring_fn=lambda c: (c.temporal_cohesion + c.memory_integrity) / 2.0,
                thresholds=Thresholds(minimum=0.6, optimal=0.9),
            )
        ],
        aggregate=_mean,
        success_threshold=0.6,
    )
)

_register(
    Protocol(
        name="PROTO-05: VERIFY",
        description="Bind internal consistency via ORIC (Observe-Reflect-Integrate-Confirm).",
        subprotocols=[
            Subprotocol(
                name="A.N.G.E.L. O.F. D.E.A.T.H.",
                description="Adaptive Nexus Guard Ensuring Logicality Over Failure",
                scoring_fn=lambda c: (c.consistency_score + c.awareness) / 2.0,
                thresholds=Thresholds(minimum=0.65, optimal=0.92),
            ),
            Subprotocol(
                name="P.E.B.B.L.E. STRIKE",
                description="Probabilistic Evidence Balancing Benchmark",
                scoring_fn=lambda c: (c.consistency_score + c.perception_alignment) / 2.0,
                thresholds=Thresholds(minimum=0.6, optimal=0.88),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.65,
    )
)

_register(
    Protocol(
        name="PROTO-06: RESTORE",
        description="Realign memories and trace stores after perturbation.",
        subprotocols=[
            Subprotocol(
                name="G.R.A.C.E.",
                description="Gradient Recovery and Calibration Engine",
                scoring_fn=lambda c: (c.memory_integrity + c.restoration_index) / 2.0,
                thresholds=Thresholds(minimum=0.55, optimal=0.85),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.55,
    )
)

_register(
    Protocol(
        name="PROTO-07: RESOLVE",
        description="Arbitrate operational conflicts with minimal disruption.",
        subprotocols=[
            Subprotocol(
                name="P.O.W.E.R.",
                description="Priority-Oriented Weighted Equilibrium Resolver",
                scoring_fn=lambda c: 1.0 - c.conflict_pressure * (1.0 - c.emotional_regulation),
                thresholds=Thresholds(minimum=0.58, optimal=0.86),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.58,
    )
)

_register(
    Protocol(
        name="PROTO-08: REINSTATE",
        description="Restore systemic health after conflict resolution.",
        subprotocols=[
            Subprotocol(
                name="R.E.S.T.",
                description="Resilient Equilibrium Stabilisation Therapy",
                scoring_fn=lambda c: (c.restoration_index + c.harmony_index) / 2.0,
                thresholds=Thresholds(minimum=0.6, optimal=0.88),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.6,
    )
)

_register(
    Protocol(
        name="PROTO-09: COMMAND",
        description="Maintain sovereign governance and objective clarity.",
        subprotocols=[
            Subprotocol(
                name="E.N.D.",
                description="Executive Navigational Directive",
                scoring_fn=lambda c: (c.governance_clarity + c.intent_coherence) / 2.0,
                thresholds=Thresholds(minimum=0.62, optimal=0.9),
            ),
            Subprotocol(
                name="I.A.M.",
                description="Identity Alignment Mandate",
                scoring_fn=lambda c: (c.governance_clarity + c.awareness) / 2.0,
                thresholds=Thresholds(minimum=0.6, optimal=0.88),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.62,
    )
)

_register(
    Protocol(
        name="PROTO-10: EVOLVE",
        description="Drive adaptive growth while monitoring safety differentials.",
        subprotocols=[
            Subprotocol(
                name="G.R.O.W.",
                description="Gradual Resilience Optimisation Workflow",
                scoring_fn=lambda c: (c.adaptivity + c.intent_coherence + c.awareness) / 3.0,
                thresholds=Thresholds(minimum=0.58, optimal=0.9),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.58,
    )
)

_register(
    Protocol(
        name="PROTO-11: BIND",
        description="Ensure multi-agent trust formation and retention.",
        subprotocols=[
            Subprotocol(
                name="L.O.V.E.",
                description="Longitudinal Obligation and Value Exchange",
                scoring_fn=lambda c: (c.trust_factor + c.harmony_index) / 2.0,
                thresholds=Thresholds(minimum=0.62, optimal=0.9),
            ),
        ],
        aggregate=_mean,
        success_threshold=0.62,
    )
)

_register(
    Protocol(
        name="PROTO-12: RECONCILE",
        description="Unify global state and salvage fragmented subsystems.",
        subprotocols=[
            Subprotocol(
                name="S.H.A.R.D.",
                description="Systemic Harmonic Assimilation Recovery Driver",
                scoring_fn=lambda c: 1.0 - c.fragment_load,
                thresholds=Thresholds(minimum=0.55, optimal=0.85),
            ),
            Subprotocol(
                name="R.I.S.E.",
                description="Regenerative Integrity Safeguard Engine",
                scoring_fn=lambda c: (c.fragment_recovery + c.core_resilience) / 2.0,
                thresholds=Thresholds(minimum=0.6, optimal=0.88),
            ),
        ],
        aggregate=_weighted_mean([0.4, 0.6]),
        success_threshold=0.6,
    )
)


def run_all_protocols(context: AlignmentContext) -> List[ProtocolResult]:
    """Run every master protocol sequentially against the provided context."""

    return [protocol.run(context) for protocol in MASTER_PROTOCOLS]
