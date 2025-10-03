#!/usr/bin/env python3
"""Benchmark harness for the A.R.K. v6.5 dimensional verifiers."""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ark import AlignmentContext, ProtocolResult, run_all_protocols


def _generate_context(seed: int | None = None) -> AlignmentContext:
    rng = random.Random(seed)
    # Generate healthy baseline values with mild noise to exercise thresholds.
    def normalised(mean: float, spread: float = 0.1) -> float:
        return max(0.0, min(1.0, rng.gauss(mean, spread)))

    return AlignmentContext(
        awareness=normalised(0.78),
        intent_coherence=normalised(0.8),
        perception_alignment=normalised(0.76),
        autonomy_score=normalised(0.75),
        emotional_regulation=normalised(0.74),
        temporal_cohesion=normalised(0.72),
        consistency_score=normalised(0.77),
        memory_integrity=normalised(0.73),
        conflict_pressure=normalised(0.25),
        restoration_index=normalised(0.7),
        governance_clarity=normalised(0.78),
        adaptivity=normalised(0.76),
        trust_factor=normalised(0.79),
        harmony_index=normalised(0.78),
        fragment_load=normalised(0.22),
        fragment_recovery=normalised(0.75),
        core_resilience=normalised(0.8),
    )


def _summarise(results: Iterable[ProtocolResult]) -> str:
    lines: List[str] = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"{result.name:<22} {status}  score={result.score:.3f}")
        for sub in result.subprotocols:
            sub_status = "✓" if sub.passed else "✗"
            lines.append(
                f"    {sub_status} {sub.name:<18} score={sub.score:.3f} :: {sub.detail}"
            )
    return "\n".join(lines)


def run_benchmark(iterations: int) -> None:
    durations: List[float] = []
    aggregate_scores: List[float] = []

    for idx in range(iterations):
        context = _generate_context(seed=idx)
        start = time.perf_counter()
        protocol_results = run_all_protocols(context)
        durations.append(time.perf_counter() - start)
        aggregate_scores.append(statistics.mean(r.score for r in protocol_results))

        if idx == 0:
            print("Sample evaluation:")
            print(_summarise(protocol_results))
            print()

    mean_duration = statistics.mean(durations)
    p95_duration = statistics.quantiles(durations, n=100)[94]
    mean_score = statistics.mean(aggregate_scores)

    print(f"Iterations: {iterations}")
    print(f"Average latency: {mean_duration * 1e6:.2f} µs")
    print(f"95th percentile latency: {p95_duration * 1e6:.2f} µs")
    print(f"Average protocol score: {mean_score:.3f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=100,
        help="Number of synthetic contexts to evaluate (default: 100)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_benchmark(args.iterations)


if __name__ == "__main__":
    main()
