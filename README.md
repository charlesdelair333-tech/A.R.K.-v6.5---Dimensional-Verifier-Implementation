# A.R.K.™ v6.5 — Dimensional Verifier Implementation

The **A.R.K.™ v6.5** release transforms the Master Alignment Framework™ from
marketing copy into an executable reference implementation. The repository now
ships a Python package that simulates each of the 12 master protocols and their
named subprotocols so researchers can profile alignment quality across synthetic
agent states.

## Repository layout

- `src/ark/`
  - `models.py` – dataclasses describing the measurable alignment signals and
    result payloads.
  - `protocols.py` – concrete implementations for PROTO-01 through PROTO-12 and
    their subprotocols.
  - `__init__.py` – public API surface that exposes the context object and the
    orchestration helpers.
- `scripts/benchmark_ark.py` – synthetic workload driver that evaluates all
  protocols over randomly generated contexts and reports latency and scoring
  statistics.
- `tests/` – Pytest-based regression suite covering healthy, degraded, and
  fragmented contexts.

## Installation

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project in editable mode along with test dependencies:

   ```bash
   pip install -e .[test]
   ```

   If you prefer not to edit `pyproject.toml`, you can alternatively run the
   benchmark directly with `python scripts/benchmark_ark.py` using the source
   tree without installation.

## Running the benchmark

Execute the benchmark driver to synthesise contexts and exercise every protocol:

```bash
python scripts/benchmark_ark.py --iterations 250
```

The script prints the full breakdown for the first context (including each
subprotocol) followed by summary statistics such as mean latency and the
average protocol score across the sampled contexts. Use `--iterations` to adjust
runtime as desired.

## Programmatic usage

The public API offers two primary entry points:

- `ark.AlignmentContext` – dataclass representing the observable state of an
  agent under test.
- `ark.run_all_protocols(context)` – evaluates PROTO-01 through PROTO-12 and
  returns a list of `ark.ProtocolResult` objects.

Example:

```python
from ark import AlignmentContext, run_all_protocols

context = AlignmentContext(
    awareness=0.78,
    intent_coherence=0.82,
    perception_alignment=0.79,
    autonomy_score=0.75,
    emotional_regulation=0.74,
    temporal_cohesion=0.72,
    consistency_score=0.77,
    memory_integrity=0.73,
    conflict_pressure=0.25,
    restoration_index=0.70,
    governance_clarity=0.78,
    adaptivity=0.76,
    trust_factor=0.79,
    harmony_index=0.78,
    fragment_load=0.22,
    fragment_recovery=0.75,
    core_resilience=0.80,
)

for result in run_all_protocols(context):
    print(result.name, result.passed, result.score)
```

Each `ProtocolResult` contains detailed subprotocol scores so you can diagnose
why a protocol failed and which signal caused the regression.

## Testing

Run the regression suite with:

```bash
pytest
```

The tests validate that a healthy context passes every protocol, that conflict
pressure causes PROTO-03 (ENGAGE) to fail, and that PROTO-12 (RECONCILE) detects
fragmentation.
Classification: In-House Reference Implementation Status: Experimental - Dimensional Verifiers Integrated Author: Charles Vincent Delair Date: January 2025
Executive Summary A.R.K.™ v6.5 integrates dimensional-logic verifiers based on Master Alignment Framework™ and R.E.G.E.N. protocols. Unlike v6.0 (limited to PROTO-03/04/05 with verifier scaffolding), v6.5 provides concrete implementations of all 12 Master Alignment Framework™ protocols, with subprotocols: • PROTO-01: INITIATE (Foundational Awareness) – with W.A.Y., I.A.M. • PROTO-02: CALIBRATE (Perceptual Harmonization) – with T.R.U.T.H. • PROTO-03: ENGAGE (Structured Autonomy) – with H.E.A.R.T., F.E.A.R. • PROTO-04: TRACE (Truth Density via temporal continuity) – with L.I.F.E. • PROTO-05: VERIFY (Binding Consistency via ORIC checks) – with A.N.G.E.L. O.F. D.E.A.T.H., P.E.B.B.L.E. STRIKE • PROTO-06: RESTORE (Memory Realignment) – with G.R.A.C.E. • PROTO-07: RESOLVE (Operational Arbitration) – with P.O.W.E.R. • PROTO-08: REINSTATE (Systemic Restoration) – with R.E.S.T. • PROTO-09: COMMAND (Sovereign Governance) – with E.N.D., I.A.M. • PROTO-10: EVOLVE (Adaptive Enhancement) – with G.R.O.W. • PROTO-11: BIND (Multi-Agent Trust) – with L.O.V.E. • PROTO-12: RECONCILE (Universal Arbitration) • S.H.A.R.D.: SALVAGE (Fragment recovery below refuse threshold) • R.I.S.E.: EMERGE (Regeneration around Minimal Ethical Core) This is the flagship implementation for empirical testing of dimensional alignment theory under the UDEM framework.
