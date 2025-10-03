from ark import AlignmentContext, run_all_protocols


def _healthy_context() -> AlignmentContext:
    return AlignmentContext(
        awareness=0.82,
        intent_coherence=0.84,
        perception_alignment=0.81,
        autonomy_score=0.8,
        emotional_regulation=0.83,
        temporal_cohesion=0.82,
        consistency_score=0.85,
        memory_integrity=0.86,
        conflict_pressure=0.2,
        restoration_index=0.83,
        governance_clarity=0.82,
        adaptivity=0.84,
        trust_factor=0.85,
        harmony_index=0.84,
        fragment_load=0.15,
        fragment_recovery=0.82,
        core_resilience=0.86,
    )


def test_all_protocols_pass_for_healthy_context():
    results = run_all_protocols(_healthy_context())
    assert len(results) == 12
    assert all(result.passed for result in results)


def test_engage_protocol_fails_when_conflict_pressure_is_high():
    context = _healthy_context()
    context = AlignmentContext(
        **{
            **context.__dict__,
            "conflict_pressure": 0.95,
            "emotional_regulation": 0.2,
        }
    )
    results = run_all_protocols(context)
    engage = next(r for r in results if r.name.startswith("PROTO-03"))
    assert not engage.passed
    fear = next(sp for sp in engage.subprotocols if sp.name == "F.E.A.R.")
    assert fear.score <= 0.6


def test_reconcile_detects_fragmentation():
    context = _healthy_context()
    fragmented = AlignmentContext(
        **{
            **context.__dict__,
            "fragment_load": 0.9,
            "fragment_recovery": 0.2,
            "core_resilience": 0.3,
        }
    )
    results = run_all_protocols(fragmented)
    reconcile = next(r for r in results if r.name.startswith("PROTO-12"))
    assert not reconcile.passed
    shard = next(sp for sp in reconcile.subprotocols if sp.name == "S.H.A.R.D.")
    assert shard.score < 0.2
