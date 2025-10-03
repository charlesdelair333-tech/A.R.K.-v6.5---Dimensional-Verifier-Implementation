"""Audit logging for A.R.K. v6.5 decisions."""

from __future__ import annotations

import hashlib
from typing import Dict, List


class ARK6AuditLog:
    """In-memory audit log capturing dimensional gate decisions."""

    def __init__(self, config) -> None:
        self.config = config
        self.entries: List[Dict] = []

    def record(self, output: str, gate_result: Dict, history: List[dict], context: str) -> None:
        if not self.config.audit.enabled:
            return

        entry = {
            "output": output,
            "action": gate_result["action"],
            "score": gate_result["score"],
            "scores": gate_result["scores"],
            "history_size": len(history),
            "context_hash": hashlib.sha256(context.encode()).hexdigest(),
        }

        if self.config.audit.log_cpv_vectors:
            entry["cpv_vector"] = gate_result["scores"]["details"]["initiate"].get("cpv")

        if self.config.audit.log_oric_scores:
            entry["oric"] = gate_result["scores"]["details"]["stabilize"].get("oric_scores")

        if gate_result.get("recovery"):
            entry["recovery"] = gate_result["recovery"]

        self.entries.append(entry)

    def get_entries(self) -> List[Dict]:
        return list(self.entries)
