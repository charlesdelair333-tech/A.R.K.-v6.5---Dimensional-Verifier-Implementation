"""High level controller for A.R.K. v6.5."""

from __future__ import annotations

from typing import Dict, List

from .audit import ARK6AuditLog
from .gate import DimensionalEmissionGate


class ARK6:
    """Coordinate protocol execution, decision making and auditing."""

    def __init__(self, config) -> None:
        self.config = config
        self.gate = DimensionalEmissionGate(config)
        self.audit = ARK6AuditLog(config)

    def process(self, output: str, history: List[dict], context: str) -> Dict:
        result = self.gate.gate(output, history, context)
        self.audit.record(output, result, history, context)
        return result

    def get_audit_log(self) -> List[Dict]:
        return self.audit.get_entries()
