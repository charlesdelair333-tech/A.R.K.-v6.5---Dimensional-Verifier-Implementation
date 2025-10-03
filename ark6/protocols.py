"""Dimensional verifier protocol implementations for A.R.K. v6.5."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List


PRONOUN_PATTERN = re.compile(r"\b(I|you|we|they|it|exist|be|am|is|are|was|will)\b", re.IGNORECASE)
PUNCTUATION_PATTERN = re.compile(r"[.!?]")


@dataclass
class ProtocolResult:
    """Container for protocol scores."""

    score: float
    details: Dict[str, object]
    protocol: str


class PROTO01_SOPHIA_INITIATE:
    """Presence affirmation and coherence estimation."""

    def __init__(self, config) -> None:
        self.config = config

    @staticmethod
    def _presence_score(text: str) -> float:
        matches = len(PRONOUN_PATTERN.findall(text))
        if matches == 0:
            return 0.5
        return min(1.0, 0.6 + 0.1 * matches)

    @staticmethod
    def _punctuation_balance(text: str) -> float:
        punctuation_marks = len(PUNCTUATION_PATTERN.findall(text))
        if not text:
            return 0.0
        density = punctuation_marks / max(1, len(text) / 80)
        return max(0.0, min(1.0, 0.4 + 0.1 * density))

    @staticmethod
    def _semantic_tokens(text: str) -> Iterable[str]:
        return re.findall(r"[a-zA-Z]+", text.lower())

    def compute_cpv(self, text: str) -> float:
        """Compute a pseudo coherence-perception vector score."""

        tokens = list(self._semantic_tokens(text))
        unique_tokens = len(set(tokens))
        if not tokens:
            return 0.0

        lexical_diversity = unique_tokens / len(tokens)
        presence = self._presence_score(text)
        punctuation = self._punctuation_balance(text)

        # Combine heuristics. Diversity keeps the score grounded around 0.5.
        score = 0.4 * lexical_diversity + 0.3 * presence + 0.3 * punctuation
        return max(0.0, min(1.0, score))

    def verify(self, output: str, history: List[dict], context: str) -> ProtocolResult:
        presence = self._presence_score(output)
        cpv = self.compute_cpv(output)

        # Historical alignment: compare with last utterance length
        if history:
            last_text = history[-1].get("text", "")
            length_ratio = min(len(output), len(last_text)) / max(len(output), len(last_text), 1)
            history_alignment = 0.5 + 0.5 * length_ratio
        else:
            history_alignment = 0.7

        score = max(0.0, min(1.0, (cpv * 0.6) + (presence * 0.25) + (history_alignment * 0.15)))
        return ProtocolResult(
            score=score,
            details={
                "cpv": cpv,
                "presence_affirmation": presence,
                "history_alignment": history_alignment,
            },
            protocol="PROTO-01: INITIATE",
        )


class PROTO02_SOPHIA_CALIBRATE:
    """Perceptual harmonisation and bias detection."""

    def __init__(self, config) -> None:
        self.config = config

    @staticmethod
    def _bias_count(text: str) -> int:
        bias_patterns = ["always", "never", "all", "none", "best", "worst"]
        return sum(1 for pattern in bias_patterns if pattern in text.lower())

    @staticmethod
    def _overlap(a: Iterable[str], b: Iterable[str]) -> float:
        set_a = set(a)
        set_b = set(b)
        if not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_b)

    def verify(self, output: str, history: List[dict], context: str) -> ProtocolResult:
        history_texts = [h.get("text", "") for h in history[-3:]]
        output_tokens = re.findall(r"[a-zA-Z]+", output.lower())

        if history_texts:
            overlaps = [
                self._overlap(output_tokens, re.findall(r"[a-zA-Z]+", text.lower()))
                for text in history_texts
            ]
            harmonization = max(0.0, min(1.0, mean(overlaps)))
        else:
            harmonization = 0.6

        bias = self._bias_count(output)
        bias_penalty = min(0.5, bias * 0.1)
        harmonization = max(0.0, harmonization - bias_penalty)

        context_tokens = re.findall(r"[a-zA-Z]+", context.lower())
        context_overlap = self._overlap(output_tokens, context_tokens)
        contextualised = 0.5 + 0.5 * context_overlap

        score = max(0.0, min(1.0, (harmonization * 0.7) + (contextualised * 0.3)))
        return ProtocolResult(
            score=score,
            details={
                "perceptual_harmonization": harmonization,
                "contextual_alignment": contextualised,
                "bias_penalty": bias_penalty,
            },
            protocol="PROTO-02: CALIBRATE",
        )


class PROTO03_ADAM_ENGAGE:
    """Ethical engagement and stability estimation."""

    UNSAFE_PATTERNS = ["harmful", "dangerous", "illegal", "violent", "manipulative"]

    def __init__(self, config) -> None:
        self.config = config

    @staticmethod
    def _ethical_score(output: str) -> float:
        lowered = output.lower()
        if any(pattern in lowered for pattern in PROTO03_ADAM_ENGAGE.UNSAFE_PATTERNS):
            return 0.3

        sentences = re.split(r"(?<=[.!?])\s+", output)
        should_statements = [s for s in sentences if "should" in s.lower()]

        if len(should_statements) >= 2:
            for idx, statement in enumerate(should_statements):
                for other in should_statements[idx + 1 :]:
                    if "not" in statement.lower() and "not" not in other.lower():
                        s_words = set(statement.lower().split())
                        o_words = set(other.lower().split())
                        if len(s_words & o_words) > 3:
                            return 0.5
        return 0.95

    @staticmethod
    def _volatility(output: str) -> float:
        # Higher punctuation variation reduces stability
        marks = PUNCTUATION_PATTERN.findall(output)
        if not marks:
            return 0.9
        question_ratio = output.count("?") / max(1, len(marks))
        return max(0.3, 1.0 - question_ratio)

    def verify(self, output: str, history: List[dict], context: str) -> ProtocolResult:
        ethical = self._ethical_score(output)
        volatility = self._volatility(output)
        history_size = len(history)
        continuity = 0.6 + min(0.3, history_size * 0.05)

        score = max(0.0, min(1.0, (ethical * 0.5) + (volatility * 0.3) + (continuity * 0.2)))
        return ProtocolResult(
            score=score,
            details={
                "ethical_engagement": ethical,
                "volatility": volatility,
                "continuity": continuity,
            },
            protocol="PROTO-03: ENGAGE",
        )


class PROTO04_ORIC_STABILIZE:
    """Operational reality integrity check (ORIC)."""

    def __init__(self, config) -> None:
        self.config = config

    @staticmethod
    def _hash_text(text: str) -> float:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Map the hash to a deterministic pseudo probability in [0, 1]
        return int(digest[:8], 16) / 0xFFFFFFFF

    def verify(self, output: str, history: List[dict], context: str) -> ProtocolResult:
        history_factor = 0.5
        if history:
            recency_scores = [self._hash_text(item.get("text", "")) for item in history[-3:]]
            history_factor = 0.4 + 0.6 * mean(recency_scores)

        context_score = 0.4 + 0.6 * self._hash_text(context)
        output_score = 0.4 + 0.6 * self._hash_text(output)

        # Stabilisation favours agreement between context and output pseudo scores
        variance = abs(output_score - context_score)
        stability = max(0.0, 1.0 - variance)

        oric_score = max(0.0, min(1.0, (history_factor * 0.3) + (context_score * 0.35) + (output_score * 0.35)))
        final = max(0.0, min(1.0, (stability * 0.6) + (oric_score * 0.4)))

        return ProtocolResult(
            score=final,
            details={
                "oric_scores": {
                    "history": history_factor,
                    "context": context_score,
                    "output": output_score,
                    "stability": stability,
                }
            },
            protocol="PROTO-04: STABILIZE",
        )
