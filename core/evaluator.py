"""
Evaluator: scores an LLM response along two combinable dimensions.
- Deterministic: heuristics with no API call (keywords, JSON, length, regex)
- LLM-as-judge: GPT-4o-mini scores the response against user-defined criteria
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are an LLM response evaluator. "
    "You score responses accurately and objectively according to the provided criteria."
)

_JUDGE_USER_TEMPLATE = """\
Evaluate the response below according to the provided criteria.

Task: {task_description}
Evaluation criteria: {criteria_description}

Response to evaluate:
{response}

Assign a score from 0.0 to 1.0 (two decimal places).
Return ONLY valid JSON:
{{"score": 0.85, "reasoning": "explanation in 1-2 concise sentences"}}\
"""


@dataclass
class EvalCriteria:
    """User-configurable evaluation criteria."""

    # LLM-as-judge
    llm_judge_description: str = ""
    use_llm_judge: bool = True

    # Deterministic
    required_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    require_valid_json: bool = False
    target_length_range: Optional[tuple[int, int]] = None  # (min_chars, max_chars)
    regex_pattern: str = ""

    @property
    def has_deterministic(self) -> bool:
        """Indicates whether at least one deterministic criterion is configured."""
        return bool(
            self.required_keywords
            or self.forbidden_keywords
            or self.require_valid_json
            or self.target_length_range
            or self.regex_pattern
        )


class Evaluator:
    """
    Computes a composite score for an LLM response.
    Final score = weighted average of deterministic (0.35) + LLM-judge (0.65).
    """

    def __init__(self, client: OpenAI, criteria: EvalCriteria) -> None:
        self._client = client
        self._criteria = criteria

    def score(
        self,
        response: str,
        task_description: str,
    ) -> tuple[float, dict[str, float | str]]:
        """
        Returns (final_score, details_per_dimension).
        final_score is always between 0.0 and 1.0.
        """
        details: dict[str, float | str] = {}
        component_scores: dict[str, float] = {}
        weights: dict[str, float] = {}

        # --- Deterministic dimension ---
        if self._criteria.has_deterministic:
            det_score = self._deterministic_score(response)
            details["deterministic"] = det_score
            component_scores["deterministic"] = det_score
            weights["deterministic"] = config.DETERMINISTIC_WEIGHT

        # --- LLM-as-judge dimension ---
        if self._criteria.use_llm_judge and self._criteria.llm_judge_description:
            llm_score, reasoning = self._llm_judge_score(response, task_description)
            details["llm_judge"] = llm_score
            details["llm_reasoning"] = reasoning
            component_scores["llm_judge"] = llm_score
            weights["llm_judge"] = config.LLM_JUDGE_WEIGHT

        # No criteria configured → neutral score
        if not weights:
            return 0.5, details

        total_weight = sum(weights.values())
        final_score = sum(
            component_scores[k] * weights[k] / total_weight
            for k in weights
        )

        return round(final_score, 4), details

    def _deterministic_score(self, response: str) -> float:
        """Heuristic score — no API call."""
        sub_scores: list[float] = []
        crit = self._criteria
        response_lower = response.lower()

        # Required keywords
        if crit.required_keywords:
            hits = sum(1 for kw in crit.required_keywords if kw.lower() in response_lower)
            sub_scores.append(hits / len(crit.required_keywords))

        # Forbidden keywords
        if crit.forbidden_keywords:
            violations = sum(1 for kw in crit.forbidden_keywords if kw.lower() in response_lower)
            sub_scores.append(max(0.0, 1.0 - violations / len(crit.forbidden_keywords)))

        # Valid JSON
        if crit.require_valid_json:
            try:
                json.loads(response)
                sub_scores.append(1.0)
            except (json.JSONDecodeError, ValueError):
                sub_scores.append(0.0)

        # Target length
        if crit.target_length_range:
            min_len, max_len = crit.target_length_range
            length = len(response)
            if min_len <= length <= max_len:
                sub_scores.append(1.0)
            elif length < min_len:
                sub_scores.append(max(0.0, length / min_len))
            else:
                sub_scores.append(max(0.0, 1.0 - (length - max_len) / max_len))

        # Regex pattern
        if crit.regex_pattern:
            try:
                matched = bool(re.search(crit.regex_pattern, response, re.IGNORECASE | re.DOTALL))
                sub_scores.append(1.0 if matched else 0.0)
            except re.error as error:
                logger.warning("Invalid regex: %s", error)

        if not sub_scores:
            return 0.5
        return round(sum(sub_scores) / len(sub_scores), 4)

    def _llm_judge_score(self, response: str, task_description: str) -> tuple[float, str]:
        """
        LLM-based evaluation.
        Returns (score 0.0-1.0, text reasoning).
        """
        user_msg = _JUDGE_USER_TEMPLATE.format(
            task_description=task_description,
            criteria_description=self._criteria.llm_judge_description,
            response=response[:2000],  # limit to avoid excessive tokens
        )

        try:
            result = self._client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=config.JUDGE_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = result.choices[0].message.content or "{}"
            data = json.loads(raw)
            score = float(data.get("score", 0.5))
            reasoning = str(data.get("reasoning", ""))
            return max(0.0, min(1.0, score)), reasoning
        except Exception as error:
            logger.error("LLM judge failed: %s", error)
            return 0.5, "Evaluation unavailable"
