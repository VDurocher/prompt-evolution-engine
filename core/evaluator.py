"""
Évaluateur : score une réponse LLM selon deux dimensions combinables.
- Déterministe : heuristiques sans appel API (mots-clés, JSON, longueur, regex)
- LLM-as-judge : GPT-4o-mini note la réponse selon des critères définis par l'utilisateur
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
    "Tu es un évaluateur de réponses LLM. "
    "Tu notes avec précision et objectivité selon les critères fournis."
)

_JUDGE_USER_TEMPLATE = """\
Évalue la réponse ci-dessous selon les critères fournis.

Tâche : {task_description}
Critères d'évaluation : {criteria_description}

Réponse à évaluer :
{response}

Attribue un score de 0.0 à 1.0 (deux décimales).
Retourne UNIQUEMENT du JSON valide :
{{"score": 0.85, "reasoning": "explication en 1-2 phrases concises"}}\
"""


@dataclass
class EvalCriteria:
    """Critères d'évaluation configurables par l'utilisateur."""

    # LLM-as-judge
    llm_judge_description: str = ""
    use_llm_judge: bool = True

    # Déterministes
    required_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    require_valid_json: bool = False
    target_length_range: Optional[tuple[int, int]] = None  # (min_chars, max_chars)
    regex_pattern: str = ""

    @property
    def has_deterministic(self) -> bool:
        """Indique si au moins un critère déterministe est configuré."""
        return bool(
            self.required_keywords
            or self.forbidden_keywords
            or self.require_valid_json
            or self.target_length_range
            or self.regex_pattern
        )


class Evaluator:
    """
    Calcule un score composite pour une réponse LLM.
    Score final = moyenne pondérée déterministe (0.35) + LLM-judge (0.65).
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
        Retourne (score_final, détails_par_dimension).
        score_final est toujours entre 0.0 et 1.0.
        """
        details: dict[str, float | str] = {}
        component_scores: dict[str, float] = {}
        weights: dict[str, float] = {}

        # --- Dimension déterministe ---
        if self._criteria.has_deterministic:
            det_score = self._deterministic_score(response)
            details["deterministic"] = det_score
            component_scores["deterministic"] = det_score
            weights["deterministic"] = config.DETERMINISTIC_WEIGHT

        # --- Dimension LLM-as-judge ---
        if self._criteria.use_llm_judge and self._criteria.llm_judge_description:
            llm_score, reasoning = self._llm_judge_score(response, task_description)
            details["llm_judge"] = llm_score
            details["llm_reasoning"] = reasoning
            component_scores["llm_judge"] = llm_score
            weights["llm_judge"] = config.LLM_JUDGE_WEIGHT

        # Aucun critère configuré → score neutre
        if not weights:
            return 0.5, details

        total_weight = sum(weights.values())
        final_score = sum(
            component_scores[k] * weights[k] / total_weight
            for k in weights
        )

        return round(final_score, 4), details

    def _deterministic_score(self, response: str) -> float:
        """Score heuristique — aucun appel API."""
        sub_scores: list[float] = []
        crit = self._criteria
        response_lower = response.lower()

        # Mots-clés requis
        if crit.required_keywords:
            hits = sum(1 for kw in crit.required_keywords if kw.lower() in response_lower)
            sub_scores.append(hits / len(crit.required_keywords))

        # Mots-clés interdits
        if crit.forbidden_keywords:
            violations = sum(1 for kw in crit.forbidden_keywords if kw.lower() in response_lower)
            sub_scores.append(max(0.0, 1.0 - violations / len(crit.forbidden_keywords)))

        # JSON valide
        if crit.require_valid_json:
            try:
                json.loads(response)
                sub_scores.append(1.0)
            except (json.JSONDecodeError, ValueError):
                sub_scores.append(0.0)

        # Longueur cible
        if crit.target_length_range:
            min_len, max_len = crit.target_length_range
            length = len(response)
            if min_len <= length <= max_len:
                sub_scores.append(1.0)
            elif length < min_len:
                sub_scores.append(max(0.0, length / min_len))
            else:
                sub_scores.append(max(0.0, 1.0 - (length - max_len) / max_len))

        # Pattern regex
        if crit.regex_pattern:
            try:
                matched = bool(re.search(crit.regex_pattern, response, re.IGNORECASE | re.DOTALL))
                sub_scores.append(1.0 if matched else 0.0)
            except re.error as error:
                logger.warning("Regex invalide : %s", error)

        if not sub_scores:
            return 0.5
        return round(sum(sub_scores) / len(sub_scores), 4)

    def _llm_judge_score(self, response: str, task_description: str) -> tuple[float, str]:
        """
        Évaluation par LLM.
        Retourne (score 0.0-1.0, raisonnement texte).
        """
        user_msg = _JUDGE_USER_TEMPLATE.format(
            task_description=task_description,
            criteria_description=self._criteria.llm_judge_description,
            response=response[:2000],  # limite pour éviter les tokens excessifs
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
            logger.error("Échec du LLM judge : %s", error)
            return 0.5, "Évaluation indisponible"
