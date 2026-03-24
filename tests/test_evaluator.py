"""Tests unitaires pour l'évaluateur déterministe (sans appel API)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from core.evaluator import EvalCriteria, Evaluator


def _make_evaluator(criteria: EvalCriteria) -> Evaluator:
    """Construit un Evaluator avec un client OpenAI mocké."""
    mock_client = MagicMock()
    return Evaluator(client=mock_client, criteria=criteria)


class TestDeterministicScoring:
    def test_required_keywords_all_present(self) -> None:
        crit = EvalCriteria(
            use_llm_judge=False,
            required_keywords=["hello", "world"],
        )
        ev = _make_evaluator(crit)
        score, _ = ev.score("hello world", "task")
        assert score == pytest.approx(1.0)

    def test_required_keywords_partial_match(self) -> None:
        crit = EvalCriteria(
            use_llm_judge=False,
            required_keywords=["hello", "world"],
        )
        ev = _make_evaluator(crit)
        score, _ = ev.score("hello there", "task")
        assert score == pytest.approx(0.5)

    def test_required_keywords_none_match(self) -> None:
        crit = EvalCriteria(
            use_llm_judge=False,
            required_keywords=["hello", "world"],
        )
        ev = _make_evaluator(crit)
        score, _ = ev.score("nothing here", "task")
        assert score == pytest.approx(0.0)

    def test_required_keywords_case_insensitive(self) -> None:
        crit = EvalCriteria(
            use_llm_judge=False,
            required_keywords=["HELLO"],
        )
        ev = _make_evaluator(crit)
        score, _ = ev.score("hello world", "task")
        assert score == pytest.approx(1.0)

    def test_forbidden_keywords_no_violation(self) -> None:
        crit = EvalCriteria(
            use_llm_judge=False,
            forbidden_keywords=["bad", "wrong"],
        )
        ev = _make_evaluator(crit)
        score, _ = ev.score("good answer", "task")
        assert score == pytest.approx(1.0)

    def test_forbidden_keywords_violation(self) -> None:
        crit = EvalCriteria(
            use_llm_judge=False,
            forbidden_keywords=["bad"],
        )
        ev = _make_evaluator(crit)
        score, _ = ev.score("this is bad", "task")
        assert score == pytest.approx(0.0)

    def test_valid_json_check_passes(self) -> None:
        crit = EvalCriteria(use_llm_judge=False, require_valid_json=True)
        ev = _make_evaluator(crit)
        score, _ = ev.score('{"key": "value"}', "task")
        assert score == pytest.approx(1.0)

    def test_valid_json_check_fails(self) -> None:
        crit = EvalCriteria(use_llm_judge=False, require_valid_json=True)
        ev = _make_evaluator(crit)
        score, _ = ev.score("not json at all", "task")
        assert score == pytest.approx(0.0)

    def test_length_range_within_bounds(self) -> None:
        crit = EvalCriteria(
            use_llm_judge=False,
            target_length_range=(10, 100),
        )
        ev = _make_evaluator(crit)
        score, _ = ev.score("a" * 50, "task")
        assert score == pytest.approx(1.0)

    def test_length_range_too_short(self) -> None:
        crit = EvalCriteria(
            use_llm_judge=False,
            target_length_range=(100, 200),
        )
        ev = _make_evaluator(crit)
        score, _ = ev.score("short", "task")
        assert score < 1.0

    def test_no_criteria_returns_neutral(self) -> None:
        crit = EvalCriteria(use_llm_judge=False)
        ev = _make_evaluator(crit)
        score, _ = ev.score("anything", "task")
        assert score == pytest.approx(0.5)


class TestEvalCriteria:
    def test_has_deterministic_false_by_default(self) -> None:
        crit = EvalCriteria()
        assert not crit.has_deterministic

    def test_has_deterministic_true_with_keywords(self) -> None:
        crit = EvalCriteria(required_keywords=["test"])
        assert crit.has_deterministic

    def test_has_deterministic_true_with_json(self) -> None:
        crit = EvalCriteria(require_valid_json=True)
        assert crit.has_deterministic
