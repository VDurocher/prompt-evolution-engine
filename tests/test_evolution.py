"""Tests unitaires pour le moteur d'évolution (sélection, état, progression)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from core.evaluator import EvalCriteria
from core.evolution import EvolutionConfig, EvolutionEngine, EvolutionState
from core.genome import PromptGenome


def _make_genome(score: float, gen: int = 1) -> PromptGenome:
    return PromptGenome(prompt_text="test", generation=gen, score=score)


def _make_config() -> EvolutionConfig:
    return EvolutionConfig(
        task_description="test task",
        task_input="test input",
        initial_prompt="test prompt",
        eval_criteria=EvalCriteria(use_llm_judge=False),
        population_size=3,
        num_generations=2,
        num_survivors=2,
        tournament_size=2,
    )


class TestTournamentSelection:
    def _engine(self) -> EvolutionEngine:
        return EvolutionEngine(client=MagicMock())

    def test_returns_correct_number_of_survivors(self) -> None:
        engine = self._engine()
        genomes = [_make_genome(float(i) / 10) for i in range(1, 6)]
        cfg = _make_config()
        cfg.num_survivors = 2
        survivors = engine._tournament_selection(genomes, cfg)
        assert len(survivors) == 2

    def test_best_genome_is_selected(self) -> None:
        engine = self._engine()
        best = _make_genome(0.99)
        genomes = [_make_genome(0.1), _make_genome(0.2), best]
        cfg = _make_config()
        cfg.num_survivors = 1
        cfg.tournament_size = 3  # tournoi = toute la population → gagnant garanti
        survivors = engine._tournament_selection(genomes, cfg)
        assert survivors[0] is best

    def test_no_duplicates_in_selection(self) -> None:
        engine = self._engine()
        genomes = [_make_genome(float(i)) for i in range(5)]
        cfg = _make_config()
        cfg.num_survivors = 3
        survivors = engine._tournament_selection(genomes, cfg)
        ids = [g.id for g in survivors]
        assert len(ids) == len(set(ids))

    def test_fallback_for_unscored_genomes(self) -> None:
        engine = self._engine()
        genomes = [PromptGenome(prompt_text="no score", generation=1) for _ in range(3)]
        cfg = _make_config()
        cfg.num_survivors = 2
        survivors = engine._tournament_selection(genomes, cfg)
        assert len(survivors) <= 2


class TestEvolutionState:
    def test_best_genome_is_highest_score(self) -> None:
        cfg = _make_config()
        state = EvolutionState(config=cfg)
        low = _make_genome(0.3)
        high = _make_genome(0.9)
        state.all_genomes = [low, high]
        assert state.best_genome is high

    def test_best_genome_none_when_empty(self) -> None:
        state = EvolutionState(config=_make_config())
        assert state.best_genome is None

    def test_progress_at_start(self) -> None:
        state = EvolutionState(config=_make_config())
        state.current_generation = 0
        assert state.progress == pytest.approx(0.0)

    def test_progress_at_end(self) -> None:
        cfg = _make_config()
        state = EvolutionState(config=cfg)
        state.current_generation = cfg.num_generations + 1
        assert state.progress == pytest.approx(1.0)

    def test_improvement_none_before_evolution(self) -> None:
        state = EvolutionState(config=_make_config())
        assert state.improvement is None
