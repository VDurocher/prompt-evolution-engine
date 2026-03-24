"""Tests unitaires pour les structures de données de base."""

from __future__ import annotations

import pytest
from core.genome import GenerationResult, PromptGenome


class TestPromptGenome:
    def test_default_id_generated(self) -> None:
        g = PromptGenome(prompt_text="test", generation=0)
        assert len(g.id) == 8

    def test_ids_are_unique(self) -> None:
        g1 = PromptGenome(prompt_text="a", generation=0)
        g2 = PromptGenome(prompt_text="a", generation=0)
        assert g1.id != g2.id

    def test_is_scored_false_by_default(self) -> None:
        g = PromptGenome(prompt_text="test", generation=0)
        assert not g.is_scored

    def test_is_scored_true_after_assignment(self) -> None:
        g = PromptGenome(prompt_text="test", generation=0, score=0.85)
        assert g.is_scored

    def test_score_display_none(self) -> None:
        g = PromptGenome(prompt_text="test", generation=0)
        assert g.score_display == "—"

    def test_score_display_formatted(self) -> None:
        g = PromptGenome(prompt_text="test", generation=0, score=0.8)
        assert g.score_display == "0.800"

    def test_short_text_no_truncation(self) -> None:
        text = "short"
        g = PromptGenome(prompt_text=text, generation=0)
        assert g.short_text(100) == text

    def test_short_text_truncated(self) -> None:
        text = "a" * 200
        g = PromptGenome(prompt_text=text, generation=0)
        result = g.short_text(50)
        assert result.endswith("…")
        assert len(result) == 51  # 50 chars + ellipsis

    def test_parent_ids_default_empty(self) -> None:
        g = PromptGenome(prompt_text="test", generation=0)
        assert g.parent_ids == []

    def test_technique_tags_default_empty(self) -> None:
        g = PromptGenome(prompt_text="test", generation=0)
        assert g.technique_tags == []


class TestGenerationResult:
    def _make_genome(self, score: float, gen: int = 1) -> PromptGenome:
        return PromptGenome(prompt_text="test", generation=gen, score=score)

    def test_from_genomes_computes_stats(self) -> None:
        genomes = [self._make_genome(0.2), self._make_genome(0.6), self._make_genome(0.8)]
        result = GenerationResult.from_genomes(1, genomes)
        assert result.max_score == pytest.approx(0.8)
        assert result.min_score == pytest.approx(0.2)
        assert result.mean_score == pytest.approx(0.5333, abs=1e-3)

    def test_from_genomes_selects_best(self) -> None:
        best = self._make_genome(0.95)
        genomes = [self._make_genome(0.3), best, self._make_genome(0.5)]
        result = GenerationResult.from_genomes(1, genomes)
        assert result.best_genome is best

    def test_from_genomes_empty_fallback(self) -> None:
        genome = PromptGenome(prompt_text="no score", generation=1)
        result = GenerationResult.from_genomes(1, [genome])
        assert result.max_score == 0.0
        assert result.mean_score == 0.0

    def test_generation_stored(self) -> None:
        genomes = [self._make_genome(0.5)]
        result = GenerationResult.from_genomes(3, genomes)
        assert result.generation == 3
