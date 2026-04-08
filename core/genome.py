"""
Core data structures for the evolution engine.
A PromptGenome represents a prompt at a given point in the evolution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptGenome:
    """
    Atomic unit of evolution: a prompt with all its metadata.
    Knows where it came from (parent_ids) and how it was created (technique_tags).
    """

    prompt_text: str
    generation: int
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_ids: list[str] = field(default_factory=list)
    technique_tags: list[str] = field(default_factory=list)
    rationale: str = ""                          # explanation of the mutation
    score: Optional[float] = None               # None = not yet evaluated
    score_details: dict[str, float] = field(default_factory=dict)
    response_sample: Optional[str] = None       # response produced during evaluation

    @property
    def is_scored(self) -> bool:
        return self.score is not None

    @property
    def score_display(self) -> str:
        """Score formatted for display."""
        if self.score is None:
            return "—"
        return f"{self.score:.3f}"

    def short_text(self, max_chars: int = 100) -> str:
        """Truncated version of the prompt for compact display."""
        if len(self.prompt_text) <= max_chars:
            return self.prompt_text
        return self.prompt_text[:max_chars] + "…"


@dataclass
class GenerationResult:
    """Statistical aggregate for a complete generation."""

    generation: int
    genomes: list[PromptGenome]
    best_genome: PromptGenome
    mean_score: float
    max_score: float
    min_score: float

    @classmethod
    def from_genomes(cls, generation: int, genomes: list[PromptGenome]) -> "GenerationResult":
        """Builds a result from a list of scored genomes."""
        scored = [g for g in genomes if g.score is not None]
        if not scored:
            # Fallback case: all genomes unscored
            placeholder = genomes[0] if genomes else PromptGenome(prompt_text="", generation=generation)
            return cls(
                generation=generation,
                genomes=genomes,
                best_genome=placeholder,
                mean_score=0.0,
                max_score=0.0,
                min_score=0.0,
            )

        scores = [g.score for g in scored if g.score is not None]
        best = max(scored, key=lambda g: g.score or 0.0)

        return cls(
            generation=generation,
            genomes=genomes,
            best_genome=best,
            mean_score=round(sum(scores) / len(scores), 4),
            max_score=round(max(scores), 4),
            min_score=round(min(scores), 4),
        )
