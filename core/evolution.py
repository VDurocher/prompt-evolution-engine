"""
Evolution engine: orchestrates mutations, executions, scoring and selection.
Implemented as a generator to allow live updates in Streamlit.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Generator
from dataclasses import dataclass, field

from openai import OpenAI

import config
from core.evaluator import EvalCriteria, Evaluator
from core.executor import Executor
from core.genome import GenerationResult, PromptGenome
from core.mutator import Mutator

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Complete parameters for an evolution session."""

    task_description: str       # description of the target task
    task_input: str             # concrete input on which prompts are tested
    initial_prompt: str         # starting point of the evolution
    eval_criteria: EvalCriteria # scoring criteria

    population_size: int = config.DEFAULT_POPULATION_SIZE
    num_generations: int = config.DEFAULT_NUM_GENERATIONS
    num_survivors: int = config.DEFAULT_NUM_SURVIVORS
    tournament_size: int = config.DEFAULT_TOURNAMENT_SIZE


@dataclass
class EvolutionState:
    """Complete state of an evolution session — snapshot after each generation."""

    config: EvolutionConfig
    all_genomes: list[PromptGenome] = field(default_factory=list)
    generation_results: list[GenerationResult] = field(default_factory=list)
    current_generation: int = 0
    is_complete: bool = False

    @property
    def best_genome(self) -> PromptGenome | None:
        """Best genome across all generations."""
        scored = [g for g in self.all_genomes if g.score is not None]
        if not scored:
            return None
        return max(scored, key=lambda g: g.score or 0.0)

    @property
    def progress(self) -> float:
        """Progress between 0.0 and 1.0."""
        total = self.config.num_generations + 1
        return min(1.0, self.current_generation / total)

    @property
    def improvement(self) -> float | None:
        """Score delta between the initial prompt and the current best prompt."""
        if len(self.generation_results) < 1:
            return None
        initial_score = self.generation_results[0].best_genome.score
        best = self.best_genome
        if initial_score is None or best is None or best.score is None:
            return None
        return round(best.score - initial_score, 4)


class EvolutionEngine:
    """
    Genetic prompt evolution engine.

    Algorithm:
    1. Evaluates the initial prompt (generation 0)
    2. For each generation:
       a. The Mutator generates N variants from the survivors
       b. The Executor runs each variant on the task
       c. The Evaluator scores each response
       d. Tournament selection → selects the best survivors
    3. Yields the state after each generation (Streamlit live updates)
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, cfg: EvolutionConfig) -> Generator[EvolutionState, None, None]:
        """
        Runs the full evolution.
        Yields an updated EvolutionState after each generation.
        """
        state = EvolutionState(config=cfg)
        mutator = Mutator(self._client)
        evaluator = Evaluator(self._client, cfg.eval_criteria)
        executor = Executor(self._client)

        # --- Generation 0: evaluate initial prompt ---
        logger.info("Generation 0 — evaluating initial prompt")
        initial = PromptGenome(
            prompt_text=cfg.initial_prompt,
            generation=0,
            technique_tags=["original"],
            rationale="Initial prompt provided by the user",
        )
        self._evaluate_genome(initial, executor, evaluator, cfg)
        state.all_genomes.append(initial)
        state.generation_results.append(GenerationResult.from_genomes(0, [initial]))
        state.current_generation = 1
        yield state

        survivors = [initial]

        # --- Generations 1..N ---
        for gen in range(1, cfg.num_generations + 1):
            logger.info("Generation %d — mutation + scoring", gen)

            # Mutation: generate N variants from survivors
            variants = mutator.generate_variants(
                parents=survivors,
                task_description=cfg.task_description,
                n=cfg.population_size,
                current_generation=gen,
            )

            # Execute + score each variant
            for genome in variants:
                self._evaluate_genome(genome, executor, evaluator, cfg)
                state.all_genomes.append(genome)

            # Tournament selection
            survivors = self._tournament_selection(variants, cfg)

            state.generation_results.append(GenerationResult.from_genomes(gen, variants))
            state.current_generation = gen + 1
            yield state

        state.is_complete = True
        yield state

    def _evaluate_genome(
        self,
        genome: PromptGenome,
        executor: Executor,
        evaluator: Evaluator,
        cfg: EvolutionConfig,
    ) -> None:
        """Runs the prompt on the task and scores the response. Mutates genome in place."""
        response = executor.run(genome.prompt_text, cfg.task_input)
        genome.response_sample = response
        score, details = evaluator.score(response, cfg.task_description)
        genome.score = score
        genome.score_details = details

    def _tournament_selection(
        self,
        genomes: list[PromptGenome],
        cfg: EvolutionConfig,
    ) -> list[PromptGenome]:
        """
        Tournament selection:
        - Draws tournament_size candidates at random from the pool
        - The best of the group wins and is removed from the pool
        - Repeats until num_survivors winners are selected
        """
        scored = [g for g in genomes if g.score is not None]
        if not scored:
            return genomes[: cfg.num_survivors]

        pool = list(scored)
        winners: list[PromptGenome] = []

        for _ in range(min(cfg.num_survivors, len(pool))):
            if not pool:
                break
            k = min(cfg.tournament_size, len(pool))
            candidates = random.sample(pool, k)
            winner = max(candidates, key=lambda g: g.score or 0.0)
            winners.append(winner)
            pool.remove(winner)

        return winners
