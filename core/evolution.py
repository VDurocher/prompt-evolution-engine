"""
Moteur d'évolution : orchestre mutations, exécutions, scoring et sélection.
Implémenté comme générateur pour permettre les mises à jour live dans Streamlit.
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
    """Paramètres complets d'une session d'évolution."""

    task_description: str       # description de la tâche cible
    task_input: str             # entrée concrète sur laquelle les prompts sont testés
    initial_prompt: str         # point de départ de l'évolution
    eval_criteria: EvalCriteria # critères de scoring

    population_size: int = config.DEFAULT_POPULATION_SIZE
    num_generations: int = config.DEFAULT_NUM_GENERATIONS
    num_survivors: int = config.DEFAULT_NUM_SURVIVORS
    tournament_size: int = config.DEFAULT_TOURNAMENT_SIZE


@dataclass
class EvolutionState:
    """État complet d'une session d'évolution — snapshot après chaque génération."""

    config: EvolutionConfig
    all_genomes: list[PromptGenome] = field(default_factory=list)
    generation_results: list[GenerationResult] = field(default_factory=list)
    current_generation: int = 0
    is_complete: bool = False

    @property
    def best_genome(self) -> PromptGenome | None:
        """Meilleur génome toutes générations confondues."""
        scored = [g for g in self.all_genomes if g.score is not None]
        if not scored:
            return None
        return max(scored, key=lambda g: g.score or 0.0)

    @property
    def progress(self) -> float:
        """Progression entre 0.0 et 1.0."""
        total = self.config.num_generations + 1
        return min(1.0, self.current_generation / total)

    @property
    def improvement(self) -> float | None:
        """Delta de score entre le prompt initial et le meilleur prompt actuel."""
        if len(self.generation_results) < 1:
            return None
        initial_score = self.generation_results[0].best_genome.score
        best = self.best_genome
        if initial_score is None or best is None or best.score is None:
            return None
        return round(best.score - initial_score, 4)


class EvolutionEngine:
    """
    Moteur d'évolution génétique de prompts.

    Algorithme :
    1. Évalue le prompt initial (génération 0)
    2. Pour chaque génération :
       a. Le Mutator génère N variants depuis les survivants
       b. L'Executor exécute chaque variant sur la tâche
       c. L'Evaluator score chaque réponse
       d. Tournament selection → sélectionne les meilleurs survivants
    3. Yield l'état après chaque génération (live updates Streamlit)
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, cfg: EvolutionConfig) -> Generator[EvolutionState, None, None]:
        """
        Exécute l'évolution complète.
        Yield un EvolutionState mis à jour après chaque génération.
        """
        state = EvolutionState(config=cfg)
        mutator = Mutator(self._client)
        evaluator = Evaluator(self._client, cfg.eval_criteria)
        executor = Executor(self._client)

        # --- Génération 0 : évaluation du prompt initial ---
        logger.info("Génération 0 — évaluation du prompt initial")
        initial = PromptGenome(
            prompt_text=cfg.initial_prompt,
            generation=0,
            technique_tags=["original"],
            rationale="Prompt de départ fourni par l'utilisateur",
        )
        self._evaluate_genome(initial, executor, evaluator, cfg)
        state.all_genomes.append(initial)
        state.generation_results.append(GenerationResult.from_genomes(0, [initial]))
        state.current_generation = 1
        yield state

        survivors = [initial]

        # --- Générations 1..N ---
        for gen in range(1, cfg.num_generations + 1):
            logger.info("Génération %d — mutation + scoring", gen)

            # Mutation : génère N variants depuis les survivants
            variants = mutator.generate_variants(
                parents=survivors,
                task_description=cfg.task_description,
                n=cfg.population_size,
                current_generation=gen,
            )

            # Exécution + scoring de chaque variant
            for genome in variants:
                self._evaluate_genome(genome, executor, evaluator, cfg)
                state.all_genomes.append(genome)

            # Sélection par tournoi
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
        """Exécute le prompt sur la tâche et score la réponse. Modifie genome en place."""
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
        Sélection par tournoi :
        - Tire tournament_size candidats au hasard dans le pool
        - Le meilleur du groupe gagne et est retiré du pool
        - Répète jusqu'à num_survivors gagnants
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
