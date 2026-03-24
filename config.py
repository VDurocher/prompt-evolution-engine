"""
Configuration globale du Prompt Evolution Engine.
Toutes les constantes ajustables sont ici — modifier avant de lancer.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# --- Modèles (gpt-4o-mini par défaut pour économiser les crédits) ---
MUTATION_MODEL: str = "gpt-4o-mini"
EXECUTOR_MODEL: str = "gpt-4o-mini"
JUDGE_MODEL: str = "gpt-4o-mini"

# --- Paramètres d'évolution ---
DEFAULT_POPULATION_SIZE: int = 5   # variants générés par génération
DEFAULT_NUM_GENERATIONS: int = 4   # nombre de générations
DEFAULT_NUM_SURVIVORS: int = 2     # meilleurs prompts conservés pour la mutation suivante
DEFAULT_TOURNAMENT_SIZE: int = 3   # candidats tirés au sort dans le tournoi de sélection

# --- Températures LLM ---
MUTATION_TEMPERATURE: float = 0.9   # haute : encourage la créativité des mutations
EXECUTOR_TEMPERATURE: float = 0.3   # basse : réponses cohérentes et reproductibles
JUDGE_TEMPERATURE: float = 0.1      # très basse : jugement stable et objectif

# --- Pondération des évaluateurs (somme = 1.0) ---
DETERMINISTIC_WEIGHT: float = 0.35
LLM_JUDGE_WEIGHT: float = 0.65

# --- Techniques de mutation disponibles ---
MUTATION_TECHNIQUES: list[str] = [
    "chain_of_thought",   # raisonnement étape par étape
    "few_shot",           # exemples input/output injectés
    "persona",            # rôle d'expert précis
    "xml_structure",      # instructions structurées en XML
    "constraints",        # contraintes de format/longueur explicites
    "reformulation",      # réécriture plus claire des instructions
    "socratic",           # questions de clarification dans le prompt
    "step_back",          # réflexion de haut niveau avant de répondre
]
