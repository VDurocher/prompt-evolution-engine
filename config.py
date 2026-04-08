"""
Global configuration for the Prompt Evolution Engine.
All adjustable constants are here — modify before running.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# --- Models (gpt-4o-mini by default to save credits) ---
MUTATION_MODEL: str = "gpt-4o-mini"
EXECUTOR_MODEL: str = "gpt-4o-mini"
JUDGE_MODEL: str = "gpt-4o-mini"

# --- Evolution parameters ---
DEFAULT_POPULATION_SIZE: int = 5   # variants generated per generation
DEFAULT_NUM_GENERATIONS: int = 4   # number of generations
DEFAULT_NUM_SURVIVORS: int = 2     # best prompts kept for the next mutation
DEFAULT_TOURNAMENT_SIZE: int = 3   # candidates randomly drawn in the tournament selection

# --- LLM temperatures ---
MUTATION_TEMPERATURE: float = 0.9   # high: encourages creativity in mutations
EXECUTOR_TEMPERATURE: float = 0.3   # low: consistent and reproducible responses
JUDGE_TEMPERATURE: float = 0.1      # very low: stable and objective scoring

# --- Evaluator weights (sum = 1.0) ---
DETERMINISTIC_WEIGHT: float = 0.35
LLM_JUDGE_WEIGHT: float = 0.65

# --- Available mutation techniques ---
MUTATION_TECHNIQUES: list[str] = [
    "chain_of_thought",   # step-by-step reasoning
    "few_shot",           # injected input/output examples
    "persona",            # precise expert role
    "xml_structure",      # XML-structured instructions
    "constraints",        # explicit format/length constraints
    "reformulation",      # clearer rewrite of instructions
    "socratic",           # clarifying questions in the prompt
    "step_back",          # high-level reflection before answering
]
