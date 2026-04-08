"""
LLM Mutator: generates N variants of a prompt by applying different
prompt engineering techniques to each variant.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

import config
from core.genome import PromptGenome

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in prompt engineering. Your mission: improve prompts
by applying precise and varied techniques to maximise the quality
of LLM responses on a given task.\
"""

_USER_TEMPLATE = """\
Here are the current best prompts, sorted by descending score:

{parent_prompts}

Target task: {task_description}

Generate exactly {n} new variants by applying DIFFERENT techniques.
Use as many distinct techniques as possible from:

- chain_of_thought   : Add "Let's think step by step" or an explicit reasoning structure
- few_shot           : Inject 1-2 concrete input/output examples adapted to the task
- persona            : Assign a precise and credible expert role
- xml_structure      : Structure instructions with clear XML tags
- constraints        : Add explicit format/length/style constraints
- reformulation      : Rewrite instructions in a clearer and more direct way
- socratic           : Guide via progressive clarifying questions
- step_back          : Start with high-level reflection before answering

Return ONLY valid JSON, with no text before or after:
{{
  "variants": [
    {{
      "prompt": "the complete prompt here",
      "technique": "chain_of_thought",
      "rationale": "short explanation of the change made"
    }}
  ]
}}\
"""


class Mutator:
    """Generates prompt variants via an LLM call."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def generate_variants(
        self,
        parents: list[PromptGenome],
        task_description: str,
        n: int,
        current_generation: int,
    ) -> list[PromptGenome]:
        """
        Generates N variants from the parent genomes.
        Each variant inherits the parent_ids of all parents.
        """
        # Build the parent list for the meta-prompt
        sorted_parents = sorted(parents, key=lambda g: g.score or 0.0, reverse=True)
        parent_prompts = "\n\n".join(
            f"[Score: {p.score_display}]\n{p.prompt_text}"
            for p in sorted_parents
        )

        user_msg = _USER_TEMPLATE.format(
            parent_prompts=parent_prompts,
            task_description=task_description,
            n=n,
        )

        variants_data: list[dict] = []
        try:
            response = self._client.chat.completions.create(
                model=config.MUTATION_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=config.MUTATION_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            variants_data = data.get("variants", [])
        except Exception as error:
            logger.error("LLM mutation failed: %s", error)

        parent_ids = [p.id for p in parents]
        variants: list[PromptGenome] = []

        for item in variants_data[:n]:
            if not isinstance(item, dict) or not item.get("prompt"):
                continue
            variants.append(
                PromptGenome(
                    prompt_text=str(item["prompt"]),
                    generation=current_generation,
                    parent_ids=parent_ids,
                    technique_tags=[str(item.get("technique", "unknown"))],
                    rationale=str(item.get("rationale", "")),
                )
            )

        # Fallback: fill with copies of the best parent if not enough variants
        while len(variants) < n and sorted_parents:
            best_parent = sorted_parents[0]
            variants.append(
                PromptGenome(
                    prompt_text=best_parent.prompt_text,
                    generation=current_generation,
                    parent_ids=parent_ids,
                    technique_tags=["fallback"],
                    rationale="Copy of best parent (mutation failed)",
                )
            )

        return variants[:n]
