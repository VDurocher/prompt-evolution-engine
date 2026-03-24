"""
Mutateur LLM : génère N variants d'un prompt en appliquant des techniques
de prompt engineering différentes sur chaque variant.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

import config
from core.genome import PromptGenome

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Tu es un expert en prompt engineering. Ta mission : améliorer des prompts
en appliquant des techniques précises et variées pour maximiser la qualité
des réponses LLM sur une tâche donnée.\
"""

_USER_TEMPLATE = """\
Voici les meilleurs prompts actuels, triés par score décroissant :

{parent_prompts}

Tâche cible : {task_description}

Génère exactement {n} nouveaux variants en appliquant des techniques DIFFÉRENTES.
Utilise autant de techniques distinctes que possible parmi :

- chain_of_thought   : Ajouter "Raisonnons étape par étape" ou une structure de raisonnement explicite
- few_shot           : Injecter 1-2 exemples concrets input/output adaptés à la tâche
- persona            : Assigner un rôle d'expert précis et crédible
- xml_structure      : Structurer les instructions avec des balises XML claires
- constraints        : Ajouter des contraintes de format/longueur/style explicites
- reformulation      : Réécrire les instructions de façon plus claire et directe
- socratic           : Guider via des questions de clarification progressives
- step_back          : Commencer par une réflexion de haut niveau avant de répondre

Retourne UNIQUEMENT du JSON valide, sans texte avant ou après :
{{
  "variants": [
    {{
      "prompt": "le prompt complet ici",
      "technique": "chain_of_thought",
      "rationale": "explication courte de la modification apportée"
    }}
  ]
}}\
"""


class Mutator:
    """Génère des variants d'un prompt via un appel LLM."""

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
        Génère N variants à partir des génomes parents.
        Chaque variant hérite des parent_ids de tous les parents.
        """
        # Construit la liste des parents pour le meta-prompt
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
            logger.error("Échec de la mutation LLM : %s", error)

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

        # Fallback : complète avec des copies du meilleur parent si pas assez de variants
        while len(variants) < n and sorted_parents:
            best_parent = sorted_parents[0]
            variants.append(
                PromptGenome(
                    prompt_text=best_parent.prompt_text,
                    generation=current_generation,
                    parent_ids=parent_ids,
                    technique_tags=["fallback"],
                    rationale="Copie du meilleur parent (mutation échouée)",
                )
            )

        return variants[:n]
