"""
Exécuteur : applique un prompt sur une entrée cible via l'API OpenAI.
Le prompt est utilisé comme system message, l'entrée comme user message.
"""

from __future__ import annotations

import logging

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class Executor:
    """Exécute un prompt contre une tâche cible et retourne la réponse brute."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, prompt: str, task_input: str) -> str:
        """
        Envoie [system: prompt, user: task_input] au modèle.
        Retourne la réponse texte ou une chaîne vide en cas d'erreur.
        """
        try:
            response = self._client.chat.completions.create(
                model=config.EXECUTOR_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": task_input},
                ],
                temperature=config.EXECUTOR_TEMPERATURE,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as error:
            logger.error("Échec d'exécution du prompt : %s", error)
            return ""
