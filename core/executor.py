"""
Executor: applies a prompt to a target input via the OpenAI API.
The prompt is used as the system message, the input as the user message.
"""

from __future__ import annotations

import logging

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class Executor:
    """Runs a prompt against a target task and returns the raw response."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, prompt: str, task_input: str) -> str:
        """
        Sends [system: prompt, user: task_input] to the model.
        Returns the text response or an empty string on error.
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
            logger.error("Prompt execution failed: %s", error)
            return ""
