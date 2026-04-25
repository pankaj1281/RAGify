"""Low-level LLM helper for the NVIDIA-hosted API.

Uses the OpenAI-compatible NVIDIA inference endpoint.  The API key is
read from the ``OPENAI_API_KEY`` environment variable so that a single
variable covers both OpenAI and NVIDIA access.
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_BASE_URL = "https://integrate.api.nvidia.com/v1"
_MODEL = "meta/llama3-70b-instruct"
_MAX_TOKENS = 512
_TEMPERATURE = 0.2


def generate_answer(prompt: str) -> str:
    """Call the NVIDIA API and return the generated answer.

    The prompt is sent as the sole user message.  The NVIDIA endpoint is
    OpenAI-compatible, so the standard ``openai`` SDK is used with a custom
    ``base_url``.

    Args:
        prompt: The full prompt text to send to the model.

    Returns:
        The model's response as a stripped string.

    Raises:
        ValueError: If ``OPENAI_API_KEY`` is not set.
        Exception: Re-raises any error returned by the API so callers can
            handle or log it as appropriate.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment or .env file."
        )

    client = OpenAI(api_key=api_key, base_url=_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("NVIDIA API call failed: %s", exc)
        raise
