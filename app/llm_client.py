import os
from typing import List, Dict, Any

# Stub: replace with actual SDK calls in your environment.
# Keep contract: `call_llm(messages) -> dict`

LLM_MODEL = os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini")

def call_llm(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Implement actual LLM call here.
    For the datathon, this stub can be replaced with the provided endpoint.
    Must return JSON-deserializable content (we assume tool prompts enforce JSON).
    """
    # Placeholder: echo-style fake result to keep pipeline testable.
    # In real implementation, you:
    # - send `messages` to LLM
    # - parse JSON from response
    return {"mock": True, "messages": messages}

def call_llm_with_vision(messages: List[Dict], images: List[str]) -> Dict:
    """
    images: list of base64-encoded image strings
    """
    # Gemini can handle images natively
    # This is MUCH faster than OCR + separate LLM calls
