"""
Abstract LLM Client Base Class

Provides unified interface for multiple LLM providers with automatic failover.
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All providers (Gemini, DeepSeek, etc.) implement this interface,
    allowing seamless switching and automatic failover.
    """

    def __init__(self, model: str, enable_cache: bool = True):
        self.model = model
        self.enable_cache = enable_cache
        self._quota_exceeded = False
        self.provider_name = "Unknown"

    @abstractmethod
    def call_with_retry(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Call LLM API with caching and retry logic.

        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0 for reproducibility)
            max_retries: Maximum retry attempts
            system_instruction: Optional system instruction

        Returns:
            Model response text

        Raises:
            RuntimeError: If quota exceeded or max retries reached
        """
        pass

    def is_available(self) -> bool:
        """Check if this provider is currently available (not quota-exceeded)."""
        return not self._quota_exceeded

    def get_provider_name(self) -> str:
        """Get human-readable provider name."""
        return self.provider_name
