"""
DeepSeek API Client for Clinical Fairness Analysis

Implements BaseLLMClient interface using DeepSeek's OpenAI-compatible API.
Provides unlimited requests for testing without rate limits.
"""

import json
import hashlib
import logging
import time
from typing import Optional
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import requests

from .llm_client_base import BaseLLMClient
from .gemini_client import LLMCallLog

logger = logging.getLogger(__name__)

class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API client compatible with the 4-tier system.

    Features:
    - OpenAI-compatible API
    - No rate limiting (unlike Gemini free tier)
    - Same caching and logging as Gemini
    - Seamless drop-in replacement
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        cache_dir: str = "llm_cache",
        log_dir: str = "llm_logs",
        enable_cache: bool = True
    ):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key
            model: Model name (default: deepseek-chat)
            cache_dir: Directory for response cache
            log_dir: Directory for call logs
            enable_cache: Enable response caching
        """

        super().__init__(model=model, enable_cache=enable_cache)
        self.provider_name = "DeepSeek"

        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.cache_dir = Path(cache_dir)
        self.log_dir = Path(log_dir)

        self.cache_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        self._cache = {}
        self._load_cache()

        logger.info(f"DeepSeek client initialized: {model}")

    def _load_cache(self):
        """Load cached responses from disk."""
        if not self.enable_cache:
            return

        for cache_file in self.cache_dir.glob("deepseek_*.json"):
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                    self._cache[entry['cache_key']] = entry['response']
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        """Generate cache key from prompt and temperature."""
        content = f"deepseek::{prompt}::temp={temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def _log_call(self, log_entry: LLMCallLog):
        """Log API call for reproducibility."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = self.log_dir / f"deepseek_{timestamp_str}.json"

        with open(log_path, 'w') as f:
            json.dump(asdict(log_entry), f, indent=2)

    def call_with_retry(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Call DeepSeek API with caching and retry logic.

        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0 for reproducibility)
            max_retries: Maximum retry attempts
            system_instruction: Optional system instruction

        Returns:
            Model response text
        """
        if self._quota_exceeded:
            raise RuntimeError("DeepSeek quota exceeded - switch to Gemini or use fallback mode")

        cache_key = self._get_cache_key(prompt, temperature)
        start_time = time.time()

        if self.enable_cache and cache_key in self._cache:
            logger.info(f"Cache hit for prompt (hash: {cache_key[:8]})")
            response = self._cache[cache_key]

            log_entry = LLMCallLog(
                timestamp=datetime.now().isoformat(),
                prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                response=response[:200] + "..." if len(response) > 200 else response,
                model=self.model,
                temperature=temperature,
                cache_hit=True,
                latency_ms=(time.time() - start_time) * 1000
            )
            self._log_call(log_entry)

            return response

        for attempt in range(max_retries):
            try:
                messages = []
                if system_instruction:
                    messages.append({
                        "role": "system",
                        "content": system_instruction
                    })

                messages.append({
                    "role": "user",
                    "content": prompt
                })

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2048
                }

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                result = response.json()
                response_text = result['choices'][0]['message']['content']
                latency_ms = (time.time() - start_time) * 1000

                if self.enable_cache:
                    self._cache[cache_key] = response_text
                    cache_path = self.cache_dir / f"deepseek_{cache_key}.json"
                    with open(cache_path, 'w') as f:
                        json.dump({
                            'cache_key': cache_key,
                            'prompt': prompt,
                            'response': response_text,
                            'temperature': temperature,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)

                log_entry = LLMCallLog(
                    timestamp=datetime.now().isoformat(),
                    prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    response=response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    model=self.model,
                    temperature=temperature,
                    cache_hit=False,
                    latency_ms=latency_ms
                )
                self._log_call(log_entry)

                logger.info(f"DeepSeek API call successful (latency: {latency_ms:.0f}ms)")
                return response_text

            except Exception as e:
                error_str = str(e).lower()

                if "quota" in error_str or "429" in error_str or "rate_limit" in error_str:
                    self._quota_exceeded = True
                    logger.error("DeepSeek quota exceeded - switching to fallback")
                    raise RuntimeError("DeepSeek quota exceeded")

                logger.warning(f"DeepSeek API call failed (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} retry attempts failed")
                    raise RuntimeError(f"DeepSeek API failed after {max_retries} attempts: {e}")

def create_deepseek_client(config_path: str = "config/api_keys.yaml") -> DeepSeekClient:
    """
    Create DeepSeek client from config file.

    Args:
        config_path: Path to API keys configuration

    Returns:
        Configured DeepSeekClient instance
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return DeepSeekClient(
        api_key=config.get('deepseek_api_key', ''),
        model=config.get('deepseek_model', 'deepseek-chat')
    )
