"""
Gemini 2.0 Flash API Client for Clinical Fairness Analysis

Provides:
- Structured output with Pydantic validation
- Caching for cost optimization
- Comprehensive logging for reproducibility
- Fallback mechanisms for API failures
- Temperature=0 for reproducible clinical reasoning
- Automatic rate limiting for free tier (2 RPM)
"""

import json
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
import yaml

from .llm_client_base import BaseLLMClient

logger = logging.getLogger(__name__)

@dataclass
class LLMCallLog:
    """Log entry for LLM API call."""
    timestamp: str
    prompt: str
    response: str
    model: str
    temperature: float
    cache_hit: bool
    latency_ms: float
    token_count: Optional[int] = None

class GeminiClient(BaseLLMClient):
    """
    Production-ready Gemini API client with clinical safety features.

    Implements BaseLLMClient interface for seamless provider switching.

    Features:
    - Response caching by prompt hash
    - Comprehensive logging for peer review
    - Retry logic with exponential backoff
    - Structured output validation
    - Temperature=0 for reproducibility
    - Auto rate limiting (2 RPM for free tier)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        cache_dir: str = "llm_cache",
        log_dir: str = "llm_logs",
        enable_cache: bool = True
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key
            model: Model name (default: gemini-2.0-flash for speed/cost)
            cache_dir: Directory for response cache
            log_dir: Directory for call logs
            enable_cache: Enable response caching
        """
        super().__init__(model=model, enable_cache=enable_cache)
        self.provider_name = "Gemini"

        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.log_dir = Path(log_dir)

        self.cache_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        try:
            import warnings
            warnings.filterwarnings('ignore', category=FutureWarning)

            from google import genai

            client = genai.Client(api_key=api_key)
            self.client = client
            self.model_name = model
            logger.info(f"Gemini client initialized: {model}")
        except ImportError as e:
            logger.error(f"google-genai not installed: {e}")
            logger.error("Install with: pip install google-genai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

        self._cache: Dict[str, str] = {}
        self._quota_exceeded = False
        self._load_cache()

    def _load_cache(self):
        """Load cached responses from disk."""
        if not self.enable_cache:
            return

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                    self._cache[entry['cache_key']] = entry['response']
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        """Generate cache key from prompt and temperature."""
        content = f"{prompt}::temp={temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def _log_call(self, log_entry: LLMCallLog):
        """Log API call for reproducibility."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = self.log_dir / f"{timestamp_str}.json"

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
        Call Gemini API with caching and retry logic.

        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0 for reproducibility)
            max_retries: Maximum retry attempts
            system_instruction: Optional system instruction

        Returns:
            Model response text
        """
        if self._quota_exceeded:
            raise RuntimeError("Quota exceeded - using fallback mode. Wait until quota resets (midnight PST) or use a different API key.")

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
                config = {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }

                if system_instruction:
                    config["system_instruction"] = system_instruction

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )

                response_text = response.text
                latency_ms = (time.time() - start_time) * 1000

                if self.enable_cache:
                    self._cache[cache_key] = response_text
                    cache_path = self.cache_dir / f"{cache_key}.json"
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

                logger.info(f"Gemini API call successful (latency: {latency_ms:.0f}ms)")

                logger.info("Rate limiting: waiting 30s for Gemini free tier (2 RPM)...")
                time.sleep(30)

                return response_text

            except Exception as e:
                error_str = str(e).lower()

                if "quota" in error_str or "429" in error_str or "resource_exhausted" in error_str:
                    self._quota_exceeded = True
                    logger.error("QUOTA EXCEEDED - Switching to fallback mode permanently")
                    logger.error("  Quota resets at midnight PST (Pacific Standard Time)")
                    logger.error("  All subsequent calls will use fallback mechanisms")
                    raise RuntimeError("Quota exceeded - using fallback mode")

                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {e}")

    def call_with_structured_output(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.3,
        system_instruction: Optional[str] = None
    ) -> BaseModel:
        """
        Call Gemini with structured JSON output validated by Pydantic.

        Args:
            prompt: User prompt
            response_model: Pydantic model for response validation
            temperature: Sampling temperature
            system_instruction: Optional system instruction

        Returns:
            Validated Pydantic model instance
        """
        schema = response_model.model_json_schema()
        enhanced_prompt = f"""{prompt}

REQUIRED OUTPUT FORMAT (JSON):
{json.dumps(schema, indent=2)}

Return ONLY valid JSON matching this schema. No markdown, no explanations."""

        response_text = self.call_with_retry(
            enhanced_prompt,
            temperature=temperature,
            system_instruction=system_instruction
        )

        try:
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            response_dict = json.loads(response_text)

            validated = response_model(**response_dict)
            return validated

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
        except Exception as e:
            logger.error(f"Failed to validate response: {e}")
            raise

def load_gemini_config(config_path: str = "config/api_keys.yaml") -> Dict[str, Any]:
    """Load Gemini configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def create_gemini_client(config_path: str = "config/api_keys.yaml") -> GeminiClient:
    """Create Gemini client from configuration file."""
    config = load_gemini_config(config_path)

    api_key = config.get('gemini_api_key')
    if not api_key:
        raise ValueError("gemini_api_key not found in config")

    model = config.get('gemini_model', 'gemini-2.0-flash')

    return GeminiClient(
        api_key=api_key,
        model=model,
        enable_cache=config.get('enable_cache', True)
    )

def create_smart_llm_client(config_path: str = "config/api_keys.yaml") -> 'BaseLLMClient':
    """
    Create LLM client with automatic provider selection and failover.

    Auto-switching logic:
    1. Try DeepSeek first (if API key available) - unlimited, fast
    2. Fall back to Gemini (if DeepSeek unavailable/exceeded) - free tier with rate limits
    3. Both clients support automatic fallback to templates if quota exceeded

    Args:
        config_path: Path to API keys configuration

    Returns:
        BaseLLMClient instance (either DeepSeekClient or GeminiClient)
    """
    config = load_gemini_config(config_path)

    preferred_provider = config.get('llm_provider', 'auto').lower()

    deepseek_key = config.get('deepseek_api_key', '').strip()
    gemini_key = config.get('gemini_api_key', '').strip()

    if preferred_provider == 'deepseek' or (preferred_provider == 'auto' and deepseek_key):
        if deepseek_key:
            try:
                from .deepseek_client import DeepSeekClient
                logger.info("Using DeepSeek (unlimited requests, fast)")
                return DeepSeekClient(
                    api_key=deepseek_key,
                    model=config.get('deepseek_model', 'deepseek-chat'),
                    enable_cache=config.get('enable_cache', True)
                )
            except Exception as e:
                logger.warning(f"Failed to create DeepSeek client: {e}")
                logger.info("Falling back to Gemini...")

    if preferred_provider == 'gemini' or (preferred_provider == 'auto' and gemini_key):
        if gemini_key:
            logger.info("Using Gemini (free tier with 30s rate limiting)")
            return GeminiClient(
                api_key=gemini_key,
                model=config.get('gemini_model', 'gemini-2.0-flash'),
                enable_cache=config.get('enable_cache', True)
            )

    raise ValueError(
        "No valid LLM provider configured. Please add either:\n"
        "  - deepseek_api_key (recommended for testing)\n"
        "  - gemini_api_key (required for demo)\n"
        "to config/api_keys.yaml"
    )
