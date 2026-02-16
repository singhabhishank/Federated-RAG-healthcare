"""
LLM Provider Interface
Supports multiple LLM providers: OpenAI, Qwen, and local models
"""

from typing import Dict, Optional, List
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Qwen (using transformers) - only needed for local Qwen
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    # Transformers not needed for Qwen API, so this is OK


class LLMProvider:
    """Base class for LLM providers."""
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate response from prompt."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-3.5-turbo, etc.)."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (if None, uses environment variable)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 1000)
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate response using OpenAI API."""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")


class QwenAPIProvider(LLMProvider):
    """Qwen API provider (uses Qwen API service, similar to OpenAI API)."""
    
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Qwen API provider.
        
        Args:
            model: Model name (e.g., "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-235B-A22B-fp8-tput")
            api_key: Qwen API key (if None, uses environment variable)
            base_url: API base URL (if None, uses default or environment variable)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.model = model
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.aimlapi.com/v1"
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 1000)
        
        if not self.api_key:
            raise ValueError(
                "Qwen API key not provided. Set QWEN_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure OpenAI client to use Qwen API
        openai.api_key = self.api_key
        if hasattr(openai, 'api_base'):
            openai.api_base = self.base_url
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate response using Qwen API."""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            # Use OpenAI-compatible API
            # Try new OpenAI client API first
            try:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except AttributeError:
                # Fallback to old OpenAI API format
                openai.api_key = self.api_key
                openai.api_base = self.base_url
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            # Check if it's a credit limit error (402)
            if "402" in error_str or "credits" in error_str.lower() or "can only afford" in error_str.lower():
                # Extract the affordable token count from error message
                import re
                match = re.search(r'can only afford (\d+)', error_str)
                if match:
                    affordable_tokens = int(match.group(1))
                    # Use 90% of affordable tokens to be safe
                    safe_tokens = max(50, int(affordable_tokens * 0.9))
                    if safe_tokens < max_tokens:
                        print(f"⚠️  Credit limit: Retrying with {safe_tokens} tokens (can afford {affordable_tokens})...")
                        # Retry with reduced tokens
                        try:
                            client = openai.OpenAI(
                                api_key=self.api_key,
                                base_url=self.base_url
                            )
                            response = client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=safe_tokens
                            )
                            return response.choices[0].message.content
                        except Exception as retry_error:
                            raise Exception(f"Qwen API error (retry failed): {retry_error}")
            raise Exception(f"Qwen API error: {e}")


class GroqAPIProvider(LLMProvider):
    """Groq API provider (fast inference with generous free tier)."""
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Groq API provider.
        
        Args:
            model: Model name (e.g., "llama-3.3-70b-versatile", "mixtral-8x7b-32768")
            api_key: Groq API key (if None, uses environment variable)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 2000)
        
        if not self.api_key:
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate response using Groq API with rate limit handling."""
        import time
        import re
        
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Retry logic for rate limits
        max_retries = 3
        retry_delay = 60  # Start with 60 seconds
        
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                
                # Check for authentication/authorization errors (401, 403)
                if "401" in error_str or "403" in error_str or "Forbidden" in error_str or "Unauthorized" in error_str:
                    raise Exception(
                        f"Groq API authentication failed (403 Forbidden). "
                        f"This usually means:\n"
                        f"  1. Invalid or expired API key\n"
                        f"  2. API key doesn't have access to the requested model\n"
                        f"  3. Account suspended or billing issues\n\n"
                        f"To fix:\n"
                        f"  1. Get a new API key from: https://console.groq.com/keys\n"
                        f"  2. Set it: export GROQ_API_KEY='your-key-here'\n"
                        f"  3. Verify: python test_groq_api.py\n\n"
                        f"Original error: {e}"
                    )
                
                # Check for rate limit (429 error)
                if "429" in error_str or "rate_limit" in error_str.lower() or "Rate limit" in error_str:
                    if attempt < max_retries - 1:
                        # Try to extract wait time from error message
                        wait_match = re.search(r"try again in ([\d.]+)s", error_str)
                        if wait_match:
                            wait_time = float(wait_match.group(1))
                            wait_time = min(wait_time + 10, 600)  # Add 10s buffer, max 10 minutes
                        else:
                            wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                        
                        print(f"⚠️  Rate limit reached. Waiting {wait_time:.0f} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(
                            f"Groq API rate limit exceeded after {max_retries} attempts. "
                            f"Free tier limit: 100,000 tokens/day. "
                            f"Please wait or upgrade at https://console.groq.com/settings/billing"
                        )
                else:
                    # Other errors - raise immediately
                    raise Exception(f"Groq API error: {e}")
        
        raise Exception(f"Groq API error: Failed after {max_retries} attempts")


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider (access to multiple models via unified API)."""
    
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenRouter API provider.
        
        Args:
            model: Model name (e.g., "openai/gpt-4o-mini", "meta-llama/llama-3.1-70b-instruct")
            api_key: OpenRouter API key (if None, uses environment variable)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 2000)
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate response using OpenRouter API."""
        import time
        
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        max_retries = 3
        retry_delay = 60
        
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                
                # Check for authentication/authorization errors
                if "401" in error_str or "403" in error_str or "Forbidden" in error_str or "Unauthorized" in error_str:
                    raise Exception(
                        f"OpenRouter API authentication failed. "
                        f"This usually means:\n"
                        f"  1. Invalid or expired API key\n"
                        f"  2. Insufficient credits\n"
                        f"  3. Account issues\n\n"
                        f"To fix:\n"
                        f"  1. Check API key: https://openrouter.ai/keys\n"
                        f"  2. Check credits: https://openrouter.ai/activity\n"
                        f"  3. Set key: export OPENROUTER_API_KEY='your-key-here'\n\n"
                        f"Original error: {e}"
                    )
                
                # Check for rate limit
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"⚠️  Rate limit reached. Waiting {wait_time:.0f} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"OpenRouter API rate limit exceeded after {max_retries} attempts.")
                else:
                    raise Exception(f"OpenRouter API error: {e}")
        
        raise Exception(f"OpenRouter API error: Failed after {max_retries} attempts")


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider (OpenAI-compatible HTTP API)."""
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DeepSeek API provider.
        
        Args:
            model: Model name (e.g., "deepseek-chat", "deepseek-reasoner")
            api_key: DeepSeek API key (if None, uses environment variable)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.model = model
        # Prefer dedicated DEEPSEEK_API_KEY, fall back to OPENAI_API_KEY for compatibility
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.deepseek.com"
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 2000)
        
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate response using DeepSeek API."""
        import time
        
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        max_retries = 3
        retry_delay = 30
        
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages={ "role": "user", "content": prompt } if isinstance(messages, dict) else messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                
                # Authentication/authorization issues
                if "401" in error_str or "403" in error_str or "Unauthorized" in error_str or "Forbidden" in error_str:
                    raise Exception(
                        "DeepSeek API authentication failed. This usually means:\n"
                        "  1. Invalid or expired API key\n"
                        "  2. Key does not have access to the requested model\n"
                        "  3. Account / billing issues\n\n"
                        "To fix:\n"
                        "  1. Get a key from: https://platform.deepseek.com/api_keys\n"
                        "  2. Set it in your shell: export DEEPSEEK_API_KEY='your-key-here'\n\n"
                        f"Original error: {e}"
                    )
                
                # Simple retry on transient errors / rate limits
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"⚠️  DeepSeek rate limit or transient error. Waiting {wait_time:.0f}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(
                            f"DeepSeek API rate limit / transient error after {max_retries} attempts. "
                            "Please wait or adjust usage."
                        )
                else:
                    raise Exception(f"DeepSeek API error: {e}")
        
        raise Exception(f"DeepSeek API error: Failed after {max_retries} attempts")


class QwenProvider(LLMProvider):
    """Qwen 3 local provider (runs locally, privacy-preserving)."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",  # Default to available Qwen model
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Qwen provider.
        
        Args:
            model_name: HuggingFace model name
                Options:
                - "Qwen/Qwen2.5-7B-Instruct" (recommended, good balance)
                - "Qwen/Qwen2.5-14B-Instruct" (better quality, more memory)
                - "Qwen/Qwen2.5-32B-Instruct" (best quality, requires more resources)
                - "Qwen/Qwen2.5-72B-Instruct" (excellent, very resource intensive)
            device: Device to use ("cuda", "cpu", "auto")
        """
        if not QWEN_AVAILABLE:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 1000)
        
        print(f"Loading Qwen model: {model_name} on {self.device}...")
        print("(This may take a few minutes on first run)")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Qwen model loaded successfully!")
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate response using Qwen model."""
        # Format prompt with system message if provided
        if system_message:
            formatted_prompt = f"System: {system_message}\n\nUser: {prompt}\n\nAssistant:"
        else:
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        temperature = kwargs.get("temperature", self.temperature)
        max_new_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()


def create_llm_provider(
    provider: str = "openai",
    model: Optional[str] = None,
    use_api: bool = False,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM provider.
    
    Args:
        provider: Provider name ("openai", "qwen", "qwen_api", "groq", "local")
        model: Model name (optional, uses defaults if not provided)
        use_api: For Qwen, whether to use API (True) or local model (False)
        **kwargs: Additional provider-specific arguments
        
    Returns:
        LLMProvider instance
    """
    if provider.lower() == "openai":
        model = model or "gpt-4"
        return OpenAIProvider(model=model, **kwargs)
    
    elif provider.lower() == "groq":
        # Groq API (fast inference, generous free tier)
        model = model or "llama-3.3-70b-versatile"
        return GroqAPIProvider(model=model, **kwargs)
    
    elif provider.lower() == "openrouter":
        # OpenRouter API (access to multiple models)
        model = model or "openai/gpt-4o-mini"
        return OpenRouterProvider(model=model, **kwargs)
    
    elif provider.lower() == "deepseek":
        # DeepSeek API (OpenAI-compatible)
        model = model or "deepseek-chat"
        return DeepSeekProvider(model=model, **kwargs)
    
    elif provider.lower() == "qwen_api":
        # Qwen via API (uses API key)
        model = model or "Qwen/Qwen2.5-7B-Instruct"
        return QwenAPIProvider(model=model, **kwargs)
    
    elif provider.lower() in ["qwen", "local"]:
        # Qwen local model (runs locally, no API key needed)
        model = model or "Qwen/Qwen2.5-7B-Instruct"
        return QwenProvider(model_name=model, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Use 'openai', 'groq', 'openrouter', 'deepseek', 'qwen' (local), or 'qwen_api' (API)"
        )


# Recommended Qwen models for healthcare
QWEN_MODELS = {
    "small": "Qwen/Qwen2.5-7B-Instruct",      # Good balance, ~14GB RAM
    "medium": "Qwen/Qwen2.5-14B-Instruct",    # Better quality, ~28GB RAM
    "large": "Qwen/Qwen2.5-32B-Instruct",    # Best quality, ~64GB RAM
    "xlarge": "Qwen/Qwen2.5-72B-Instruct"    # Excellent, ~144GB RAM
}

