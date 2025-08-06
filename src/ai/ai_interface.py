"""
AI Interface Module
Foundation for AI model communication including OpenAI, Azure OpenAI, and local models.

This module provides:
- Unified interface for multiple AI providers
- Configuration management for API keys and endpoints
- Token usage tracking and cost estimation
- Error handling and retry logic
- Model capability detection and validation
- Conversation context management
"""

import os
import json
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Optional dependencies - will be imported when needed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    LOCAL_OLLAMA = "local_ollama"
    LOCAL_LLAMACPP = "local_llamacpp"
    GOOGLE_GEMINI = "google_gemini"


class ModelCapability(Enum):
    """AI model capabilities."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    LONG_CONTEXT = "long_context"
    STREAMING = "streaming"


@dataclass
class AIModelConfig:
    """Configuration for AI model."""
    provider: AIProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 60
    max_retries: int = 3
    capabilities: List[ModelCapability] = field(default_factory=list)
    cost_per_1k_tokens: float = 0.0
    context_window: int = 4000


@dataclass
class AIMessage:
    """AI conversation message."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIResponse:
    """AI response data."""
    content: str
    model: str
    provider: AIProvider
    tokens_used: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_estimate: float = 0.0
    response_time: float = 0.0
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIProviderInterface(ABC):
    """Abstract interface for AI providers."""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.conversation_history: List[AIMessage] = []
        
    @abstractmethod
    async def generate_response(self, messages: List[AIMessage]) -> AIResponse:
        """Generate response from AI model."""
        pass
        
    @abstractmethod
    async def stream_response(self, messages: List[AIMessage]) -> AsyncGenerator[str, None]:
        """Stream response from AI model."""
        pass
        
    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate provider configuration."""
        pass
        
    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        message = AIMessage(role=role, content=content)
        self.conversation_history.append(message)
        
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        
    def get_context_window_usage(self) -> float:
        """Get percentage of context window used."""
        total_tokens = sum(msg.tokens or len(msg.content.split()) for msg in self.conversation_history)
        return (total_tokens / self.config.context_window) * 100


class OpenAIProvider(AIProviderInterface):
    """OpenAI API provider."""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
            
        self.client = openai.OpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.api_base,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
    def validate_configuration(self) -> bool:
        """Validate OpenAI configuration."""
        try:
            # Test with a simple request
            response = self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI configuration validation failed: {e}")
            return False
            
    async def generate_response(self, messages: List[AIMessage]) -> AIResponse:
        """Generate response from OpenAI."""
        start_time = time.time()
        
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content or ""
            tokens_prompt = response.usage.prompt_tokens if response.usage else 0
            tokens_completion = response.usage.completion_tokens if response.usage else 0
            tokens_total = response.usage.total_tokens if response.usage else 0
            
            # Calculate cost estimate
            cost_estimate = (tokens_total / 1000) * self.config.cost_per_1k_tokens
            
            return AIResponse(
                content=content,
                model=self.config.model_name,
                provider=self.config.provider,
                tokens_used=tokens_total,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                cost_estimate=cost_estimate,
                response_time=response_time,
                finish_reason=response.choices[0].finish_reason,
                metadata={"response_id": response.id}
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
            
    async def stream_response(self, messages: List[AIMessage]) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


class AzureOpenAIProvider(AIProviderInterface):
    """Azure OpenAI API provider."""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
            
        self.client = openai.AzureOpenAI(
            api_key=config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=config.api_version or "2024-02-01",
            azure_endpoint=config.api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
    def validate_configuration(self) -> bool:
        """Validate Azure OpenAI configuration."""
        try:
            # Test with a simple request
            response = self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Azure OpenAI configuration validation failed: {e}")
            return False
            
    async def generate_response(self, messages: List[AIMessage]) -> AIResponse:
        """Generate response from Azure OpenAI."""
        start_time = time.time()
        
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content or ""
            tokens_prompt = response.usage.prompt_tokens if response.usage else 0
            tokens_completion = response.usage.completion_tokens if response.usage else 0
            tokens_total = response.usage.total_tokens if response.usage else 0
            
            # Calculate cost estimate
            cost_estimate = (tokens_total / 1000) * self.config.cost_per_1k_tokens
            
            return AIResponse(
                content=content,
                model=self.config.model_name,
                provider=self.config.provider,
                tokens_used=tokens_total,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                cost_estimate=cost_estimate,
                response_time=response_time,
                finish_reason=response.choices[0].finish_reason,
                metadata={"response_id": response.id}
            )
            
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise
            
    async def stream_response(self, messages: List[AIMessage]) -> AsyncGenerator[str, None]:
        """Stream response from Azure OpenAI."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Azure OpenAI streaming error: {e}")
            raise


class OllamaProvider(AIProviderInterface):
    """Local Ollama provider."""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests library not available. Install with: pip install requests")
            
        self.base_url = config.api_base or "http://localhost:11434"
        
    def validate_configuration(self) -> bool:
        """Validate Ollama configuration."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama configuration validation failed: {e}")
            return False
            
    async def generate_response(self, messages: List[AIMessage]) -> AIResponse:
        """Generate response from Ollama."""
        start_time = time.time()
        
        try:
            # Convert messages to Ollama format
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            response_time = time.time() - start_time
            result = response.json()
            
            return AIResponse(
                content=result.get("response", ""),
                model=self.config.model_name,
                provider=self.config.provider,
                tokens_used=result.get("eval_count", 0),
                tokens_prompt=result.get("prompt_eval_count", 0),
                tokens_completion=result.get("eval_count", 0),
                cost_estimate=0.0,  # Local model, no cost
                response_time=response_time,
                finish_reason=result.get("done_reason"),
                metadata=result
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
            
    async def stream_response(self, messages: List[AIMessage]) -> AsyncGenerator[str, None]:
        """Stream response from Ollama."""
        try:
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
            
    def _messages_to_prompt(self, messages: List[AIMessage]) -> str:
        """Convert messages to single prompt for Ollama."""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
                
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)


class GeminiProvider(AIProviderInterface):
    """Google Gemini provider."""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not available. Install with: pip install google-generativeai")
            
        # Configure Gemini with API key from config or environment
        api_key = config.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Google Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.model_name)
        
    def validate_configuration(self) -> bool:
        """Validate Gemini configuration."""
        try:
            # Test with a simple generation
            response = self.model.generate_content("Hello")
            return hasattr(response, 'text')
        except Exception as e:
            logger.error(f"Gemini configuration validation failed: {e}")
            return False
            
    async def generate_response(self, messages: List[AIMessage]) -> AIResponse:
        """Generate response from Gemini."""
        start_time = time.time()
        
        try:
            # Convert messages to Gemini format
            prompt = self._messages_to_prompt(messages)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_time = time.time() - start_time
            
            # Estimate tokens (Gemini doesn't provide exact counts for free tier)
            content = response.text if hasattr(response, 'text') else ""
            estimated_tokens = len(content.split()) * 1.3  # Rough estimate
            
            return AIResponse(
                content=content,
                model=self.config.model_name,
                provider=self.config.provider,
                tokens_used=int(estimated_tokens),
                tokens_prompt=len(prompt.split()) * 1.3,
                tokens_completion=int(estimated_tokens),
                cost_estimate=0.0,  # Free tier, no cost
                response_time=response_time,
                finish_reason="stop",
                metadata={"response": response}
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
            
    async def stream_response(self, messages: List[AIMessage]) -> AsyncGenerator[str, None]:
        """Stream response from Gemini."""
        try:
            prompt = self._messages_to_prompt(messages)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
            
            # Generate streaming response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
            
    def _messages_to_prompt(self, messages: List[AIMessage]) -> str:
        """Convert messages to single prompt for Gemini."""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System Instructions: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
                
        return "\n\n".join(prompt_parts)


class AIInterfaceManager:
    """Main interface manager for AI providers."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.providers: Dict[str, AIProviderInterface] = {}
        self.active_provider: Optional[str] = None
        self.config_path = config_path or "ai_config.json"
        self.usage_stats: Dict[str, Any] = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "provider_usage": {}
        }
        
        # Load configuration
        self._load_configuration()
        
    def _load_configuration(self) -> None:
        """Load AI configuration from file."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Load provider configurations
                for provider_name, provider_config in config_data.get("providers", {}).items():
                    try:
                        self.add_provider(provider_name, provider_config)
                    except Exception as e:
                        logger.warning(f"Failed to load provider {provider_name}: {e}")
                        
                # Set active provider
                self.active_provider = config_data.get("active_provider")
                
                # Load usage stats
                self.usage_stats.update(config_data.get("usage_stats", {}))
                
            else:
                logger.info(f"No AI configuration file found at {self.config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load AI configuration: {e}")
            
    def save_configuration(self) -> None:
        """Save AI configuration to file."""
        try:
            config_data = {
                "active_provider": self.active_provider,
                "providers": {},
                "usage_stats": self.usage_stats
            }
            
            # Save provider configurations (without sensitive data)
            for name, provider in self.providers.items():
                config_data["providers"][name] = {
                    "provider": provider.config.provider.value,
                    "model_name": provider.config.model_name,
                    "api_base": provider.config.api_base,
                    "api_version": provider.config.api_version,
                    "max_tokens": provider.config.max_tokens,
                    "temperature": provider.config.temperature,
                    "timeout": provider.config.timeout,
                    "max_retries": provider.config.max_retries,
                    "capabilities": [cap.value for cap in provider.config.capabilities],
                    "cost_per_1k_tokens": provider.config.cost_per_1k_tokens,
                    "context_window": provider.config.context_window
                }
                
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save AI configuration: {e}")
            
    def add_provider(self, name: str, config: Union[Dict[str, Any], AIModelConfig]) -> None:
        """Add AI provider."""
        try:
            if isinstance(config, dict):
                # Convert dict to AIModelConfig
                provider_type = AIProvider(config["provider"])
                model_config = AIModelConfig(
                    provider=provider_type,
                    model_name=config["model_name"],
                    api_key=config.get("api_key"),
                    api_base=config.get("api_base"),
                    api_version=config.get("api_version"),
                    max_tokens=config.get("max_tokens", 4000),
                    temperature=config.get("temperature", 0.1),
                    timeout=config.get("timeout", 60),
                    max_retries=config.get("max_retries", 3),
                    capabilities=[ModelCapability(cap) for cap in config.get("capabilities", [])],
                    cost_per_1k_tokens=config.get("cost_per_1k_tokens", 0.0),
                    context_window=config.get("context_window", 4000)
                )
            else:
                model_config = config
                
            # Create provider instance
            if model_config.provider == AIProvider.OPENAI:
                provider = OpenAIProvider(model_config)
            elif model_config.provider == AIProvider.AZURE_OPENAI:
                provider = AzureOpenAIProvider(model_config)
            elif model_config.provider == AIProvider.LOCAL_OLLAMA:
                provider = OllamaProvider(model_config)
            elif model_config.provider == AIProvider.GOOGLE_GEMINI:
                provider = GeminiProvider(model_config)
            else:
                raise ValueError(f"Unsupported provider: {model_config.provider}")
                
            # Validate configuration
            if provider.validate_configuration():
                self.providers[name] = provider
                logger.info(f"Added AI provider: {name} ({model_config.provider.value})")
                
                # Set as active if it's the first provider
                if not self.active_provider:
                    self.active_provider = name
            else:
                logger.error(f"Provider validation failed: {name}")
                
        except Exception as e:
            logger.error(f"Failed to add provider {name}: {e}")
            raise
            
    def set_active_provider(self, name: str) -> None:
        """Set active AI provider."""
        if name in self.providers:
            self.active_provider = name
            logger.info(f"Set active AI provider: {name}")
        else:
            raise ValueError(f"Provider not found: {name}")
            
    def get_active_provider(self) -> Optional[AIProviderInterface]:
        """Get active AI provider."""
        if self.active_provider and self.active_provider in self.providers:
            return self.providers[self.active_provider]
        return None
        
    async def generate_response(self, 
                              messages: List[AIMessage], 
                              provider_name: Optional[str] = None) -> AIResponse:
        """Generate response using specified or active provider."""
        provider = self.providers.get(provider_name or self.active_provider)
        if not provider:
            raise ValueError("No active AI provider configured")
            
        response = await provider.generate_response(messages)
        
        # Update usage statistics
        self._update_usage_stats(provider_name or self.active_provider, response)
        
        return response
        
    async def stream_response(self, 
                            messages: List[AIMessage], 
                            provider_name: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream response using specified or active provider."""
        provider = self.providers.get(provider_name or self.active_provider)
        if not provider:
            raise ValueError("No active AI provider configured")
            
        async for chunk in provider.stream_response(messages):
            yield chunk
            
    def _update_usage_stats(self, provider_name: str, response: AIResponse) -> None:
        """Update usage statistics."""
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_tokens"] += response.tokens_used
        self.usage_stats["total_cost"] += response.cost_estimate
        
        if provider_name not in self.usage_stats["provider_usage"]:
            self.usage_stats["provider_usage"][provider_name] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0
            }
            
        provider_stats = self.usage_stats["provider_usage"][provider_name]
        provider_stats["requests"] += 1
        provider_stats["tokens"] += response.tokens_used
        provider_stats["cost"] += response.cost_estimate
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.usage_stats.copy()
        
    def list_providers(self) -> Dict[str, Dict[str, Any]]:
        """List all configured providers."""
        providers_info = {}
        
        for name, provider in self.providers.items():
            providers_info[name] = {
                "provider": provider.config.provider.value,
                "model": provider.config.model_name,
                "active": name == self.active_provider,
                "capabilities": [cap.value for cap in provider.config.capabilities],
                "context_window": provider.config.context_window,
                "cost_per_1k_tokens": provider.config.cost_per_1k_tokens
            }
            
        return providers_info


# Convenience functions for common operations
async def simple_chat_completion(prompt: str, 
                                system_message: Optional[str] = None,
                                provider_name: Optional[str] = None) -> str:
    """Simple chat completion function."""
    manager = AIInterfaceManager()
    
    messages = []
    if system_message:
        messages.append(AIMessage(role="system", content=system_message))
    messages.append(AIMessage(role="user", content=prompt))
    
    response = await manager.generate_response(messages, provider_name)
    return response.content


def create_default_config() -> Dict[str, Any]:
    """Create default AI configuration."""
    return {
        "providers": {
            "openai_gpt4": {
                "provider": "openai",
                "model_name": "gpt-4",
                "max_tokens": 4000,
                "temperature": 0.1,
                "timeout": 60,
                "max_retries": 3,
                "capabilities": ["text_generation", "code_generation", "function_calling"],
                "cost_per_1k_tokens": 0.03,
                "context_window": 8000
            },
            "openai_gpt35": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "max_tokens": 4000,
                "temperature": 0.1,
                "timeout": 60,
                "max_retries": 3,
                "capabilities": ["text_generation", "code_generation", "function_calling"],
                "cost_per_1k_tokens": 0.002,
                "context_window": 4000
            },
            "local_ollama": {
                "provider": "local_ollama",
                "model_name": "codellama:13b",
                "api_base": "http://localhost:11434",
                "max_tokens": 4000,
                "temperature": 0.1,
                "timeout": 120,
                "max_retries": 3,
                "capabilities": ["text_generation", "code_generation"],
                "cost_per_1k_tokens": 0.0,
                "context_window": 4000
            },
            "gemini_flash": {
                "provider": "google_gemini", 
                "model_name": "gemini-2.5-flash",
                "api_key": None,  # Set via environment variable GOOGLE_API_KEY
                "max_tokens": 8000,
                "temperature": 0.1,
                "timeout": 60,
                "max_retries": 3,
                "capabilities": ["text_generation", "code_generation"],
                "cost_per_1k_tokens": 0.0,
                "context_window": 1000000
            }
        },
        "active_provider": "gemini_flash",
        "usage_stats": {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "provider_usage": {}
        }
    }
