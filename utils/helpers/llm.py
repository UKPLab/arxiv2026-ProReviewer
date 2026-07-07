"""Simple litellm wrapper for unified LLM calls across the project.

This module provides a thin wrapper that:
- Loads model configurations from config.toml
- Supports both API models (OpenAI, DeepSeek, etc.) via litellm
- Supports local models via vLLM
- Provides a consistent interface for all LLM calls

Usage:
    from llm_helper import call_llm, get_content, get_tool_calls

    # API model (uses litellm)
    response = call_llm("openai/gpt-4", messages=[...])
    text = get_content(response)

    # Local vLLM model with tool calling
    tools = [{"type": "function", "function": {...}}]
    response = call_llm("qwen3-8b", messages=[...], tools=tools)
    
    # Extract tool calls
    tool_calls = get_tool_calls(response)
    if tool_calls:
        # Process tool calls
        pass
    else:
        # Get text content
        text = get_content(response)
"""

import os
import json
import re
import toml
import litellm
litellm.drop_params = True
import asyncio
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Optional, Any, Tuple
from .token_tracker import token_tracker

# Try to import vllm, fallback gracefully if not installed
try:
    from vllm import LLM as VllmEngine, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    VllmEngine = None
    SamplingParams = None
    LoRARequest = None


def load_model_config(config_path: str = "config.toml") -> Dict[str, Dict]:
    """Load model configurations from TOML file.

    Args:
        config_path: Path to config.toml file

    Returns:
        Dictionary mapping config names to configuration dictionaries
    """
    # Try to find config.toml in current directory or same directory as this file
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, searching in current directory")
        current_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        if os.path.exists(current_dir_path):
            config_path = current_dir_path

    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, returning empty dictionary")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return toml.load(f)
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}


# Load configs globally
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.toml")
MODEL_CONFIGS = load_model_config(config_path=config_path)
# exit()

# Cache for vLLM engines (singleton pattern)
_VLLM_ENGINES: Dict[str, Any] = {}

# Cache for AsyncOpenAI clients keyed by (base_url, api_key)
_ASYNC_CLIENTS: Dict[Tuple[str, str], "AsyncOpenAI"] = {}


def _is_local_model(model_path: str) -> bool:
    """Check if a model path is a local file system path.
    
    Args:
        model_path: Model path to check
        
    Returns:
        True if the path is a local file system path, False otherwise
    """
    # Check if it exists as a path (relative path that exists)
    if os.path.exists(model_path):
        return True
    return False


def _extract_thinking_tokens(text: str) -> Tuple[str, str]:
    """Extract thinking tokens from model response.
    
    Extracts <think>...</think> sections from the response text.
    
    Args:
        text: Raw response text that may contain thinking tokens
        
    Returns:
        Tuple of (cleaned_text, thinking_content)
    """
    if not text:
        return text, ""
    
    # Extract thinking content
    thinking_matches = re.findall(r'<think[^>]*>(.*?)</think[^>]*>', text, flags=re.DOTALL | re.IGNORECASE)
    thinking_content = "\n".join(thinking_matches) if thinking_matches else ""
    
    # Remove thinking sections from text
    cleaned_text = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return cleaned_text, thinking_content


def _call_vllm(
    model_path: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict]] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> Any:
    """Call local vLLM model with optional tool calling support."""
    if not VLLM_AVAILABLE:
        raise ImportError("vllm is not installed. Install it with: pip install vllm")

    # Check if LoRA adapter is specified in config
    lora_path = config.get("review_lora") if config else None

    # Create a unique cache key including LoRA adapter if present
    cache_key = model_path
    if lora_path:
        cache_key = f"{model_path}:lora:{lora_path}"

    # Initialize engine once per model path (cached)
    if cache_key not in _VLLM_ENGINES:
        gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9) if config else 0.9
        tensor_parallel_size = config.get("tensor_parallel_size", 1) if config else 1

        engine_kwargs = {
            "model": model_path,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
        }

        # Enable LoRA if adapter is specified
        if lora_path:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = config.get("max_lora_rank", 64) if config else 64

        _VLLM_ENGINES[cache_key] = VllmEngine(**engine_kwargs)

    engine = _VLLM_ENGINES[cache_key]
    
    # Get sampling parameters
    temp = temperature if temperature is not None else (config.get("temperature", 0.7) if config else 0.7)
    max_toks = max_tokens if max_tokens is not None else (config.get("max_tokens", 2048) if config else 2048)
    top_p = config.get("top_p", 0.95) if config else 0.95
    top_k = config.get("top_k", -1) if config else -1
    min_p = config.get("min_p", 0.0) if config else 0.0
    
    sampling_params = SamplingParams(
        max_tokens=max_toks,
        temperature=temp if temp > 0 else 0.0,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        min_p=min_p if min_p > 0 else 0.0,
    )

    # Create LoRARequest if LoRA adapter is specified
    lora_request = None
    if lora_path:
        lora_request = LoRARequest(
            lora_name="review_lora",
            lora_int_id=1,
            lora_path=lora_path
        )

    # Use chat() method for tool calling support, generate() for regular calls
    if tools:
        # vLLM's chat() method handles tool calling automatically
        chat_kwargs = {"messages": messages, "sampling_params": sampling_params, "tools": tools}
        if lora_request:
            chat_kwargs["lora_request"] = lora_request
        outputs = engine.chat(**chat_kwargs)
        response_text = outputs[0].outputs[0].text.strip()

        # Extract thinking content before parsing
        response_text, thinking_content = _extract_thinking_tokens(response_text)

        # Parse tool calls from JSON response (vLLM returns JSON array of tool calls)
        tool_calls = None
        try:
            # Response is a JSON array of tool calls: [{"name": "...", "arguments": {...}}, ...]
            parsed_tool_calls = json.loads(response_text)
            if isinstance(parsed_tool_calls, list) and len(parsed_tool_calls) > 0:
                # Convert to OpenAI-compatible format
                tool_calls = []
                for i, call in enumerate(parsed_tool_calls):
                    if isinstance(call, dict) and "name" in call:
                        arguments = call.get("arguments", {})
                        if isinstance(arguments, dict):
                            arguments_str = json.dumps(arguments)
                        else:
                            arguments_str = str(arguments)

                        tool_calls.append({
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": call.get("name", ""),
                                "arguments": arguments_str
                            }
                        })
                # Clear response text when tool calls are present
                response_text = ""
        except (json.JSONDecodeError, KeyError, TypeError):
            # Response is not tool calls, treat as regular text
            tool_calls = None
    else:
        # Regular chat
        chat_kwargs = {"messages": messages, "sampling_params": sampling_params}
        if lora_request:
            chat_kwargs["lora_request"] = lora_request
        outputs = engine.chat(**chat_kwargs)
        response_text = outputs[0].outputs[0].text
        # Extract thinking content from regular response
        response_text, thinking_content = _extract_thinking_tokens(response_text)
        tool_calls = None
    
    # Create a response object compatible with litellm format
    class VllmUsage:
        """Usage object for vLLM responses."""
        def __init__(self, prompt_tokens: int, completion_tokens: int):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens

    class VllmResponse:
        def __init__(self, content: str, tool_calls=None, thinking=None, usage=None):
            class Message:
                def __init__(self, content: str, tool_calls=None, thinking=None):
                    self.content = content
                    self.tool_calls = tool_calls
                    self.thinking = thinking

            class Choice:
                def __init__(self, content: str, tool_calls=None, thinking=None):
                    self.message = Message(content, tool_calls, thinking)

            self.choices = [Choice(content, tool_calls, thinking)]
            self.usage = usage

    # Extract token counts from vLLM output
    output = outputs[0].outputs[0]
    prompt_tokens = len(outputs[0].prompt_token_ids) if hasattr(outputs[0], 'prompt_token_ids') else 0
    completion_tokens = len(output.token_ids) if hasattr(output, 'token_ids') else 0
    usage = VllmUsage(prompt_tokens, completion_tokens)

    response = VllmResponse(response_text, tool_calls, thinking_content or None, usage)

    # Record token usage
    token_tracker.record(response.usage, model_path)

    return response


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict]] = None,
    **kwargs
) -> Any:
    """Call LLM with automatic routing between litellm (API models) and vLLM (local models).

    Automatically detects whether to use:
    - litellm for API models (OpenAI, DeepSeek, etc.)
    - vLLM for local models (file system paths)

    Args:
        model: Model identifier. Can be:
            - API model string: "openai/gpt-4", "deepseek/deepseek-chat"
            - Config name: "gpt-51", "qwen3-8b" (loads from config.toml)
            - Local model path: "/path/to/model" (uses vLLM)
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Sampling temperature (overrides config value)
        max_tokens: Maximum tokens to generate (overrides config value)
        tools: Optional tool definitions for function calling (supported for both API and local models)
        **kwargs: Additional parameters (api_key, base_url for API models, etc.)

    Returns:
        Response object with .choices[0].message.content (compatible with litellm format)

    Examples:
        # API model (uses litellm)
        response = call_llm("openai/gpt-4", messages=[{"role": "user", "content": "Hi"}])

        # Config name (loads from config.toml, routes based on model path)
        response = call_llm("gpt-51", messages=[...])  # API model
        response = call_llm("qwen3-8b", messages=[...])  # Local model (uses vLLM)

        # Direct local model path (uses vLLM)
        response = call_llm("/storage/models/Qwen3-8B", messages=[...])

        # With tools for function calling (both API and local models)
        response = call_llm("gpt-51", messages=[...], tools=[{...}])  # API model
        response = call_llm("qwen3-8b", messages=[...], tools=[{...}])  # Local model with tools
    """
    config = None
    actual_model = model
    is_local = False
    
    # Check if model is a config name from config.toml
    if model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model]
        actual_model = config.get("model", model)
        
        # Use config values as defaults (can be overridden by parameters)
        if temperature is None:
            temperature = config.get("temperature")
        if max_tokens is None:
            max_tokens = config.get("max_tokens")
    else:
        # Model is a direct model string (e.g., "openai/gpt-4", "/path/to/model")
        actual_model = model
    
    # Check if config specifies a remote server (base_url takes priority over local loading)
    base_url = config.get("base_url") if config else None
    is_local = _is_local_model(actual_model) if not base_url else False

    # Route to appropriate backend
    if is_local:
        return _call_vllm(
            model_path=actual_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            config=config,
            **kwargs
        )
    else:
        if base_url:
            api_key = config.get("api_key", "EMPTY")
            lora_module = config.get("lora_module")
            model_name = lora_module if lora_module else actual_model.removeprefix("openai/")

            client = OpenAI(base_url=base_url, api_key=api_key)
            params = {"model": model_name, "messages": messages}
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if tools:
                params["tools"] = tools

            # Pass through standard OpenAI sampling params
            for key in ("top_p", "presence_penalty", "frequency_penalty"):
                val = config.get(key) if config else None
                if val is not None:
                    params[key] = val

            # vLLM-specific params go in extra_body
            extra_body = {}
            for key in ("top_k", "min_p", "repetition_penalty"):
                val = config.get(key) if config else None
                if val is not None:
                    extra_body[key] = val
            if config and config.get("enable_thinking") is not None:
                extra_body["chat_template_kwargs"] = {"enable_thinking": config["enable_thinking"]}
            if extra_body:
                params["extra_body"] = extra_body

            response = client.chat.completions.create(**params)

            if hasattr(response, 'usage') and response.usage:
                token_tracker.record(response.usage, model_name)

            return response

        # API model - use litellm
        if config:
            api_key = config.get("api_key")
            if api_key and "api_key" not in kwargs:
                kwargs["api_key"] = api_key

        # Build parameters for litellm.completion()
        params = {
            "model": actual_model,
            "messages": messages,
        }

        # Add optional parameters if provided
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools

        # Filter out vLLM-specific parameters that are not supported by API models
        vllm_only_params = {"top_k", "min_p", "gpu_memory_utilization", "tensor_parallel_size"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in vllm_only_params}

        params.update(filtered_kwargs)
        response = litellm.completion(**params)

        if hasattr(response, 'usage') and response.usage:
            token_tracker.record(response.usage, actual_model)

        return response


def get_content(response: Any) -> str:
    """Extract text content from LLM response.

    Args:
        response: LLM response object (from call_llm)

    Returns:
        Response text content as string (empty if only tool calls present)

    Example:
        response = call_llm("gpt-51", messages=[...])
        text = get_content(response)
        
    Note:
        For tool calling scenarios, use response.choices[0].message.tool_calls
        to access tool calls directly. get_content() may return empty string
        when tool calls are present.
    """
    return response.choices[0].message.content


def get_tool_calls(response: Any) -> Optional[List[Dict]]:
    """Extract tool calls from LLM response.

    Args:
        response: LLM response object (from call_llm)

    Returns:
        List of tool call dictionaries, or None if no tool calls present

    Example:
        response = call_llm("qwen3-8b", messages=[...], tools=tools)
        tool_calls = get_tool_calls(response)
        if tool_calls:
            for call in tool_calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
    """
    message = response.choices[0].message
    return getattr(message, 'tool_calls', None)


async def acall_llm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict]] = None,
    **kwargs
) -> Any:
    """Async version of call_llm for concurrent API calls.

    This function supports async API calls via litellm.acompletion.
    For local vLLM models, it runs the synchronous vLLM call in a thread pool.

    Args:
        model: Model identifier (same as call_llm)
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Sampling temperature (overrides config value)
        max_tokens: Maximum tokens to generate (overrides config value)
        tools: Optional tool definitions for function calling
        **kwargs: Additional parameters

    Returns:
        Response object (same format as call_llm)

    Examples:
        # Single async call
        response = await acall_llm("gpt-51", messages=[...])

        # Multiple concurrent calls
        responses = await asyncio.gather(
            acall_llm("gpt-51", messages1),
            acall_llm("gpt-51", messages2),
            acall_llm("gpt-51", messages3)
        )
    """
    config = None
    actual_model = model
    is_local = False

    # Check if model is a config name from config.toml
    if model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model]
        actual_model = config.get("model", model)

        # Use config values as defaults (can be overridden by parameters)
        if temperature is None:
            temperature = config.get("temperature")
        if max_tokens is None:
            max_tokens = config.get("max_tokens")
    else:
        # Model is a direct model string
        actual_model = model

    # Check if config specifies a remote server (base_url takes priority over local loading)
    base_url = config.get("base_url") if config else None
    is_local = _is_local_model(actual_model) if not base_url else False

    # Route to appropriate backend
    if is_local:
        # For local vLLM, run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: _call_vllm(
                model_path=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                config=config,
                **kwargs
            )
        )
    else:
        if base_url:
            api_key = config.get("api_key", "EMPTY")
            lora_module = config.get("lora_module")
            model_name = lora_module if lora_module else actual_model.removeprefix("openai/")

            cache_key = (base_url, api_key)
            if cache_key not in _ASYNC_CLIENTS:
                _ASYNC_CLIENTS[cache_key] = AsyncOpenAI(base_url=base_url, api_key=api_key)
            client = _ASYNC_CLIENTS[cache_key]
            params = {"model": model_name, "messages": messages}
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if tools:
                params["tools"] = tools
            if config:
                if config.get("top_p") is not None:
                    params["top_p"] = config["top_p"]
                extra_body = {k: config[k] for k in ("top_k", "min_p", "presence_penalty", "repetition_penalty") if k in config}
                if config.get("enable_thinking") is not None:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": config["enable_thinking"]}
                if extra_body:
                    params["extra_body"] = extra_body

            response = await client.chat.completions.create(**params)

            if hasattr(response, 'usage') and response.usage:
                token_tracker.record(response.usage, model_name)

            return response

        # API model - use litellm async
        if config:
            api_key = config.get("api_key")
            if api_key and "api_key" not in kwargs:
                kwargs["api_key"] = api_key

        # Build parameters for litellm.acompletion()
        params = {
            "model": actual_model,
            "messages": messages,
        }

        # Add optional parameters if provided
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools

        # Pass extra_body options from config if available
        if config:
            extra_body = params.get("extra_body") or {}
            if config.get("prompt_cache_retention"):
                extra_body["prompt_cache_retention"] = config["prompt_cache_retention"]
            # DeepSeek thinking control: thinking="disabled" -> {"thinking": {"type": "disabled"}}
            if config.get("thinking") and "deepseek" in (config.get("model") or ""):
                extra_body["thinking"] = {"type": config["thinking"]}
            if extra_body:
                params["extra_body"] = extra_body

        vllm_only_params = {"top_k", "min_p", "gpu_memory_utilization", "tensor_parallel_size"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in vllm_only_params}

        params.update(filtered_kwargs)

        response = await litellm.acompletion(**params)

        if hasattr(response, 'usage') and response.usage:
            token_tracker.record(response.usage, actual_model)

        return response
