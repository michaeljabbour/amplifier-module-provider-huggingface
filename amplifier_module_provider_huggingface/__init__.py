"""
HuggingFace Inference API provider module for Amplifier.
Integrates with HuggingFace models via the OpenAI-compatible chat completions endpoint.
"""

__all__ = ["mount", "HuggingFaceProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCall

logger = logging.getLogger(__name__)

# Default base URL for HuggingFace Serverless Inference API
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"

# Default model (widely available, high quality)
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Default request parameters
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120.0


class HuggingFaceChatResponse(ChatResponse):
    """Extended ChatResponse with HuggingFace-specific metadata."""

    raw_response: dict[str, Any] | None = None
    model_name: str | None = None
    # content_blocks for streaming UI compatibility (triggers content_block:start/end events)
    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


def _truncate_values(
    obj: Any,
    max_length: int = 200,
    max_depth: int = 10,
    _depth: int = 0,
) -> Any:
    """Truncate long strings in nested structures for logging.

    Args:
        obj: Object to truncate (dict, list, str, or other)
        max_length: Maximum length for strings before truncation
        max_depth: Maximum recursion depth
        _depth: Current recursion depth (internal)

    Returns:
        Truncated copy of the object
    """
    if _depth > max_depth:
        return "..."

    if isinstance(obj, str):
        if len(obj) > max_length:
            return obj[:max_length] + f"... ({len(obj)} chars)"
        return obj
    if isinstance(obj, dict):
        return {
            k: _truncate_values(v, max_length, max_depth, _depth + 1)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        if len(obj) > 10:
            truncated = [
                _truncate_values(item, max_length, max_depth, _depth + 1)
                for item in obj[:10]
            ]
            return truncated + [f"... ({len(obj)} items total)"]
        return [
            _truncate_values(item, max_length, max_depth, _depth + 1) for item in obj
        ]
    return obj


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the HuggingFace provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including:
            - api_key: HuggingFace API token (or set HF_TOKEN env var)
            - base_url: API base URL (default: HF Serverless Inference)
            - default_model: Model to use (default: meta-llama/Llama-3.3-70B-Instruct)
            - max_tokens: Maximum tokens (default: 4096)
            - temperature: Generation temperature (default: 0.7)
            - timeout: Request timeout in seconds (default: 120)

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get API key from config (resolved from ${HF_TOKEN} by runtime) or environment
    api_key = config.get("api_key") or os.environ.get("HF_TOKEN")

    if not api_key:
        logger.warning("No API key found for HuggingFace provider")
        return None  # Silent skip

    provider = HuggingFaceProvider(
        api_key=api_key, config=config, coordinator=coordinator
    )
    await coordinator.mount("providers", provider, name="huggingface")
    logger.info("Mounted HuggingFaceProvider")

    # Return cleanup function
    async def cleanup():
        # AsyncInferenceClient doesn't require explicit cleanup
        pass

    return cleanup


class HuggingFaceProvider:
    """HuggingFace Inference API integration.

    Supports both Serverless Inference API and dedicated Inference Endpoints
    via the OpenAI-compatible chat completions format.
    """

    name = "huggingface"
    api_label = "HuggingFace"

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        """
        Initialize HuggingFace provider.

        The SDK client is created lazily on first use, allowing get_info()
        to work without a valid API key.

        Args:
            api_key: HuggingFace API token (can be None for get_info() calls)
            config: Additional configuration
            coordinator: Module coordinator for event emission
        """
        self._api_key = api_key
        self._client: Any | None = None  # Lazy init: AsyncInferenceClient
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration with sensible defaults
        self.base_url = self.config.get("base_url", DEFAULT_BASE_URL)
        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        self.max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = float(self.config.get("timeout", DEFAULT_TIMEOUT))
        self.debug = self.config.get("debug", False)
        self.raw_debug = self.config.get(
            "raw_debug", False
        )  # Enable ultra-verbose raw API I/O logging

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

    @property
    def client(self) -> Any:
        """Lazily initialize the HuggingFace AsyncInferenceClient on first access."""
        if self._client is None:
            from huggingface_hub import AsyncInferenceClient  # pyright: ignore[reportMissingImports]

            self._client = AsyncInferenceClient(
                token=self._api_key,
                timeout=self.timeout,
            )
        return self._client

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="huggingface",
            display_name="HuggingFace",
            credential_env_vars=["HF_TOKEN"],
            capabilities=["tools"],
            defaults={
                "model": DEFAULT_MODEL,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": 0.7,
                "timeout": DEFAULT_TIMEOUT,
                "context_window": 128000,
                "max_output_tokens": 4096,
            },
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your HuggingFace API token",
                    env_var="HF_TOKEN",
                ),
                ConfigField(
                    id="base_url",
                    display_name="API Base URL",
                    field_type="text",
                    prompt="API base URL (change for Inference Endpoints)",
                    required=False,
                    default=DEFAULT_BASE_URL,
                ),
                ConfigField(
                    id="default_model",
                    display_name="Default Model",
                    field_type="text",
                    prompt="Default model to use",
                    required=False,
                    default=DEFAULT_MODEL,
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available HuggingFace models for chat completion.

        Returns a curated list of popular models known to work well with the
        Inference API. HuggingFace has thousands of models, so we don't attempt
        to enumerate them all via API.
        """
        # Curated list of popular chat-capable models on HuggingFace
        curated_models = [
            ModelInfo(
                id="meta-llama/Llama-3.3-70B-Instruct",
                display_name="Llama 3.3 70B Instruct",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["tools"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="meta-llama/Llama-3.1-70B-Instruct",
                display_name="Llama 3.1 70B Instruct",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["tools"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="meta-llama/Llama-3.1-8B-Instruct",
                display_name="Llama 3.1 8B Instruct",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["tools", "fast"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="Qwen/Qwen2.5-72B-Instruct",
                display_name="Qwen 2.5 72B Instruct",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["tools"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="Qwen/Qwen2.5-Coder-32B-Instruct",
                display_name="Qwen 2.5 Coder 32B Instruct",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["tools"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                display_name="Mixtral 8x7B Instruct",
                context_window=32768,
                max_output_tokens=4096,
                capabilities=["tools"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="mistralai/Mistral-7B-Instruct-v0.3",
                display_name="Mistral 7B Instruct v0.3",
                context_window=32768,
                max_output_tokens=4096,
                capabilities=["tools", "fast"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="microsoft/Phi-3-mini-4k-instruct",
                display_name="Phi 3 Mini 4K Instruct",
                context_window=4096,
                max_output_tokens=2048,
                capabilities=["fast"],
                defaults={"temperature": 0.7, "max_tokens": 2048},
            ),
            ModelInfo(
                id="deepseek-ai/DeepSeek-R1",
                display_name="DeepSeek R1",
                context_window=128000,
                max_output_tokens=8192,
                capabilities=["tools", "thinking"],
                defaults={"temperature": 0.7, "max_tokens": 8192},
            ),
        ]

        return curated_models

    async def complete(
        self, request: ChatRequest, **kwargs: Any
    ) -> HuggingFaceChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            HuggingFaceChatResponse with content blocks, tool calls, usage
        """
        return await self._complete_chat_request(request, **kwargs)

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs: Any
    ) -> HuggingFaceChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            HuggingFaceChatResponse with content blocks
        """
        logger.info(
            f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages"
        )

        # Validate tool call sequences and repair if needed
        missing = self._find_missing_tool_results(request.messages)
        extra_tool_messages: list[dict[str, Any]] = []

        if missing:
            logger.warning(
                f"[PROVIDER] HuggingFace: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for call_id, _, _ in missing]}"
            )

            # Inject synthetic results and track repaired IDs to prevent infinite loops
            for call_id, tool_name, _ in missing:
                extra_tool_messages.append(
                    self._create_synthetic_result(call_id, tool_name)
                )
                # Track this ID so we don't detect it as missing again in future iterations
                self._repaired_tool_ids.add(call_id)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for call_id, tool_name, _ in missing
                        ],
                    },
                )

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        # Build OpenAI-compatible messages list
        hf_messages: list[dict[str, Any]] = []

        # Add system messages with native role
        for sys_msg in system_msgs:
            content = sys_msg.content if isinstance(sys_msg.content, str) else ""
            hf_messages.append({"role": "system", "content": content})

        # Convert developer messages to XML-wrapped user messages
        for dev_msg in developer_msgs:
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            wrapped = f"<context_file>\n{content}\n</context_file>"
            hf_messages.append({"role": "user", "content": wrapped})

        # Convert conversation messages
        conversation_msgs = self._convert_messages(
            [m.model_dump() for m in conversation]
        )
        hf_messages.extend(conversation_msgs)

        # Append synthetic tool results for any missing tool calls
        hf_messages.extend(extra_tool_messages)

        # Prepare request parameters
        model = kwargs.get("model", self.default_model)

        # Build params dict with 3-tier precedence: request -> kwargs -> instance default
        temperature = request.temperature or kwargs.get("temperature", self.temperature)
        max_tokens = request.max_output_tokens or kwargs.get(
            "max_tokens", self.max_tokens
        )

        # Build the API call kwargs
        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": hf_messages,
            "max_tokens": max_tokens,
        }

        # Only include temperature if set (some models may not support it)
        if temperature is not None:
            api_kwargs["temperature"] = temperature

        # Add tools if provided
        if request.tools:
            api_kwargs["tools"] = self._format_tools_from_request(request.tools)

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "huggingface",
                    "model": model,
                    "message_count": len(hf_messages),
                },
            )

            # DEBUG level: Truncated request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "huggingface",
                        "request": _truncate_values(
                            {
                                "model": model,
                                "messages": hf_messages,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            }
                        ),
                    },
                )

            # RAW level: Full request payload (if raw_debug enabled)
            if self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": "huggingface",
                        "request": api_kwargs,
                    },
                )

        start_time = time.time()

        # Call HuggingFace API via OpenAI-compatible endpoint
        try:
            response = await self.client.chat.completions.create(**api_kwargs)

            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from HuggingFace API")

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # Build usage info
                usage_info: dict[str, int] = {}
                usage_obj = getattr(response, "usage", None)
                if usage_obj:
                    if hasattr(usage_obj, "prompt_tokens"):
                        usage_info["input"] = usage_obj.prompt_tokens or 0
                    if hasattr(usage_obj, "completion_tokens"):
                        usage_info["output"] = usage_obj.completion_tokens or 0

                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "huggingface",
                        "model": model,
                        "usage": usage_info,
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Truncated response (if debug enabled)
                if self.debug:
                    response_dict = (
                        response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response)
                    )
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "huggingface",
                            "response": _truncate_values(response_dict),
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

                # RAW level: Full response (if raw_debug enabled)
                if self.raw_debug:
                    response_raw = (
                        response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response)
                    )
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": "huggingface",
                            "response": response_raw,
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # Convert to HuggingFaceChatResponse
            return self._convert_to_chat_response(response)

        except TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[PROVIDER] HuggingFace API call timed out after {self.timeout}s"
            )

            # Emit timeout event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "huggingface",
                        "model": model,
                        "status": "timeout",
                        "duration_ms": elapsed_ms,
                        "error": f"Request timed out after {self.timeout}s",
                    },
                )
            raise TimeoutError(f"HuggingFace API call timed out after {self.timeout}s")

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] HuggingFace API error: {e}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "huggingface",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
            raise

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from provider response.

        Args:
            response: Provider response

        Returns:
            List of tool calls
        """
        return response.tool_calls or []

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[str, str, dict]]:
        """Find tool calls without corresponding results.

        Scans message history to detect tool calls that were never answered
        with a tool result message.

        Filters out tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops across LLM iterations.

        Args:
            messages: List of conversation messages

        Returns:
            List of (call_id, tool_name, tool_arguments) tuples for unpaired calls
        """
        tool_calls: dict[str, tuple[str, dict]] = {}  # {call_id: (name, args)}
        tool_results: set[str] = set()  # {call_id}

        for msg in messages:
            if msg.role == "assistant":
                # Check for tool calls in content blocks
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "type") and block.type == "tool_use":
                            block_id = getattr(block, "id", "")
                            block_name = getattr(block, "name", "unknown")
                            block_input = getattr(block, "input", {})
                            if block_id:
                                tool_calls[block_id] = (block_name, block_input)
                        elif hasattr(block, "id") and hasattr(block, "name"):
                            # ToolCallBlock style
                            block_id = getattr(block, "id", "")
                            block_name = getattr(block, "name", "unknown")
                            block_input = getattr(block, "input", {})
                            if block_id:
                                tool_calls[block_id] = (block_name, block_input)
                # Also check tool_calls field
                if hasattr(msg, "tool_calls") and msg.tool_calls:  # pyright: ignore[reportAttributeAccessIssue]
                    for tc in msg.tool_calls:  # pyright: ignore[reportAttributeAccessIssue]
                        tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
                        tc_name = (
                            tc.name
                            if hasattr(tc, "name")
                            else tc.get("name", "unknown")
                        )
                        tc_args = (
                            tc.arguments
                            if hasattr(tc, "arguments")
                            else tc.get("arguments", {})
                        )
                        if tc_id:
                            tool_calls[tc_id] = (tc_name, tc_args)
            elif msg.role == "tool":
                # Tool result - mark as received
                tool_call_id = msg.tool_call_id if hasattr(msg, "tool_call_id") else ""
                if tool_call_id:
                    tool_results.add(tool_call_id)

        # Bound memory: clear tracking set if it grows too large
        if len(self._repaired_tool_ids) > 1000:
            self._repaired_tool_ids.clear()

        # Exclude IDs that have already been repaired to prevent infinite loops
        return [
            (call_id, name, args)
            for call_id, (name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> dict[str, Any]:
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.

        Args:
            call_id: The ID of the tool call that needs a result
            tool_name: The name of the tool that was called

        Returns:
            Dict in tool message format with error content
        """
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": (
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
        }

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert Amplifier message format to OpenAI-compatible format.

        Handles the conversion of:
        - Tool calls in assistant messages (Amplifier format -> OpenAI format)
        - Tool result messages
        - Developer messages (converted to XML-wrapped user messages)
        - Regular user/assistant/system messages
        - Structured content blocks (list of text/image blocks) -> plain string
        """
        hf_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Handle structured content (list of content blocks from Amplifier)
            # Convert to plain string for the OpenAI-compatible endpoint
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        # TextContent block: {"type": "text", "text": "..."}
                        if block.get("type") == "text" and "text" in block:
                            text_parts.append(block["text"])
                        # Image content block - not supported, describe as text
                        elif block.get("type") == "image":
                            source = block.get("source", {})
                            if source.get("type") == "url":
                                text_parts.append(
                                    f"[Image URL: {source.get('url', '')}]"
                                )
                        # ToolCallContent, ThinkingContent, etc. - handled by role-specific logic
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts) if text_parts else ""

            if role == "developer":
                # Developer messages -> XML-wrapped user messages (context files)
                wrapped = f"<context_file>\n{content}\n</context_file>"
                hf_messages.append({"role": "user", "content": wrapped})

            elif role == "assistant":
                # Check for tool_calls in Amplifier format
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Convert Amplifier tool_calls to OpenAI format
                    openai_tool_calls = []
                    for tc in msg["tool_calls"]:
                        tc_args = tc.get("arguments", {})
                        # OpenAI format requires arguments as JSON string
                        if isinstance(tc_args, dict):
                            tc_args_str = json.dumps(tc_args)
                        else:
                            tc_args_str = str(tc_args) if tc_args else "{}"

                        openai_tool_calls.append(
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", "") or tc.get("tool", ""),
                                    "arguments": tc_args_str,
                                },
                            }
                        )

                    hf_messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": openai_tool_calls,
                        }
                    )
                else:
                    # Check for tool_call blocks in content
                    tool_call_blocks = []
                    if isinstance(msg.get("content"), list):
                        for block in msg["content"]:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "tool_call"
                            ):
                                tc_input = block.get("input", {})
                                if isinstance(tc_input, dict):
                                    tc_args_str = json.dumps(tc_input)
                                else:
                                    tc_args_str = str(tc_input) if tc_input else "{}"
                                tool_call_blocks.append(
                                    {
                                        "id": block.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name", ""),
                                            "arguments": tc_args_str,
                                        },
                                    }
                                )

                    if tool_call_blocks:
                        hf_messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_call_blocks,
                            }
                        )
                    else:
                        # Regular assistant message
                        hf_messages.append({"role": "assistant", "content": content})

            elif role == "tool":
                # Tool result message
                hf_messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": msg.get("tool_call_id", ""),
                    }
                )

            else:
                # User, system, etc.
                hf_messages.append({"role": role, "content": content})

        return hf_messages

    def _format_tools_from_request(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to OpenAI function format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of OpenAI-formatted tool definitions
        """
        hf_tools: list[dict[str, Any]] = []
        for tool in tools:
            hf_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    },
                }
            )
        return hf_tools

    def _convert_to_chat_response(self, response: Any) -> HuggingFaceChatResponse:
        """Convert HuggingFace OpenAI-compatible response to HuggingFaceChatResponse.

        Handles the standard OpenAI Chat Completions response format:
        - response.choices[0].message.content -> TextBlock
        - response.choices[0].message.tool_calls -> ToolCallBlock / ToolCall
        - response.usage.prompt_tokens / completion_tokens -> Usage

        Args:
            response: HuggingFace API response (OpenAI Chat Completions format)

        Returns:
            HuggingFaceChatResponse with content blocks
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks: list[Any] = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        tool_calls: list[ToolCall] = []
        text_accumulator: list[str] = []

        # Extract from the first choice (standard OpenAI format)
        choices = getattr(response, "choices", None) or []
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None)

            if message:
                # Extract text content
                text_content = getattr(message, "content", None)
                if text_content:
                    content_blocks.append(TextBlock(text=text_content))
                    text_accumulator.append(text_content)
                    event_blocks.append(TextContent(text=text_content))

                # Extract tool calls
                msg_tool_calls = getattr(message, "tool_calls", None)
                if msg_tool_calls:
                    for tc in msg_tool_calls:
                        tool_id = getattr(tc, "id", "") or f"call_{uuid4().hex[:8]}"
                        function = getattr(tc, "function", None)
                        if function:
                            tool_name = getattr(function, "name", "")
                            tool_args_raw = getattr(function, "arguments", "{}")
                            # Parse arguments from JSON string to dict
                            if isinstance(tool_args_raw, str):
                                try:
                                    tool_args = json.loads(tool_args_raw)
                                except json.JSONDecodeError:
                                    logger.debug(
                                        "Failed to decode tool call arguments: %s",
                                        tool_args_raw,
                                    )
                                    tool_args = {}
                            elif isinstance(tool_args_raw, dict):
                                tool_args = tool_args_raw
                            else:
                                tool_args = {}

                            content_blocks.append(
                                ToolCallBlock(
                                    id=tool_id, name=tool_name, input=tool_args
                                )
                            )
                            tool_calls.append(
                                ToolCall(
                                    id=tool_id, name=tool_name, arguments=tool_args
                                )
                            )
                            event_blocks.append(
                                ToolCallContent(
                                    id=tool_id, name=tool_name, arguments=tool_args
                                )
                            )

            # Extract finish reason
            finish_reason = getattr(choice, "finish_reason", None)
        else:
            finish_reason = None

        # Build usage info
        usage_obj = getattr(response, "usage", None)
        input_tokens = 0
        output_tokens = 0
        if usage_obj:
            input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        combined_text = "\n\n".join(text_accumulator).strip()

        # Get model name from response
        response_model = getattr(response, "model", None)

        return HuggingFaceChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=finish_reason,
            raw_response=(
                response.model_dump()
                if self.raw_debug and hasattr(response, "model_dump")
                else None
            ),
            model_name=response_model,
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
        )
