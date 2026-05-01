from __future__ import annotations

import unittest
from pathlib import Path

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from client.llm_client import LLMClient, _merge_tool_call_delta_payload
from client.response import ToolCall, ToolResultMessage
from config.config import Config
from context.manager import ContextManager


class GeminiCompatTests(unittest.TestCase):
    def test_config_builds_gemini_request_overrides(self) -> None:
        config = Config(
            cwd=Path("."),
            active_model_profile="gemini",
            models={
                "gemini": {
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "api_key_env": "GEMINI_API_KEY",
                    "gemini": {
                        "reasoning_effort": "low",
                        "cached_content": "cachedContents/demo",
                        "thinking_config": {
                            "include_thoughts": True,
                        },
                    },
                }
            },
        )

        self.assertTrue(config.profile_uses_gemini_openai_compat("gemini"))
        self.assertEqual(
            config.request_overrides,
            {
                "reasoning_effort": "low",
                "extra_body": {
                    "google": {
                        "cached_content": "cachedContents/demo",
                        "thinking_config": {
                            "include_thoughts": True,
                        },
                    }
                },
            },
        )

    def test_config_rejects_conflicting_gemini_reasoning_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            Config(
                cwd=Path("."),
                active_model_profile="gemini",
                models={
                    "gemini": {
                        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                        "gemini": {
                            "reasoning_effort": "low",
                            "thinking_config": {
                                "thinking_level": "medium",
                            },
                        },
                    }
                },
            )

    def test_client_build_chat_kwargs_includes_gemini_overrides(self) -> None:
        config = Config(
            cwd=Path("."),
            active_model_profile="gemini",
            models={
                "gemini": {
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "api_key_env": "GEMINI_API_KEY",
                    "model": {
                        "name": "gemini-2.5-pro",
                        "temperature": 0.2,
                        "max_output_tokens": 4096,
                    },
                    "gemini": {
                        "reasoning_effort": "medium",
                        "thinking_config": {
                            "include_thoughts": True,
                        },
                    },
                }
            },
        )

        client = LLMClient(config)
        kwargs = client._build_chat_kwargs(
            [{"role": "user", "content": "hello"}],
            tools=[
                {
                    "name": "echo",
                    "description": "Echo input",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            stream=True,
        )

        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertEqual(
            kwargs["extra_body"],
            {"google": {"thinking_config": {"include_thoughts": True}}},
        )
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})
        self.assertEqual(kwargs["tool_choice"], "auto")

    def test_tool_call_to_openai_dict_preserves_extra_content(self) -> None:
        payload = {
            "id": "function-call-1",
            "type": "function",
            "extra_content": {
                "google": {
                    "thought_signature": "<Signature A>",
                }
            },
            "function": {
                "name": "check_flight",
                "arguments": '{"flight":"AA100"}',
            },
        }

        tool_call = ToolCall(
            call_id="function-call-1",
            name="check_flight",
            arguments={"flight": "AA100"},
            raw_payload=payload,
        )

        self.assertEqual(tool_call.to_openai_dict(), payload)

    def test_tool_result_message_includes_name(self) -> None:
        message = ToolResultMessage(
            tool_call_id="function-call-1",
            name="check_flight",
            content='{"status":"delayed"}',
        )

        self.assertEqual(
            message.to_openai_message(),
            {
                "role": "tool",
                "tool_call_id": "function-call-1",
                "name": "check_flight",
                "content": '{"status":"delayed"}',
            },
        )

    def test_context_manager_serializes_named_tool_result(self) -> None:
        manager = ContextManager(
            Config(cwd=Path(".")),
            user_memory=None,
            tools=None,
        )

        manager.add_tool_result(
            "function-call-1",
            '{"status":"delayed"}',
            name="check_flight",
        )

        messages = manager.get_messages()
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "function-call-1")
        self.assertEqual(messages[-1]["name"], "check_flight")

    def test_merge_tool_call_delta_payload_keeps_gemini_signature(self) -> None:
        payload = {
            "id": "",
            "type": "function",
            "function": {
                "name": "",
                "arguments": "",
            },
        }
        delta = ChoiceDeltaToolCall(
            index=0,
            id="function-call-1",
            type="function",
            extra_content={"google": {"thought_signature": "<Signature A>"}},
            function={
                "name": "check_flight",
                "arguments": '{"flight":"AA100"}',
            },
        )

        merged = _merge_tool_call_delta_payload(payload, delta)

        self.assertEqual(merged["id"], "function-call-1")
        self.assertEqual(
            merged["extra_content"]["google"]["thought_signature"],
            "<Signature A>",
        )
        self.assertEqual(merged["function"]["name"], "check_flight")
        self.assertEqual(merged["function"]["arguments"], '{"flight":"AA100"}')


if __name__ == "__main__":
    unittest.main()
