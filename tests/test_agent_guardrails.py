from __future__ import annotations

import unittest

from agent.agent import (
    _message_likely_requires_tools,
    _should_fail_text_only_workspace_response,
    _should_retry_text_only_workspace_response,
)


class AgentGuardrailTests(unittest.TestCase):
    def test_workspace_build_request_is_detected_as_tool_oriented(self) -> None:
        self.assertTrue(
            _message_likely_requires_tools(
                "build me a nextjs web page with some 3d abstract animations"
            )
        )
        self.assertFalse(_message_likely_requires_tools("hello there"))

    def test_small_local_model_gets_extra_retry_on_low_effort_text(self) -> None:
        should_retry = _should_retry_text_only_workspace_response(
            user_message="build me a nextjs web page with some 3d abstract animations",
            response_text="Hello! How can I assist you today?",
            tool_calls_present=False,
            retry_count=1,
            model_name="qwen2.5-coder:1.5b",
            base_url="http://localhost:11434/v1",
        )

        self.assertTrue(should_retry)

    def test_non_local_or_stronger_model_stops_retrying_sooner(self) -> None:
        should_retry = _should_retry_text_only_workspace_response(
            user_message="build me a nextjs web page with some 3d abstract animations",
            response_text="Hello! How can I assist you today?",
            tool_calls_present=False,
            retry_count=1,
            model_name="qwen2.5-coder:3b",
            base_url="http://localhost:11434/v1",
        )

        self.assertFalse(should_retry)

    def test_low_effort_text_only_reply_fails_after_retries(self) -> None:
        self.assertTrue(
            _should_fail_text_only_workspace_response(
                user_message="build me a nextjs web page with some 3d abstract animations",
                response_text="build_nextjs_webpage_nextjs_with_3d_asthetics_abstract_animations_using_three_js",
                tool_calls_present=False,
            )
        )

    def test_substantive_small_model_attempt_is_not_auto_failed(self) -> None:
        self.assertFalse(
            _should_fail_text_only_workspace_response(
                user_message="build me a nextjs web page with some 3d abstract animations",
                response_text=(
                    "I could not directly edit the workspace, but here is a concrete Next.js page structure, "
                    "a Three.js scene plan, animation layers, component breakdown, and the files that should "
                    "be created so you can wire it in manually."
                ),
                tool_calls_present=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
