from __future__ import annotations

from types import SimpleNamespace
import unittest

from utils.model_health import ModelHealthChecker


class _FakeCompletions:
    def __init__(self, response):
        self._response = response

    async def create(self, **_: object):
        return self._response


class _FakeClient:
    def __init__(self, response) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


class ModelHealthCheckerTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_model_marks_tool_capable_model_as_working(self) -> None:
        checker = ModelHealthChecker(
            provider="default",
            base_url="http://localhost:11434/v1",
            api_key="unused",
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[{"id": "call-1"}],
                    )
                )
            ]
        )

        record = await checker._probe_model(_FakeClient(response), "qwen2.5-coder:3b")

        self.assertEqual(record.status, "working")
        self.assertIsNone(record.detail)

    async def test_probe_model_marks_plain_text_tool_smoke_test_as_limited(self) -> None:
        checker = ModelHealthChecker(
            provider="default",
            base_url="http://localhost:11434/v1",
            api_key="unused",
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello! How can I assist you today?",
                        tool_calls=[],
                    )
                )
            ]
        )

        record = await checker._probe_model(_FakeClient(response), "qwen2.5-coder:1.5b")

        self.assertEqual(record.status, "limited")
        self.assertIsNotNone(record.detail)
        assert record.detail is not None
        self.assertIn("plain text instead of a tool call", record.detail)
        self.assertIn("Hello! How can I assist you today?", record.detail)


if __name__ == "__main__":
    unittest.main()
