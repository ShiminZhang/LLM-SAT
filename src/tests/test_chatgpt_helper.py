"""Tests for OpenAI Responses request construction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llmsat.utils import chatgpt_helper


class RecordingResponses:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="candidate")


class TestChatGPTResponse(unittest.TestCase):
    def test_gpt56_uses_reasoning_effort_without_temperature(self) -> None:
        responses = RecordingResponses()
        client = SimpleNamespace(responses=responses)
        with patch.object(
            chatgpt_helper, "_get_openai_client", return_value=client
        ), patch.dict(
            chatgpt_helper.os.environ,
            {"OPENAI_REASONING_EFFORT": "high"},
        ):
            output = chatgpt_helper.get_response_from_chatgpt(
                "prompt", model="gpt-5.6-luna", temperature=0.7
            )

        self.assertEqual(output, "candidate")
        self.assertEqual(responses.request["reasoning"], {"effort": "high"})
        self.assertNotIn("temperature", responses.request)

    def test_legacy_model_keeps_temperature(self) -> None:
        responses = RecordingResponses()
        client = SimpleNamespace(responses=responses)
        with patch.object(
            chatgpt_helper, "_get_openai_client", return_value=client
        ), patch.dict(
            chatgpt_helper.os.environ,
            {"OPENAI_REASONING_EFFORT": "high"},
        ):
            chatgpt_helper.get_response_from_chatgpt(
                "prompt", model="gpt-4.1", temperature=0.3
            )

        self.assertEqual(responses.request["temperature"], 0.3)
        self.assertNotIn("reasoning", responses.request)


if __name__ == "__main__":
    unittest.main()
