from __future__ import annotations

from src.clients.qwen_local_client import QwenLocalClient


def test_extract_message_text_ignores_null_content_and_falls_back_to_choice_text() -> (
    None
):
    client = QwenLocalClient(model_name="qwen-test", config={})
    response_json = {
        "choices": [
            {
                "message": {
                    "content": None,
                },
                "text": '{"decision":"merge","patch":{"type":"chart"}}',
            }
        ]
    }

    assert (
        client._extract_message_text(response_json)
        == '{"decision":"merge","patch":{"type":"chart"}}'
    )


def test_extract_message_text_reads_nested_content_list() -> None:
    client = QwenLocalClient(model_name="qwen-test", config={})
    response_json = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "output_text", "text": '{"decision":"keep_mineru"}'}
                    ]
                }
            }
        ]
    }

    assert client._extract_message_text(response_json) == '{"decision":"keep_mineru"}'
