from __future__ import annotations

import json

import requests

from src.schema import ImageTask
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


def test_build_payload_disables_thinking_for_flowchart_adjudication(tmp_path) -> None:
    client = QwenLocalClient(model_name="qwen-test", config={})
    image_path = tmp_path / "demo.png"
    image_path.write_bytes(b"fake-image")
    payload = client._build_payload(
        image_task=ImageTask(
            image_id="img-1",
            image_path=str(image_path),
            file_name="demo.png",
            file_ext=".png",
        ),
        prompt="prompt",
        context={"mode": "flowchart_adjudication"},
        disable_thinking=True,
    )

    assert payload["extra_body"]["enable_thinking"] is False


def test_qwen_client_falls_back_when_disable_thinking_is_rejected(
    monkeypatch,
    tmp_path,
) -> None:
    class DummyResponse:
        def __init__(
            self,
            *,
            status_code: int,
            json_payload: dict[str, object] | None = None,
            text: str = "",
        ) -> None:
            self.status_code = status_code
            self._json_payload = json_payload
            self.text = text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")

        def json(self) -> dict[str, object]:
            if self._json_payload is None:
                raise ValueError("missing json")
            return self._json_payload

    calls: list[dict[str, object]] = []

    def fake_post(url, headers, json, timeout):  # type: ignore[no-untyped-def]
        del url, headers, timeout
        calls.append(json)
        if len(calls) == 1:
            return DummyResponse(status_code=400, text="unsupported extra_body")
        return DummyResponse(
            status_code=200,
            json_payload={
                "choices": [
                    {
                        "message": {
                            "content": '{"decision":"keep_mineru","patch":{},"reason":"ok"}'
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    monkeypatch.setattr("src.clients.qwen_local_client.requests.post", fake_post)

    image_path = tmp_path / "demo.png"
    image_path.write_bytes(b"fake-image")
    client = QwenLocalClient(model_name="qwen-test", config={})
    output = client.analyze(
        image_task=ImageTask(
            image_id="img-flow",
            image_path=str(image_path),
            file_name="demo.png",
            file_ext=".png",
        ),
        prompt="prompt",
        context={
            "mode": "flowchart_adjudication",
            "issue_payload": {"issue_id": "flow-1"},
        },
    )

    assert output.success is True
    assert len(calls) == 2
    assert calls[0]["extra_body"]["enable_thinking"] is False
    assert "extra_body" not in calls[1]
    assert output.parsed["_request_control"]["thinking_mode"] == (
        "disabled_requested_fallback_to_default"
    )
    assert output.parsed["_request_control"]["disable_thinking_fallback_used"] is True
    assert json.loads(output.raw_text)["decision"] == "keep_mineru"
