"""Tests for the Claude runner that need no network: prompt assembly, caching,
and response handling against a fake client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from icdprompt.data import ClinicalRecord
from icdprompt.infer import (
    MIN_MAX_TOKENS,
    ClaudeRunner,
    GenerationConfig,
    ModelUnavailableError,
    cache_key,
)
from icdprompt.prompts import get_template


def _record(note: str = "Admitted with fever and cough. Treated with antibiotics.") -> ClinicalRecord:
    return ClinicalRecord(
        record_id="abc123def456",
        split="dev",
        note=note,
        label_code="J18",
        label_title="Pneumonia, organism unspecified",
        label_evidence="Discharge Diagnosis: Pneumonia",
        label_matched_text="pneumonia",
        n_chars=len(note),
    )


def _message(text: str, stop_reason: str = "end_turn", output_tokens: int = 7):
    """Build an object shaped like an ``anthropic.types.Message``."""
    return SimpleNamespace(
        model="claude-test",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=321, output_tokens=output_tokens),
    )


@dataclass
class FakeMessages:
    reply: str = "J18"
    calls: list[dict] = field(default_factory=list)

    def create(self, **params):
        self.calls.append(params)
        return _message(self.reply)


@dataclass
class FakeClient:
    messages: FakeMessages = field(default_factory=FakeMessages)


def _runner(tmp_path: Path, client: FakeClient, **config) -> ClaudeRunner:
    return ClaudeRunner(
        model_id="claude-test",
        config=GenerationConfig(use_batch=False, **config),
        cache_dir=tmp_path / "raw",
        client=client,
    )


def test_system_turn_is_lifted_out_of_messages(tmp_path: Path) -> None:
    client = FakeClient()
    runner = _runner(tmp_path, client)
    template = get_template("persona_clinical_coder")
    assert template.system, "the persona template is expected to carry a system turn"

    runner.run(template, [_record()], progress=False)

    (params,) = client.messages.calls
    assert params["system"] == template.system.strip()
    assert [m["role"] for m in params["messages"]] == ["user"]
    assert params["output_config"] == {"effort": "low"}
    assert params["max_tokens"] == MIN_MAX_TOKENS


def test_text_blocks_are_joined_and_thinking_skipped(tmp_path: Path) -> None:
    client = FakeClient(FakeMessages(reply="CODE: J18"))
    runner = _runner(tmp_path, client)
    (generation,) = runner.run(get_template("zero_shot_strict_format"), [_record()], progress=False)
    assert generation.text == "CODE: J18"
    assert generation.n_prompt_tokens == 321
    assert generation.n_new_tokens == 7
    assert generation.stop_reason == "end_turn"
    assert generation.from_cache is False


def test_second_run_is_served_from_cache(tmp_path: Path) -> None:
    client = FakeClient()
    runner = _runner(tmp_path, client)
    template = get_template("zero_shot_minimal")
    first = runner.run(template, [_record()], progress=False)
    second = runner.run(template, [_record()], progress=False)
    assert len(client.messages.calls) == 1
    assert second[0].from_cache is True
    assert second[0].text == first[0].text
    cached = json.loads(next((tmp_path / "raw").glob("*.json")).read_text())
    assert cached["request"]["messages"][0]["role"] == "user"
    assert cached["config"]["effort"] == "low"


def test_cache_key_changes_with_effort_and_prompt() -> None:
    template = get_template("zero_shot_minimal")
    record = _record()
    low = cache_key("m", template, record, GenerationConfig(effort="low"), "prompt")
    high = cache_key("m", template, record, GenerationConfig(effort="high"), "prompt")
    other_prompt = cache_key("m", template, record, GenerationConfig(effort="low"), "prompt2")
    assert len({low, high, other_prompt}) == 3


def test_refusal_is_recorded_not_raised(tmp_path: Path) -> None:
    class Refusing(FakeMessages):
        def create(self, **params):
            self.calls.append(params)
            return _message("", stop_reason="refusal", output_tokens=0)

    client = FakeClient(Refusing())
    runner = _runner(tmp_path, client)
    (generation,) = runner.run(get_template("zero_shot_minimal"), [_record()], progress=False)
    assert generation.text == ""
    assert generation.stop_reason == "refusal"


def test_authentication_error_becomes_model_unavailable(tmp_path: Path) -> None:
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    class Broken(FakeMessages):
        def create(self, **params):
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            response = httpx2.Response(401, request=request)
            raise anthropic.AuthenticationError("bad key", response=response, body=None)

    runner = _runner(tmp_path, FakeClient(Broken()))
    with pytest.raises(ModelUnavailableError):
        runner.run(get_template("zero_shot_minimal"), [_record()], progress=False)


def test_config_rejects_small_budgets_and_unknown_effort() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(max_tokens=16)
    with pytest.raises(ValueError):
        GenerationConfig(effort="turbo")
