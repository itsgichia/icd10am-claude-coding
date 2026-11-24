"""Claude API inference with an on-disk generation cache.

Every raw model output is cached under ``results/raw/`` keyed by a hash of
(model, template, record, decoding parameters, rendered prompt).  Re-running a
study, adding a template, or fixing the post-processor therefore costs nothing
for work already done, and the cache is the audit trail: each file holds the
exact request that produced the output.

Two transports are available.  The Message Batches API is the default because
it halves the token price and nothing in this study is latency-sensitive.  The
synchronous Messages API is kept for smoke tests and small runs.

Design decisions that matter for the measurements:

* **Thinking is left at the model's default and effort is set explicitly.**
  Current Claude models reason before answering; the budget is controlled by
  ``output_config.effort``, recorded in every cache entry and every result row.
* **No server-side refusal fallback.**  A refusal is recorded as a generation
  with empty text and ``stop_reason == "refusal"``.  Routing it to a different
  model would mix models inside one result row, which this study must not do.
* **Raw text is scored, not structured output.**  The prompt strategies under
  test include their own output contracts; constraining the response format at
  the API would decide the experiment in advance.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .data import ClinicalRecord
from .prompts import PromptTemplate

__all__ = [
    "DEFAULT_MODEL_ID",
    "MODEL_IDS",
    "GenerationConfig",
    "Generation",
    "ModelUnavailableError",
    "ClaudeRunner",
    "cache_key",
]

#: Default model.  The study runs the three current Claude tiers; Opus is the
#: default because it is the most capable one and therefore the ceiling.
DEFAULT_MODEL_ID: Final[str] = "claude-opus-5-5"

#: The three tiers compared in the study, smallest first.
MODEL_IDS: Final[tuple[str, ...]] = (
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
)

#: Lower bound on ``max_tokens``.  Reasoning tokens count toward the limit, so
#: the template budgets written for a non-reasoning model (16 tokens for a bare
#: code) would truncate the answer before it was written.
MIN_MAX_TOKENS: Final[int] = 4096


class ModelUnavailableError(RuntimeError):
    """Raised when the API cannot be reached or refuses the credentials."""


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters.

    Attributes:
        effort: ``output_config.effort``; ``low`` is appropriate for a
            single-label classification and keeps reasoning spend small.
        max_tokens: Output ceiling per request, including reasoning tokens.
        use_batch: Send uncached requests through the Message Batches API.
        poll_seconds: Interval between batch status checks.
        batch_timeout_seconds: Give up waiting on a batch after this long.
    """

    effort: str = "low"
    max_tokens: int = MIN_MAX_TOKENS
    use_batch: bool = True
    poll_seconds: float = 30.0
    batch_timeout_seconds: float = 6 * 3600.0

    def __post_init__(self) -> None:
        if self.effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"unknown effort level {self.effort!r}")
        if self.max_tokens < MIN_MAX_TOKENS:
            raise ValueError(f"max_tokens must be at least {MIN_MAX_TOKENS}")


@dataclass(frozen=True)
class Generation:
    """One raw model output.

    Attributes:
        record_id: Record the output belongs to.
        template: Template name used.
        text: Concatenated text blocks of the response.
        n_prompt_tokens: Input tokens billed.
        n_new_tokens: Output tokens billed, reasoning included.
        seconds: Wall-clock time attributed to this record (0 for batches).
        from_cache: Whether the output was read from disk.
        stop_reason: The API's ``stop_reason``; ``refusal`` and ``max_tokens``
            are the two that change how a result should be read.
    """

    record_id: str
    template: str
    text: str
    n_prompt_tokens: int = 0
    n_new_tokens: int = 0
    seconds: float = 0.0
    from_cache: bool = False
    stop_reason: str = ""


def cache_key(
    model_id: str,
    template: PromptTemplate,
    record: ClinicalRecord,
    config: GenerationConfig,
    prompt_text: str,
) -> str:
    """Return the cache key for one (model, template, record, params) combination.

    Args:
        model_id: Claude model identifier.
        template: Template being run.
        record: Record being coded.
        config: Decoding parameters.
        prompt_text: The fully rendered request, so wording changes invalidate.

    Returns:
        A 16-character hex digest.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "template": template.name,
            "record": record.record_id,
            "effort": config.effort,
            "max_tokens": config.max_tokens,
            "prompt_sha": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _text_of(message: Any) -> str:
    """Join the text blocks of a response; thinking blocks are skipped."""
    return "".join(block.text for block in message.content if block.type == "text").strip()


class ClaudeRunner:
    """Runs prompt templates over records through the Claude API, with caching.

    Args:
        model_id: Claude model identifier.
        config: Decoding parameters.
        cache_dir: Directory for raw outputs.  Defaults to ``results/raw``
            beside the package.
        client: An ``anthropic.Anthropic`` instance; built lazily when omitted
            so prompts can be rendered and the cache read without credentials.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        config: GenerationConfig | None = None,
        cache_dir: Path | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.config = config or GenerationConfig()
        self.cache_dir = cache_dir or Path(__file__).resolve().parent.parent / "results" / "raw"
        self._client = client

    # ------------------------------------------------------------------ client
    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as error:  # pragma: no cover - environment
                raise ModelUnavailableError("the anthropic package is not installed") from error
            self._client = anthropic.Anthropic()
        return self._client

    def is_available(self) -> bool:
        """Return whether a request can be made with the current credentials."""
        try:
            self._ensure_client().models.retrieve(self.model_id)
        except Exception:
            return False
        return True

    # ------------------------------------------------------------------- cache
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(key).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ----------------------------------------------------------------- prompts
    def _request_params(self, template: PromptTemplate, record: ClinicalRecord) -> dict[str, Any]:
        """Build the Messages API parameters for one record.

        The template's optional system turn becomes the ``system`` field; the
        single user turn is the message list.
        """
        messages = template.render_messages(record.note)
        params: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.config.max_tokens,
            "output_config": {"effort": self.config.effort},
            "messages": [m for m in messages if m["role"] != "system"],
        }
        system = [m["content"] for m in messages if m["role"] == "system"]
        if system:
            params["system"] = "\n\n".join(system)
        return params

    @staticmethod
    def _prompt_text(params: dict[str, Any]) -> str:
        """Serialise the prompt-bearing part of a request for the cache key."""
        return json.dumps(
            {"system": params.get("system"), "messages": params["messages"]}, sort_keys=True
        )

    def _from_cache(
        self, template: PromptTemplate, record: ClinicalRecord, cached: dict[str, Any]
    ) -> Generation:
        usage = cached.get("usage") or {}
        return Generation(
            record_id=record.record_id,
            template=template.name,
            text=str(cached.get("text", "")),
            n_prompt_tokens=int(usage.get("input_tokens", 0)),
            n_new_tokens=int(usage.get("output_tokens", 0)),
            seconds=float(cached.get("seconds", 0.0)),
            from_cache=True,
            stop_reason=str(cached.get("stop_reason", "")),
        )

    def _store(
        self,
        template: PromptTemplate,
        record: ClinicalRecord,
        params: dict[str, Any],
        message: Any,
        seconds: float,
    ) -> Generation:
        """Write one response to the cache and return it as a generation."""
        usage = {
            "input_tokens": int(getattr(message.usage, "input_tokens", 0)),
            "output_tokens": int(getattr(message.usage, "output_tokens", 0)),
        }
        stop_reason = str(message.stop_reason or "")
        stop_details = getattr(message, "stop_details", None)
        text = _text_of(message)
        payload = {
            "model": self.model_id,
            "response_model": getattr(message, "model", self.model_id),
            "template": template.name,
            "record_id": record.record_id,
            "config": asdict(self.config),
            "request": {"system": params.get("system"), "messages": params["messages"]},
            "text": text,
            "stop_reason": stop_reason,
            "refusal_category": getattr(stop_details, "category", None) if stop_details else None,
            "usage": usage,
            "seconds": seconds,
        }
        self._write_cache(
            cache_key(self.model_id, template, record, self.config, self._prompt_text(params)),
            payload,
        )
        return Generation(
            record_id=record.record_id,
            template=template.name,
            text=text,
            n_prompt_tokens=usage["input_tokens"],
            n_new_tokens=usage["output_tokens"],
            seconds=seconds,
            from_cache=False,
            stop_reason=stop_reason,
        )

    # --------------------------------------------------------------------- run
    def run(
        self,
        template: PromptTemplate,
        records: Sequence[ClinicalRecord],
        progress: bool = True,
    ) -> list[Generation]:
        """Run one template over records, using and filling the cache.

        Args:
            template: Template to run.
            records: Records to code.
            progress: Whether to print progress lines.

        Returns:
            One :class:`Generation` per record, in input order.

        Raises:
            ModelUnavailableError: If uncached records remain and the API
                cannot be used, or if some records produced no response.
        """
        results: dict[str, Generation] = {}
        pending: list[tuple[ClinicalRecord, dict[str, Any]]] = []

        for record in records:
            params = self._request_params(template, record)
            cached = self._read_cache(
                cache_key(self.model_id, template, record, self.config, self._prompt_text(params))
            )
            if cached is not None:
                results[record.record_id] = self._from_cache(template, record, cached)
            else:
                pending.append((record, params))

        if pending:
            if progress:
                transport = "batch" if self.config.use_batch else "sync"
                print(
                    f"  [{template.name}] {len(results)} cached, {len(pending)} to generate "
                    f"({transport}, {self.model_id}, effort {self.config.effort})",
                    flush=True,
                )
            generate = self._generate_batch if self.config.use_batch else self._generate_sync
            for generation in generate(template, pending, progress):
                results[generation.record_id] = generation

        missing = [r.record_id for r in records if r.record_id not in results]
        if missing:
            raise ModelUnavailableError(
                f"{len(missing)} record(s) produced no response for {template.name}; "
                "re-run to retry them"
            )
        return [results[record.record_id] for record in records]

    # ------------------------------------------------------------- transports
    def _generate_sync(
        self,
        template: PromptTemplate,
        pending: Sequence[tuple[ClinicalRecord, dict[str, Any]]],
        progress: bool,
    ) -> list[Generation]:
        """One synchronous request per record."""
        import anthropic

        client = self._ensure_client()
        outputs: list[Generation] = []
        for index, (record, params) in enumerate(pending, start=1):
            began = time.perf_counter()
            try:
                message = client.messages.create(**params)
            except anthropic.AuthenticationError as error:
                raise ModelUnavailableError(f"authentication failed: {error}") from error
            except anthropic.NotFoundError as error:
                raise ModelUnavailableError(f"model {self.model_id!r} not found: {error}") from error
            except anthropic.RateLimitError as error:
                raise ModelUnavailableError(f"rate limited after retries: {error}") from error
            except anthropic.APIStatusError as error:
                raise ModelUnavailableError(f"API error {error.status_code}: {error}") from error
            except anthropic.APIConnectionError as error:
                raise ModelUnavailableError(f"could not reach the API: {error}") from error
            elapsed = time.perf_counter() - began
            outputs.append(self._store(template, record, params, message, elapsed))
            if progress and (index % 10 == 0 or index == len(pending)):
                print(f"  [{template.name}] {index}/{len(pending)} ({elapsed:.1f}s last)", flush=True)
        return outputs

    def _generate_batch(
        self,
        template: PromptTemplate,
        pending: Sequence[tuple[ClinicalRecord, dict[str, Any]]],
        progress: bool,
    ) -> list[Generation]:
        """Submit all uncached records as one batch, wait, and collect.

        Errored, canceled or expired requests are reported and left uncached so
        that the next run retries exactly those records.
        """
        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        client = self._ensure_client()
        by_id = {record.record_id: (record, params) for record, params in pending}
        requests = [
            Request(custom_id=record.record_id, params=MessageCreateParamsNonStreaming(**params))
            for record, params in pending
        ]
        began = time.perf_counter()
        try:
            batch = client.messages.batches.create(requests=requests)
        except anthropic.AuthenticationError as error:
            raise ModelUnavailableError(f"authentication failed: {error}") from error
        except anthropic.APIStatusError as error:
            raise ModelUnavailableError(f"batch rejected ({error.status_code}): {error}") from error
        except anthropic.APIConnectionError as error:
            raise ModelUnavailableError(f"could not reach the API: {error}") from error
        if progress:
            print(f"  [{template.name}] batch {batch.id} submitted ({len(requests)} requests)", flush=True)

        while batch.processing_status != "ended":
            if time.perf_counter() - began > self.config.batch_timeout_seconds:
                raise ModelUnavailableError(f"batch {batch.id} did not finish in time")
            time.sleep(self.config.poll_seconds)
            batch = client.messages.batches.retrieve(batch.id)
            if progress:
                counts = batch.request_counts
                print(
                    f"  [{template.name}] {batch.processing_status}: {counts.succeeded} done, "
                    f"{counts.processing} processing, {counts.errored} errored",
                    flush=True,
                )

        outputs: list[Generation] = []
        failures: dict[str, str] = {}
        for result in client.messages.batches.results(batch.id):
            record, params = by_id[result.custom_id]
            if result.result.type == "succeeded":
                outputs.append(self._store(template, record, params, result.result.message, 0.0))
            elif result.result.type == "errored":
                failures[result.custom_id] = result.result.error.type
            else:
                failures[result.custom_id] = result.result.type
        if failures and progress:
            kinds = sorted(set(failures.values()))
            print(f"  [{template.name}] {len(failures)} request(s) failed: {kinds}", flush=True)
        return outputs
