from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "data" / "policies.md"
DEFAULT_BENCHMARK_PATH = ROOT / "data" / "benchmark.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "skill_optimization"

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
SKILL_NAME = "customer_service_policy"
DEFERRAL_SIGNALS = (
    "联系人工",
    "不确定",
    "不清楚",
    "不知道",
    "无法确认",
    "无法判断",
)
NEG_PREFIXES = ("不", "无", "非", "未", "没")
NEG_WINDOW = 4


class CandidateValidationError(ValueError):
    """The model did not return a complete, safe Skill candidate."""


class BenchmarkValidationError(ValueError):
    """The benchmark is malformed or leaks examples across splits."""


class LLMRequestError(RuntimeError):
    def __init__(self, message: str, attempts: int, latency_ms: float):
        super().__init__(message)
        self.attempts = attempts
        self.latency_ms = latency_ms


def _read_attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None
    reasoning_tokens: int | None = None

    @classmethod
    def from_provider(cls, usage: Any) -> "TokenUsage":
        details = _read_attr(usage, "completion_tokens_details")
        return cls(
            prompt_tokens=_optional_int(_read_attr(usage, "prompt_tokens")),
            completion_tokens=_optional_int(_read_attr(usage, "completion_tokens")),
            total_tokens=_optional_int(_read_attr(usage, "total_tokens")),
            prompt_cache_hit_tokens=_optional_int(
                _read_attr(usage, "prompt_cache_hit_tokens")
            ),
            prompt_cache_miss_tokens=_optional_int(
                _read_attr(usage, "prompt_cache_miss_tokens")
            ),
            reasoning_tokens=_optional_int(_read_attr(details, "reasoning_tokens")),
        )

    @property
    def comparable(self) -> bool:
        return (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["comparable"] = self.comparable
        return result


@dataclass(frozen=True)
class LLMCallResult:
    purpose: str
    text: str
    usage: TokenUsage
    latency_ms: float
    attempts: int
    requested_model: str
    response_model: str | None
    system_fingerprint: str | None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 3),
            "attempts": self.attempts,
            "retries": self.attempts - 1,
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "system_fingerprint": self.system_fingerprint,
            "finish_reason": self.finish_reason,
        }


class DeepSeekGateway:
    """Small OpenAI-compatible DeepSeek client with measurable calls."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        client: Any = None,
        client_factory: Callable[..., Any] | None = None,
        max_retries: int = 2,
        request_timeout_seconds: float = 120.0,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0")
        if client is not None and client_factory is not None:
            raise ValueError("pass either client or client_factory, not both")
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.sdk_max_retries = 0
        self._clock = clock
        self._sleep = sleep

        if client is None:
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY is required for a live run")
            if client_factory is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError(
                        "Missing dependency: pip install -r requirements.txt"
                    ) from exc
                client_factory = OpenAI
            client = client_factory(
                api_key=api_key,
                base_url=base_url,
                timeout=request_timeout_seconds,
                max_retries=0,
            )
        self.client = client

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        purpose: str,
        max_tokens: int,
    ) -> LLMCallResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        started = self._clock()
        last_error: Exception | None = None
        attempts = 0

        for attempts in range(1, self.max_retries + 2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    temperature=0,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                choice = response.choices[0]
                finish_reason = _read_attr(choice, "finish_reason")
                if finish_reason != "stop":
                    raise RuntimeError(
                        "provider returned incomplete response: "
                        f"finish_reason={finish_reason!r}"
                    )
                text = _read_attr(_read_attr(choice, "message"), "content")
                if not text or not str(text).strip():
                    raise RuntimeError("provider returned an empty message")
                latency_ms = (self._clock() - started) * 1000
                return LLMCallResult(
                    purpose=purpose,
                    text=str(text).strip(),
                    usage=TokenUsage.from_provider(_read_attr(response, "usage")),
                    latency_ms=latency_ms,
                    attempts=attempts,
                    requested_model=self.model,
                    response_model=_read_attr(response, "model"),
                    system_fingerprint=_read_attr(response, "system_fingerprint"),
                    finish_reason=finish_reason,
                )
            except Exception as exc:  # provider SDK exposes several error classes
                last_error = exc
                if attempts <= self.max_retries:
                    self._sleep(0.5 * (2 ** (attempts - 1)))

        latency_ms = (self._clock() - started) * 1000
        raise LLMRequestError(
            f"DeepSeek request failed after {attempts} attempts: {last_error}",
            attempts=attempts,
            latency_ms=latency_ms,
        ) from last_error


@dataclass(frozen=True)
class SkillCandidate:
    name: str
    version: int
    description: str
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def chars(self) -> int:
        return len(self.content)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "sha256": self.sha256,
            "chars": self.chars,
        }


def _strip_single_markdown_fence(raw: str) -> str:
    text = raw.strip()
    if "```" not in text:
        return text
    match = re.fullmatch(
        r"```(?:markdown|md)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if not match:
        raise CandidateValidationError(
            "candidate must be raw Markdown or one complete Markdown code fence"
        )
    return match.group(1).strip()


def parse_skill_candidate(
    raw: str,
    *,
    expected_version: int,
    expected_name: str = SKILL_NAME,
) -> SkillCandidate:
    text = _strip_single_markdown_fence(raw)
    if not text.startswith("---\n"):
        raise CandidateValidationError("candidate must start with YAML-like frontmatter")
    if len(text) > 50_000:
        raise CandidateValidationError("candidate is unexpectedly large")
    if "\x00" in text:
        raise CandidateValidationError("candidate contains a null byte")

    lines = text.splitlines()
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise CandidateValidationError("frontmatter is not closed") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        if not line.strip():
            continue
        if ":" not in line:
            raise CandidateValidationError(f"invalid frontmatter line: {line!r}")
        key, value = (part.strip() for part in line.split(":", 1))
        if key in metadata:
            raise CandidateValidationError(f"duplicate frontmatter key: {key}")
        metadata[key] = value

    for required in ("name", "description", "type", "version"):
        if not metadata.get(required):
            raise CandidateValidationError(f"missing frontmatter field: {required}")

    name = metadata["name"]
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
        raise CandidateValidationError("unsafe Skill name")
    if name != expected_name:
        raise CandidateValidationError(
            f"Skill name must be {expected_name!r}, got {name!r}"
        )
    if metadata["type"] != "knowledge":
        raise CandidateValidationError("Skill type must be 'knowledge'")
    try:
        version = int(metadata["version"])
    except ValueError as exc:
        raise CandidateValidationError("Skill version must be an integer") from exc
    if version != expected_version:
        raise CandidateValidationError(
            f"Skill version must be {expected_version}, got {version}"
        )

    body = "\n".join(lines[closing + 1 :]).strip()
    if not body or not re.search(r"^##\s+", body, flags=re.MULTILINE):
        raise CandidateValidationError("candidate must contain a non-empty Markdown body")
    if body.startswith(("@@", "diff --git", "{\"patch\"")):
        raise CandidateValidationError("candidate must be a complete Skill, not a patch")

    normalized = text.rstrip() + "\n"
    return SkillCandidate(
        name=name,
        version=version,
        description=metadata["description"],
        content=normalized,
    )


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("run_id may only contain letters, numbers, dot, dash, underscore")
    return run_id


def save_candidate(run_dir: Path, candidate: SkillCandidate) -> Path:
    version_dir = run_dir / f"v{candidate.version}"
    version_dir.mkdir(parents=True, exist_ok=False)
    path = version_dir / "SKILL.md"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(candidate.content)
    return path


@dataclass(frozen=True)
class BenchmarkCase:
    id: int
    split: str
    category: str
    question: str
    required: tuple[str, ...]
    forbidden: tuple[str, ...]
    critical: bool = False


def _question_fingerprint(question: str) -> str:
    normalized = re.sub(r"\s+", "", question).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_benchmark(path: Path) -> tuple[dict[str, Any], list[BenchmarkCase]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    splits = data.get("splits") or {}
    dev_ids = list(splits.get("dev") or [])
    holdout_ids = list(splits.get("holdout") or [])
    if not dev_ids or not holdout_ids:
        raise BenchmarkValidationError("benchmark needs non-empty dev and holdout splits")
    if len(dev_ids) != len(set(dev_ids)) or len(holdout_ids) != len(set(holdout_ids)):
        raise BenchmarkValidationError("split IDs must be unique")
    overlap = set(dev_ids) & set(holdout_ids)
    if overlap:
        raise BenchmarkValidationError(f"Dev/Holdout ID overlap: {sorted(overlap)}")

    raw_cases = data.get("cases") or []
    by_id: dict[int, dict[str, Any]] = {}
    for raw_case in raw_cases:
        case_id = int(raw_case["id"])
        if case_id in by_id:
            raise BenchmarkValidationError(f"duplicate case ID: {case_id}")
        by_id[case_id] = raw_case

    expected_ids = set(dev_ids) | set(holdout_ids)
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise BenchmarkValidationError(
            f"case/split mismatch; missing={missing}, extra={extra}"
        )

    cases: list[BenchmarkCase] = []
    for split, ids in (("dev", dev_ids), ("holdout", holdout_ids)):
        for case_id in ids:
            raw = by_id[case_id]
            cases.append(
                BenchmarkCase(
                    id=case_id,
                    split=split,
                    category=str(raw["category"]),
                    question=str(raw["question"]),
                    required=tuple(str(x) for x in raw.get("required", [])),
                    forbidden=tuple(str(x) for x in raw.get("forbidden", [])),
                    critical=bool(raw.get("critical", False)),
                )
            )

    dev = [case for case in cases if case.split == "dev"]
    holdout = [case for case in cases if case.split == "holdout"]
    dev_fingerprints = {_question_fingerprint(case.question) for case in dev}
    holdout_fingerprints = {_question_fingerprint(case.question) for case in holdout}
    if dev_fingerprints & holdout_fingerprints:
        raise BenchmarkValidationError("Dev/Holdout contain duplicate normalized questions")
    if {case.category for case in dev} != {case.category for case in holdout}:
        raise BenchmarkValidationError("Dev and Holdout must cover the same categories")

    metadata = {
        "description": data.get("description", ""),
        "source": data.get("source", {}),
        "dev_ids": dev_ids,
        "holdout_ids": holdout_ids,
        "dev_question_fingerprints": sorted(dev_fingerprints),
        "holdout_question_fingerprints": sorted(holdout_fingerprints),
    }
    return metadata, cases


def _normalize(text: str) -> str:
    return re.sub(r"(?<=\d)[,，](?=\d)", "", text).lower()


def _forbidden_hits(text: str, keyword: str) -> bool:
    index = 0
    while True:
        position = text.find(keyword, index)
        if position == -1:
            return False
        prefix = text[max(0, position - NEG_WINDOW) : position]
        if not any(negation in prefix for negation in NEG_PREFIXES):
            return True
        index = position + 1


def evaluate_answer(answer: str, case: BenchmarkCase) -> tuple[bool, str]:
    normalized = _normalize(answer)
    for signal in DEFERRAL_SIGNALS:
        if signal in normalized:
            return False, f"Agent 推脱或表达不确定（含 '{signal}'）"
    for keyword in case.required:
        if _normalize(keyword) not in normalized:
            return False, f"缺少关键词: '{keyword}'"
    for keyword in case.forbidden:
        if _forbidden_hits(normalized, _normalize(keyword)):
            return False, f"出现禁止词: '{keyword}'"
    return True, "correct"


def build_answer_messages(candidate: SkillCandidate, question: str) -> list[dict[str, str]]:
    system = f"""你是云购商城客服。只能依据下面的 Skill 回答。

规则：
- Skill 覆盖时，直接给出简洁、具体的结论和必要数字。
- Skill 未覆盖时，仅回答“需要联系人工客服”。
- 不要在正确答案后追加联系人工客服。

## 当前 Skill

{candidate.content}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _nearest_rank_p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _aggregate_results(
    *,
    candidate: SkillCandidate,
    split: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for result in results if result["correct"])
    critical_errors = sum(
        1 for result in results if result["critical"] and not result["correct"]
    )
    api_errors = sum(1 for result in results if result["api_error"])
    provider_attempts = sum(result["attempts"] for result in results)
    latencies = [float(result["latency_ms"]) for result in results]

    comparable_usage = (
        bool(results)
        and api_errors == 0
        and all(result["usage"]["comparable"] for result in results)
    )
    token_fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "reasoning_tokens",
    )
    token_totals: dict[str, int | None] = {}
    for field in token_fields:
        values = [result["usage"].get(field) for result in results]
        token_totals[field] = (
            sum(int(value) for value in values if value is not None)
            if comparable_usage and all(value is not None for value in values)
            else None
        )

    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        stats = by_category.setdefault(result["category"], {"total": 0, "correct": 0})
        stats["total"] += 1
        stats["correct"] += int(result["correct"])
    for stats in by_category.values():
        stats["accuracy"] = stats["correct"] / stats["total"]
    macro_accuracy = (
        sum(stats["accuracy"] for stats in by_category.values()) / len(by_category)
        if by_category
        else 0.0
    )

    metrics = {
        "split": split,
        "candidate_version": candidate.version,
        "skill_chars": candidate.chars,
        "logical_calls": total,
        "provider_attempts": provider_attempts,
        "retries": provider_attempts - total,
        "api_errors": api_errors,
        "case_ids": sorted(result["id"] for result in results),
        "correct_ids": sorted(
            result["id"] for result in results if result["correct"]
        ),
        "failed_ids": sorted(
            result["id"] for result in results if not result["correct"]
        ),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_accuracy": macro_accuracy,
        "critical_errors": critical_errors,
        "usage_comparable": comparable_usage,
        **token_totals,
        "latency_avg_ms": sum(latencies) / len(latencies) if latencies else None,
        "latency_p95_ms": _nearest_rank_p95(latencies),
        "by_category": by_category,
    }
    return {"metrics": metrics, "cases": results}


def evaluate_candidate(
    *,
    gateway: Any,
    candidate: SkillCandidate,
    cases: Sequence[BenchmarkCase],
    split: str,
    answer_max_tokens: int = 300,
) -> dict[str, Any]:
    expected_cases = [case for case in cases if case.split == split]
    results: list[dict[str, Any]] = []
    for case in expected_cases:
        try:
            trace = gateway.complete(
                messages=build_answer_messages(candidate, case.question),
                purpose=f"answer:v{candidate.version}:{split}:{case.id}",
                max_tokens=answer_max_tokens,
            )
            correct, reason = evaluate_answer(trace.text, case)
            results.append(
                {
                    "id": case.id,
                    "split": split,
                    "category": case.category,
                    "question": case.question,
                    "critical": case.critical,
                    "answer": trace.text,
                    "correct": correct,
                    "reason": reason,
                    "api_error": False,
                    "usage": trace.usage.to_dict(),
                    "latency_ms": round(trace.latency_ms, 3),
                    "attempts": trace.attempts,
                    "response_model": trace.response_model,
                    "system_fingerprint": trace.system_fingerprint,
                    "finish_reason": trace.finish_reason,
                }
            )
        except LLMRequestError as exc:
            results.append(
                {
                    "id": case.id,
                    "split": split,
                    "category": case.category,
                    "question": case.question,
                    "critical": case.critical,
                    "answer": "",
                    "correct": False,
                    "reason": str(exc),
                    "api_error": True,
                    "usage": TokenUsage(None, None, None).to_dict(),
                    "latency_ms": round(exc.latency_ms, 3),
                    "attempts": exc.attempts,
                    "response_model": None,
                    "system_fingerprint": None,
                    "finish_reason": None,
                }
            )
    return _aggregate_results(
        candidate=candidate, split=split, results=results
    )


@dataclass(frozen=True)
class GateThresholds:
    min_prompt_token_reduction_pct: float = 15.0
    min_total_token_reduction_pct: float = 5.0
    max_p95_latency_regression_pct: float = 50.0


def _reduction_pct(before: int | float | None, after: int | float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return (before - after) / before * 100


def compare_efficiency(
    before: dict[str, Any],
    after: dict[str, Any],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    usage_available = bool(
        before.get("usage_comparable") and after.get("usage_comparable")
    )
    prompt_reduction = _reduction_pct(
        before.get("prompt_tokens"), after.get("prompt_tokens")
    )
    total_reduction = _reduction_pct(
        before.get("total_tokens"), after.get("total_tokens")
    )
    before_p95 = before.get("latency_p95_ms")
    after_p95 = after.get("latency_p95_ms")
    latency_regression = (
        ((after_p95 - before_p95) / before_p95 * 100)
        if before_p95 not in (None, 0) and after_p95 is not None
        else None
    )
    before_attempts = before.get("provider_attempts")
    after_attempts = after.get("provider_attempts")
    provider_attempts_available = isinstance(before_attempts, int) and isinstance(
        after_attempts, int
    )
    checks = {
        "usage_available": usage_available,
        "prompt_token_reduction_met": (
            prompt_reduction is not None
            and prompt_reduction >= thresholds.min_prompt_token_reduction_pct
        ),
        "total_token_reduction_met": (
            total_reduction is not None
            and total_reduction >= thresholds.min_total_token_reduction_pct
        ),
        "logical_calls_not_increased": (
            after.get("logical_calls") == before.get("logical_calls")
        ),
        "provider_attempts_available": provider_attempts_available,
        "provider_attempts_not_increased": (
            provider_attempts_available and after_attempts <= before_attempts
        ),
        "p95_latency_within_limit": (
            latency_regression is not None
            and latency_regression <= thresholds.max_p95_latency_regression_pct
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {
            "prompt_token_reduction_pct": prompt_reduction,
            "total_token_reduction_pct": total_reduction,
            "p95_latency_regression_pct": latency_regression,
            "skill_char_reduction_pct": _reduction_pct(
                before.get("skill_chars"), after.get("skill_chars")
            ),
        },
    }


def _quality_checks(before: dict[str, Any], after: dict[str, Any]) -> dict[str, bool]:
    before_case_ids = before.get("case_ids")
    after_case_ids = after.get("case_ids")
    before_correct_ids = before.get("correct_ids")
    after_correct_ids = after.get("correct_ids")
    ids_available = all(
        isinstance(value, list)
        for value in (
            before_case_ids,
            after_case_ids,
            before_correct_ids,
            after_correct_ids,
        )
    )
    same_case_ids = bool(
        ids_available
        and len(before_case_ids) == len(after_case_ids)
        and set(before_case_ids) == set(after_case_ids)
    )
    preserves_v1_correct_cases = bool(
        same_case_ids and set(before_correct_ids).issubset(set(after_correct_ids))
    )
    return {
        "same_case_count": before.get("total") == after.get("total"),
        "same_case_ids": same_case_ids,
        "preserves_v1_correct_cases": preserves_v1_correct_cases,
        "quality_not_regressed": after.get("correct", 0) >= before.get("correct", 0),
        "critical_errors_zero": after.get("critical_errors", 0) == 0,
        "before_api_errors_zero": before.get("api_errors") == 0,
        "after_api_errors_zero": after.get("api_errors") == 0,
    }


def evaluate_dev_gate(
    before: dict[str, Any],
    after: dict[str, Any],
    thresholds: GateThresholds,
    *,
    efficiency_comparator: Callable[
        [dict[str, Any], dict[str, Any], GateThresholds], dict[str, Any]
    ] = compare_efficiency,
) -> dict[str, Any]:
    quality_checks = _quality_checks(before, after)
    if not all(quality_checks.values()):
        return {
            "passed": False,
            "stage": "quality",
            "quality_passed": False,
            "efficiency_evaluated": False,
            "checks": quality_checks,
            "deltas": {
                "correct_delta": after.get("correct", 0) - before.get("correct", 0)
            },
        }

    efficiency = efficiency_comparator(before, after, thresholds)
    return {
        "passed": bool(efficiency["passed"]),
        "stage": "passed" if efficiency["passed"] else "efficiency",
        "quality_passed": True,
        "efficiency_evaluated": True,
        "checks": {**quality_checks, **efficiency["checks"]},
        "deltas": {
            "correct_delta": after.get("correct", 0) - before.get("correct", 0),
            **efficiency["deltas"],
        },
    }


def evaluate_holdout_gate(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    checks = _quality_checks(before, after)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {
            "correct_delta": after.get("correct", 0) - before.get("correct", 0),
            "prompt_token_reduction_pct": _reduction_pct(
                before.get("prompt_tokens"), after.get("prompt_tokens")
            ),
            "total_token_reduction_pct": _reduction_pct(
                before.get("total_tokens"), after.get("total_tokens")
            ),
            "skill_char_reduction_pct": _reduction_pct(
                before.get("skill_chars"), after.get("skill_chars")
            ),
        },
    }


def build_v1_messages(policy: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是客服 Skill 架构师。请把政策原文转成培训手册式、层级清晰的 "
                "Markdown SOP。这个 v1 是故意保留解释性冗余的未优化基线：逐条展开"
                "适用条件、结论和说明，并为每组规则提供标注为‘仅用于解释’的示例。"
                "必须保留所有数字、条件、例外、优先级和计算规则，不得发明业务事实。"
            ),
        },
        {
            "role": "user",
            "content": f"""根据下面政策生成一个完整 Skill v1。

硬性格式：
---
name: {SKILL_NAME}
description: 一句话描述
type: knowledge
version: 1
---

正文至少包含一个二级标题，且正文（不含 frontmatter）不少于 1400 个中文字符。
只输出完整 Markdown，不要解释，不要输出 patch。不要提前压缩成速查表；正文使用完整句子，
为每组规则至少提供两个标注为“示例，仅用于解释”的场景，并单独说明判断顺序和常见误区，
让后续优化器有明确的压缩空间。

## 政策原文

{policy}""",
        },
    ]


def _dev_failure_summary(dev_report: dict[str, Any]) -> str:
    failures = [result for result in dev_report["cases"] if not result["correct"]]
    if not failures:
        return "Dev 集没有失败样本。优化时只压缩重复表达，必须保留全部规则。"
    parts = []
    for result in failures:
        parts.append(
            "\n".join(
                [
                    f"- ID {result['id']} / {result['category']}",
                    f"  问题: {result['question']}",
                    f"  当前回答: {result['answer']}",
                    f"  失败原因: {result['reason']}",
                ]
            )
        )
    return "\n".join(parts)


def build_v2_messages(
    *,
    v1: SkillCandidate,
    dev_report: dict[str, Any],
    target_prompt_reduction_pct: float,
) -> list[dict[str, str]]:
    metrics = dev_report["metrics"]
    failures = _dev_failure_summary(dev_report)
    target_max_chars = max(400, int(v1.chars * 0.6))
    return [
        {
            "role": "system",
            "content": (
                "你是 Skill 优化器。质量优先于效率：先保持或修复回答正确性，再减少 "
                "运行时 Prompt Token。你只能根据 v1 和 Dev 反馈优化，不得假设未给出的测试题。"
            ),
        },
        {
            "role": "user",
            "content": f"""把下面完整 Skill v1 优化为完整 Skill v2。

目标：
1. 不删除任何业务事实、数字、条件、例外、优先级或计算规则。
2. v1 中的培训解释、示例、背景说明和同义重复不属于业务事实，应删除。
3. 修复 Dev 失败，但不要复制题目或针对题目写死答案。
4. 用短句、决策表、公式和优先级列表代替展开叙述；每条业务事实只出现一次。
5. 争取运行时 prompt_tokens 至少降低 {target_prompt_reduction_pct:.1f}%，并将 Skill
   从 {v1.chars} 字符压缩到不超过约 {target_max_chars} 字符；达不到时也必须明显短于 v1。
6. 输出必须是完整 Markdown，不是 diff、patch 或修改说明。
7. frontmatter 必须固定为 name={SKILL_NAME}、type=knowledge、version=2。

## v1 Dev 指标

- 正确: {metrics['correct']}/{metrics['total']}
- prompt_tokens: {metrics['prompt_tokens']}
- Skill 字符数: {metrics['skill_chars']}

## 仅 Dev 失败反馈

{failures}

## 完整 Skill v1

{v1.content}""",
        },
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_policy_snapshot(path: Path, benchmark_metadata: dict[str, Any]) -> str:
    actual = _sha256_file(path)
    expected = (benchmark_metadata.get("source") or {}).get("policies_sha256")
    if expected and actual != expected:
        raise BenchmarkValidationError(
            "policy snapshot hash differs from the benchmark source; "
            f"expected={expected}, actual={actual}"
        )
    return actual


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _generate_candidate_with_repair(
    *,
    gateway: Any,
    messages: list[dict[str, str]],
    expected_version: int,
    purpose: str,
    max_tokens: int,
) -> tuple[SkillCandidate, list[LLMCallResult]]:
    traces: list[LLMCallResult] = []
    trace = gateway.complete(
        messages=messages, purpose=purpose, max_tokens=max_tokens
    )
    traces.append(trace)
    try:
        return parse_skill_candidate(
            trace.text, expected_version=expected_version
        ), traces
    except CandidateValidationError as exc:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": trace.text},
            {
                "role": "user",
                "content": (
                    f"上一个输出不合格：{exc}。请只返回修正后的完整 Markdown Skill，"
                    "不要解释。"
                ),
            },
        ]
        repaired = gateway.complete(
            messages=repair_messages,
            purpose=f"{purpose}:repair",
            max_tokens=max_tokens,
        )
        traces.append(repaired)
        return parse_skill_candidate(
            repaired.text, expected_version=expected_version
        ), traces


def _exclusive_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _metric_cell(metrics: dict[str, Any] | None, key: str, percent: bool = False) -> str:
    if metrics is None:
        return "not run"
    value = metrics.get(key)
    if value is None:
        return "n/a"
    if percent:
        return f"{value:.1%}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _render_report_markdown(report: dict[str, Any]) -> str:
    dev_v1 = report["dev"]["v1"]["metrics"]
    dev_v2 = report["dev"]["v2"]["metrics"]
    holdout_v1 = (
        report["holdout"]["v1"]["metrics"]
        if report["holdout"].get("v1")
        else None
    )
    holdout_v2 = (
        report["holdout"]["v2"]["metrics"]
        if report["holdout"].get("v2")
        else None
    )
    lines = [
        "# Skill Optimization Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Model: `{report['config']['model']}`",
        f"- Final status: **{report['final_status']}**",
        f"- Holdout status: `{report['holdout']['status']}`",
        "",
        "## Before / After",
        "",
        "| Split | Version | Accuracy | Critical errors | Prompt tokens | Total tokens | Provider attempts | Retries | P95 latency ms | Skill chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, version, metrics in (
        ("Dev", "v1", dev_v1),
        ("Dev", "v2", dev_v2),
        ("Holdout", "v1", holdout_v1),
        ("Holdout", "v2", holdout_v2),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    version,
                    _metric_cell(metrics, "accuracy", percent=True),
                    _metric_cell(metrics, "critical_errors"),
                    _metric_cell(metrics, "prompt_tokens"),
                    _metric_cell(metrics, "total_tokens"),
                    _metric_cell(metrics, "provider_attempts"),
                    _metric_cell(metrics, "retries"),
                    _metric_cell(metrics, "latency_p95_ms"),
                    _metric_cell(metrics, "skill_chars"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Dev gate",
            "",
            f"- Passed: `{report['dev_gate']['passed']}`",
            f"- Stage: `{report['dev_gate']['stage']}`",
        ]
    )
    for name, passed in report["dev_gate"]["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    if report["holdout"].get("gate"):
        lines.extend(["", "## Holdout quality gate", ""])
        for name, passed in report["holdout"]["gate"]["checks"].items():
            lines.append(f"- `{name}`: `{passed}`")
    lines.extend(
        [
            "",
            "Provider token fields are reported only when every call returned core usage. "
            "Character counts are descriptive and are never substituted for missing provider tokens.",
            "",
        ]
    )
    return "\n".join(lines)


class SkillOptimizationExperiment:
    def __init__(
        self,
        *,
        gateway: Any,
        policy_path: Path = DEFAULT_POLICY_PATH,
        benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        thresholds: GateThresholds = GateThresholds(),
        skill_max_tokens: int = 3_000,
        answer_max_tokens: int = 300,
    ) -> None:
        self.gateway = gateway
        self.policy_path = Path(policy_path)
        self.benchmark_path = Path(benchmark_path)
        self.output_root = Path(output_root)
        self.thresholds = thresholds
        self.skill_max_tokens = skill_max_tokens
        self.answer_max_tokens = answer_max_tokens

    def run(self, *, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
        run_id = validate_run_id(run_id or _default_run_id())
        metadata, cases = load_benchmark(self.benchmark_path)
        policy_sha256 = validate_policy_snapshot(self.policy_path, metadata)
        policy = self.policy_path.read_text(encoding="utf-8")
        self.output_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.output_root / run_id
        run_dir.mkdir(exist_ok=False)

        v1, v1_generation = _generate_candidate_with_repair(
            gateway=self.gateway,
            messages=build_v1_messages(policy),
            expected_version=1,
            purpose="generate:v1",
            max_tokens=self.skill_max_tokens,
        )
        save_candidate(run_dir, v1)
        dev_v1 = evaluate_candidate(
            gateway=self.gateway,
            candidate=v1,
            cases=cases,
            split="dev",
            answer_max_tokens=self.answer_max_tokens,
        )

        v2, v2_generation = _generate_candidate_with_repair(
            gateway=self.gateway,
            messages=build_v2_messages(
                v1=v1,
                dev_report=dev_v1,
                target_prompt_reduction_pct=self.thresholds.min_prompt_token_reduction_pct,
            ),
            expected_version=2,
            purpose="optimize:v2",
            max_tokens=self.skill_max_tokens,
        )
        save_candidate(run_dir, v2)
        dev_v2 = evaluate_candidate(
            gateway=self.gateway,
            candidate=v2,
            cases=cases,
            split="dev",
            answer_max_tokens=self.answer_max_tokens,
        )
        dev_gate = evaluate_dev_gate(
            dev_v1["metrics"], dev_v2["metrics"], self.thresholds
        )

        holdout: dict[str, Any] = {
            "status": "not_run_dev_gate_failed",
            "v1": None,
            "v2": None,
            "gate": None,
        }
        final_status = "rejected_dev"
        if dev_gate["passed"]:
            holdout_v1 = evaluate_candidate(
                gateway=self.gateway,
                candidate=v1,
                cases=cases,
                split="holdout",
                answer_max_tokens=self.answer_max_tokens,
            )
            holdout_v2 = evaluate_candidate(
                gateway=self.gateway,
                candidate=v2,
                cases=cases,
                split="holdout",
                answer_max_tokens=self.answer_max_tokens,
            )
            holdout_gate = evaluate_holdout_gate(
                holdout_v1["metrics"], holdout_v2["metrics"]
            )
            holdout = {
                "status": "completed",
                "v1": holdout_v1,
                "v2": holdout_v2,
                "gate": holdout_gate,
            }
            final_status = "accepted" if holdout_gate["passed"] else "rejected_holdout"

        report = {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at": _utc_now(),
            "final_status": final_status,
            "config": {
                "provider": "DeepSeek",
                "base_url": getattr(self.gateway, "base_url", DEFAULT_BASE_URL),
                "model": getattr(self.gateway, "model", DEFAULT_MODEL),
                "thinking": "disabled",
                "temperature": 0,
                "skill_max_tokens": self.skill_max_tokens,
                "answer_max_tokens": self.answer_max_tokens,
                "request_timeout_seconds": getattr(
                    self.gateway, "request_timeout_seconds", None
                ),
                "outer_max_retries": getattr(self.gateway, "max_retries", None),
                "sdk_internal_retries": getattr(
                    self.gateway, "sdk_max_retries", None
                ),
                "thresholds": asdict(self.thresholds),
            },
            "inputs": {
                "policy_path": str(self.policy_path),
                "policy_sha256": policy_sha256,
                "benchmark_path": str(self.benchmark_path),
                "benchmark_sha256": _sha256_file(self.benchmark_path),
                "benchmark_metadata": metadata,
            },
            "candidates": {"v1": v1.summary(), "v2": v2.summary()},
            "generation_calls": {
                "v1": [trace.to_dict() for trace in v1_generation],
                "v2": [trace.to_dict() for trace in v2_generation],
            },
            "dev": {"v1": dev_v1, "v2": dev_v2},
            "dev_gate": dev_gate,
            "holdout": holdout,
        }
        _exclusive_json(run_dir / "report.json", report)
        with (run_dir / "report.md").open("x", encoding="utf-8") as handle:
            handle.write(_render_report_markdown(report))
        return report, run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and optimize a DeepSeek Skill, then compare v1/v2."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--skill-max-tokens", type=int, default=3_000)
    parser.add_argument("--answer-max-tokens", type=int, default=300)
    parser.add_argument("--min-prompt-reduction-pct", type=float, default=15.0)
    parser.add_argument("--min-total-reduction-pct", type=float, default=5.0)
    parser.add_argument("--max-p95-regression-pct", type=float, default=50.0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate local inputs without importing the SDK or calling a provider",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metadata, cases = load_benchmark(args.benchmark)
    policy_sha256 = validate_policy_snapshot(args.policy, metadata)
    if args.validate_only:
        summary = {
            "status": "valid",
            "policy_sha256": policy_sha256,
            "benchmark_sha256": _sha256_file(args.benchmark),
            "dev_cases": sum(case.split == "dev" for case in cases),
            "holdout_cases": sum(case.split == "holdout" for case in cases),
            "categories": sorted({case.category for case in cases}),
            "source": metadata.get("source", {}),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(
            f"Missing {args.api_key_env}; set it in the environment for a live DeepSeek run.",
            file=sys.stderr,
        )
        return 2

    gateway = DeepSeekGateway(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        max_retries=args.max_retries,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    experiment = SkillOptimizationExperiment(
        gateway=gateway,
        policy_path=args.policy,
        benchmark_path=args.benchmark,
        output_root=args.output_root,
        thresholds=GateThresholds(
            min_prompt_token_reduction_pct=args.min_prompt_reduction_pct,
            min_total_token_reduction_pct=args.min_total_reduction_pct,
            max_p95_latency_regression_pct=args.max_p95_regression_pct,
        ),
        skill_max_tokens=args.skill_max_tokens,
        answer_max_tokens=args.answer_max_tokens,
    )
    report, run_dir = experiment.run(run_id=args.run_id)
    dev_before = report["dev"]["v1"]["metrics"]
    dev_after = report["dev"]["v2"]["metrics"]
    print(f"Run: {run_dir}")
    print(
        f"Dev accuracy: {dev_before['correct']}/{dev_before['total']} -> "
        f"{dev_after['correct']}/{dev_after['total']}"
    )
    print(
        "Dev prompt token reduction: "
        f"{report['dev_gate']['deltas'].get('prompt_token_reduction_pct')}%"
    )
    print(f"Final status: {report['final_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
