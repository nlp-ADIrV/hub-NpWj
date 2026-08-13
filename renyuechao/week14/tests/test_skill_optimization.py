from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.skill_optimization import (
    BenchmarkValidationError,
    CandidateValidationError,
    DeepSeekGateway,
    GateThresholds,
    LLMCallResult,
    LLMRequestError,
    SkillCandidate,
    SkillOptimizationExperiment,
    TokenUsage,
    evaluate_answer,
    evaluate_dev_gate,
    evaluate_holdout_gate,
    load_benchmark,
    parse_skill_candidate,
    save_candidate,
    validate_policy_snapshot,
    validate_run_id,
)


ROOT = Path(__file__).resolve().parents[1]


def skill_text(version: int, *, verbose: bool) -> str:
    padding = (
        "\n".join(
            f"- 冗余说明{i}：所有判断都必须再次核对完整政策条件。" for i in range(24)
        )
        if verbose
        else "- 按用户、商品、时限和例外顺序判断。"
    )
    return f"""---
name: customer_service_policy
description: 云购商城客服完整政策
type: knowledge
version: {version}
---

## 决策流程
{padding}

## 政策事实
- 普通商品30天；银卡60天；金卡90天；白卡仍为30天。
- 数字商品不可退；质量问题提交工单和截图。
- 48小时内未发货可取消，24小时退款；超过48小时走退货退款。
- 积分每1元退80积分，混合支付分别原路退回。
"""


class QueueCompletions:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeOpenAIClient:
    def __init__(self, items):
        self.chat = SimpleNamespace(completions=QueueCompletions(items))


def fake_response(text: str, *, usage=True, finish_reason="stop"):
    usage_obj = None
    if usage:
        usage_obj = SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_cache_hit_tokens=20,
            prompt_cache_miss_tokens=100,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage_obj,
        model="deepseek-v4-flash-0731",
        system_fingerprint="fp-test",
    )


def gate_metrics(
    *,
    case_ids=(1, 2),
    correct_ids=(1, 2),
    api_errors=0,
    critical_errors=0,
    usage_comparable=True,
    prompt_tokens=100,
    total_tokens=120,
    provider_attempts=None,
    latency_p95_ms=10.0,
    skill_chars=100,
):
    case_ids = list(case_ids)
    correct_ids = list(correct_ids)
    if provider_attempts is None:
        provider_attempts = len(case_ids)
    return {
        "total": len(case_ids),
        "correct": len(correct_ids),
        "case_ids": case_ids,
        "correct_ids": correct_ids,
        "failed_ids": sorted(set(case_ids) - set(correct_ids)),
        "critical_errors": critical_errors,
        "api_errors": api_errors,
        "usage_comparable": usage_comparable,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "logical_calls": len(case_ids),
        "provider_attempts": provider_attempts,
        "retries": provider_attempts - len(case_ids),
        "latency_p95_ms": latency_p95_ms,
        "skill_chars": skill_chars,
    }


class ScriptedGateway:
    base_url = "offline://fake-deepseek"
    model = "deepseek-v4-flash"

    def __init__(
        self, cases, *, break_v2_dev_id=None, break_v2_holdout_id=None
    ):
        self.answers = {
            case.question: " ".join(case.required) or "符合政策"
            for case in cases
        }
        self.break_v2_dev_id = break_v2_dev_id
        self.break_v2_holdout_id = break_v2_holdout_id
        self.calls = []

    def complete(self, *, messages, purpose, max_tokens):
        self.calls.append(
            {"messages": messages, "purpose": purpose, "max_tokens": max_tokens}
        )
        if purpose.startswith("generate:v1"):
            text = skill_text(1, verbose=True)
        elif purpose.startswith("optimize:v2"):
            text = skill_text(2, verbose=False)
        else:
            question = messages[-1]["content"]
            text = self.answers[question]
            if purpose == f"answer:v2:dev:{self.break_v2_dev_id}":
                text = "错误答案"
            if purpose == f"answer:v2:holdout:{self.break_v2_holdout_id}":
                text = "错误答案"

        prompt_chars = sum(len(message["content"]) for message in messages)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, len(text) // 4)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=prompt_tokens,
            reasoning_tokens=0,
        )
        version_latency = 20.0 if ":v1:" in purpose else 18.0
        return LLMCallResult(
            purpose=purpose,
            text=text,
            usage=usage,
            latency_ms=version_latency,
            attempts=1,
            requested_model=self.model,
            response_model="deepseek-v4-flash-0731",
            system_fingerprint="offline-fake",
            finish_reason="stop",
        )


class TestDeepSeekGateway(unittest.TestCase):
    def test_exact_model_parameters_usage_and_latency(self):
        client = FakeOpenAIClient([fake_response("ok")])
        clock_values = iter([10.0, 10.125])
        gateway = DeepSeekGateway(
            client=client,
            model="deepseek-v4-flash",
            max_retries=0,
            clock=lambda: next(clock_values),
            sleep=lambda _: None,
        )
        result = gateway.complete(
            messages=[{"role": "user", "content": "hello"}],
            purpose="test",
            max_tokens=123,
        )

        request = client.chat.completions.calls[0]
        self.assertEqual(request["model"], "deepseek-v4-flash")
        self.assertEqual(request["temperature"], 0)
        self.assertFalse(request["stream"])
        self.assertEqual(request["max_tokens"], 123)
        self.assertEqual(
            request["extra_body"], {"thinking": {"type": "disabled"}}
        )
        self.assertEqual(result.latency_ms, 125.0)
        self.assertEqual(result.usage.prompt_tokens, 120)
        self.assertEqual(result.usage.prompt_cache_hit_tokens, 20)
        self.assertEqual(result.usage.reasoning_tokens, 0)
        self.assertTrue(result.usage.comparable)
        self.assertEqual(result.finish_reason, "stop")

    def test_sdk_retries_are_disabled_and_timeout_is_explicit(self):
        captured = {}

        def client_factory(**kwargs):
            captured.update(kwargs)
            return FakeOpenAIClient([])

        gateway = DeepSeekGateway(
            api_key="local-test-key",
            client_factory=client_factory,
            request_timeout_seconds=17.5,
        )
        self.assertEqual(captured["max_retries"], 0)
        self.assertEqual(captured["timeout"], 17.5)
        self.assertEqual(gateway.sdk_max_retries, 0)

    def test_non_stop_finish_reason_is_rejected(self):
        client = FakeOpenAIClient(
            [fake_response("syntactically valid but truncated", finish_reason="length")]
        )
        clock_values = iter([1.0, 1.1])
        gateway = DeepSeekGateway(
            client=client,
            max_retries=0,
            clock=lambda: next(clock_values),
        )
        with self.assertRaises(LLMRequestError) as context:
            gateway.complete(
                messages=[{"role": "user", "content": "hello"}],
                purpose="truncated",
                max_tokens=10,
            )
        self.assertIn("finish_reason='length'", str(context.exception))

    def test_missing_core_usage_is_not_comparable(self):
        client = FakeOpenAIClient([fake_response("ok", usage=False)])
        clock_values = iter([1.0, 1.1])
        result = DeepSeekGateway(
            client=client,
            max_retries=0,
            clock=lambda: next(clock_values),
        ).complete(
            messages=[{"role": "user", "content": "hello"}],
            purpose="test",
            max_tokens=10,
        )
        self.assertFalse(result.usage.comparable)
        self.assertIsNone(result.usage.prompt_tokens)
        self.assertIsNone(result.usage.total_tokens)

    def test_retry_attempts_are_recorded(self):
        client = FakeOpenAIClient([RuntimeError("temporary"), fake_response("ok")])
        clock_values = iter([2.0, 2.3])
        result = DeepSeekGateway(
            client=client,
            max_retries=1,
            clock=lambda: next(clock_values),
            sleep=lambda _: None,
        ).complete(
            messages=[{"role": "user", "content": "hello"}],
            purpose="retry",
            max_tokens=10,
        )
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(client.chat.completions.calls), 2)


class TestCandidateValidation(unittest.TestCase):
    def test_accepts_raw_and_single_fenced_markdown(self):
        raw = skill_text(1, verbose=False)
        self.assertEqual(parse_skill_candidate(raw, expected_version=1).version, 1)
        fenced = f"```markdown\n{raw}```"
        self.assertEqual(parse_skill_candidate(fenced, expected_version=1).name,
                         "customer_service_policy")

    def test_rejects_patch_wrong_version_and_unsafe_name(self):
        wrong_version = skill_text(1, verbose=False)
        unsafe_name = skill_text(2, verbose=False).replace(
            "name: customer_service_policy", "name: ../outside"
        )
        patch = "---\nname: customer_service_policy\ndescription: x\ntype: knowledge\nversion: 2\n---\n\n@@ -1 +1 @@"
        for raw, version in ((wrong_version, 2), (unsafe_name, 2), (patch, 2)):
            with self.subTest(raw=raw[:40]):
                with self.assertRaises(CandidateValidationError):
                    parse_skill_candidate(raw, expected_version=version)

    def test_candidates_are_separate_and_never_overwritten(self):
        v1 = parse_skill_candidate(skill_text(1, verbose=True), expected_version=1)
        v2 = parse_skill_candidate(skill_text(2, verbose=False), expected_version=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            v1_path = save_candidate(run_dir, v1)
            v1_bytes = v1_path.read_bytes()
            v2_path = save_candidate(run_dir, v2)
            self.assertNotEqual(v1_path, v2_path)
            self.assertEqual(v1_path.read_bytes(), v1_bytes)
            with self.assertRaises(FileExistsError):
                save_candidate(run_dir, v1)

    def test_unsafe_run_id_is_rejected(self):
        for run_id in ("../bad", "/absolute", "bad/name", "bad\\name"):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    validate_run_id(run_id)


class TestIsolationAndGate(unittest.TestCase):
    def test_benchmark_is_disjoint_and_stratified(self):
        metadata, cases = load_benchmark(ROOT / "data" / "benchmark.json")
        dev = [case for case in cases if case.split == "dev"]
        holdout = [case for case in cases if case.split == "holdout"]
        self.assertEqual(len(dev), 12)
        self.assertEqual(len(holdout), 12)
        self.assertFalse(set(metadata["dev_ids"]) & set(metadata["holdout_ids"]))
        self.assertEqual(
            {case.category for case in dev}, {case.category for case in holdout}
        )

    def test_duplicate_question_across_splits_is_rejected(self):
        payload = {
            "splits": {"dev": [1], "holdout": [2]},
            "cases": [
                {"id": 1, "category": "x", "question": "same question"},
                {"id": 2, "category": "x", "question": " same   question "},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "benchmark.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BenchmarkValidationError):
                load_benchmark(path)

    def test_policy_snapshot_hash_is_enforced(self):
        metadata, _ = load_benchmark(ROOT / "data" / "benchmark.json")
        self.assertEqual(
            validate_policy_snapshot(ROOT / "data" / "policies.md", metadata),
            metadata["source"]["policies_sha256"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            changed = Path(temp_dir) / "policies.md"
            changed.write_text("changed", encoding="utf-8")
            with self.assertRaises(BenchmarkValidationError):
                validate_policy_snapshot(changed, metadata)

    def test_uncertain_answer_is_not_counted_as_policy_denial(self):
        _, cases = load_benchmark(ROOT / "data" / "benchmark.json")
        case = next(case for case in cases if case.id == 23)
        correct, reason = evaluate_answer("不确定，需要再核实。", case)
        self.assertFalse(correct)
        self.assertIn("不确定", reason)

    def test_quality_failure_short_circuits_efficiency(self):
        before = gate_metrics(case_ids=range(10), correct_ids=range(10))
        after = gate_metrics(case_ids=range(10), correct_ids=range(9))

        def must_not_run(*_):
            raise AssertionError("efficiency comparator must not run")

        gate = evaluate_dev_gate(
            before,
            after,
            GateThresholds(),
            efficiency_comparator=must_not_run,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["efficiency_evaluated"])
        self.assertEqual(gate["stage"], "quality")

    def test_swapped_correct_case_short_circuits_efficiency(self):
        before = gate_metrics(correct_ids=(1,))
        after = gate_metrics(correct_ids=(2,))

        def must_not_run(*_):
            raise AssertionError("efficiency comparator must not run")

        gate = evaluate_dev_gate(
            before,
            after,
            GateThresholds(),
            efficiency_comparator=must_not_run,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["preserves_v1_correct_cases"])
        self.assertFalse(gate["efficiency_evaluated"])

    def test_missing_usage_fails_closed(self):
        before = gate_metrics(
            usage_comparable=False,
            prompt_tokens=None,
            total_tokens=None,
        )
        after = {**before, "skill_chars": 50}
        gate = evaluate_dev_gate(before, after, GateThresholds())
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["quality_passed"])
        self.assertFalse(gate["checks"]["usage_available"])

    def test_more_provider_attempts_fails_efficiency_gate(self):
        before = gate_metrics(provider_attempts=2)
        after = gate_metrics(
            prompt_tokens=80,
            total_tokens=100,
            provider_attempts=3,
            skill_chars=70,
        )
        gate = evaluate_dev_gate(before, after, GateThresholds())
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["quality_passed"])
        self.assertFalse(gate["checks"]["provider_attempts_not_increased"])

    def test_equal_nonzero_holdout_api_errors_fail_closed(self):
        before = gate_metrics(correct_ids=(1,), api_errors=1)
        after = gate_metrics(correct_ids=(1,), api_errors=1)
        gate = evaluate_holdout_gate(before, after)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["before_api_errors_zero"])
        self.assertFalse(gate["checks"]["after_api_errors_zero"])


class TestOfflineEndToEnd(unittest.TestCase):
    def _run(
        self,
        output_root: Path,
        *,
        break_v2_dev_id=None,
        break_v2_holdout_id=None,
    ):
        _, cases = load_benchmark(ROOT / "data" / "benchmark.json")
        gateway = ScriptedGateway(
            cases,
            break_v2_dev_id=break_v2_dev_id,
            break_v2_holdout_id=break_v2_holdout_id,
        )
        experiment = SkillOptimizationExperiment(
            gateway=gateway,
            policy_path=ROOT / "data" / "policies.md",
            benchmark_path=ROOT / "data" / "benchmark.json",
            output_root=output_root,
        )
        report, run_dir = experiment.run(run_id="offline-test")
        return report, run_dir, gateway

    def test_complete_run_writes_comparable_report_without_holdout_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report, run_dir, gateway = self._run(Path(temp_dir))
            self.assertEqual(report["final_status"], "accepted")
            self.assertEqual(report["holdout"]["status"], "completed")
            self.assertTrue((run_dir / "v1" / "SKILL.md").is_file())
            self.assertTrue((run_dir / "v2" / "SKILL.md").is_file())
            self.assertTrue((run_dir / "report.json").is_file())
            self.assertTrue((run_dir / "report.md").is_file())
            self.assertNotEqual(
                (run_dir / "v1" / "SKILL.md").read_bytes(),
                (run_dir / "v2" / "SKILL.md").read_bytes(),
            )
            self.assertGreater(
                report["dev_gate"]["deltas"]["prompt_token_reduction_pct"], 15
            )

            optimizer_call = next(
                call for call in gateway.calls if call["purpose"] == "optimize:v2"
            )
            optimizer_prompt = "\n".join(
                message["content"] for message in optimizer_call["messages"]
            )
            for case in load_benchmark(ROOT / "data" / "benchmark.json")[1]:
                if case.split == "holdout":
                    self.assertNotIn(case.question, optimizer_prompt)
            for call in gateway.calls:
                if call["purpose"].startswith("answer:"):
                    self.assertEqual(len(call["messages"]), 2)

    def test_failed_dev_gate_does_not_call_holdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report, _, gateway = self._run(Path(temp_dir), break_v2_dev_id=38)
            self.assertEqual(report["final_status"], "rejected_dev")
            self.assertEqual(
                report["holdout"]["status"], "not_run_dev_gate_failed"
            )
            self.assertFalse(
                any(":holdout:" in call["purpose"] for call in gateway.calls)
            )

    def test_holdout_regression_rejects_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report, _, _ = self._run(
                Path(temp_dir), break_v2_holdout_id=2
            )
            self.assertEqual(report["final_status"], "rejected_holdout")
            self.assertFalse(report["holdout"]["gate"]["passed"])
            self.assertFalse(
                report["holdout"]["gate"]["checks"][
                    "preserves_v1_correct_cases"
                ]
            )


if __name__ == "__main__":
    unittest.main()
