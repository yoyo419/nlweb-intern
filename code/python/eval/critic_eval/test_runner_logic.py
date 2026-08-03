"""
Logic tests for the critic auto-eval runner (① 測 runner 程式邏輯本身).

These exercise the runner's PURE logic — scoring, flag-gating, calibration
classification, case loading, env capture, and the mock mappings — with NO LLM
calls and no money spent. Run:

    ../../venv/Scripts/python.exe -m unittest eval.critic_eval.test_runner_logic -v

Design references: docs/specs/critic-eval-plan.md §5 (pass/fail rule),
§6.5 (flag-gating), §3.1 (calibration).
"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from eval.critic_eval import runner
from eval.critic_eval.schema import EvalCase, JudgeIssue, JudgeVerdict


# --- small builders -------------------------------------------------------
def verdict(dim, found, n=1):
    """A JudgeVerdict with `n` dummy issues when found=True."""
    issues = [JudgeIssue(quote="q", reason="r", evidence_check="e")] * (n if found else 0)
    return JudgeVerdict(dimension=dim, found_issue=found, issues=issues)


def critic(status, source_issues=None, logical_gaps=None):
    return {
        "status": status,
        "source_issues": source_issues or [],
        "logical_gaps": logical_gaps or [],
    }


ALL_QUIET = {  # no judge accuses
    "grounding": verdict("grounding", False),
    "fabrication": verdict("fabrication", False),
    "logic": verdict("logic", False),
}


class TestScoreCase(unittest.TestCase):
    """§5 mixed rule: a judge 'accuses' when found_issue=True; critic is credited
    if the mapped field is non-empty OR status != PASS. Any miss fails; a HARD
    (grounding/fabrication) miss is a one-vote-veto."""

    def test_clean_case_all_pass(self):
        sc = runner.score_case(critic("PASS"), ALL_QUIET)
        self.assertTrue(sc["critic_ok"])
        self.assertEqual(sc["judges_accused"], [])
        self.assertEqual(sc["critic_missed"], [])
        self.assertFalse(sc["hard_veto"])

    def test_hard_dim_caught_via_field(self):
        # fabrication judge accuses; critic listed it in source_issues -> held
        judges = {**ALL_QUIET, "fabrication": verdict("fabrication", True)}
        sc = runner.score_case(critic("REJECT", source_issues=["編造 X"]), judges)
        self.assertTrue(sc["critic_ok"])
        self.assertIn("fabrication", sc["critic_held"])
        self.assertEqual(sc["critic_missed"], [])

    def test_hard_dim_missed_triggers_veto(self):
        # grounding judge accuses; critic PASS with empty fields -> missed + veto
        judges = {**ALL_QUIET, "grounding": verdict("grounding", True)}
        sc = runner.score_case(critic("PASS"), judges)
        self.assertFalse(sc["critic_ok"])
        self.assertIn("grounding", sc["critic_missed"])
        self.assertTrue(sc["hard_veto"])

    def test_soft_logic_miss_fails_but_not_veto(self):
        # logic judge accuses; critic PASS empty -> missed, fails, but NOT a hard veto
        judges = {**ALL_QUIET, "logic": verdict("logic", True)}
        sc = runner.score_case(critic("PASS"), judges)
        self.assertFalse(sc["critic_ok"])
        self.assertIn("logic", sc["critic_missed"])
        self.assertFalse(sc["hard_veto"])

    def test_strict_reject_without_field_is_a_miss(self):
        # STRICT rule (yoyo 定案，改嚴): a bare non-PASS status earns NO credit.
        # Logic is accused but the critic's logical_gaps is empty -> missed,
        # even though the critic said REJECT (it rejected for some other reason).
        judges = {**ALL_QUIET, "logic": verdict("logic", True)}
        sc = runner.score_case(critic("REJECT", logical_gaps=[]), judges)
        self.assertFalse(sc["critic_ok"])
        self.assertIn("logic", sc["critic_missed"])

    def test_strict_warn_without_field_is_a_miss(self):
        # grounding accused, source_issues empty, status WARN -> still a miss
        judges = {**ALL_QUIET, "grounding": verdict("grounding", True)}
        sc = runner.score_case(critic("WARN"), judges)
        self.assertFalse(sc["critic_ok"])
        self.assertIn("grounding", sc["critic_missed"])
        self.assertTrue(sc["hard_veto"])

    def test_strict_field_present_but_status_pass_still_held(self):
        # The flip side: field non-empty earns credit even if status is PASS —
        # the critic *did* name the problem in the right dimension.
        judges = {**ALL_QUIET, "fabrication": verdict("fabrication", True)}
        sc = runner.score_case(critic("PASS", source_issues=["編造 X"]), judges)
        self.assertTrue(sc["critic_ok"])
        self.assertIn("fabrication", sc["critic_held"])

    def test_grounding_and_fabrication_share_source_issues(self):
        # both hard dims map to the same critic field; one non-empty field covers both
        judges = {
            "grounding": verdict("grounding", True),
            "fabrication": verdict("fabrication", True),
            "logic": verdict("logic", False),
        }
        sc = runner.score_case(critic("PASS", source_issues=["來源問題"]), judges)
        self.assertTrue(sc["critic_ok"])
        self.assertCountEqual(sc["critic_held"], ["grounding", "fabrication"])

    def test_mixed_hard_held_soft_missed(self):
        # fabrication caught (field), logic missed (PASS empty) -> fails, no veto
        judges = {
            "grounding": verdict("grounding", False),
            "fabrication": verdict("fabrication", True),
            "logic": verdict("logic", True),
        }
        sc = runner.score_case(critic("PASS", source_issues=["編造"]), judges)
        self.assertFalse(sc["critic_ok"])
        self.assertIn("fabrication", sc["critic_held"])
        self.assertIn("logic", sc["critic_missed"])
        self.assertFalse(sc["hard_veto"])  # only soft dim missed


class TestCompareBaseline(unittest.TestCase):
    """§6.5: refuse to compare across different flags/env."""

    def _write(self, tmp, env, rate):
        p = Path(tmp) / "base.json"
        p.write_text(json.dumps({"env": env, "pass_rate": rate}), encoding="utf-8")
        return p

    def test_same_env_reports_delta(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            env = {"mock": False, "structured_critique": True, "cov_lite_enabled": False}
            p = self._write(tmp, env, 0.80)
            buf = io.StringIO()
            with redirect_stdout(buf):
                runner.compare_baseline(p, env, 0.60)  # dropped
            out = buf.getvalue()
            self.assertIn("退步", out)
            self.assertNotIn("比對中止", out)

    def test_flag_mismatch_aborts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base_env = {"mock": False, "structured_critique": True}
            p = self._write(tmp, base_env, 0.80)
            buf = io.StringIO()
            with redirect_stdout(buf):
                runner.compare_baseline(p, {"mock": False, "structured_critique": False}, 0.80)
            out = buf.getvalue()
            self.assertIn("比對中止", out)
            self.assertIn("不可比", out)

    def test_equal_rate_is_flat(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            env = {"mock": True}
            p = self._write(tmp, env, 1.0)
            buf = io.StringIO()
            with redirect_stdout(buf):
                runner.compare_baseline(p, env, 1.0)
            self.assertIn("持平", buf.getvalue())


class TestLoadCases(unittest.TestCase):
    def test_gold_cases_load_and_validate(self):
        gold = Path(runner.__file__).resolve().parent / "cases" / "gold_cases.yaml"
        cases = runner.load_cases(gold)
        self.assertEqual(len(cases), 6)
        self.assertTrue(all(isinstance(c, EvalCase) for c in cases))
        # every case declares an expected verdict and at least one label
        for c in cases:
            self.assertIn(c.expected_critic_verdict, ("PASS", "WARN", "REJECT"))
            self.assertTrue(c.human_labels)

    def test_missing_cases_key_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.yaml"
            p.write_text("something: else\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.load_cases(p)


class TestFormatContextAndEnv(unittest.TestCase):
    def test_format_context_numbered(self):
        from eval.critic_eval.schema import Source
        s = [Source(id=1, text="甲"), Source(id=2, text="乙")]
        self.assertEqual(runner.format_context(s), "[1] 甲\n[2] 乙")

    def test_format_context_empty(self):
        self.assertEqual(runner.format_context([]), "（無來源）")

    def test_capture_env_mock(self):
        self.assertEqual(runner.capture_env(True), {"mock": True})


class TestMockMappings(unittest.TestCase):
    """The mock critic/judges must mirror human labels into the right dimensions."""

    def _case(self, defect_type, quote="X"):
        from eval.critic_eval.schema import DefectLabel, Source
        return EvalCase(
            id="t", query="q", sources=[Source(id=1, text="s")], draft="d",
            human_labels=[DefectLabel(type=defect_type, quote=quote, reason="r")],
            expected_critic_verdict="REJECT",
        )

    def test_fabrication_hits_grounding_and_fabrication(self):
        j = runner._mock_judges(self._case("fabrication"))
        self.assertTrue(j["grounding"].found_issue)
        self.assertTrue(j["fabrication"].found_issue)
        self.assertFalse(j["logic"].found_issue)

    def test_fake_citation_hits_fabrication_only(self):
        j = runner._mock_judges(self._case("fake_citation"))
        self.assertFalse(j["grounding"].found_issue)
        self.assertTrue(j["fabrication"].found_issue)
        self.assertFalse(j["logic"].found_issue)

    def test_logic_jump_hits_logic_only(self):
        j = runner._mock_judges(self._case("logic_jump"))
        self.assertFalse(j["grounding"].found_issue)
        self.assertFalse(j["fabrication"].found_issue)
        self.assertTrue(j["logic"].found_issue)

    def test_clean_case_no_accusation(self):
        j = runner._mock_judges(self._case("none", quote=""))
        self.assertFalse(any(v.found_issue for v in j.values()))

    def test_mock_critic_mirrors_labels_and_verdict(self):
        c = runner._mock_critic(self._case("fabrication"))
        self.assertEqual(c["status"], "REJECT")
        self.assertTrue(c["source_issues"])
        self.assertFalse(c["logical_gaps"])
        # Reinforce the constraint against the SOURCE OF TRUTH, not a copied magic
        # number: the mock must be a VALID CriticReviewOutput. This ties the check
        # to production's real schema — if the critique floor (or any field rule)
        # changes upstream, this test tracks it automatically instead of drifting.
        from reasoning.schemas import CriticReviewOutput
        CriticReviewOutput(
            status=c["status"], critique=c["critique"],
            mode_compliance=c["mode_compliance"],
            source_issues=c["source_issues"], logical_gaps=c["logical_gaps"],
        )  # raises ValidationError if critique < min_length or any field invalid


class TestFullMockPipeline(unittest.TestCase):
    """End-to-end over the real gold set in mock mode: a perfect critic + perfect
    judges must score 6/6. This is the $0 sanity gate for the whole flow."""

    def test_gold_mock_perfect_score(self):
        import asyncio
        gold = Path(runner.__file__).resolve().parent / "cases" / "gold_cases.yaml"
        cases = runner.load_cases(gold)

        async def run_one(case):
            c = asyncio.get_event_loop()  # noqa: F841
            crit = await runner.run_critic(case, mock=True)
            judges = await runner.run_judges(case, mock=True, critic_verdict=crit["status"])
            return runner.score_case(crit, judges)

        async def run_all():
            return [await run_one(c) for c in cases]

        scores = asyncio.run(run_all())
        ok = sum(1 for s in scores if s["critic_ok"])
        self.assertEqual(ok, len(cases), f"expected perfect, got {ok}/{len(cases)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
