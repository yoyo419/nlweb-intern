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
import contextlib
from contextlib import redirect_stdout
from pathlib import Path

from core.temporal_anchor import TEMPORAL_ANCHOR_RULE
from eval.critic_eval import runner
from eval.critic_eval.schema import EvalCase, JudgeIssue, JudgeVerdict, Source


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
        # prod 對齊 (boss 2026-08-12 #2): 前面多了當前時間 header + 「可用資料來源」段，
        # 薄來源仍維持終端的「[id] 內容」段落。
        from eval.critic_eval.schema import Source
        s = [Source(id=1, text="甲"), Source(id=2, text="乙")]
        out = runner.format_context(s)
        self.assertIn("## 當前時間", out)
        self.assertTrue(out.rstrip().endswith("[1] 甲\n[2] 乙"))

    def test_format_context_empty(self):
        # 無來源仍會有當前時間 header（prod header 永遠非空），本體為「（無來源）」。
        out = runner.format_context([])
        self.assertIn("## 當前時間", out)
        self.assertTrue(out.rstrip().endswith("（無來源）"))

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


class TestJudgesOnlyMode(unittest.TestCase):
    """校準模式不跑 critic：省每案一次高階呼叫，但不得因此產生假的分數。"""

    def test_placeholder_critic_credits_nothing(self):
        """佔位 critic 的欄位必須全空——否則 judges-only 會憑空給 critic 分數。"""
        ph = runner._JUDGES_ONLY_CRITIC
        self.assertIsNone(ph["status"])
        self.assertEqual(ph["source_issues"], [])
        self.assertEqual(ph["logical_gaps"], [])
        # 用它去算分，任何評審指控都算漏抓（不會假裝守住）
        judges = {"grounding": JudgeVerdict(dimension="grounding", found_issue=True,
                                            issues=[JudgeIssue(quote="q", reason="r",
                                                               evidence_check="e")]),
                  "fabrication": JudgeVerdict(dimension="fabrication", found_issue=False),
                  "logic": JudgeVerdict(dimension="logic", found_issue=False)}
        sc = runner.score_case(dict(ph), judges)
        self.assertFalse(sc["critic_ok"])

    def test_judges_only_rejects_gate_and_compare(self):
        """沒有通過率就沒有可比的東西——不可讓 gate 拿 judges-only 的跑法當基準。"""
        import sys as _sys
        for extra in (["--gate"], ["--compare", "b.json"]):
            argv = ["runner", "--judges-only"] + extra
            old = _sys.argv
            _sys.argv = argv
            try:
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                    with contextlib.redirect_stderr(io.StringIO()):
                        runner.main()
                self.assertNotEqual(cm.exception.code, 0)
            finally:
                _sys.argv = old


class TestFormatContextProdParity(unittest.TestCase):
    """機械防線 (boss 2026-08-12 feedback #2): the string fed to the critic must be
    structurally aligned with prod orchestrator._format_context_shared — specifically
    it MUST carry the 當前時間 header (else the critic can't judge 今天/最近/今年), and
    render prod-style 網站 - 標題 (日期) headers when the fixture carries those fields.

    mutation check: delete the header line in runner._current_time_header, or the
    site/title branch in format_context, and the matching assertion below goes red.
    """

    def test_current_time_header_present(self):
        """『當前時間』區塊 + 『可用資料來源』段首必須存在（prod 的第三樣、影響最大）。"""
        out = runner.format_context([Source(id=1, text="內容")])
        self.assertIn("## 當前時間", out)
        self.assertIn("當前時間", out)
        self.assertIn("## 可用資料來源", out)
        # 時間提示語，讓 critic 知道怎麼用這個時間
        self.assertIn("最近", out)

    def test_rich_source_renders_site_title_date(self):
        """帶 site/title/date 的來源 → 渲染成『[id] 網站 - 標題 (YYYY-MM-DD)』。"""
        out = runner.format_context([
            Source(id=1, text="台積電第一季 EPS 8.70 元",
                   site="經濟日報", title="台積電法說會", date_published="2026-08-10T09:00:00"),
        ])
        self.assertIn("[1] 經濟日報 - 台積電法說會 (2026-08-10)", out)
        self.assertIn("台積電第一季 EPS 8.70 元", out)

    def test_thin_source_degrades_to_terse(self):
        """只有 id+text 的薄來源 → 維持『[id] 內容』，不硬塞 Unknown/No title。"""
        out = runner.format_context([Source(id=2, text="純文字來源")])
        self.assertIn("[2] 純文字來源", out)
        self.assertNotIn("Unknown", out)
        self.assertNotIn("No title", out)

    def test_source_relative_year_anchored_to_publication_date(self):
        """來源內文的「今年」＝該來源發布年（prod _format_context_shared 同口徑）。

        沒有這道錨定，2020 年的來源寫「今年」→ critic 會拿「當前時間」換算成今年，
        判 grounding 時就對不上真實年份。
        """
        out = runner.format_context([
            Source(id=1, text="今年營收成長 12%", site="經濟日報",
                   title="法說會", date_published="2020-05-06"),
        ])
        self.assertIn("今年（2020年）", out)
        self.assertIn(TEMPORAL_ANCHOR_RULE, out)

    def test_citation_marker_preserved(self):
        """[id] 引用標記務必保留——CoV 假引用查核與 grounding 都靠它。"""
        out = runner.format_context([
            Source(id=1, text="a"), Source(id=2, text="b"), Source(id=3, text="c"),
        ])
        for i in (1, 2, 3):
            self.assertIn(f"[{i}]", out)


def _res(case_id, ok):
    """一筆結果／基準紀錄（gate 只看 id 與 score.critic_ok）。"""
    return {"id": case_id, "score": {"critic_ok": ok}}


class TestGateDecision(unittest.TestCase):
    """§5 整組層 gate：可比性 → BLOCKED、退步 → FAIL、其餘 PASS。

    機械防線重點在「不可比不得靜靜放行」與「總分持平也要抓逐案退步」，
    這兩條是 gate 失效最容易發生的地方。
    mutation 驗法：把 evaluate_gate 的 blockers 判斷或 regressed 判斷拿掉，
    對應的測試必須轉紅。
    """

    ENV = {"mock": False, "structured_critique": True, "context_format": "prod-parity-v1"}

    def _base(self, rate, results=None, env=None, cases_file="gold_cases.yaml"):
        return {"env": env if env is not None else self.ENV, "cases_file": cases_file,
                "pass_rate": rate, "results": results or []}

    # --- BLOCKED：不可比 ---------------------------------------------------
    def test_no_baseline_is_blocked_not_pass(self):
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 1.0, [], None)
        self.assertEqual(g["verdict"], runner.GATE_BLOCKED)
        self.assertTrue(g["reasons"])

    def test_env_mismatch_is_blocked(self):
        other = dict(self.ENV, structured_critique=False)
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 1.0, [], self._base(1.0, env=other))
        self.assertEqual(g["verdict"], runner.GATE_BLOCKED)
        self.assertIn("structured_critique", " ".join(g["reasons"]))

    def test_missing_context_format_in_old_baseline_is_blocked(self):
        """舊基準沒有 context_format（產生於舊版 format_context）→ 不可比。

        context 形狀變了 critic 看到的東西就變了，分數差不能算退步也不能算持平。
        """
        old = {k: v for k, v in self.ENV.items() if k != "context_format"}
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 1.0, [], self._base(1.0, env=old))
        self.assertEqual(g["verdict"], runner.GATE_BLOCKED)
        self.assertIn("context_format", " ".join(g["reasons"]))

    def test_different_cases_file_is_blocked(self):
        g = runner.evaluate_gate(self.ENV, "gold_extended.yaml", 1.0, [],
                                 self._base(1.0, cases_file="gold_cases.yaml"))
        self.assertEqual(g["verdict"], runner.GATE_BLOCKED)
        self.assertIn("考卷不同", " ".join(g["reasons"]))

    def test_baseline_without_pass_rate_is_blocked(self):
        base = self._base(1.0)
        base.pop("pass_rate")
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 1.0, [], base)
        self.assertEqual(g["verdict"], runner.GATE_BLOCKED)

    # --- FAIL：退步 --------------------------------------------------------
    def test_pass_rate_drop_fails(self):
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 0.60, [_res("a", False)],
                                 self._base(0.80, [_res("a", True)]))
        self.assertEqual(g["verdict"], runner.GATE_FAIL)
        self.assertLess(g["delta"], 0)

    def test_flat_rate_but_case_swap_still_fails(self):
        """一升一降 → 總通過率持平，但確實有案例從守住變成漏抓，必須 FAIL。

        只看總分的 gate 會在這裡放行，這是它最危險的盲點。
        """
        now = [_res("a", False), _res("b", True)]
        base = self._base(0.50, [_res("a", True), _res("b", False)])
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 0.50, now, base)
        self.assertEqual(g["verdict"], runner.GATE_FAIL)
        self.assertEqual(g["regressed"], ["a"])
        self.assertEqual(g["improved"], ["b"])

    def test_drop_beyond_tolerance_fails(self):
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 0.80, [], self._base(1.0),
                                 tolerance=0.10)
        self.assertEqual(g["verdict"], runner.GATE_FAIL)

    # --- PASS --------------------------------------------------------------
    def test_identical_run_passes(self):
        now = [_res("a", True), _res("b", True)]
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 1.0, now, self._base(1.0, now))
        self.assertEqual(g["verdict"], runner.GATE_PASS)
        self.assertEqual(g["reasons"], [])

    def test_improvement_passes(self):
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 1.0, [_res("a", True)],
                                 self._base(0.50, [_res("a", False)]))
        self.assertEqual(g["verdict"], runner.GATE_PASS)
        self.assertEqual(g["improved"], ["a"])

    def test_drop_within_tolerance_passes(self):
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 0.95, [], self._base(1.0),
                                 tolerance=0.10)
        self.assertEqual(g["verdict"], runner.GATE_PASS)

    def test_case_only_on_one_side_is_reported_not_judged(self):
        """基準沒跑過的新案不算退步，但要被點名，避免題目換了卻無聲比。"""
        g = runner.evaluate_gate(self.ENV, "gold_cases.yaml", 1.0,
                                 [_res("a", True), _res("new", True)],
                                 self._base(1.0, [_res("a", True)]))
        self.assertEqual(g["verdict"], runner.GATE_PASS)
        self.assertEqual(g["unmatched"], ["new"])

    # --- 出口碼 ------------------------------------------------------------
    def test_exit_codes_are_distinct_and_nonzero_on_problem(self):
        self.assertEqual(runner.GATE_EXIT[runner.GATE_PASS], 0)
        self.assertNotEqual(runner.GATE_EXIT[runner.GATE_FAIL], 0)
        self.assertNotEqual(runner.GATE_EXIT[runner.GATE_BLOCKED], 0)
        self.assertNotEqual(runner.GATE_EXIT[runner.GATE_FAIL],
                            runner.GATE_EXIT[runner.GATE_BLOCKED])

    def test_capture_env_stamps_context_format(self):
        """context 格式版本必須進 env，gate 才擋得掉跨格式比對（靠機制不靠人記得）。"""
        env = runner.capture_env(False)
        self.assertEqual(env.get("context_format"), runner.CONTEXT_FORMAT_VERSION)


class TestCompareBaselinePrintsGate(unittest.TestCase):
    """compare_baseline 只是 evaluate_gate 的列印皮，判定不得另立一套規則。"""

    def test_returns_gate_verdict_and_prints_case_regression(self):
        import tempfile
        env = {"mock": True}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "base.json"
            p.write_text(json.dumps({"env": env, "cases_file": "c.yaml", "pass_rate": 0.50,
                                     "results": [_res("a", True), _res("b", False)]}),
                         encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                gate = runner.compare_baseline(p, env, 0.50, "c.yaml",
                                               [_res("a", False), _res("b", True)])
            out = buf.getvalue()
        self.assertEqual(gate["verdict"], runner.GATE_FAIL)
        self.assertIn("持平", out)          # 總分持平
        self.assertIn("逐案退步", out)      # 但逐案抓到退步
        self.assertIn("a", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
