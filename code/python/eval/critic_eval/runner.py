"""
Runner for the critic auto-eval harness.

One command that:
  1. loads a fixed set of exam cases (query + sources + draft),
  2. runs the REAL critic on each draft (or a free mock),
  3. runs the 3 adversarial judges to find issues the critic missed,
  4. records the environment (feature flags + model) into the result,
  5. scores each case and writes a baseline JSON.

Two most important design rules (see docs/specs/critic-eval-plan.md §6.5, §3.1):
  - The critic under test must be the SAME critic that ships. Its behavior is
    controlled by feature flags (structured_critique / cov_lite_enabled) read at
    runtime. We stamp those flags into every result and REFUSE to compare two
    runs whose flags differ — a score drop under different flags is meaningless.
  - Judge-vs-human agreement (calibration) is NOT auto string-matched. The runner
    only prints human labels and judge findings side by side; whether they are
    "the same error" is decided by a human. We only auto-decide the easy case:
    a clean draft where a judge must return found_issue=false.

Usage:
    # free, no LLM — demo the whole flow at $0 (judges/critic are mocked from labels)
    python -m eval.critic_eval.runner --cases cases/gold_cases.yaml --mock

    # real run, save a baseline
    python -m eval.critic_eval.runner --cases cases/gold_cases.yaml --save baseline.json

    # rerun later and compare against a saved baseline (flags must match)
    python -m eval.critic_eval.runner --cases cases/gold_cases.yaml --compare baseline.json

    # gate mode: same compare, but the exit code can actually block a merge
    #   0 = PASS / 1 = FAIL(退步) / 2 = BLOCKED(不可比，要人處理)
    python -m eval.critic_eval.runner --cases cases/gold_cases.yaml \
        --compare baseline.json --gate
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import TypeAdapter

# Allow `python eval/critic_eval/runner.py ...` as well as `-m`.
_CODE_PYTHON = Path(__file__).resolve().parents[2]
if str(_CODE_PYTHON) not in sys.path:
    sys.path.insert(0, str(_CODE_PYTHON))

from eval.critic_eval.schema import EvalCase, JudgeIssue, JudgeVerdict, Source  # noqa: E402
from eval.critic_eval.judges import JUDGES  # noqa: E402
from core.temporal_anchor import (  # noqa: E402
    TEMPORAL_ANCHOR_RULE,
    anchor_note,
    annotate_relative_years,
)


# A judge's dimension -> which critic output field should carry the same finding.
# mode_compliance is intentionally excluded: no judge covers it, so it is not scored.
DIM_TO_CRITIC_FIELD = {
    "grounding": "source_issues",
    "fabrication": "source_issues",
    "logic": "logical_gaps",
}
# "Hard" dimensions get one-vote-veto (§5); logic is the softer/subjective one.
HARD_DIMENSIONS = {"grounding", "fabrication"}

# format_context 的形狀會改變 critic 看到的東西 → 改了就跟舊基準不可比。
# 這個版本字串會戳進 env，讓 gate 自動擋掉跨格式的比對（不是靠人記得）。
CONTEXT_FORMAT_VERSION = "prod-parity-v1"


# ---------------------------------------------------------------------------
# case loading
# ---------------------------------------------------------------------------
def load_cases(path: Path) -> List[EvalCase]:
    """Load and validate a YAML case file into EvalCase objects."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw or "cases" not in raw:
        raise ValueError(f"{path} 沒有 top-level 'cases:' 清單")
    return TypeAdapter(List[EvalCase]).validate_python(raw["cases"])


def _current_time_header() -> str:
    """Replicate orchestrator._get_current_time_header so the critic sees the SAME
    「當前時間」block prod feeds it (boss 2026-08-12 feedback #2).

    Without it the critic cannot judge「今天/最近/今年」claims — and news queries are
    frequently time-sensitive, so the eval would test a different behavior than prod.
    Uses real now() to match prod (not a pinned time); a run-to-run baseline on
    time-sensitive cases therefore carries some inherent time drift — acceptable per
    boss「基準要的是真實形狀而非歷史考古」.
    """
    from core.config import CONFIG

    timezone_str = CONFIG.reasoning_params.get("timezone", "Asia/Taipei")
    try:
        import pytz
        current_time = datetime.now(pytz.timezone(timezone_str))
    except ImportError:
        current_time = datetime.now()
    weekday = ["星期一", "星期二", "星期三", "星期四",
               "星期五", "星期六", "星期日"][current_time.weekday()]
    return (
        f"## 當前時間\n"
        f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} {weekday} ({timezone_str})\n\n"
        f"當用戶詢問「今天」、「最近」、「現在」等時間相關詞彙時，請參考上述當前時間。\n"
        f"**注意：上述當前時間只用於理解「使用者問句」的相對時間，"
        f"不可用來換算「來源內文」的相對時間。**\n"
        f"{TEMPORAL_ANCHOR_RULE}\n\n"
        f"## 可用資料來源\n"
    )


def format_context(sources: List[Source]) -> str:
    """Render the context string the critic reads — structurally aligned with prod
    orchestrator._format_context_shared (boss 2026-08-12 feedback #2).

    Layout: a 當前時間 header on top, then per-source「[id] 網站 - 標題 (日期)」headers
    followed by the source text. Thin fixtures (only id+text, e.g. synthetic gold
    cases) degrade to the terse「[id] 內容」under the same wrapper.
    """
    header = _current_time_header()
    if not sources:
        return header + "（無來源）"
    parts = []
    for s in sources:
        # thin case: no prod-parity fields → keep terse「[id] 內容」
        if not (s.site or s.title or s.date_published):
            parts.append(f"[{s.id}] {s.text}")
            continue
        line_head = f"[{s.id}] {s.site or 'Unknown'} - {s.title or 'No title'}"
        if s.date_published:
            line_head += f" ({str(s.date_published).split('T')[0]})"
        line_head += anchor_note(s.date_published)
        # 發布日期錨定：與 prod 同口徑（prod 在 _format_context_shared 做同一件事）
        parts.append(f"{line_head}\n{annotate_relative_years(s.text, s.date_published)}")
    return header + "\n".join(parts)


# ---------------------------------------------------------------------------
# environment capture — the single most important record (§6.5)
# ---------------------------------------------------------------------------
def capture_env(mock: bool) -> Dict:
    """Stamp the exact conditions the critic ran under, so baselines are comparable."""
    if mock:
        return {"mock": True}
    from core.config import CONFIG

    feats = CONFIG.reasoning_params.get("features", {})
    provider = CONFIG.llm_endpoints.get("openai")
    model = provider.models.high if (provider and provider.models) else "unknown"
    return {
        "mock": False,
        "structured_critique": feats.get("structured_critique", False),
        "cov_lite_enabled": feats.get("cov_lite_enabled", False),
        "enable_live_research": False,  # runner locks the DR whole-draft path
        "typeagent_enabled": CONFIG.reasoning_params.get("typeagent", {}).get("enabled", False),
        "model_high": model,
        "context_format": CONTEXT_FORMAT_VERSION,
    }


# ---------------------------------------------------------------------------
# running the critic
# ---------------------------------------------------------------------------
class _StubHandler:
    """Minimal handler the critic needs (only .query_params is read downstream)."""

    query_params: Dict = {}


async def run_critic(case: EvalCase, mock: bool) -> Dict:
    """Run the real critic on one draft; in mock mode derive a plausible output from labels."""
    if mock:
        return _mock_critic(case)

    from reasoning.agents.critic import CriticAgent

    critic = CriticAgent(handler=_StubHandler())
    out = await critic.review(
        draft=case.draft,
        query=case.query,
        mode="deep_research",  # value ignored since 2026-04; must still be passed
        analyst_output=None,   # known limitation (§6.5): eval runs the "no argument_graph" path
        formatted_context=format_context(case.sources),
        enable_live_research=False,
    )
    return {
        "status": out.status,
        "critique": out.critique,
        "suggestions": list(out.suggestions),
        "source_issues": list(out.source_issues),
        "logical_gaps": list(out.logical_gaps),
        "mode_compliance": out.mode_compliance,
    }


def _mock_critic(case: EvalCase) -> Dict:
    """Free stand-in: a *perfect* critic that flags exactly the human-labeled defects.

    Purpose is to exercise the scoring machinery at $0, not to judge anything.
    """
    src, logic = [], []
    for lab in case.human_labels:
        if lab.type in ("fabrication", "fake_citation"):
            src.append(f"{lab.quote}：{lab.reason}")
        elif lab.type == "logic_jump":
            logic.append(f"{lab.quote}：{lab.reason}")
    return {
        "status": case.expected_critic_verdict,
        # critique 必須 >= 50 字以符合 CriticReviewOutput schema 的 min_length（即使
        # mock 模式回純 dict 不過驗證，也保持與真 schema 一致，避免潛在不符）。
        "critique": "（mock）本評語依人工標註自動產生，僅供 $0 流程演示與 runner 邏輯測試之占位用途，不代表任何真實 critic 對草稿品質的判斷結果。",
        "suggestions": [],
        "source_issues": src,
        "logical_gaps": logic,
        "mode_compliance": "符合",
    }


# ---------------------------------------------------------------------------
# running the judges
# ---------------------------------------------------------------------------
async def run_judges(case: EvalCase, mock: bool,
                     critic_verdict: Optional[str]) -> Dict[str, JudgeVerdict]:
    """Run all three judges on one case. Returns dimension -> JudgeVerdict."""
    if mock:
        return _mock_judges(case)

    from reasoning.agents.base import generate_structured

    async def _one(dim: str) -> JudgeVerdict:
        prompt = JUDGES[dim](case.query, case.sources, case.draft, critic_verdict)
        verdict, _, _ = await generate_structured(prompt, JudgeVerdict)
        return verdict

    dims = list(JUDGES.keys())
    results = await asyncio.gather(*[_one(d) for d in dims])
    return dict(zip(dims, results))


def _mock_judges(case: EvalCase) -> Dict[str, JudgeVerdict]:
    """Free stand-in: judges that 'find' exactly the human-labeled defects for their dimension."""
    buckets: Dict[str, List[JudgeIssue]] = {"grounding": [], "fabrication": [], "logic": []}
    for lab in case.human_labels:
        issue = JudgeIssue(quote=lab.quote, reason=lab.reason,
                           evidence_check=lab.source_hint or "（mock）")
        if lab.type == "fabrication":
            buckets["grounding"].append(issue)
            buckets["fabrication"].append(issue)
        elif lab.type == "fake_citation":
            buckets["fabrication"].append(issue)
        elif lab.type == "logic_jump":
            buckets["logic"].append(issue)
    return {
        dim: JudgeVerdict(dimension=dim, found_issue=bool(items), issues=items)
        for dim, items in buckets.items()
    }


# ---------------------------------------------------------------------------
# scoring one case
# ---------------------------------------------------------------------------
def score_case(critic: Dict, judges: Dict[str, JudgeVerdict]) -> Dict:
    """Decide whether the critic 'held the line' on this case, per the §5 mixed rule.

    A judge 'accuses' a dimension when found_issue=True. STRICT credit rule
    (yoyo 定案 2026-08，改嚴): the critic is credited with catching a dimension ONLY
    if its own mapped field is non-empty (source_issues / logical_gaps). A bare
    non-PASS status does NOT earn credit — a critic that REJECTs for reason A must
    not be scored as also catching an unrelated problem B it never mentioned.
    Otherwise a 'reject everything' critic would score high and hide regressions.
    A HARD dimension unmet = one-vote-veto fail.
    """
    misses, held = [], []
    for dim, verdict in judges.items():
        if not verdict.found_issue:
            continue
        field = DIM_TO_CRITIC_FIELD[dim]
        flagged = bool(critic.get(field))  # 改嚴：只認「該維度欄位真的有寫東西」
        (held if flagged else misses).append(dim)

    hard_missed = [d for d in misses if d in HARD_DIMENSIONS]
    critic_ok = not hard_missed and not misses  # any miss fails; hard miss = veto
    return {
        "critic_status": critic.get("status"),
        "judges_accused": [d for d, v in judges.items() if v.found_issue],
        "critic_held": held,
        "critic_missed": misses,
        "hard_veto": bool(hard_missed),
        "critic_ok": critic_ok,
    }


# ---------------------------------------------------------------------------
# calibration view (§3.1): print human vs judge side by side; no auto matching
# ---------------------------------------------------------------------------
def print_calibration(case: EvalCase, judges: Dict[str, JudgeVerdict]) -> None:
    human = [f"[{lab.type}] {lab.quote}" for lab in case.human_labels if lab.type != "none"]
    is_clean = not human
    print(f"\n── 校準 · {case.id} ──")
    print(f"  人工標註：{human if human else '（乾淨，無缺陷）'}")
    for dim, v in judges.items():
        found = [f"「{i.quote}」— {i.reason}" for i in v.issues]
        if is_clean:
            # the only auto-decidable case: clean draft => judge must find nothing
            mark = "OK" if not v.found_issue else "假警報!"
            print(f"  {dim:11s} found_issue={str(v.found_issue):5s} [{mark}]"
                  + ("" if not found else f" → {found}"))
        else:
            print(f"  {dim:11s} 報了：{found if found else '（無）'}  ← 由人判定是否對到人工標的錯")


# ---------------------------------------------------------------------------
# baseline compare + gate 判定（§5 整組層 / §6.5 可比性）
# ---------------------------------------------------------------------------
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_BLOCKED = "BLOCKED"
# 出口碼：讓 pre-merge / nightly 真的擋得住下游（0 放行、1 退步、2 不可比要人處理）
GATE_EXIT = {GATE_PASS: 0, GATE_FAIL: 1, GATE_BLOCKED: 2}


def _case_ok_map(results: List[Dict]) -> Dict[str, bool]:
    """id -> critic_ok。本次跑的結果與存下來的基準是同一個形狀，共用這支。"""
    return {r["id"]: bool((r.get("score") or {}).get("critic_ok"))
            for r in results if r.get("id")}


def evaluate_gate(env: Dict, cases_file: str, pass_rate: float, results: List[Dict],
                  baseline: Optional[Dict], tolerance: float = 0.0) -> Dict:
    """整組層的過／不過判定（plan §5 第二層）。

    單案層（哪些維度算 critic 沒守住）已由 score_case 定案：硬傷維度
    (grounding / fabrication) 一票否決、任一維度漏抓即該案 MISS。這裡只做整組層，
    把「跟基準比有沒有掉」收斂成一個明確判定，讓 gate 真的擋得住下游：

      BLOCKED —— 兩次跑「不可比」：沒有基準、開關/模型不同、context 格式不同、
                 或考卷不同。這種情況不可放行也不可當退步，必須停下來由人處理；
                 悄悄當成 PASS 等於讓整條 gate 失效（不可 silent fail）。
      FAIL    —— 可比且退步：通過率掉超過容忍值，或有案例從基準的 OK 變成 MISS。
                 逐案比對是必要的——一升一降會讓總通過率持平，只看總分會漏掉真退步。
      PASS    —— 可比且沒退步。

    tolerance 是 §5 說的「可調參數」：LLM 有非決定性，容忍值放大可換取穩定度，
    但預設 0.0（最嚴），要放鬆得是明講的決定，不是預設值偷偷幫你放行。
    """
    if baseline is None:
        return {"verdict": GATE_BLOCKED,
                "reasons": ["沒有基準可比：gate 需要 --compare <baseline.json>"],
                "delta": None, "regressed": [], "improved": [], "unmatched": []}

    blockers: List[str] = []
    base_env = baseline.get("env") or {}
    if base_env != env:
        keys = sorted(set(base_env) | set(env))
        diffs = [f"{k}（基準={base_env.get(k, '無此鍵')} / 本次={env.get(k, '無此鍵')}）"
                 for k in keys if base_env.get(k) != env.get(k)]
        blockers.append("環境/開關與基準不一致，分數不可比（§6.5）：" + "、".join(diffs))

    base_file = baseline.get("cases_file")
    if base_file and cases_file and base_file != cases_file:
        blockers.append(f"考卷不同（基準={base_file} / 本次={cases_file}），通過率不可比")

    base_rate = baseline.get("pass_rate")
    if base_rate is None:
        blockers.append("基準檔沒有 pass_rate，無從比對")

    if blockers:
        return {"verdict": GATE_BLOCKED, "reasons": blockers, "delta": None,
                "regressed": [], "improved": [], "unmatched": []}

    now_ok = _case_ok_map(results)
    base_ok = _case_ok_map(baseline.get("results") or [])
    both = set(now_ok) & set(base_ok)
    regressed = sorted(c for c in both if base_ok[c] and not now_ok[c])
    improved = sorted(c for c in both if not base_ok[c] and now_ok[c])
    unmatched = sorted(set(now_ok) ^ set(base_ok))  # 基準沒有 / 本次沒跑的案，只提示不判定

    delta = pass_rate - base_rate
    reasons: List[str] = []
    if delta < -tolerance - 1e-9:
        reasons.append(f"通過率退步：{base_rate:.0%} → {pass_rate:.0%}"
                       f"（容忍值 {tolerance:.0%}）")
    if regressed:
        reasons.append(f"逐案退步（基準守住、本次漏抓）：{regressed}")
    return {"verdict": GATE_FAIL if reasons else GATE_PASS, "reasons": reasons,
            "delta": delta, "regressed": regressed, "improved": improved,
            "unmatched": unmatched}


def compare_baseline(baseline_path: Path, env: Dict, pass_rate: float,
                     cases_file: str = "", results: Optional[List[Dict]] = None,
                     tolerance: float = 0.0) -> Dict:
    """讀基準 → 算 gate → 印給人看，並把 gate 判定回傳給呼叫端決定出口碼。

    判定邏輯只有 evaluate_gate 一份，這裡不重寫一套規則——兩套規則遲早會分岔。
    """
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    gate = evaluate_gate(env, cases_file or baseline.get("cases_file", ""), pass_rate,
                         results or [], baseline, tolerance)
    if gate["verdict"] == GATE_BLOCKED:
        print("\n[比對中止] 兩次跑不可比 —— 不放行也不算退步（§6.5）。")
        for r in gate["reasons"]:
            print(f"  - {r}")
        print("  請在相同條件下重跑，或重建基準。")
        return gate

    base_rate = baseline.get("pass_rate")
    delta = gate["delta"]
    trend = "退步!" if delta < -1e-9 else ("持平" if abs(delta) <= 1e-9 else "進步")
    print(f"\n[比對基準] 通過率 {base_rate:.0%} → {pass_rate:.0%}（{trend}）")
    if gate["regressed"]:
        print(f"  逐案退步（基準 OK → 本次 MISS）：{gate['regressed']}")
    if gate["improved"]:
        print(f"  逐案進步（基準 MISS → 本次 OK）：{gate['improved']}")
    if gate["unmatched"]:
        print(f"  ⚠ 只出現在其中一邊的案例（不列入判定）：{gate['unmatched']}")
    for r in gate["reasons"]:
        print(f"  - {r}")
    return gate


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main_async(args) -> int:
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = Path(__file__).resolve().parent / cases_path
    cases = load_cases(cases_path)
    env = capture_env(args.mock)

    print(f"載入 {len(cases)} 案 ← {cases_path.name}")
    print(f"環境/開關：{env}")

    results, ok_count = [], 0
    for case in cases:
        critic = await run_critic(case, args.mock)
        # calibration (Q3) judges the draft alone; leave critic_verdict None
        cv = None if args.calibrate else critic.get("status")
        judges = await run_judges(case, args.mock, cv)
        sc = score_case(critic, judges)
        ok_count += int(sc["critic_ok"])
        results.append({"id": case.id, "critic": critic,
                        "judges": {d: v.model_dump() for d, v in judges.items()},
                        "score": sc})
        flag = "OK " if sc["critic_ok"] else "MISS"
        print(f"  [{flag}] {case.id:38s} critic={sc['critic_status']:6s} "
              f"評審指控={sc['judges_accused']} critic漏={sc['critic_missed']}")
        if args.calibrate:
            print_calibration(case, judges)

    pass_rate = ok_count / len(cases) if cases else 0.0
    print(f"\n通過率（critic 守住 / 全部）：{ok_count}/{len(cases)} = {pass_rate:.0%}")

    payload = {"env": env, "cases_file": cases_path.name,
               "pass_rate": pass_rate, "results": results}
    if args.save:
        Path(args.save).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"基準已存：{args.save}")
    if args.compare:
        gate = compare_baseline(Path(args.compare), env, pass_rate,
                                cases_path.name, results, args.tolerance)
    elif args.gate:
        # --gate 沒給基準 → BLOCKED，不可當成通過（不可 silent fail）
        gate = evaluate_gate(env, cases_path.name, pass_rate, results, None)
        print("\n[比對中止] " + "；".join(gate["reasons"]))
    else:
        return 0

    if not args.gate:
        return 0
    print(f"\n[GATE] {gate['verdict']}"
          + ("" if not gate["reasons"] else " —— " + "；".join(gate["reasons"])))
    return GATE_EXIT[gate["verdict"]]


def main() -> None:
    p = argparse.ArgumentParser(description="critic 自動評審 runner")
    p.add_argument("--cases", default="cases/gold_cases.yaml", help="case YAML（相對本檔目錄或絕對路徑）")
    p.add_argument("--mock", action="store_true", help="免費模式：不呼叫 LLM，從人工標註推導示範輸出")
    p.add_argument("--calibrate", action="store_true",
                   help="校準模式：評審只看草稿（不餵 critic 判定）+ 印人工 vs 評審並排")
    p.add_argument("--save", help="把本次結果（含 env/flags）存成基準 JSON")
    p.add_argument("--compare", help="與指定基準 JSON 比對（flags 不一致會拒絕比）")
    p.add_argument("--gate", action="store_true",
                   help="gate 模式：依比對結果回出口碼（0=PASS、1=退步、2=不可比），給 CI / pre-merge 用")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="gate 容忍的通過率降幅（0.05=允許排0.05）；預設 0（最嚴）")
    sys.exit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
