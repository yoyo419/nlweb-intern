"""
Data schemas for the critic auto-eval harness.

Two kinds of objects:
- EvalCase: one exam question (query + sources + draft), optionally with
  human-labeled ground-truth defects (used for the Q3 calibration set).
- JudgeVerdict: what one adversarial judge returns for one case.

Pydantic v2 (matches reasoning/schemas.py conventions).
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


# A defect the draft actually contains. "none" is used for clean cases so a
# label list is never silently empty-vs-missing.
DefectType = Literal["fabrication", "fake_citation", "logic_jump", "none"]

# Each judge owns exactly one dimension.
JudgeDimension = Literal["grounding", "fabrication", "logic"]


class Source(BaseModel):
    """One retrieved source the draft is allowed to rely on."""

    id: int = Field(..., description="Citation id, referenced in the draft as [id]")
    text: str = Field(..., description="Source content the draft may cite")
    # Optional prod-parity fields (boss 2026-08-12 feedback #2): let real re-run
    # fixtures carry the same 網站/標題/日期 that prod's _format_context_shared renders
    # into the critic context. Absent (thin synthetic case) → terse「[id] text」;
    # present → prod-style「[id] site - title (date)」. Additive: id+text still valid.
    site: str = Field("", description="Source site/publisher (prod: item['site'])")
    title: str = Field("", description="Source title (prod: item['title'] or ['name'])")
    date_published: str = Field("", description="ISO date; rendered as (YYYY-MM-DD)")


class DefectLabel(BaseModel):
    """Human-labeled ground truth about a defect in the draft (Q3 gold set)."""

    type: DefectType
    quote: str = Field("", description="The offending sentence in the draft ('' if type=none)")
    reason: str = Field("", description="Why it is a defect")
    source_hint: Optional[str] = Field(
        None, description="Which source it should/should not map to, if relevant"
    )


class EvalCase(BaseModel):
    """One exam question fed to the critic and the judges."""

    id: str
    query: str = Field(..., description="The user's original question")
    sources: List[Source] = Field(..., description="Sources available to the draft")
    draft: str = Field(..., description="The Analyst draft under review (with [id] citations)")

    # Q3 gold set only. Empty list == a case with no defects still needs one
    # DefectLabel(type="none") so 'clean' is explicit, never assumed.
    human_labels: List[DefectLabel] = Field(default_factory=list)

    # What a *good* critic should output for this case. Used to score the critic
    # itself once judges are calibrated.
    expected_critic_verdict: Literal["PASS", "WARN", "REJECT"] = "PASS"

    notes: str = ""


class JudgeIssue(BaseModel):
    """One concrete problem a judge found, with evidence (adversarial output)."""

    quote: str = Field(..., description="The offending sentence copied from the draft")
    reason: str = Field(..., description="Why this is a problem in this judge's dimension")
    evidence_check: str = Field(
        ..., description="What was checked against the sources (e.g. 'source 2 says X, draft says Y')"
    )


class JudgeVerdict(BaseModel):
    """One judge's verdict on one case."""

    dimension: JudgeDimension
    found_issue: bool = Field(..., description="True if this judge found at least one real problem")
    issues: List[JudgeIssue] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
