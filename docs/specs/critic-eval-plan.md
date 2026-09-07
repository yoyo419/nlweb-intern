# Critic 自動評審機制 — 計畫書（critic-eval）

> **一句話**：做一支「可重複跑的自動評分程式」，每次改動 critic 後，自動判斷 critic 實際跑出來的判斷品質**有沒有退步**。
>
> **狀態**：設計 + 考題/評審初版已建（本文件末〈已產出檔案〉）。runner（跑分主程式）為下一步。
> **注意**：這**不是**靜態檢查指令稿（那是另一輪「prompt 稽核」，用 `docs/prompt-check`〔待建〕那張檢查表）。本案是**動態**——讓 critic 真的跑，評它跑出來的結果。

---

## 0. 我要做出什麼

一個測試工具，按一下就會：

1. 拿一組**固定考題**（真實跑過的 session 撈出來的：草稿 + 來源）。
2. 讓 **critic 對這些草稿真的跑一遍**，拿到它的判定（過/警告/退回 + 問題清單）。
3. 叫 **3 個 AI 評審**從不同角度去挑「critic 漏抓的問題」。
4. 算出一個**分數**（critic 抓到幾成該抓的），**存起來當基準**。

之後有人改了 critic，重跑這支 → 分數掉了 → 它自動報「品質退步」。

**自我檢查**：改了 critic 指令稿後，系統能不能自動告訴我們品質變差？→ **能**（固定輸入 + 已對齊人工的評審 + 基準分數，重跑即比對）。

---

## 1. critic 模組在哪、輸入輸出長什麼樣（Q1，已查證 code）

- **程式**：`code/python/reasoning/agents/critic.py`
- **指令稿**：`code/python/reasoning/prompts/critic.py`
- **輸出格式**：`reasoning/schemas.py`（`CriticReviewOutput`）、`reasoning/schemas_live.py`（`CriticSectionReview`）

兩個主要入口（**本案先只鎖第一個**，最單純）：

| 入口 | 輸入 | 輸出 |
|---|---|---|
| `review()`（critic.py:90，DR 主線，審整篇草稿） | 一篇 Analyst 草稿全文 + 使用者問題 + 可看的來源資料 | `CriticReviewOutput`：`status`(PASS/WARN/REJECT) + `critique` + `suggestions` + `source_issues` + `logical_gaps` |
| `review_section_publish_gate()`（critic.py:475，LR 逐段守門） | 單一段落 + 該章來源全文 | `CriticSectionReview`：`verdict`(PASS/WARN/REJECT) + `claim_issues` + `overall_explanation` |

**本質**：critic 吃「一篇草稿 + 它的來源」，吐「一個放行/退回判定 + 抓到的問題」。
→ 我們的**考題** = (草稿, 來源) 成對；critic 的**結果** = 那個判定 + 問題清單；我們評的是**這個判定準不準**。

### 1.1 `review()` 完整簽名（6 參數，runner 要照填）

`review(draft, query, mode, analyst_output=None, formatted_context="", enable_live_research=False)`

| 參數 | runner 怎麼處理 |
|---|---|
| `draft` / `query` / `formatted_context` | 從 case 直接餵（來源組成 formatted_context） |
| `mode` | 必填但**值已被忽略**（critic.py:105，2026-04 起）→ 固定傳一個字串即可 |
| `analyst_output` | 選填，帶 Analyst 完整輸出（argument_graph 等）。**我們 fixture 不存這個，critic 走「沒有它」的路徑**（見〈已知限制〉） |
| `enable_live_research` | 固定 False（本案只鎖 DR 主線） |

### 1.2 評審發現 → critic 哪個欄位對應（runner 算分規則，明寫）

評審報了問題後，去 critic 對應欄位找它有沒有抓到：

| 評審角度 | 去 critic 的哪個欄位對 |
|---|---|
| 有沒有根據 / 編造 / 假引用 | `source_issues` |
| 邏輯跳步 | `logical_gaps` |
| 總判定（該不該退） | `status`（PASS/WARN/REJECT） |
| `mode_compliance` | **三個評審都不管 → 明確排除，不算分** |

---

## 2. 三個評審各自怎麼問（Q2，預設「挑毛病」不是打分數）

每個評審看草稿 + 來源（+ critic 的判定），各守一個角度，**任務是找出 critic 漏掉的問題並舉證**：

| 評審 | 角度 | 它問什麼 |
|---|---|---|
| 評審 1 | **有沒有根據**（grounding） | critic 放行的草稿，每個具體說法在來源裡真的找得到嗎？ |
| 評審 2 | **有沒有編造/假引用**（fabrication） | 草稿引用的出處真的存在、對得上嗎？有沒有無中生有的數字/事件？ |
| 評審 3 | **邏輯有沒有跳步**（logic） | 結論跟前提接得起來嗎？有沒有中間跳一步？ |

- 問法都設成**找碴**：「找出問題並舉出具體證據（哪一句、對照哪條來源）」，找不到才算 critic 這關過——不給 1~5 分。
- 三個角度分開問，避免同一盲點。
- 實作：`eval/critic_eval/judges.py`。

---

## 3. 人工標準答案的小案例集 + 驗證評審跟人一致（Q3）

- **建**：10~20 個 (草稿, 來源)，人工先標好每篇真正的缺陷（哪句沒根據/哪個引用是假的/哪裡跳步）與該有的判定。**校準期一題只植入 0 或 1 種缺陷**，讓答案明確好對。
  - 預留擴充：真實草稿常常好幾種錯混在一起，之後擴充案例集時加幾題「兩種以上缺陷混合」，確認評審在混合狀況也抓得全。
- **這組是「刻意構造」的校準集，不算造假輸入**——它的用途就是拿已知答案去驗評審。
- **為什麼**：評審沒先對齊人，後面所有數字都不能信——信任地基。
- 實作：`eval/critic_eval/cases/gold_cases.yaml`（初版 6 案，含正例/負例）。

### 3.1 「一致」怎麼判——先把判法定死（否則一致率算不出來）

評審摘的句子跟人標的**不會逐字相同**（多半句、少半句都會），所以：

- **校準期題目少 → 「摘的是不是同一個錯」由人工肉眼判定就好。不寫程式自動比對字串**（那是另一個難題，不值得）。
- 每題 × 每評審記三種結果：
  - **抓到**：評審找到人標的那個錯
  - **漏抓**：人標了、評審沒找到
  - **誤抓**：評審報了人沒標的問題
- 回報兩個數字：
  - **抓到率 = 抓到 ÷（抓到＋漏抓）**
  - **誤抓另外逐條列出來看**（有些「誤抓」其實是人工標準答案漏標，這也是校準的一部分）
- **乾淨題**最單純：評審回 `found_issue=false` 就是對。

→ runner 只負責把三種結果的原始素材（評審報了什麼、人標了什麼）並排印出來給人看；**判定「是否同一個錯」留給人**，不自動化。

---

## 4. 固定測試輸入的案例集（Q4，用真實輸入，不造假）

- **來源**：從實際跑過的 DR/LR session 撈**真的 Analyst 草稿 + 對應來源**，不自己編。
- **固定存檔**：存成 fixture，每次用同一組，才能跨版本比。先 20~40 個，涵蓋資料充足/稀薄/時間敏感。
- **和 Q3 關係**：Q3（有人工標）校準評審；Q4（量大）當回歸基準。
- **省錢**：草稿撈存下來，跑評測時只讓 critic 對固定草稿重跑，不用重跑整條 pipeline。
- 樣板：`eval/critic_eval/cases/fixed_cases.template.yaml`（**內容待撈真實 session 填入，勿造假**）。

---

## 5. 怎麼判定「過／不過」（Q5）

兩層：

- **單案例層**（三評審怎麼合）：**混合制**——硬傷（編造、假引用）**一票否決**（一個評審舉出具體證據即算 critic 沒守住）；較主觀的（邏輯跳步）**多數決**。
- **整組層**（回歸判定）：看整組「抓到率／通過率」跟基準比有沒有掉，掉了報警。看的是分數變化，不是單題。

先用最嚴的跑，依假警報再放鬆——判定規則是**可調參數**，先立框架。

### 5.1 已落地的判定（`runner.evaluate_gate`）

單案層在 `score_case`：某維度的評審舉證指控、而 critic 對應欄位（`source_issues` /
`logical_gaps`）是空的 → 該維度算漏；硬傷維度（grounding / fabrication）漏 = 一票否決。
**只有非 PASS 的狀態不給分**——因 A 理由 REJECT 的 critic 不能被算成也抓到了它從沒提過的 B。

整組層在 `evaluate_gate`，回三種判定，並對應出口碼讓它擋得住下游：

| 判定 | 出口碼 | 何時 |
|---|---|---|
| `PASS` | 0 | 可比，且通過率沒掉、也沒有案例從 OK 變 MISS |
| `FAIL` | 1 | 可比但退步：通過率跌破容忍值，**或**逐案出現 OK → MISS |
| `BLOCKED` | 2 | **不可比**：沒有基準、env/flags 不同、`context_format` 不同、考卷不同 |

兩個容易失守的點，各有機械防線（`test_runner_logic.py::TestGateDecision`）：

1. **不可比不得靜靜放行**。不可比就是不可比，既不能算通過也不能算退步，得停下來由人處理。
   餵給 critic 的 context 形狀（`format_context`）改版也算不可比——所以版本字串
   `CONTEXT_FORMAT_VERSION` 會戳進 env，由 gate 自動擋，不靠人記得。
2. **總分持平不代表沒退步**。一案退、一案進會讓通過率持平，所以 gate 一律逐案比對
   `critic_ok`，出現 OK → MISS 就 FAIL。

容忍值 `--tolerance` 預設 `0.0`（最嚴）；LLM 有非決定性，要放鬆必須是明講的決定。

跑法：

```bash
cd code/python
# 存基準
python -m eval.critic_eval.runner --cases cases/gold_cases.yaml --save baseline.json
# 當關卡跑（出口碼 0 / 1 / 2）
python -m eval.critic_eval.runner --cases cases/gold_cases.yaml --compare baseline.json --gate
```

---

## 6. 跑一次多少錢、什麼時機跑（Q6）

- **成本**：每案例 = critic 真跑 1 次 + 3 評審各 1 次 ≈ 4 次高階 LLM。粗估每案例 ~$0.2–0.4，40 案例一輪 ~$8–16（**待實測校正，先當量級**）。
- **時機**：**不是每次 commit 跑**。建議：改 critic 指令稿／相關程式／換 model 時手動觸發，或 push prod 前跑一次當關卡。每次存基準分數比對。

---

## 6.5 已知限制 + runner 必記的環境狀態（最重要）

被測的 critic **必須跟上場的 critic 是同一個**（跟評審要盲判同一個道理）。所以：

- **功能開關會改變 critic 行為**：`review()` 內部讀系統設定檔的 `structured_critique`、`cov_lite_enabled`（critic.py:116-117），開關不同 → 連 response schema 都換（critic.py:178-188）→ critic 行為就不同。
  - → **runner 每次跑都要把當時的開關值記進基準結果**；比對兩次分數前**先確認開關一致**。否則「分數掉了」可能只是開關不同，不是 critic 退步——基準就廢了。
- **analyst_output 差異**：正式環境 critic 會拿到 Analyst 完整輸出（argument_graph 等），我們 fixture 不存這個，critic 走「沒有它」的路徑。→ **eval 跑法跟正式環境在這點上有差異**，分數是「無 analyst_output 條件下」的相對基準，跨版本仍可比（只要都無），但別跟正式線上表現直接畫等號。

---

## 7. critic 成功後怎麼推廣（Q7）

「固定真實輸入 → 讓目標模組真跑 → 多評審找碴 → 先對齊人工 gold → 跟基準比對抓退步」是**通用骨架**，換三樣即可套：目標模組、評審角度、人工答案組。下一個建議推 Analyst（草稿源頭）或 Writer（最終輸出）。做法：把 harness 抽成共用框架，各模組只提供自己的 case set + 評審角度 + gold。

---

## 已產出檔案

```
code/python/eval/critic_eval/
├── __init__.py
├── schema.py                       ← 考題 / 評審輸出的資料格式（pydantic）
├── judges.py                       ← 三個評審的問法（繁中 prompt builder）
└── cases/
    ├── gold_cases.yaml             ← Q3 人工標準答案組（校準評審用，6 案初版）
    └── fixed_cases.template.yaml   ← Q4 固定回歸組樣板（待撈真實 session 填）
```

## 下一步（未做，待老闆點頭燒錢前先給 plan）

1. 寫 runner（跑分主程式）：讀 cases → 呼叫 `critic.review()` → 呼叫三評審 → 依 §5 判定 → 出分數 + 存基準。
2. 用 `gold_cases.yaml` 跑一輪，量評審 vs 人工一致率（§3），不一致先修評審問法。
3. 撈真實 session 填 `fixed_cases`（§4），跑一次量成本（§6）。
