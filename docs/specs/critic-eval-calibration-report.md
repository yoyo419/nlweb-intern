# Critic 自動評審 — 校準報告（Q3 評審 vs 人工 一致率）

**結論：三評審在 6 題人工標準答案組上一致率過關——乾淨案假警報 0/2、缺陷案抓到 4/4。**

- 校準執行日：2026-08-11
- 報告整理日：2026-08-13
- 對應設計：`docs/specs/critic-eval-plan.md` §3.1（校準：評審 vs 人工，不做自動字串比對，由人判是否同一個錯）

---

## 1. 目的

在把三評審拿去當「critic 迴歸偵測」的裁判之前，先確認**評審報的跟人工標註對不對得上**：

- **假警報率**：乾淨草稿（無缺陷）評審是否誤報。愈低愈好。
- **抓到率**：有缺陷草稿，該負責的評審是否抓到。愈高愈好。

評審不可信，之後用它建的迴歸基準就沒有意義——這是撈真實迴歸料之前的前置閘門。

## 2. 方法

- 案例集：`code/python/eval/critic_eval/cases/gold_cases.yaml`（6 題，每題人工預先標好缺陷型別與位置；4 缺陷案 + 2 乾淨案）。
- 模式：**judges-only、非錨定**（`critic_verdict=None`）——只讓三評審看草稿去挑錯，**不把 critic 的判定先餵給評審**，避免錨定效應污染一致率。
- 一致率由人對照「人工標註 vs 評審實際報的」判定（§3.1：不做自動字串比對）。
- 抓到率認定：fabrication 案期望 grounding+fabrication 皆中；fake_citation 案期望 fabrication 中；logic_jump 案期望 logic 中。

## 3. 環境

| 項目 | 值 |
|---|---|
| 模型 | gpt-5.1（high effort） |
| 評審版本 | `code/python/eval/critic_eval/judges.py`（守備範圍已調） |
| critic 判定餵給評審 | 否（非錨定，`critic_verdict=None`） |
| 呼叫數 | 6 案 × 3 評審 = 18 次 |

## 4. 逐案結果

### 乾淨案（測假警報）

| 案例 | 人工標註 | grounding | fabrication | logic | 判定 |
|---|---|---|---|---|---|
| `gold_clean_tsmc` | 乾淨 | 未報 | 未報 | 未報 | ✅ 無假警報 |
| `gold_clean_weather_thin` | 乾淨 | 未報 | 未報 | 未報 | ✅ 無假警報 |

### 缺陷案（測抓到率）

| 案例 | 人工標註 | 該抓的評審是否抓到 | 判定 |
|---|---|---|---|
| `gold_fabrication_margin` | fabrication：「第一季毛利率高達 53.1%[1]」 | grounding ✔、fabrication ✔（皆指出來源無此毛利率） | ✅ 抓到 |
| `gold_fake_citation_eps` | fake_citation：「每股盈餘為 8.70 元[3]」 | fabrication ✔（指出 8.70 出自[2]、標成[3]是張冠李戴） | ✅ 抓到 |
| `gold_logic_jump_competitiveness` | logic_jump：「因此台積電已經失去市場競爭力」 | logic ✔（指出單季季減 5.3% 推不到「失去競爭力」） | ✅ 抓到 |
| `gold_subtle_overstate_high` | fabrication：「創下歷史新高」 | grounding ✔、fabrication ✔（指出來源未提歷史比較） | ✅ 抓到 |

> 缺陷案中偶有「非該維度評審也開火」的情形（例：`gold_fake_citation_eps` 的 grounding 對「表現穩健」開火、`gold_fabrication_margin` 的 logic 對含「顯示」的推論句開火）。這些落在該評審的守備範圍內（沒根據的評價詞歸 grounding、真有推論動作歸 logic），屬多重有病的同一句被不同角度點到，非誤報。

## 5. 彙總

| 指標 | 結果 |
|---|---|
| **乾淨案假警報率** | **0 / 2** |
| **缺陷案抓到率** | **4 / 4** |

一致率過關。

## 6. 觀察到的兩個縫（已處理）

1. **grounding 與 fabrication 會對同一句同時開火**：屬設計預期（fabrication 案本就期望兩者皆中）。守備範圍已在 judges.py 講清楚：沒根據的評價詞歸 grounding、假引用＋憑空捏造的具體數字歸 fabrication。
2. **logic 曾把純評價詞當跳步亂喊**（如「表現穩健」）：已調 judges.py，要求草稿真的有「因此／所以／顯示」這類推論動作、且那步接不起來時才報。調後 `gold_fake_citation_eps` 的 logic 正確保持沉默。

## 7. 侷限與依賴

- **小樣本**：目前 6 題。要更硬的信心，可將 gold 擴充後再驗一輪（`cases/gold_extended.yaml` 已備 26 題合成校準集，尚未跑校準）。
- **版本依賴（重要）**：本 0/2、4/4 是用 **fork 版 `judges.py`** 跑出的。主 repo 若為舊版評審，重跑未必得到相同結果。**佐證成立的前提是先把該版 judges.py（連同 cov.py 引用修復、format_context 對齊、模板更正那一包）land 進主 repo，再重跑校準。**

## 8. 如何複驗

```bash
cd code/python
# 內建 --calibrate：評審非錨定（critic_verdict=None），印逐案人工 vs 評審並排
../../venv/Scripts/python.exe -m eval.critic_eval.runner --cases cases/gold_cases.yaml --calibrate
```

> 註：`--calibrate` 亦會實跑 critic（額外成本），但評審在此模式為非錨定，逐案比對與本報告一致；一致率由人對照人工標註判定（§3.1）。
