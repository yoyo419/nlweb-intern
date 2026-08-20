# E2E 測試文件

> **程式碼改動在 E2E 測試通過前不算完成。**
> 
> 完整 pipeline：`Unit Test → Smoke Test → Agent E2E (DevTools) → 修 bugs → 寫到本文件 → CEO 人工 E2E → Pass = 完成`
> 
> Agent 測試結果記錄在本文件最後面。人工 checklist 在各段落。
> 詳細流程規則見 `memory/delegation-patterns.md`「E2E Gate」段落。

---

# Login 系統 E2E 測試 Checklist

>  

---

# 來源發布日期錨定 E2E（fix/source-date-anchoring）

> 症狀：2025 年發布的報導內文寫「今年」，系統答案卻寫成當前年（2026）。
> 修法：`core/temporal_anchor.py` 由 code 換算年份後就地標註（今年 → 今年（2025年）），
> prompt 的【時間換算規則】只當第二道防線。詳見 commit `0f043cc`。

## Agent 驗證結果（2026-08-21）

**方法：prompt-level replay + 舊碼對照組。** 不跑完整 pipeline——完整跑會多燒
50–80 次 ranking 呼叫（每次送整篇全文），與「模型會不會把來源的今年寫錯」這個命題無關。
改為重放四個真正產文的 prompt，各打一次 gpt-5.1；對照組在修改前的 commit（`9b85089`）
另開 worktree 跑同一支腳本，因此對照的是真的舊碼，不是手工重建的行為。

素材：一篇 2025-09-12 的中央社報導，內文「今年公費流感疫苗將於10月1日開打，較去年提前兩週。
今年採購量為690萬劑，去年為650萬劑。」正解為 2025-10-01 / 2024 對照。

| 出口 | 修改前 | 修改後 |
| --- | --- | --- |
| DR analyst | 寫「2025 年 10 月 1 日」（舊版 context 標頭本來就有日期） | 寫「2025 年」 |
| 搜尋 summarize | 「今年…690萬劑」未換算，另註記來源發布於 2025-09-12 | 「2025年10月1日開打…2025年690萬劑、2024年650萬劑」 |
| DR writer | **通篇「今年／去年」無年份**（prompt 內完全沒有發布日期） | 「2025 年…較 2024 年」 |
| LR writer section | **通篇「今年／去年」無年份** | 「2025 年…2024 年」 |

LR writer 修改前的 narration 自述：「來源中**未明示年份**，只能以『今年／去年』表述，
避免擅自推斷具體西元年。」——模型自己指出拿不到年份，對應診斷的結構性缺口
（該 prompt 的 `prompt_有發布日期 = false`）。修改後同一位置變成「先將報導中的
『今年』『去年』明確換算為 2025 年與 2024 年」。

新引入的失敗模式（照抄系統標註括號「今年（2025年）」）四個出口都未發生。

**保留事項（不可當成已驗證）：**

1. 修改前沒有任何出口真的輸出「2026」字面。實際壞掉的形態是「只寫今年、不給年份」，
   2026 年的讀者會讀成 2026 —— 與回報現象一致，但本次未證明模型會主動打出 2026。
2. 卡片摘要那條（ranking，走 low 檔 gpt-4o-mini）未驗，行為可能與 gpt-5.1 不同。
3. 未跑真實資料庫的完整 pipeline（需 Docker + 有 2025 年且內文含「今年」的文章）。

## CEO 人工 E2E checklist（未完成）

前置：Docker Desktop → `bash scripts/dev-up.sh` → `cd code/python && uv run python app-aiohttp.py`

先用 SQL 挑會踩雷的素材（發布年 ≠ 當年，標註才會觸發）：

```sql
SELECT url, title, date_published,
       substring(content from greatest(position('今年' in content) - 40, 1) for 100) AS 前後文
FROM articles
WHERE date_published < '2026-01-01' AND content LIKE '%今年%'
ORDER BY date_published DESC LIMIT 10;
```

- [ ] 一般搜尋：用該文主題查詢，**卡片摘要**的年份等於來源發布年
- [ ] 一般搜尋：**綜合摘要**的年份等於來源發布年
- [ ] 三者皆 fail 條件：出現當前年、原樣抄出「今年（YYYY年）」、只寫「今年」不帶年份
- [ ] Deep Research 跑一次，報告內文年份正確
- [ ] Live Research 跑一次，章節內文年份正確
