"""來源內文相對時間的「發布日期錨定」。

修的症狀：新聞內文寫的「今年」指的是**該報導發布那一年**（例：2025 年的報導寫
「今年流感疫苗 10 月開打」＝2025 年），但答案輸出卻寫成當前年份（2026）。

根因：送進 LLM 的 prompt 兩端同時存在
  (a) 「今天日期是 YYYY-MM-DD」的全域時間 header（各 prompt 都注入），
  (b) 來源內文的「今年 / 去年 / 明年」相對詞，且多數渲染點根本沒把該來源的
      發布日期一起送進去（LR analyst 的搜尋結果、LR writer evidence 視圖皆無日期）。
LLM 於是把 (b) 的相對詞掛到 (a) 的今天 → 年份整個偏移。

修法紀律（不靠 prompt 自己算年份）：年份換算由 code 算好，就地標註在相對詞後面，
LLM 只要照抄絕對年份，不做算術；prompt 規則只當第二道防線。標註只在
「來源發布年 ≠ 當前年」時發生 —— 那正是會出錯的情況，同年來源不加噪音。
"""

from datetime import datetime
from typing import Optional
import re

# 相對年份詞 → 相對於發布年的位移。只收「年」層級的詞：月/週層級（本月、上週）
# 的錨定由 prompt 規則負責，機械標註不介入（歧義高、誤標成本大於收益）。
RELATIVE_YEAR_OFFSETS = {
    "今年": 0,
    "本年": 0,
    "去年": -1,
    "前年": -2,
    "明年": 1,
    "後年": 2,
}

# 負向 lookahead：
#   （ ( → 已被標註過（本函式可能對同一段文字跑兩次，例如 snippet 先標註再進另一個視圖）
#   度   → 「今年度」是會計年度用語，插進去會變「今年（2025年）度」
_RELATIVE_YEAR_PATTERN = re.compile(
    "(" + "|".join(RELATIVE_YEAR_OFFSETS) + ")(?![（(度])"
)

# Prompt 端第二道防線（機械標註是第一道）。供各 Python 端 prompt 組裝點引用，
# 與 config/prompts.xml 內的【時間換算規則】同義，改一邊要同步另一邊。
TEMPORAL_ANCHOR_RULE = (
    "【時間換算規則】來源內文的相對時間（今年／去年／明年／本月／上週／日前等）"
    "一律以**該來源自己的發布日期**為基準換算，不可用今天的日期換算。"
    "例：2025-09-12 發布的報導寫「今年」＝2025 年。"
    "輸出時請直接寫換算後的絕對年份（如「2025 年」），"
    "不要照抄來源摘要中系統加註的括號（如「今年（2025年）」）。"
)


def published_year(date_value) -> Optional[int]:
    """從各種發布日期形狀取出年份；取不到回 None（不猜、不 fallback 到今年）。

    接受：datetime、'2025-09-12'、'2025-09-12T08:00:00Z'、'2025/09/12'、'2025'。
    """
    if date_value is None:
        return None
    if isinstance(date_value, datetime):
        return date_value.year
    text = str(date_value).strip()
    if not text or text == "Unknown":
        return None
    match = re.match(r"(\d{4})", text)
    if not match:
        return None
    year = int(match.group(1))
    # 明顯不是發布年（爬蟲 metadata 髒資料）就當取不到，寧可不標也不標錯
    if year < 1900 or year > 2200:
        return None
    return year


def annotate_relative_years(text: str, date_value, *, now_year: Optional[int] = None) -> str:
    """把來源內文的相對年份詞就地標成絕對年份：「今年」→「今年（2025年）」。

    只在「發布年已知且 ≠ 當前年」時標註；其餘情況原文回傳（LLM 用今天換算剛好對，
    不需要標註，也避免對 90% 的新鮮來源加噪音）。
    """
    if not text:
        return text
    if not isinstance(text, str):
        # 呼叫端負責先轉字串（例：trim_json 回的是 dict，prompt 端最後也是 str(dict)）。
        # 這裡不硬轉、也不炸——真正鎖「有沒有轉」的是各出口的接線測試。
        return text
    pub_year = published_year(date_value)
    if pub_year is None:
        return text
    if now_year is None:
        now_year = datetime.now().year
    if pub_year == now_year:
        return text

    def _sub(match: "re.Match") -> str:
        term = match.group(1)
        return f"{term}（{pub_year + RELATIVE_YEAR_OFFSETS[term]}年）"

    return _RELATIVE_YEAR_PATTERN.sub(_sub, text)


def anchor_note(
    date_value, *, include_date: bool = False, now_year: Optional[int] = None
) -> str:
    """來源標頭用的錨定註記；發布年未知或同當前年 → 回空字串（呼叫端直接串接）。

    預設 '（此文中的「今年」＝2025 年）'——多數渲染點的標頭本來就印了發布日期，
    再印一次只是雜訊。標頭沒有日期的地方（如 critic reference sheet）傳
    include_date=True 取 '（發布於 2025-09-12；此文中的「今年」＝2025 年）'。
    """
    pub_year = published_year(date_value)
    if pub_year is None:
        return ""
    if now_year is None:
        now_year = datetime.now().year
    if pub_year == now_year:
        return ""
    if include_date:
        date_str = str(date_value).strip().split("T")[0]
        return f"（發布於 {date_str}；此文中的「今年」＝{pub_year} 年）"
    return f"（此文中的「今年」＝{pub_year} 年）"
