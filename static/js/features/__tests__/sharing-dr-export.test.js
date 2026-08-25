// 複製／匯出：DR / LR 報告必須進得了匯出內容（node --test 直跑）。
//
// 守的東西：DR / LR **沒有「綜合摘要」這個東西**。一般搜尋的摘要在 `data.answer`，
// DR / LR 的報告本文在 `researchReport.report`（deep-research.js:1751 push 的形狀）。
// 三個匯出 builder 曾一律只讀 data.answer → DR 複製出來只剩一個空的【讀豹分析摘要】
// 標頭（CEO 回報現象）。
//
// sharing.js 的 import 鏈在載入時就摸 localStorage/document，故先 stub 再動態 import。
import { test } from 'node:test';
import assert from 'node:assert';

const store = new Map();
globalThis.localStorage = {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k),
    clear: () => store.clear(),
};
globalThis.sessionStorage = globalThis.localStorage;
globalThis.document = {
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => ({ style: {}, classList: { add() {}, remove() {} }, appendChild() {}, setAttribute() {} }),
    addEventListener: () => {},
    body: { appendChild() {}, removeChild() {} },
};
globalThis.window = globalThis;
globalThis.navigator = { clipboard: { writeText: async () => {} } };

const sharing = await import('../sharing.js');
const { setSessionHistory } = await import('../sessions-list.js');

// deep-research.js:1751 push 進 sessionHistory 的形狀（只取匯出會讀到的欄位）
function drSession({ report = 'DR 報告本文 [1] 第二句 [2]。', sources = [], live = false, data = {} } = {}) {
    return {
        query: 'TVBS 的 AI WIZE 系統是什麼？',
        data,
        timestamp: 1,
        isDeepResearch: true,
        isLiveResearch: live,
        researchReport: { report, sources, query: 'TVBS 的 AI WIZE 系統是什麼？', timestamp: 1 },
    };
}

function searchSession(answer = '一般搜尋的綜合摘要。') {
    return { query: 'q', data: { answer, content: [] }, timestamp: 1 };
}

test('DR：報告本文必須出現在純文字複製結果裡', () => {
    setSessionHistory([drSession()]);
    const out = sharing.formatPlainText();
    assert.ok(out.includes('DR 報告本文'), '報告本文沒被複製進去');
    assert.ok(out.includes('〈深度研究報告〉'), '應標示這段是 DR 報告，不是綜合摘要');
});

test('DR：不可再出現「只有空標頭」的結果', () => {
    setSessionHistory([drSession()]);
    const out = sharing.formatPlainText();
    // 回歸鎖：標頭後面必須有實質內容，不能標頭直接接下一段/結尾
    const idx = out.indexOf('【讀豹分析摘要】');
    assert.ok(idx >= 0);
    const after = out.slice(idx + '【讀豹分析摘要】'.length).replace(/[\s〈〉]/g, '');
    assert.ok(after.length > 20, '【讀豹分析摘要】後面是空的');
});

test('完全沒有可匯出的分析內容時，不印空標頭', () => {
    setSessionHistory([{ query: 'q', data: {}, timestamp: 1 }]);
    const out = sharing.formatPlainText();
    assert.ok(!out.includes('【讀豹分析摘要】'), '沒內容還印標頭');
});

test('一般搜尋：既有行為不回歸（摘要照舊、不加研究報告標籤）', () => {
    setSessionHistory([searchSession()]);
    const out = sharing.formatPlainText();
    assert.ok(out.includes('一般搜尋的綜合摘要'));
    assert.ok(!out.includes('〈深度研究報告〉'), '一般搜尋不該被標成研究報告');
});

test('LR 走同一條路徑，標籤為「即時研究報告」', () => {
    setSessionHistory([drSession({ report: 'LR 章節內文。', live: true })]);
    const out = sharing.formatPlainText();
    assert.ok(out.includes('LR 章節內文'));
    assert.ok(out.includes('〈即時研究報告〉'));
});

test('DR + 一般搜尋混在同一 session 清單時，兩者都要在', () => {
    setSessionHistory([searchSession('搜尋摘要A'), drSession({ report: 'DR 本文B' })]);
    const out = sharing.formatPlainText();
    assert.ok(out.includes('搜尋摘要A'));
    assert.ok(out.includes('DR 本文B'));
});

test('引用來源：編號自 1 起、濾掉空值、URN 轉可讀標示', () => {
    setSessionHistory([drSession({
        sources: ['https://a.example/1', '', 'urn:llm:knowledge:x', 'private://doc'],
    })]);
    const srcs = sharing.getResearchSources();
    assert.deepStrictEqual(srcs.map(s => s.index), [1, 3, 4], '編號必須對齊報告內文的 [N]');
    const out = sharing.formatPlainText();
    assert.ok(out.includes('[1] https://a.example/1'));
    assert.ok(out.includes('讀豹背景知識（無外部連結）'));
    assert.ok(out.includes('私人文件（無外部連結）'));
});

test('無來源時不印參考資料段（一般搜尋不受影響）', () => {
    setSessionHistory([searchSession()]);
    const out = sharing.formatPlainText();
    assert.ok(!out.includes('參考資料來源'));
});

test('AI chatbot 格式：報告本文與來源清單都在', () => {
    setSessionHistory([drSession({ sources: ['https://a.example/1'] })]);
    const out = sharing.formatForAIChatbot();
    assert.ok(out.includes('DR 報告本文'));
    assert.ok(out.includes('[1] https://a.example/1'));
});

test('NotebookLM 格式：報告用 markdown 標題層級', () => {
    setSessionHistory([drSession({ sources: ['https://a.example/1'] })]);
    const out = sharing.formatForNotebookLM();
    assert.ok(out.includes('## 讀豹分析摘要'));
    assert.ok(out.includes('### 深度研究報告'));
    assert.ok(out.includes('## 參考資料來源'));
    assert.ok(out.includes('1. https://a.example/1'));
});

test('getSessionAnalysisBlock：純函式行為', () => {
    assert.strictEqual(sharing.getSessionAnalysisBlock(null), null);
    assert.strictEqual(sharing.getSessionAnalysisBlock({ data: {} }), null);
    assert.strictEqual(sharing.getSessionAnalysisBlock(searchSession('X')).label, '');
    assert.strictEqual(sharing.getSessionAnalysisBlock(drSession()).label, '深度研究報告');
});
