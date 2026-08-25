// KG SVG 匯出：XML 合法性防線的純函式測試（node --test 直跑，不需瀏覽器）。
//
// 守的東西：`.svg` 是嚴格 XML，夾帶一個 XML 非法字元 → 整份解析失敗 → 瀏覽器
// 改顯示 XML 錯誤頁並印出原始碼，使用者回報成「下載下來是一大串標籤文字」。
// node/edge label 來自 LLM 抽取 + 爬取內文，不保證乾淨，故 buildStandaloneSVG
// 必須在輸出前剔除非法字元。
//
// 紀律：本檔一律用 \uXXXX 跳脫寫控制字元，不可貼實體字元（會讓原始碼變 binary、
// diff / grep 全爛）。
import { test } from 'node:test';
import assert from 'node:assert';
import { stripXMLIllegalChars, buildStandaloneSVG, buildExportFilename, EXPORT_CSS } from '../kg-svg-export.js';

// XML 1.0 允許字元集之外者（測試端獨立寫一份，不 import 實作的 regex——
// 抄實作等於拿同一個錯誤驗自己）。
const ILLEGAL = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\uD800-\uDFFF\uFFFE\uFFFF]/;
function hasIllegal(s) {
    // 成對 surrogate 合法，先移除成對者再檢查落單者與控制碼
    return ILLEGAL.test(s.replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g, ''));
}

test('C0 控制碼被剔除', () => {
    assert.strictEqual(stripXMLIllegalChars('\u0000\u0001\u0008\u000B\u000C\u000E\u001F'), '');
    assert.strictEqual(stripXMLIllegalChars('台\u0007積\u001B電'), '台積電');
});

test('\\t \\n \\r 是 XML 合法字元，不可被剔除', () => {
    assert.strictEqual(stripXMLIllegalChars('a\tb\nc\rd'), 'a\tb\nc\rd');
});

test('U+FFFE / U+FFFF 被剔除', () => {
    assert.strictEqual(stripXMLIllegalChars('a\uFFFEb\uFFFFc'), 'abc');
});

test('中文與成對 surrogate（emoji）完整保留', () => {
    const s = '台積電 2330 半導體 \u{1F4C8} ok';
    assert.strictEqual(stripXMLIllegalChars(s), s);
});

test('落單 surrogate 被剔除（高位、低位皆是）', () => {
    assert.strictEqual(stripXMLIllegalChars('a\uD800b'), 'ab');            // 孤兒 high
    assert.strictEqual(stripXMLIllegalChars('a\uDC00b'), 'ab');            // 孤兒 low
    assert.strictEqual(stripXMLIllegalChars('a\uD800\u{1F4C8}'), 'a\u{1F4C8}'); // 孤兒後接合法 pair
});

test('null / undefined 不炸，回空字串', () => {
    assert.strictEqual(stripXMLIllegalChars(null), '');
    assert.strictEqual(stripXMLIllegalChars(undefined), '');
});

test('buildStandaloneSVG：髒 label 進去，出來的整份字串無 XML 非法字元', () => {
    const dirty = '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400">'
        + '<text class="kg-node-label">台\u0007積電</text>'
        + '<text class="kg-link-label">關\uD800聯</text></svg>';
    const out = buildStandaloneSVG(dirty);
    assert.ok(!hasIllegal(out), '匯出字串仍含 XML 非法字元');
    assert.ok(out.includes('台積電'), '中文內容不可被誤刪');
    assert.ok(out.includes('關聯'), '中文內容不可被誤刪');
});

test('buildStandaloneSVG：既有行為不回歸（XML 宣告 / xmlns 不重複 / style 注入位置）', () => {
    const raw = '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400"><g/></svg>';
    const out = buildStandaloneSVG(raw);
    assert.ok(out.startsWith('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'));
    assert.strictEqual(out.split('xmlns="http://www.w3.org/2000/svg"').length - 1, 1, 'xmlns 只能有一份');
    assert.ok(out.includes('<style>' + EXPORT_CSS + '</style>'), 'EXPORT_CSS 必須被注入');
    assert.ok(/<svg[^>]*><style>/.test(out), 'style 必須緊接 root tag，不可掉進子元素');
});

test('buildStandaloneSVG：缺 xmlns 時補上', () => {
    const out = buildStandaloneSVG('<svg width="600"><g/></svg>');
    assert.ok(out.includes('<svg xmlns="http://www.w3.org/2000/svg" width="600">'));
});

test('buildStandaloneSVG：CSS 內的 $& 不被 replace 樣板展開', () => {
    // 字串形式 replace 會把 `$&` 展開成整個 match（= 整個 <svg ...> tag）→ 匯出爛掉。
    const out = buildStandaloneSVG('<svg width="600"><g/></svg>', { css: 'text{content:"$&";}' });
    assert.ok(out.includes('<style>text{content:"$&";}</style>'), '$& 必須原樣輸出');
});

test('buildExportFilename：既有行為不回歸', () => {
    assert.strictEqual(buildExportFilename(''), 'kg-knowledge-graph.svg');
    assert.strictEqual(buildExportFilename('台積電'), 'kg-台積電.svg');
    assert.strictEqual(buildExportFilename('A/B:C*D'), 'kg-ABCD.svg');
    assert.strictEqual(buildExportFilename(' 台 積 電 '), 'kg-台-積-電.svg');
});
