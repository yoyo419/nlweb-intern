// static/js/features/kg-svg-export.js
//
// SVG「所見即所得」匯出（CEO 拍板 #3）：序列化畫面當下的焦點子圖 SVG DOM，
// lazy（點按鈕才做），前端零後端成本。純字串處理部分（buildStandaloneSVG /
// buildExportFilename）可 node:test 直測；serializeGraphSVG / triggerSVGDownload
// 觸碰 DOM，由 Agent E2E 覆蓋。
//
// 關鍵：node/edge label 樣式在 CSS class，序列化 DOM 不自動帶入 → 把關鍵樣式
// 以 <style> 注入 standalone SVG（EXPORT_CSS）。字型不 embed，離線 fall back
// 到系統中文字型（Microsoft JhengHei 等），中文仍可見（見 kg-spec §7 匯出取捨）。

// 匯出時注入的關鍵樣式（鏡像 news-search.css 的 .kg-* SVG 規則，只取離線必需者）。
// 🔧 R3 SF-R3-5：node label 只用 `.kg-node-label`（**移除 `.kg-node text`**）。
// 原因：badge 的 `<text>` append 在 `.kg-node` group 內 → 同時匹配 `.kg-node text`。
// `.kg-node text`（class+type，specificity 0,1,1）比 `.kg-hidden-badge`（class，0,1,0）高，
// 會覆蓋 badge 的 font-size:10px/font-weight:700（強制成 11px/500）→ 匯出 SVG 裡 badge 字級/字重跑掉。
// node label 本身有 `.kg-node-label` class（renderKGGraphView L790 verify），移除 `.kg-node text`
// 後 label 仍由 `.kg-node-label` 命中不掉樣式；badge 由 `.kg-hidden-badge` 命中，兩者 specificity
// 皆 0,1,0 且 selector 互斥，不再相互覆蓋。
export const EXPORT_CSS = [
    '.kg-node-label{font-size:11px;fill:#2D3436;font-weight:500;text-anchor:middle;}',
    '.kg-link{fill:none;stroke-opacity:0.7;}',
    '.kg-link-label{font-size:10px;fill:#2D3436;text-anchor:middle;dominant-baseline:central;}',
    '.kg-hidden-badge{font-size:10px;font-weight:700;fill:#2D3436;text-anchor:middle;}',
    'text{font-family:"Noto Sans TC","Microsoft JhengHei",sans-serif;}'
].join('');

const XMLNS = 'http://www.w3.org/2000/svg';

// XML 1.0 禁止的字元（除 \t \n \r 外的 C0 控制碼、U+FFFE/U+FFFF）。
// 為什麼要擋：`.svg` 是**嚴格 XML**，只要夾帶一個非法字元，整份文件解析失敗——
// 瀏覽器不會畫圖，改顯示 XML 錯誤頁並把文件原始碼一起印出來，使用者看到的就是
// 「下載下來是一大串標籤文字」。node/edge label 來自 LLM 抽取 + 爬取內文，不保證
// 乾淨（XMLSerializer 只 escape `<`/`&`/`"`，控制碼原樣輸出），故序列化後統一剔除。
const XML_ILLEGAL_CTRL = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]/g;

// 純字串：剔除會讓 SVG 無法被 XML parser 接受的字元。
// 兩段：C0 控制碼 regex 一次掃掉；孤兒 surrogate（未成對的 U+D800-DFFF，
// 同樣是 XML 非法字元）逐字掃——不用 lookbehind，避免舊版 Safari 不支援。
export function stripXMLIllegalChars(s) {
    const out = String(s == null ? '' : s).replace(XML_ILLEGAL_CTRL, '');
    // 快路徑：中文/英數皆在 BMP 且非 surrogate 區，絕大多數匯出不需逐字掃。
    if (!/[\uD800-\uDFFF]/.test(out)) return out;
    let res = '';
    for (let i = 0; i < out.length; i++) {
        const c = out.charCodeAt(i);
        if (c >= 0xD800 && c <= 0xDBFF) {
            const next = out.charCodeAt(i + 1);
            if (next >= 0xDC00 && next <= 0xDFFF) {   // 成對（emoji 等）→ 保留
                res += out[i] + out[i + 1];
                i++;
            }
            // 落單的 high surrogate → 丟棄
        } else if (c >= 0xDC00 && c <= 0xDFFF) {
            // 落單的 low surrogate → 丟棄
        } else {
            res += out[i];
        }
    }
    return res;
}

// 純字串：SVG 內容 → standalone 可離線開啟字串。
export function buildStandaloneSVG(innerSVG, opts = {}) {
    // 先清非法字元，再做任何字串加工（清洗必須涵蓋注入前的整份內容）。
    let svg = stripXMLIllegalChars(innerSVG);
    // 確保 root svg 有 xmlns（只在缺時加）。
    if (!svg.includes(`xmlns="${XMLNS}"`)) {
        svg = svg.replace(/^<svg/, `<svg xmlns="${XMLNS}"`);
    }
    // 注入 <style>（緊接在 <svg ...> 後）。用 replacer function 而非 `$1` 樣板：
    // CSS 若含 `$&`／`$'` 等 replacement pattern 會被字串形式的 replace 展開成別的內容。
    const css = stripXMLIllegalChars(opts.css || EXPORT_CSS);
    svg = svg.replace(/(<svg[^>]*>)/, (rootTag) => `${rootTag}<style>${css}</style>`);
    return `<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n${svg}`;
}

// 純字串：焦點名稱 → 安全檔名。
export function buildExportFilename(focusName) {
    const base = (focusName || '').trim();
    if (!base) return 'kg-knowledge-graph.svg';
    // 移除檔名不安全字元（/ \ : * ? " < > |），空白轉 -。
    const safe = base.replace(/[\/\\:*?"<>|]/g, '').replace(/\s+/g, '-');
    return `kg-${safe}.svg`;
}

// DOM：序列化 graph container 內的 svg → standalone 字串。
// clone 後注入 style，不污染畫面上的 live SVG。
export function serializeGraphSVG(svgEl) {
    if (!svgEl) return null;
    const clone = svgEl.cloneNode(true);
    const raw = new XMLSerializer().serializeToString(clone);
    return buildStandaloneSVG(raw, {});
}

// DOM：把文字內容下載為檔案（沿用 live-research.js:1071-1076 blob download pattern）。
export function downloadTextAsFile(text, filename, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
}
