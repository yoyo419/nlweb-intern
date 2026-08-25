// static/js/features/sharing.js
//
// D-1 Module Header — Sharing Owner (state + functions — commit 18, Phase 8)
//
//   Owned state:
//     - _shareContentOverride (string|null — explicit share content; when non-null,
//       share modal uses this instead of formatPlainText / formatForAIChatbot / etc.)
//
//   Functions migrated (8 — commit 18, Phase 8):
//     Export format builders:
//       - cleanHTMLContent (was news-search.js:5029)
//       - getTop10Articles (was 5048)
//       - formatPlainText (was 5056)
//       - formatForAIChatbot (was 5121)
//       - formatForNotebookLM (was 5184)
//       - copyAndOpen (was 5256)
//       - openFeedbackModal (was 5331)
//     Session sharing:
//       - toggleSessionSharing (was 6246)
//
//   Trigger writes (shareContentOverride):
//     - Verification-share path: button click in reasoning UI sets override to
//       formatReasoningForVerification() (news-search.js residual)
//     - Share modal close handlers clear override
//     - UserStateSync.clearUserScopedState (IIFE relocated to state-sync.js commit 11) clears on logout
//
//   External read:
//     - Share format builders use `getShareContentOverride() || formatXxx()` fallback pattern.
//
// D-3 Cross-Module Communication:
//   Static imports (commit 18):
//     - features/search.js — getConversationHistory (search history for export titles)
//     - features/chat.js — getChatHistory (free-chat for export)
//     - features/sessions-list.js — getSessionHistory (per-search results for export),
//       getSavedSessions (toggleSessionSharing target lookup)
//     - features/source-filters.js — getSourceDisplayNames (article source label mapping)
//     - utils/analytics.js — getCurrentSessionId (feedback POST body)
//
//   Window bridges accessed (KEEP-in-place owners — sweep commits 22/25):
//     - window.matchSessionId (KEEP-in-place residual per inventory)
//     - window.renderLeftSidebarSessions (sessions-list UI residual — commit 22)
//     - window.sessionManager (main.js — saveSession / setSessionVisibility)
//     - window._updateOrgSpaceBadge (sessions-list scope — commit 22; bridged for now)
//
// D-13 Compliance:
//   No top-level side effects on import. Pure declarations + exports.
//
// v4.0 Commit 6 (2026-05-24): State-only migration (shareContentOverride).
// v4.0 Commit 18 (2026-05-25, Phase 8): 8 sharing/export function bodies migrated.
//   Bridge removed: 1 — window.toggleSessionSharing (was news-search.js:6284). Re-bridge
//   kept (window.toggleSessionSharing = toggleSessionSharing) for sidebar inline-onclick
//   compatibility until commit 25 sweep. openFeedbackModal stays accessible via ES import
//   from chat.js feedback handler residual (event delegation in news-search.js uses named
//   import).

import { getConversationHistory } from './search.js?v=20260728a';
import { getChatHistory } from './chat.js?v=20260714a';
import { getSessionHistory, getSavedSessions, clearSharedSessions, renderSharedSessions } from './sessions-list.js';
import { getSourceDisplayNames } from './source-filters.js';
import { getCurrentSessionId } from '../utils/analytics.js';

// ============================================================================
// shareContentOverride — explicit share content (string|null)
// ============================================================================
let _shareContentOverride = null;

export function getShareContentOverride() {
    return _shareContentOverride;
}

export function setShareContentOverride(c) {
    _shareContentOverride = c;
}

export function clearShareContentOverride() {
    _shareContentOverride = null;
}

// ============================================================================
// Export / share helpers (commit 18 migration)
// ============================================================================

// Helper: Clean HTML content for different export formats
export function cleanHTMLContent(content, format = 'plain') {
    if (!content) return '';

    if (format === 'plain') {
        // Strip all HTML and markdown links
        return content
            .replace(/<br\s*\/?>/gi, '\n')
            .replace(/<[^>]+>/g, '')
            .replace(/\[來源\]\([^\)]+\)/g, '')
            .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1'); // Keep link text only
    } else if (format === 'markdown') {
        // Keep markdown, convert <br> to newlines
        return content.replace(/<br\s*\/?>/gi, '\n\n');
    }

    return content;
}

// Helper: Get top 10 articles from the most recent search
export function getTop10Articles() {
    const hist = getSessionHistory();
    if (hist.length === 0) return [];
    const lastSession = hist[hist.length - 1];
    if (!lastSession.data || !lastSession.data.content) return [];
    return lastSession.data.content.slice(0, 10);
}

// Helper: 一筆 session 的「分析內容」→ { label, text }；無可匯出內容回 null。
//
// 為什麼需要這層：**DR / LR 沒有「綜合摘要」這個東西**。一般搜尋的摘要在
// `session.data.answer`；DR / LR 的報告本文則在 `session.researchReport.report`
// （deep-research.js:1751 push 進 sessionHistory 的形狀）。三個匯出 builder 原本
// 一律只讀 data.answer → DR / LR 每一筆都落空，複製出來只剩一個空的
// 【讀豹分析摘要】標頭（CEO 回報現象）。
export function getSessionAnalysisBlock(session, format = 'plain') {
    if (session?.data?.answer) {
        return { label: '', text: cleanHTMLContent(session.data.answer, format) };
    }
    const report = session?.researchReport?.report;
    if (report) {
        return {
            label: session.isLiveResearch ? '即時研究報告' : '深度研究報告',
            text: cleanHTMLContent(report, format)
        };
    }
    return null;
}

// Helper: 全部 session 的分析內容（已濾掉空的）。builder 先取這個再決定要不要印標頭，
// 避免「有 session 但沒有任何可匯出內容」時印出空標頭。
export function getAnalysisBlocks(format = 'plain') {
    return getSessionHistory()
        .map(s => getSessionAnalysisBlock(s, format))
        .filter(b => b && b.text);
}

// Helper: DR / LR 的引用來源（最近一筆有來源的 research session）。
//
// 報告內文帶 `[N]` 引用標記，來源清單在 `researchReport.sources`——**URL 字串陣列**，
// 編號 = 位置 + 1（與 deep-research.js::generateCitationReferenceList 同一套編號）。
// 這與一般搜尋的文章卡（`data.content` → getTop10Articles）是不同資料，DR 沒有後者；
// 不附這段的話，貼出去的報告裡 [1][2] 全部斷鏈、無從查證。
export function getResearchSources() {
    const hist = getSessionHistory();
    for (let i = hist.length - 1; i >= 0; i--) {
        const srcs = hist[i]?.researchReport?.sources;
        if (Array.isArray(srcs) && srcs.length > 0) {
            return srcs
                .map((url, idx) => ({ index: idx + 1, url: String(url || '').trim() }))
                .filter(s => s.url);
        }
    }
    return [];
}

// Helper: 來源 URL → 可讀標示。非 http(s) 的內部 URN 直接貼出去對讀者沒有意義
// （鏡像 generateCitationReferenceList 的分類）。
export function describeSourceURL(url) {
    if (url.startsWith('urn:llm:knowledge:')) return '讀豹背景知識（無外部連結）';
    if (url.startsWith('private://')) return '私人文件（無外部連結）';
    return url;
}

// Format content for plain text export
export function formatPlainText() {
    let content = '';
    const date = new Date().toLocaleDateString('zh-TW', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    content += `台灣新聞搜尋結果\n`;
    content += `日期：${date}\n`;
    content += `${'='.repeat(50)}\n\n`;

    // Search queries
    if (getConversationHistory().length > 0) {
        content += `【搜尋查詢】\n`;
        getConversationHistory().forEach((query, idx) => {
            content += `${idx + 1}. ${query}\n`;
        });
        content += `\n`;
    }

    // AI answers from search results（含 DR / LR 報告本文）
    const analysisBlocks = getAnalysisBlocks('plain');
    if (analysisBlocks.length > 0) {
        content += `【讀豹分析摘要】\n`;
        analysisBlocks.forEach(block => {
            if (block.label) content += `〈${block.label}〉\n`;
            content += `${block.text}\n\n`;
        });
    }

    // Free conversation messages
    if (getChatHistory().length > 0) {
        content += `【自由對話紀錄】\n`;
        getChatHistory().forEach(msg => {
            const icon = msg.role === 'user' ? '👤 你' : '讀豹';
            const plainContent = cleanHTMLContent(msg.content, 'plain');
            content += `${icon}：${plainContent}\n\n`;
        });
    }

    // Top 10 articles
    const top10 = getTop10Articles();
    if (top10.length > 0) {
        content += `【相關新聞文章（${top10.length} 篇）】\n`;
        top10.forEach((article, idx) => {
            const title = article.name || article.schema_object?.headline || '無標題';
            const _drPub = article.schema_object?.publisher;
            const source = (typeof _drPub === 'object' ? _drPub?.name : _drPub) || getSourceDisplayNames()[article.site] || article.site || '未知來源';
            const date = article.schema_object?.datePublished?.split('T')[0] || '未知日期';
            const desc = article.description || article.ranking?.description || '';

            content += `${idx + 1}. ${title}\n`;
            content += `   來源：${source} | 日期：${date}\n`;
            if (desc) {
                content += `   ${desc}\n`;
            }
            content += `\n`;
        });
    }

    // DR / LR 引用來源（報告內文的 [N] 對應此清單；一般搜尋無此段）
    const plainSources = getResearchSources();
    if (plainSources.length > 0) {
        content += `【參考資料來源（${plainSources.length} 筆）】\n`;
        plainSources.forEach(s => {
            content += `[${s.index}] ${describeSourceURL(s.url)}\n`;
        });
        content += `\n`;
    }

    return content;
}

// Format content for AI chatbot (ChatGPT/Claude/Gemini)
export function formatForAIChatbot() {
    let content = '';

    // Opening context
    if (getConversationHistory().length > 0) {
        content += `我剛搜尋了關於「${getConversationHistory()[0]}」的台灣新聞，以下是搜尋結果：\n\n`;
    }

    // Search queries
    if (getConversationHistory().length > 1) {
        content += `【搜尋查詢】\n`;
        getConversationHistory().forEach((query, idx) => {
            content += `${idx + 1}. ${query}\n`;
        });
        content += `\n`;
    }

    // AI analysis（含 DR / LR 報告本文）
    const chatBlocks = getAnalysisBlocks('markdown');
    if (chatBlocks.length > 0) {
        content += `【讀豹分析摘要】\n`;
        chatBlocks.forEach(block => {
            if (block.label) content += `〈${block.label}〉\n`;
            content += `${block.text}\n\n`;
        });
    }

    // Free conversation
    if (getChatHistory().length > 0) {
        content += `【自由對話紀錄】\n`;
        getChatHistory().forEach(msg => {
            const icon = msg.role === 'user' ? '👤 你' : '讀豹';
            const cleanContent = cleanHTMLContent(msg.content, 'markdown');
            content += `${icon}：${cleanContent}\n\n`;
        });
    }

    // Articles with URLs
    const top10 = getTop10Articles();
    if (top10.length > 0) {
        content += `【相關新聞來源（${top10.length} 篇）】\n`;
        top10.forEach((article, idx) => {
            const title = article.name || article.schema_object?.headline || '無標題';
            const url = article.url || article.schema_object?.url || '';
            const _drPub2 = article.schema_object?.publisher;
            const source = (typeof _drPub2 === 'object' ? _drPub2?.name : _drPub2) || getSourceDisplayNames()[article.site] || article.site || '';
            const date = article.schema_object?.datePublished?.split('T')[0] || '';
            const desc = article.description || article.ranking?.description || '';

            content += `${idx + 1}. ${title}\n`;
            if (url) content += `   網址：${url}\n`;
            if (source || date) content += `   來源：${source} | 日期：${date}\n`;
            if (desc) content += `   摘要：${desc}\n`;
            content += `\n`;
        });
    }

    // DR / LR 引用來源（報告內文的 [N] 對應此清單；一般搜尋無此段）
    const chatSources = getResearchSources();
    if (chatSources.length > 0) {
        content += `【參考資料來源（${chatSources.length} 筆）】\n`;
        chatSources.forEach(s => {
            content += `[${s.index}] ${describeSourceURL(s.url)}\n`;
        });
        content += `\n`;
    }

    content += `---\n請基於以上資訊幫我進行分析。`;

    return content;
}

// Format content for NotebookLM (rich markdown with full details)
export function formatForNotebookLM() {
    let content = '';
    const date = new Date().toLocaleDateString('zh-TW', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    // Title
    if (getConversationHistory().length > 0) {
        content += `# 台灣新聞搜尋：${getConversationHistory()[0]}\n\n`;
    } else {
        content += `# 台灣新聞搜尋結果\n\n`;
    }

    content += `**搜尋日期**: ${date}\n\n`;
    content += `---\n\n`;

    // Search queries
    if (getConversationHistory().length > 0) {
        content += `## 搜尋查詢\n\n`;
        getConversationHistory().forEach((query, idx) => {
            content += `${idx + 1}. ${query}\n`;
        });
        content += `\n`;
    }

    // AI analysis（含 DR / LR 報告本文）
    const nbBlocks = getAnalysisBlocks('markdown');
    if (nbBlocks.length > 0) {
        content += `## 讀豹分析摘要\n\n`;
        nbBlocks.forEach(block => {
            if (block.label) content += `### ${block.label}\n\n`;
            content += `${block.text}\n\n`;
        });
    }

    // Free conversation
    if (getChatHistory().length > 0) {
        content += `## 自由對話紀錄\n\n`;
        getChatHistory().forEach(msg => {
            const role = msg.role === 'user' ? '**你**' : '**讀豹**';
            const cleanContent = cleanHTMLContent(msg.content, 'markdown');
            content += `${role}: ${cleanContent}\n\n`;
        });
    }

    // Detailed articles
    const top10 = getTop10Articles();
    if (top10.length > 0) {
        content += `## 相關新聞來源（${top10.length} 篇）\n\n`;
        top10.forEach((article, idx) => {
            const title = article.name || article.schema_object?.headline || '無標題';
            const url = article.url || article.schema_object?.url || '';
            const _drPub2 = article.schema_object?.publisher;
            const source = (typeof _drPub2 === 'object' ? _drPub2?.name : _drPub2) || getSourceDisplayNames()[article.site] || article.site || '';
            const date = article.schema_object?.datePublished?.split('T')[0] || '';
            const desc = article.description || article.ranking?.description || '';

            content += `### ${idx + 1}. ${title}\n\n`;
            if (source) content += `- **來源**: ${source}\n`;
            if (date) content += `- **日期**: ${date}\n`;
            if (url) content += `- **網址**: ${url}\n`;
            if (desc) content += `\n${desc}\n`;
            content += `\n---\n\n`;
        });
    }

    // DR / LR 引用來源（報告內文的 [N] 對應此清單；一般搜尋無此段）
    const nbSources = getResearchSources();
    if (nbSources.length > 0) {
        content += `## 參考資料來源（${nbSources.length} 筆）\n\n`;
        nbSources.forEach(s => {
            content += `${s.index}. ${describeSourceURL(s.url)}\n`;
        });
        content += `\n`;
    }

    return content;
}

// Copy to clipboard and optionally open URL
export async function copyAndOpen(text, url = null, buttonElement) {
    const originalText = buttonElement.textContent;

    try {
        await navigator.clipboard.writeText(text);

        // Visual feedback
        buttonElement.textContent = '✓ 已複製！';
        buttonElement.style.borderColor = '#059669';
        buttonElement.style.color = '#059669';

        // Open URL if provided
        if (url) {
            window.open(url, '_blank');
        }

        setTimeout(() => {
            buttonElement.textContent = originalText;
            buttonElement.style.borderColor = '';
            buttonElement.style.color = '';
        }, 2000);

    } catch (err) {
        console.error('複製失敗:', err);
        buttonElement.textContent = '✗ 複製失敗';
        setTimeout(() => {
            buttonElement.textContent = originalText;
        }, 2000);
    }
}

// Feedback modal logic (Bug #14)
export function openFeedbackModal(rating) {
    // Remove existing modal if any
    const existing = document.getElementById('feedbackModal');
    if (existing) existing.remove();

    const ratingLabel = rating === 'positive' ? '<img src="/static/images/icon-good.svg" alt="正面" class="inline-icon"> 正面回饋' : '<img src="/static/images/icon-bad.svg" alt="負面" class="inline-icon"> 負面回饋';

    const modal = document.createElement('div');
    modal.id = 'feedbackModal';
    modal.className = 'feedback-modal-overlay';
    modal.innerHTML = `
        <div class="feedback-modal">
            <div class="feedback-modal-header">
                <span>${ratingLabel}</span>
                <button class="feedback-modal-close">&times;</button>
            </div>
            <div class="feedback-modal-body">
                <textarea class="feedback-textarea"
                          placeholder="感謝提供意見，有任何正面、負面體驗，或其他意見都歡迎回饋！"
                          rows="4"></textarea>
            </div>
            <div class="feedback-modal-footer">
                <button class="feedback-cancel">取消</button>
                <button class="feedback-submit">提交回饋</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Focus textarea
    const textarea = modal.querySelector('.feedback-textarea');
    setTimeout(() => textarea.focus(), 100);

    // Close handlers
    modal.querySelector('.feedback-modal-close').addEventListener('click', () => modal.remove());
    modal.querySelector('.feedback-cancel').addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });

    // Submit handler
    modal.querySelector('.feedback-submit').addEventListener('click', async () => {
        const comment = textarea.value.trim();
        const submitBtn = modal.querySelector('.feedback-submit');
        submitBtn.disabled = true;
        submitBtn.textContent = '提交中...';

        // Gather context
        const query = document.getElementById('searchInput')?.value || '';
        const summaryEl = document.querySelector('.summary-content');
        const answerSnippet = summaryEl ? summaryEl.textContent.substring(0, 200) : '';

        try {
            // P1 E2E fix (2026-05-26): route through authenticatedFetch for 401→refresh→retry.
            const resp = await window.authManager.authenticatedFetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rating: rating,
                    query: query,
                    answer_snippet: answerSnippet,
                    comment: comment,
                    session_id: getCurrentSessionId() || ''
                })
            });
            if (resp.ok) {
                submitBtn.textContent = '已提交';
                submitBtn.style.background = '#059669';
                submitBtn.style.color = '#fff';
                setTimeout(() => modal.remove(), 1000);
            } else if (resp.status === 401) {
                submitBtn.textContent = '登入已過期，請重新登入';
                submitBtn.disabled = false;
            } else {
                submitBtn.textContent = '提交失敗，請重試';
                submitBtn.disabled = false;
            }
        } catch (err) {
            console.error('[Feedback] Submit error:', err);
            submitBtn.textContent = '提交失敗，請重試';
            submitBtn.disabled = false;
        }
    });
}

// ============================================================================
// Session sharing toggle (AC-3 auto-refresh: sidebar re-renders + badge updates)
// ============================================================================

// Toggle session sharing visibility (private <-> org)
// Optimistic UI: update state + render immediately, sync to server in background
export async function toggleSessionSharing(sessionId) {
    const session = getSavedSessions().find(s => window.matchSessionId(s.id, sessionId));
    if (!session) return;

    const isCurrentlyShared = session.visibility && session.visibility !== 'private';
    const newVisibility = isCurrentlyShared ? 'private' : 'org';

    // Optimistic: update local state + UI immediately
    session.visibility = newVisibility;
    localStorage.setItem('taiwanNewsSavedSessions', JSON.stringify(getSavedSessions()));
    // AC-3 auto-refresh — sidebar re-renders with new visibility tag
    if (typeof window.renderLeftSidebarSessions === 'function') window.renderLeftSidebarSessions();
    if (typeof window._updateOrgSpaceBadge === 'function') {
        window._updateOrgSpaceBadge(newVisibility === 'org' ? 1 : -1);
    }
    // AC-3 fix (P2 E2E finding): invalidate shared sessions cache + re-render org space
    // list so the newly shared (or unshared) session appears immediately without requiring
    // a manual tab switch. Mirror delete handler pattern (sessions-list.js wasShared block).
    clearSharedSessions();
    renderSharedSessions();

    // Background: sync to server
    let serverId = session._serverId || (typeof session.id === 'string' && session.id.includes('-') ? session.id : null);
    try {
        if (!serverId) {
            await window.sessionManager.saveSession(session);
            serverId = session._serverId;
            if (!serverId) {
                console.error('[Session] Failed to sync session to server, reverting');
                session.visibility = isCurrentlyShared ? 'org' : 'private';
                localStorage.setItem('taiwanNewsSavedSessions', JSON.stringify(getSavedSessions()));
                if (typeof window.renderLeftSidebarSessions === 'function') window.renderLeftSidebarSessions();
                if (typeof window._updateOrgSpaceBadge === 'function') {
                    window._updateOrgSpaceBadge(newVisibility === 'org' ? -1 : 1);
                }
                return;
            }
        }
        await window.sessionManager.setSessionVisibility(serverId, newVisibility);
    } catch (err) {
        console.error('[Session] Failed to set visibility, reverting:', err);
        session.visibility = isCurrentlyShared ? 'org' : 'private';
        localStorage.setItem('taiwanNewsSavedSessions', JSON.stringify(getSavedSessions()));
        if (typeof window.renderLeftSidebarSessions === 'function') window.renderLeftSidebarSessions();
        if (typeof window._updateOrgSpaceBadge === 'function') {
            window._updateOrgSpaceBadge(newVisibility === 'org' ? -1 : 1);
        }
    }
}
