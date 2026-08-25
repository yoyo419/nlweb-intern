# Knowledge Graph 視覺化規格

> 本文件描述 KG 視覺化功能的實際實作。
>
> **⚠️ 檔案位置更新（2026-06-15）**：KG 前端邏輯已於 2026-05-25（前端模組化重構 v4.0 Phase 8 commit 21）從 `static/news-search.js` 搬至獨立模組 **`static/js/features/knowledge-graph.js`**。本文件內所有 `static/news-search.js line XXXX` 的行號參照均為**歷史位置**，現行常數與函數請於 `static/js/features/knowledge-graph.js` 查找（用常數/函數名 grep，行號為邏輯參考）。CSS / HTML 結構（`news-search.css` / `news-search-prototype.html`）位置未變。
>
> **範圍分界**：本文件涵蓋「KG 資料顯示與互動」。KG 節點/邊的新增、刪除、重命名屬於編輯功能，見 `docs/archive/specs/kg-editing-spec.md`（🪦 已歸檔 2026-07-10；編輯功能本身仍上線——現行實作為 `knowledge-graph.js` closure factory + `POST /api/research/rerun`，見該檔頭 banner 與「API Contract」章）。

---

## 1. 目的與範圍

Knowledge Graph（知識圖譜）以**互動下鑽（focus + expand on demand）**視覺化 Deep Research / Live Research 報告所萃取的實體與關係。一打開預設顯示 **Top-N 核心實體骨架**（高 degree 實體 + 它們之間的關係），長尾節點隱藏；使用者點核心節點即展開其鄰居，逐層下鑽。目標：讓認知負荷與底層資料量脫鉤 —— 不論底層 10 還 500 個實體，使用者任一時刻只面對「一個焦點的鄰域」（Obsidian / Roam graph view 標準做法）。

| 功能子系統 | 文件 |
|---|---|
| KG 視覺化（本文件）| `docs/specs/kg-spec.md` |
| KG 編輯模式 | `docs/archive/specs/kg-editing-spec.md`（🪦 已歸檔；功能仍上線，現行契約見該檔頭 banner）|

---

## 2. 資料來源

### 2.1 後端生成路徑

KG 後端生成有**兩條路徑**：Deep Research（DR，單次生成）與 Live Research（LR，跨輪 merge）。

#### 2.1.1 DR 路徑（單次生成）

```
Query → Reasoning Orchestrator
  → Analyst Agent（AnalystResearchOutputEnhancedKG）
  → knowledge_graph: KnowledgeGraph（entities[], relationships[]）
  → orchestrator.py 序列化為 JSON
  → schema_object 的 knowledge_graph 欄位（SSE done 事件夾帶）
```

DR 為**單次生成**：Analyst 一次產出完整 KG，`orchestrator.py` 將其寫入 `schema_obj["knowledge_graph"]`（schema_object 頂層 key，非 metadata 子欄位；前端用 §2.2 dual-path 兼容兩種落點）。

#### 2.1.2 LR 路徑（跨輪 merge）

LR 有獨立的 KG 生成與**跨輪累積**機制，與 DR 的單次生成行為本質不同：

```
每輪 BAB mini-reasoning → Analyst 產出該輪 new_kg
  → loop_engine._merge_knowledge_graph(state.knowledge_graph, new_kg)
     （entity name dedup + relationship triple dedup；Critic REJECT 該輪則跳過 merge）
  → 累積進 state.knowledge_graph
  → Stage 6 匯出：live_research/orchestrator.py._build_kg_export_payload(state.knowledge_graph)
     → SSE knowledge_graph payload + 🪦報告末段 KG section
```

> ⚠️ **報告末段 KG JSON section 已於 2026-07-21 暫移除**（匯出只支援純文字、KG 的 raw JSON 使檔案體積雙倍且不可讀），待 KG overhaul 後恢復；**SSE knowledge_graph payload（餵前端視覺化）保留不變**。上圖「報告末段 KG section」一路即為被移除者，`_build_kg_export_payload()` 仍構築供 SSE payload 使用。

- **跨輪 merge**：`loop_engine.py` Track D 每輪呼叫 `_merge_knowledge_graph()` 將該輪 `new_kg` 併入 `state.knowledge_graph`（DR 是單次、LR 是跨輪累積）。
- **獨立 export payload**：LR 自己的序列化函式 `_build_kg_export_payload()`（`live_research/orchestrator.py`），格式與 DR 相同（entities/relationships/metadata），讀 `state.knowledge_graph`；空 KG（無 entities 也無 relationships）回 `None` → SSE `knowledge_graph=None` → 前端自動 short-circuit 不渲染。
- **共用 DR schema/validator**：LR reuse DR 的 `KnowledgeGraph` schema，dangling relationship 由同一個 `validate_relationships()` 自動 filter（見 §2.3）。
- **merge 失敗降級**：見 §2.5。

KG 生成由 `config/config_reasoning.yaml` 中 `knowledge_graph_generation: false` 控制，可透過 `enable_kg` request 參數覆蓋（DR/LR 共用此 toggle）。

### 2.2 前端接收路徑

`displayDeepResearchResults()` 從 SSE `done` 事件的 metadata 取出 KG：

```javascript
// 雙路徑向下相容
schemaObj?.knowledge_graph || metadata?.knowledge_graph
```

### 2.3 KG 資料結構（後端 Pydantic Schema）

**KnowledgeGraph**
```
entities: Entity[]
relationships: Relationship[]
```

**Entity**
```
entity_id: str (UUID)
name: str
entity_type: EntityType
description: str | None
evidence_ids: List[int]        // 引用的 citation index
confidence: "high" | "medium" | "low"
attributes: Dict[str, Any]
supporting_claims: List[str]   // 未來連結到 ArgumentNode
```

**Relationship**
```
relationship_id: str (UUID)
source_entity_id: str
target_entity_id: str
relation_type: RelationType
description: str | None
evidence_ids: List[int]
confidence: "high" | "medium" | "low"
temporal_context: Dict | None
```

**EntityType 列舉**：`person`, `organization`, `event`, `location`, `metric`, `technology`, `concept`, `product`, `facility`, `service`

**RelationType 列舉**：`causes`, `enables`, `prevents`（因果）; `precedes`, `concurrent`（時序）; `part_of`, `owns`（層級）; `supports`, `related_to`（關聯/論證）

> `supports`（論證支持關係）是為了**避免 prod LLM 自然輸出 `supports` 觸發 enum validation 而導致整包 KG 被 Pydantic reject 丟棄**而補入（`schemas_enhanced.py` RelationType，共 10 型）。讀者勿誤以為 `supports` 是非法值。

**驗證規則**：`KnowledgeGraph.validate_relationships()` 會自動過濾掉引用不存在 entity_id 的關係（log warning，不 raise error）。

### 2.4 前端序列化格式差異

前端取得的 JSON 使用 `model_dump()` 序列化，欄位名稱與 Pydantic model 完全一致（snake_case）。orchestrator 額外附加 metadata：

```json
{
  "entities": [...],
  "relationships": [...],
  "metadata": {
    "generated_at": "ISO8601",
    "entity_count": N,
    "relationship_count": N
  }
}
```

### 2.5 LR KG merge 失敗降級行為（2026-06-17, a7a15c78）

LR 跨輪 merge（§2.1.2）若失敗（如 LLM 429 貫穿整個 run），除 `logger.warning` 外，會對 user 播放**一次** per-run 降級旁白，落實 CLAUDE.md「不可 silent fail」紀律：

- 旁白文案 `lr_copy.KG_MERGE_DEGRADED_NARRATION`：「提醒：這一輪的新資料未能併入知識圖譜，圖譜可能少了這部分內容，但文字研究仍會照常進行。」
- 由 dedup flag `_kg_merge_degraded_narrated`（`loop_engine.py` per-run dedup flags 初始化）控制，**每 run 只播一次**防持續性失敗每輪轟炸；log 則每輪照記。
- 降級後**文字研究續行**，僅該輪 KG 內容未併入。
- 此降級僅 LR 路徑；DR 為單次生成無跨輪 merge，不適用。

---

## 3. Layout 演算法（中心放射式）

### 3.0 可見集與骨架（KG overhaul 2026-07）

渲染範式為互動下鑽：任一時刻只渲染「當前可見集」的子圖。

- **初始可見集 = Top-N 骨架**：`kg-skeleton.js::selectSkeleton()` 依 degree 選 top-N 核心實體（`N = clamp(round(totalEntities * 0.3), 8, 15)`，第 N 名 degree 的 tie 全收），骨架邊 = 兩端皆在骨架的邊。長尾節點隱藏（list view 仍全量可查）。
- **可見集狀態機**：`kg-visible-set.js`（`expandNode` 累積加一層鄰居 / `focusNode` 收斂單鄰域；收合 = 回骨架 reset，不提供 per-node collapse）。
- **子圖投影**：`kg-subgraph.js::projectSubgraph()` 把（完整 kg + 可見集）投影成 `{entities, relationships}`，並為每個可見節點附 `hiddenNeighborCount`（畫「+N」badge）。
- **增量佈局（D 拍板 2026-07-21）**：`kg-layout.js`。座標由**外部持有**（instance `_kgNodePositions`），佈局器 `renderKGGraphView(subgraph, container, precomputedPositions)` 只畫不自算。
  - **展開（expand）→ `placeNewNodes()`**：已在畫面上的節點座標**凍結不動**，只有這次新展開的鄰居以焦點父為錨、在既有占用位置外找角度/半徑環擺放、避讓既有（保留空間記憶）。
  - **骨架初始 / reset / focus / show-all → `layoutSkeleton()`**：整體極座標佈局回乾淨狀態（此時允許重排，鏡像原 §3.2 sector 演算法）。
  - **不用 D3 force**（2026-03 刻意移除的相依不重引）——增量佈局是「靜態極座標的增量版」。

### 3.1 概述

現行 layout 為**靜態極座標計算**，無 D3 force simulation（force simulation 已於 2026-03 重設計時移除）。所有節點位置在渲染前即已確定，不會因物理模擬而移動。佈局器輸入為**當前可見子圖 + 外部傳入的凍結座標**（非完整 KG、非每次自算），座標由 `kg-layout.js` 增量/整體佈局供給。

### 3.2 演算法步驟

```
1. 建立 degree map：degree[entity_id] = in-edges + out-edges

2. 選取中心節點：
   - 最高 degree 的 entity 為中心節點
   - Tie-break：取 entities[] 中先出現者

3. 剩餘節點按 entity_type 分組 → typeGroups{}
   - 每個 type 為一個 sector（扇形區域）
   - sectorAngle = 2π / typeCount
   - sector 起始角度從 -π/2（頂部）開始，順時針排列

4. 每個 sector 內均勻分佈節點：
   - step = sectorAngle / (groupSize + 1)
   - node[j] → angle = startAngle + (j + 1) * step

5. 決定環半徑（ring radius）：
   - useDoubleRing = (maxGroupSize > 5)
   - 單環：innerRingR = min(width, height) * 0.32
   - 雙環（交錯排列）：
       innerRingR = min(width, height) * 0.22
       outerRingR = min(width, height) * 0.40
   - 偶數 index 節點放內環，奇數放外環

6. 極座標轉笛卡爾：
   x = centerX + ringR * cos(angle)
   y = centerY + ringR * sin(angle)

7. 中心節點：固定在 (centerX, centerY)
```

### 3.3 節點大小

```
BASE_RADIUS = 14
SCALE_FACTOR = 4
MAX_RADIUS = 40

nodeRadius(id) = min(BASE_RADIUS + degree(id) * SCALE_FACTOR, MAX_RADIUS)
centerRadius   = max(nodeRadius(centerId), 24)  // 中心節點最小半徑 24
```

---

## 4. 節點設計

### 4.1 品牌色對應表

`KG_ENTITY_STYLES` 常數（`static/js/features/knowledge-graph.js` 約 line 98；歷史位置 `static/news-search.js` line 3794）：

| Entity Type | Fill | Stroke | Stroke Style | Shape |
|---|---|---|---|---|
| person | `#FDCB6E`（金）| `#2D3436` | 2px solid | circle |
| organization | `#FDCB6E`（金）| `#2D3436` | 2px dashed `4,3` | diamond |
| event | `#FFEAA7`（淺金）| `#2D3436` | 2px solid | circle |
| location | `#FFEAA7`（淺金）| `#2D3436` | 2px solid | diamond |
| metric | `#FFFFFF`（白）| `#2D3436` | 2px solid | circle |
| technology | `#FFFFFF`（白）| `#2D3436` | 2px solid | diamond |
| concept | `#B2BEC3`（灰）| `#2D3436` | 2px solid | circle |
| product | `#B2BEC3`（灰）| `#2D3436` | 2px dashed `4,3` | circle |
| 其他（fallback）| `#B2BEC3` | `#2D3436` | solid | circle |

**中心節點**（最高 degree，不論 entity_type）：
- Fill：`#FDCB6E`
- Stroke：`#2D3436`，3px solid
- Shape：強制 circle

### 4.2 Diamond 實作

Diamond 以旋轉 45° 的 `<rect>` 實作：

```javascript
el.append('rect')
  .attr('x', -size/2).attr('y', -size/2)
  .attr('width', size).attr('height', size)
  .attr('transform', 'rotate(45)')
// size = nodeRadius * 1.4（使面積與同半徑 circle 相近）
```

### 4.3 節點 Label

- 位置：節點下方 `dy = nodeRadius + 14`
- 字體：11px，font-weight 500，fill `#2D3436`
- 截斷：超過 12 字元 → 截斷加 `...`
- `pointer-events: none`（不攔截 mouse event）

---

## 5. 邊設計

### 5.1 邊類型

所有邊都是有向邊（帶箭頭），分兩種路徑：

| 情況 | 路徑類型 | 說明 |
|---|---|---|
| 邊的一端為中心節點 | 直線（`M...L...`）| `computeStraightPath()` |
| 兩端皆為非中心節點 | 二次貝茲曲線（`M...Q...`）| `computeArcPath()` |

### 5.2 直線路徑

- 方向：中心節點 → 葉節點（即使原始關係方向相反，arrow 也從中心出發）
- 起點偏移：從節點邊緣出發（不穿透節點）
- 終點偏移：距目標節點邊緣 8px（arrowhead 空間）
- Diamond 節點的 boundingR 使用 `r * 1.0`（同 circle radius）

### 5.3 弧線路徑（Bezier）

- 控制點：線段中點 + 垂直偏移
- bulge = `min(dist * 0.2, 40)`（弧度上限 40px，防過度彎曲）
- 控制點：`midpoint + normal * bulge`

### 5.4 視覺樣式

- 一般邊：stroke `#B2BEC3`，stroke-width 1.5，stroke-opacity 0.7，fill none
- 高亮邊（節點被點選時）：stroke `#FDCB6E`，stroke-width 2.5，opacity 1
- 非連接邊（暗化）：opacity 0.15

### 5.5 Arrow Markers

兩個 SVG defs marker：
- `#kg-arrow`（一般）：fill `#B2BEC3`
- `#kg-arrow-highlight`（高亮）：fill `#FDCB6E`
- `viewBox="0 -5 10 10"`，`markerWidth/Height=6`

### 5.6 邊 Label

- 位置：中心邊 → 直線中點；葉節點邊 → 貝茲中點偏移 60%
- 顯示中文 relation label（`KG_RELATION_LABELS`）
- 字體：10px，fill `#2D3436`，text-anchor middle

---

## 6. D3 Simulation

**現行 layout 無 D3 force simulation。**

節點位置由步驟 3 的靜態極座標計算決定，`d3.select().append()` 僅用於 SVG DOM 操作，不使用 `d3.forceSimulation()`。

全域變數 `kgSimulation` 保留作為清理 hook（在 `renderKGGraphView()` 入口呼叫 `kgSimulation.stop()` 以防舊 simulation 殘留），目前正常情況下永遠為 null。

### Zoom & Pan

```javascript
d3.zoom()
  .scaleExtent([0.3, 3])
  .on('zoom', (event) => { g.attr('transform', event.transform); })
```

縮放範圍：0.3x ～ 3x。

---

## 7. 互動行為

### 7.1 Click 行為（下鑽優先，退化到 Highlight）

點節點的動作由 `kg-visible-set.js::decideNodeClickAction()` 決定：
- **有隱藏鄰居（hiddenNeighborCount > 0）→ expand**：展開該節點的直接鄰居（累積進可見集），以焦點為中心增量重繪（既有節點座標凍結）。
- **無隱藏鄰居 → 退回既有 highlight/deselect**：找鄰接節點，非鄰接節點 opacity 0.2、非連接邊 opacity 0.15 stroke `#B2BEC3`、連接邊 stroke `#FDCB6E` stroke-width 2.5 marker `#kg-arrow-highlight`；再點同節點 `deselectAll()`；點 SVG 空白處 `deselectAll()`。
- **雙擊節點 → focus↔骨架**：第一次雙擊聚焦為該節點單鄰域；當前已聚焦於該節點時再雙擊回骨架（整體重排）。

click/dblclick 衝突用 delayed single-click timer 處理（`_kgClickTimer` + `DBLCLICK_DELAY_MS`）：click 延遲執行、dblclick 進來先 `clearTimeout` 取消 pending click。Edit mode 下所有下鑽 handler bypass，顯示完整圖（見 §7.6）。

### 7.2 Hover Tooltip

`mouseenter` 觸發，顯示：
- 節點名稱、entity_type 中文標籤、description（若有）、連結數（degree）

**Tooltip 定位**：以 container BoundingClientRect 為基準，計算 tipX/tipY，並 clamp 避免溢出容器（max-width 300px，估算高度 100px）。

`mouseleave` 移除 `.visible` class（opacity → 0）。

### 7.3 互動提示 Overlay

每次 `renderKGGraphView()` 後，在容器右下角顯示「點節點展開鄰居・雙擊聚焦・滾輪縮放・拖曳移動」提示，3 秒後 fade out（`.faded` class）。滑鼠移入 container 時重新顯示（opacity 0.75）。

### 7.4 View Toggle（圖形 / 列表）

`setupKGViewToggle()` 在 `#kgViewToggle` 監聽點擊：

- **圖形模式**（預設）：顯示 `#kgGraphView`，隱藏 `#kgDisplayContent`。切換時若容器寬度 > 0，重新呼叫 `renderKGGraphView()` 以正確計算尺寸。
- **列表模式**：顯示 `#kgDisplayContent`（由 `renderKGListView()` 填充的 HTML 列表），隱藏 `#kgGraphView`。

### 7.5 Collapse / Hide

- **收起（Collapse）**：`#kgToggleButton` 切換 `#kgContentWrapper` 的 display，header 保持可見。
- **隱藏（Hide）**：`#kgHideBtn` 隱藏整個 `#kgDisplayContainer`，顯示 `#kgRestoreBar`，並設定 `dataset.userHidden = 'true'`，同時寫入 `localStorage('nlweb-kg-hidden')`。
- **還原**：點擊 `#kgRestoreBar` 顯示 container，`dataset.userHidden = 'false'`，更新 localStorage。

### 7.6 Edit Mode 與下鑽互斥

Edit mode 在**完整圖**上操作（`_kgEditData` 完整 clone，不走 `projectSubgraph`），每次 edit re-render 走 `renderEditGraph()` → `layoutSkeleton` 整體佈局（不用增量凍結），使用者可編輯所有節點不受骨架隱藏影響。退出 edit（cancel/confirm）後重建可見集回骨架子圖 + 整體佈局。下鑽互動（含增量佈局凍結）僅在檢視模式生效。

### 7.7 導覽控制（回骨架 / 顯示全部）

- **回骨架**（`kgResetSkeletonBtn` / `lrKGResetSkeletonBtn`）：可見集偏離骨架時顯示，點擊還原初始骨架（整體重排）。
- **顯示全部**（`kgShowAllBtn` / `lrKGShowAllBtn`）：一次顯示全體節點（可能密集），全可見時 disabled。

### 7.8 SVG 匯出（所見即所得，lazy）

- **下載圖譜**按鈕（`kgDownloadBtn` / `lrKGDownloadBtn`）：DR/LR 對稱。點擊當下才序列化畫面上的 graph view SVG（lazy），匯出**當前可見子圖**含當下 zoom/pan 狀態（所見即所得，非完整佈局大圖）。
- **序列化**：`kg-svg-export.js::serializeGraphSVG()` clone SVG DOM → `buildStandaloneSVG()` 注入 XML 宣告 + xmlns + inline `<style>`（`EXPORT_CSS`，因 label 樣式在 CSS class 不隨 DOM 序列化）。
- **XML 合法性防線**：`buildStandaloneSVG()` 輸出前跑 `stripXMLIllegalChars()`，剔除 XML 1.0 非法字元（除 `\t\n\r` 外的 C0 控制碼、落單 surrogate、U+FFFE/U+FFFF）。`.svg` 走嚴格 XML 解析，夾帶一個非法字元就整份解析失敗 → 瀏覽器改顯示 XML 錯誤頁並印出原始碼，使用者回報形態是「下載下來是一大串標籤文字」。node/edge label 源自 LLM 抽取 + 爬取內文，`XMLSerializer` 只 escape `<`/`&`/`"`、控制碼原樣輸出，故需此防線。純函式測試見 `static/js/features/__tests__/kg-svg-export.test.js`（已做 mutation 驗證）。
- **字型取捨**：不 embed 字型；離線開啟 fall back 到系統中文字型（Microsoft JhengHei 等），中文仍可見、字體可能不同。embed Noto Sans TC（base64 woff）為未實作的 optional。
- **下載**：`downloadTextAsFile()`（沿用 `live-research.js` blob download pattern），檔名 `kg-<焦點名>.svg`（`buildExportFilename` sanitize）。
- **報告匯出不含 KG**：報告匯出保持純文字/markdown（2026-07-21 `750e1488` 已拔除報告內 KG JSON）；KG 匯出是獨立的 SVG 下載，兩者不混。
- 後端 KG SSE payload（`_build_kg_export_payload`）不變 —— 匯出全在前端。

---

## 8. Session Persistence

### 8.1 全域狀態

```javascript
let currentKGData = null;    // 當前 session 的 KG 資料（由 displayKnowledgeGraph() 設定）
let kgSimulation = null;     // 保留作清理 hook（目前永遠為 null）
```

### 8.2 Session History 內的 KG 存儲

每次新搜尋完成後，`sessionHistory.push({...})` 包含：

```javascript
knowledgeGraph: currentKGData ? JSON.parse(JSON.stringify(currentKGData)) : null
```

使用 `JSON.parse(JSON.stringify(...))` 做深拷貝，避免後續操作污染歷史記錄。

### 8.3 Session 切換還原

`restoreSession(sessionIndex)` 判斷：
- 若 `session.isDeepResearch && session.researchReport`：呼叫 `displayKnowledgeGraph(session.knowledgeGraph)`（如果有）
- 若為普通搜尋：不顯示 KG

`loadSavedSession()` 載入 localStorage 中的 session 時，同樣在 step 末尾呼叫 `displayKnowledgeGraph(session.knowledgeGraph)`。

### 8.4 跨 Session 殘留 Bug（已修復）

**問題**：切換到無 KG 的 session 時，前一 session 的 KG 視覺上仍殘留於畫面。

**根因**：`loadSavedSession()` 的普通搜尋分支（`!session.researchReport`）只清空 researchView，未將 `#kgDisplayContainer` 設回 `display: none`，也未重置 `currentKGData = null`。

**修復位置**：`cancelActiveSearch()` / session 切換初始化流程中：

```javascript
// 清理知識圖譜狀態（在 resetUIForNewSearch() 內）
currentKGData = null;
if (kgSimulation) { kgSimulation.stop(); kgSimulation = null; }
const kgContainerReset = document.getElementById('kgDisplayContainer');
if (kgContainerReset) kgContainerReset.style.display = 'none';
```

此段位於 `static/news-search.js` 第 1589-1593 行。

### 8.5 Rerun State 持久化（2026-07-15 land，KG 編輯 rerun 契約）

KG 編輯後的 `POST /api/research/rerun`（見 `docs/archive/specs/kg-editing-spec.md`「API Contract」章）需要「重跑分析所需的輸入中間狀態」（`rerunState`：`formatted_context` / `source_map` / `current_context` / `mode` 等），這與本節 §8.1-8.4 描述的「KG **輸出**資料前端顯示快取」是不同的持久化問題。

- **併入既有 `research_report` JSONB 內層**：rerun 輸入狀態不開新欄位，塞進 `search_sessions.research_report` 內層 `rerunState` key（零 DDL）。寫入時機：DR phase 1 完成快照點，`build_rerun_state_subset(state)`（`orchestrator.py`）抽精簡子集。
- **記憶體 cache 為快取層、DB 為 fallback source of truth**：`_research_state_cache`（進程記憶體 dict，TTL 3600s / LRU 50）hit 時走快取；cache miss（server 重啟 / TTL 過期 / LRU 淘汰）→ 用前端帶的 `session_id`（PG session UUID）讀 DB `research_report.rerunState` → `restore_rerun_state_from_report()` 重建 → 餵 `run_research_rerun(restored_state=...)`。
- **payload 精簡**：不存全量 `items`（只存 `items_count`，phase 2-4 只用其 `len()`）；`source_map` 的每個 item 經 `_slim_item` 剝成 5 欄（`url`/`title`/`description`/`site`/`datePublished`，rerun path 實際讀取的欄位窮舉）；`current_context` 不單存，restore 時從 `source_map` 依 citation id 排序重建。
- **query_id 對齊防線**：`rerunState` 綁定產生它的那次 DR `query_id`；DB fallback 時驗 `rerunState.query_id == 請求的 original_query_id`，不符（含舊資料缺此欄位）判無效 → 400，防同一 session 內多次 DR 時張冠李戴。
- **rerun 產出本身不 persist**（既有行為，非本次變更）：`execute_rerun` 不呼叫 `_persist_research_report`，rerun 後的報告只存在當前畫面，reload 後回退到原始版本；原 `rerunState` 不受影響（rerun 不覆蓋 `research_report`）。
- 詳細技術細節見 `docs/specs/reasoning-spec.md` §2.7、`docs/specs/session-spec.md` research_report 欄位段。

### 8.6 User Hidden Preference

`initKGVisibilityToggle()` 在 DOMContentLoaded 時讀取 `localStorage('nlweb-kg-hidden')`。若為 `'true'`，設定 `kgContainer.dataset.userHidden = 'true'`。

`displayKnowledgeGraph()` 尊重此偏好：若 `dataset.userHidden === 'true'`，仍執行完整渲染（資料準備好），但 container 保持 `display: none`，並顯示 restore bar。

---

## 9. HTML 結構

```html
<!-- KG Restore Bar -->
<div class="kg-restore-bar" id="kgRestoreBar">...</div>

<!-- KG 主容器 -->
<div class="kg-display-container" id="kgDisplayContainer" style="display: none;">
  <div class="kg-display-header">
    <div class="kg-display-title">
      <span>知識圖譜</span>
      <span class="kg-display-metadata" id="kgMetadata"></span>
    </div>
    <div class="kg-view-toggle" id="kgViewToggle">
      <button class="kg-view-btn active" data-view="graph">圖形</button>
      <button class="kg-view-btn" data-view="list">列表</button>
    </div>
    <button id="kgToggleButton">收起</button>
    <button class="kg-hide-btn" id="kgHideBtn">隱藏</button>
  </div>
  <div id="kgContentWrapper">
    <div class="kg-graph-container" id="kgGraphView">
      <div class="kg-tooltip" id="kgTooltip"></div>
      <!-- SVG 由 D3 動態注入 -->
    </div>
    <div class="kg-display-content" id="kgDisplayContent" style="display: none;">
      <!-- 列表 HTML 由 renderKGListView() 填充 -->
    </div>
    <div class="kg-legend" id="kgLegend"></div>
    <div class="kg-display-empty" id="kgDisplayEmpty" style="display: none;"></div>
  </div>
</div>
```

---

## 10. CSS 關鍵樣式

| 選擇器 | 說明 |
|---|---|
| `.kg-display-container` | 白底，圓角 12px，box-shadow，border 2px `var(--color-primary-bg)` |
| `.kg-graph-container` | 高度 400px，`overflow: hidden`，`position: relative`（tooltip 定位基準）|
| `.kg-node` | `cursor: pointer`，`transition: opacity 0.3s` |
| `.kg-node:hover circle/rect` | `filter: brightness(1.08)`，stroke-width 3px |
| `.kg-link` | `stroke-opacity: 0.7`，transition stroke/opacity 0.3s |
| `.kg-tooltip` | `position: absolute`，`pointer-events: none`，max-width 300px，opacity 0 → 1（`.visible`）|
| `.kg-interaction-hint` | 右下角提示，3s fade，hover 時重顯（opacity 0.75）|
| `.kg-view-btn.active` | background `#FFEAA7` |
| `.kg-restore-bar` | dashed border，display none，cursor pointer |

---

## 11. 已知限制

1. **EntityType 樣式不完整支援**：後端 schema 包含 `facility` 和 `service` 兩個 EntityType，前端 `KG_ENTITY_STYLES`（fill/stroke/shape）**仍未涵蓋**這兩型，fallback 為 `KG_DEFAULT_STYLE`（灰色 circle）。
   - **更新（2026-06-15）**：中文標籤部分已修復——`KG_TYPE_LABELS` 現已包含 `'facility': '設施'` 與 `'service': '服務'`（見 `static/js/features/knowledge-graph.js`），故這兩型 hover/列表會顯示正確中文標籤，僅**節點的色彩與形狀**仍走灰色 circle fallback。

2. **節點 label 截斷**：超過 12 字元強制截斷，長名稱（如機構全名）會顯示不完整。Tooltip hover 仍顯示完整名稱。

3. **無拖曳**：靜態 layout 不支援拖曳重排節點。節點位置由演算法決定，使用者無法手動調整。（KG 編輯模式同樣不啟用拖曳，見 `docs/archive/specs/kg-editing-spec.md`）

4. **Resize 不自動重繪**：切換回圖形 view 時觸發重繪，但視窗 resize 不自動觸發。

5. **~~大量節點效能（label 重疊）~~**（KG overhaul 2026-07 解決）：舊放射佈局在節點 > 50 時 label 重疊。互動下鑽範式下，任一時刻只渲染骨架/焦點鄰域（節點數受可見集控制，遠低於全量），根本上避免密集重疊。「顯示全部」按鈕仍可能觸發密集渲染（使用者主動選擇，非預設）。後端 `len(entities) > 100` 的 warning 仍保留作生成量告警。

6. **KG 生成預設關閉**：`config_reasoning.yaml` 中 `knowledge_graph_generation: false`，需 `enable_kg=true` 參數才生成。

---

## 12. 未來擴充方向

- 補齊 `facility`、`service` entity type 的 `KG_ENTITY_STYLES` 樣式（色彩/形狀）；中文標籤已於 2026-06 前補齊（`KG_TYPE_LABELS`）
- Resize Observer 自動重繪（目前切換 view 時才重繪）
- Label 避讓（SVG collision detection 或 `getBBox()`）
- KG 編輯模式（已上線，見 `docs/archive/specs/kg-editing-spec.md`）
- KG 匯出：SVG 下載已於 KG overhaul 2026-07 上線（§7.8）；JSON 下載仍為未實作 optional
- 跨議題 KG 合併與比較

---

## 13. 相關檔案

| 檔案 | 說明 |
|---|---|
| `static/js/features/knowledge-graph.js` | KG 前端全部邏輯（2026-05-25 從 `static/news-search.js` L3794-4415 搬出，獨立模組）|
| `static/js/features/kg-skeleton.js` | Top-N 骨架選擇 + 可見集初始化（純函式）|
| `static/js/features/kg-visible-set.js` | 下鑽可見集狀態機（expand/focus/click 決策，純函式；收合=回骨架）|
| `static/js/features/kg-subgraph.js` | 可見集→子圖投影 + 隱藏鄰居計數（純函式）|
| `static/js/features/kg-layout.js` | 增量佈局（骨架整體佈局 + 新節點找空位，已放凍結，純函式）|
| `static/js/features/kg-svg-export.js` | SVG 所見即所得序列化 + 下載工具（純函式部分可測）|
| `static/news-search.css` L528-670, L2474-2662, L4324-4358 | KG CSS |
| `static/news-search-prototype.html` L498-544 | KG HTML 結構 |
| `code/python/reasoning/schemas_enhanced.py` L242-334 | 後端 KG schema（EntityType/RelationType/Entity/Relationship/KnowledgeGraph/validator）|
| `code/python/reasoning/orchestrator.py` | DR KG 序列化與 SSE 傳送（`schema_obj["knowledge_graph"]`）|
| `code/python/reasoning/live_research/loop_engine.py` | LR Track D 跨輪 KG merge（`_merge_knowledge_graph`）+ merge 失敗降級旁白 |
| `code/python/reasoning/live_research/orchestrator.py` | LR Stage 6 KG export（`_build_kg_export_payload`）|
| `code/python/reasoning/live_research/lr_copy.py` | `KG_MERGE_DEGRADED_NARRATION` 降級旁白文案 |
| `docs/archive/specs/kg-editing-spec.md` | KG 編輯模式規格（🪦 已歸檔 2026-07-10；功能仍上線）|
| `docs/superpowers/plans/2026-03-27-kg-radial-mindmap.md` | 中心放射式重設計計畫（歷史參考）|
