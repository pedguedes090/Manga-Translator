(function () {
    const DATA = window.CORRECTION_DATA || {};
    const images = DATA.images || [];
    const sessionId = DATA.sessionId || '';
    // mode === 'postrender' → post-render editor (spec F4): canvas shows the
    // RENDERED image, blocks edit translated text + bbox, re-render wired to
    // POST /re-render-image (spec F5).
    const isPostrender = DATA.mode === 'postrender';
    // V3: style editor (spec F1-F14) — erased background + live translated
    // text + per-block style panel + background erase tools.
    const isStyleditor = DATA.mode === 'styleditor';
    // Global image index in the session (the API needs it even though the
    // post-render editor only ever loads ONE image per page).
    const globalImageIdx = (typeof DATA.postrenderImageIdx === 'number' && DATA.postrenderImageIdx >= 0) ? DATA.postrenderImageIdx : 0;
    // V3: all session images for the sidebar (each editor page shows one image).
    const allImages = Array.isArray(DATA.allImages) ? DATA.allImages : [];
    const DEFAULT_STYLE = {
        font: 'animeace_', font_size: 0, text_color: null,
        bold: false, italic: false, align: 'center'
    };

    let currentImageIdx = 0;
    let currentTool = 'select';
    let selectedBlockIdx = -1;
    let isDrawing = false;
    let drawStart = null;
    let drawEnd = null;
    let dragOffset = null;
    let isDragging = false;
    let dragBlockIdx = -1;
    let dragStartBbox = null;
    let dragStartSnapshot = null;
    let isOcrPending = false;
    let fitScale = 1;
    let zoomLevel = 1;
    let drawScheduled = false;
    let thumbsScheduled = false;
    let isSpaceDown = false;
    let isPanning = false;
    let panStart = null;
    let undoStack = [];
    let redoStack = [];
    let hoveredDeleteIdx = -1;
    // F1: delete only fires on a real click (mouseup within DELETE_SLOP px of
    // mousedown); a drag that starts on a block never deletes it.
    let deleteDown = null;
    // F2: active resize handle id (e.g. 'nw') while dragging.
    let resizeHandle = null;
    let resizeStartBbox = null;
    let resizeStartSnapshot = null;
    let resizeShiftHeld = false;
    // F7: double-submit guard shared by re-render / save-all.
    let isBusy = false;
    const MAX_UNDO = 60;
    const MIN_ZOOM = 0.25;
    const MAX_ZOOM = 8;
    const ZOOM_STEP = 1.2;
    const MIN_BLOCK_SIZE = 8;           // F2: min block size in IMAGE px
    const DELETE_SLOP = 6;              // F1: click slop in screen px
    const HANDLE_MOUSE_RADIUS = 10;     // F2: mouse hit radius in CSS px
    const HANDLE_TOUCH_SIZE = 44;       // F2: touch invisible hit area ≥44x44 CSS px

    // ---- V3 style editor state (F4/F5/F6/F8/F9) ----
    let brushSize = 12;                 // F4.3: 4/12/24 px (image px)
    let isErasing = false;              // brush stroke in progress
    let strokePoints = [];              // current brush stroke (image coords)
    let strokeColor = null;             // sampled fill color for the stroke
    let lastPointerPos = null;          // image coords, for the brush cursor ring
    let fontList = [];                  // from GET /api/fonts
    const fontLoads = {};               // font name -> Promise (FontFace)
    const fontFailures = {};            // font name -> true (fallback + toast once)
    const fontReady = {};               // font name -> true (loaded, no re-draw loop)
    const layoutCache = {};             // styled text layout cache (keyed)
    const edgeColorCache = {};          // erase region fill color cache (keyed)
    const autoColorCache = {};          // auto text color cache (keyed)
    const eraseLayer = document.createElement('canvas');
    const bgCanvas = document.createElement('canvas');
    const STYLE_SWATCHES = ['#000000', '#ffffff', '#5E1675', '#E53935', '#1E88E5', '#43A047', '#FDD835', '#8E24AA'];

    // The 8 resize handles (F2): corners + edge midpoints.
    const HANDLES = [
        { id: 'nw', fx: 0, fy: 0, corner: true,  cursor: 'nwse-resize' },
        { id: 'ne', fx: 1, fy: 0, corner: true,  cursor: 'nesw-resize' },
        { id: 'sw', fx: 0, fy: 1, corner: true,  cursor: 'nesw-resize' },
        { id: 'se', fx: 1, fy: 1, corner: true,  cursor: 'nwse-resize' },
        { id: 'n',  fx: 0.5, fy: 0, corner: false, cursor: 'ns-resize' },
        { id: 's',  fx: 0.5, fy: 1, corner: false, cursor: 'ns-resize' },
        { id: 'w',  fx: 0, fy: 0.5, corner: false, cursor: 'ew-resize' },
        { id: 'e',  fx: 1, fy: 0.5, corner: false, cursor: 'ew-resize' }
    ];

    const mainCanvas = document.getElementById('main-canvas');
    const ctx = mainCanvas.getContext('2d');
    const canvasOuter = document.querySelector('.canvas-outer');
    const currentImageLabel = document.getElementById('current-image-label');
    const thumbnails = () => document.querySelectorAll('.thumb-item');
    const blockProperties = document.getElementById('block-properties');
    const modifiedBlocksInput = document.getElementById('modified-blocks-input');
    const ocrStatus = document.getElementById('ocr-status');
    const zoomLabel = document.getElementById('zoom-label');
    const hintsBar = document.getElementById('corr-hints');
    const shortcutsHint = document.getElementById('shortcuts-hint');
    const toastEl = document.getElementById('toast');

    const imageCache = {};
    const thumbnailKeys = {};

    // ---- Mode cosmetics ----
    if (isStyleditor) {
        document.body.classList.add('mode-styleditor');
        const add = document.getElementById('tool-add');
        if (add) add.style.display = 'none';
        const reset = document.getElementById('tool-reset');
        if (reset) reset.style.display = 'none';
        if (shortcutsHint) {
            shortcutsHint.innerHTML =
                '<kbd>S</kbd> Chọn <kbd>D</kbd> Xoá <kbd>E</kbd> Xoá nền <kbd>B</kbd>/<kbd>I</kbd> Đậm/Nghiêng ' +
                '<kbd>L</kbd>/<kbd>C</kbd>/<kbd>R</kbd> Căn lề <kbd>[</kbd>/<kbd>]</kbd> Chọn bóng thoại ' +
                '<kbd>Ctrl+Z</kbd> Undo <kbd>Ctrl+Wheel</kbd> Zoom';
        }
    } else if (isPostrender) {
        document.body.classList.add('mode-postrender');
        const el = document.getElementById('tool-add');
        if (el) el.style.display = 'none';
        const reset = document.getElementById('tool-reset');
        if (reset) reset.style.display = 'none';
        const nav = document.querySelector('.canvas-nav');
        if (nav) nav.style.display = 'none';
        if (shortcutsHint) {
            shortcutsHint.innerHTML =
                '<kbd>S</kbd> Chọn <kbd>D</kbd> Xoá <kbd>Ctrl+Z</kbd> Undo ' +
                '<kbd>Ctrl+Shift+Z</kbd> Redo <kbd>Ctrl+Wheel</kbd> Zoom <kbd>Space</kbd> Kéo';
        }
    }

    // ---- Clean OCR text (pre-render mode only) ----
    function cleanOcrText(raw) {
        if (!raw) return '';
        let t = raw
            .replace(/[\u200b\u200c\u200d\u2060\ufeff]/g, '')
            .replace(/\r\n/g, '\n').replace(/\r/g, '\n')
            .replace(/\n{3,}/g, '\n\n')
            .replace(/[ \t]{2,}/g, ' ')
            .replace(/^\s+|\s+$/gm, '')
            .replace(/\n{2,}$/g, '\n')
            .trim();
        return t;
    }

    // ---- Undo/Redo (F6: extended snapshot {text, translated, bbox} +
    // deletedRegions so post-render undo restores translations and erased
    // regions too) ----
    function cloneBlocks(blocks) {
        return (blocks || []).map(b => ({
            text: b.text || '',
            translated: b.translated || '',
            bbox: b.bbox ? [...b.bbox] : null,
            style: b.style ? { ...b.style } : null
        }));
    }

    function snapshot() {
        return {
            imageIdx: currentImageIdx,
            images: images.map(img => ({
                blocks: cloneBlocks(img.blocks || []),
                deletedRegions: [...(img.deletedRegions || [])],
                // V3: eraseRegions is monotonic (never restored/shrunk — F4.5);
                // the *preview* lists drive eraseLayer and ARE undoable.
                eraseRegions: [...(img.eraseRegions || [])],
                erasePreviewRects: [...(img.erasePreviewRects || [])],
                eraseStrokesPreview: (img.eraseStrokesPreview || []).map(s => ({
                    points: (s.points || []).map(pt => [pt[0], pt[1]]),
                    size: s.size,
                    color: s.color
                }))
            }))
        };
    }

    function pushUndo(snap = snapshot()) {
        undoStack.push(snap);
        if (undoStack.length > MAX_UNDO) undoStack.shift();
        redoStack = [];
    }

    function restoreSnapshot(snap) {
        currentImageIdx = snap.imageIdx;
        snap.images.forEach((imgSnap, idx) => {
            setBlocks(idx, cloneBlocks(imgSnap.blocks || []));
            images[idx].deletedRegions = [...(imgSnap.deletedRegions || [])];
            if (isStyleditor) {
                // F4.5: eraseRegions is MONOTONIC — keep the current (larger)
                // set so re-render still erases undone regions. Only the
                // preview lists (eraseLayer source) are restored.
                images[idx].erasePreviewRects = [...(imgSnap.erasePreviewRects || [])];
                images[idx].eraseStrokesPreview = (imgSnap.eraseStrokesPreview || []).map(s => ({
                    points: (s.points || []).map(pt => [pt[0], pt[1]]),
                    size: s.size,
                    color: s.color
                }));
            }
            // Undo returns the block state that is NOT rendered yet → dirty.
            if ((isPostrender || isStyleditor) && idx === snap.imageIdx) {
                images[idx].dirty = true;
                if (isStyleditor) setDirtyBadge(globalImageIdx, true);
            }
        });
        selectedBlockIdx = -1;
        updateBlockEditor(-1);
        if (isStyleditor) saveDraftState();
        loadImage(currentImageIdx).then(() => {
            if (isStyleditor) { initBgCanvas(); redrawEraseLayer(); }
            fitCanvas(); requestDraw(); updateThumbnails(); updateNavButtons(); updateFooterButtons();
        });
    }

    function undo() {
        if (isBusy || undoStack.length === 0) return;
        redoStack.push(snapshot());
        restoreSnapshot(undoStack.pop());
        showToast('Đã hoàn tác ↩');
    }

    function redo() {
        if (isBusy || redoStack.length === 0) return;
        undoStack.push(snapshot());
        restoreSnapshot(redoStack.pop());
        showToast('Đã làm lại ↪');
    }

    function loadImage(idx) {
        if (imageCache[idx]) return Promise.resolve(imageCache[idx]);
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => { imageCache[idx] = img; resolve(img); };
            img.onerror = () => resolve(null);
            img.src = 'data:image/jpeg;base64,' + (images[idx] ? images[idx].data : '');
        });
    }

    // V3 styleditor: the sidebar shows a small preview for EVERY page. The
    // server sends per-page thumbnail data in allImages[idx].data (main canvas
    // only carries the current page), so thumbnails load from there.
    function loadThumbImage(idx) {
        const key = 'thumb:' + idx;
        if (imageCache[key]) return Promise.resolve(imageCache[key]);
        const src = (isStyleditor && allImages[idx] && allImages[idx].data) ? allImages[idx].data : (images[idx] ? images[idx].data : '');
        return new Promise((resolve) => {
            if (!src) { resolve(null); return; }
            const img = new Image();
            img.onload = () => { imageCache[key] = img; resolve(img); };
            img.onerror = () => resolve(null);
            img.src = 'data:image/jpeg;base64,' + src;
        });
    }

    function getBlocks(idx) { return images[idx] ? images[idx].blocks : []; }
    function setBlocks(idx, blocks) { images[idx].blocks = blocks; }
    function getCurrentBlocks() { return getBlocks(currentImageIdx); }

    function sameBbox(a, b) {
        return a && b && a.length === 4 && b.length === 4 &&
            a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    // Clamp + round a bbox to valid image bounds (F3.4/A3.5). Returns null when
    // the bbox is degenerate (x2<=x1 or y2<=y1) after clamping.
    function normalizeBbox(bbox, imgW, imgH) {
        if (!bbox || bbox.length !== 4) return null;
        let [x1, y1, x2, y2] = bbox.map(v => Math.round(Number(v) || 0));
        x1 = clamp(x1, 0, imgW - MIN_BLOCK_SIZE);
        y1 = clamp(y1, 0, imgH - MIN_BLOCK_SIZE);
        x2 = clamp(x2, x1 + MIN_BLOCK_SIZE, imgW);
        y2 = clamp(y2, y1 + MIN_BLOCK_SIZE, imgH);
        if (x2 <= x1 || y2 <= y1) return null;
        return [x1, y1, x2, y2];
    }

    function displayScale() {
        return fitScale * zoomLevel;
    }

    function updateZoomLabel() {
        if (!zoomLabel) return;
        const pct = Math.round(displayScale() * 100);
        zoomLabel.textContent = pct + '%';
    }

    function applyCanvasScale() {
        const img = imageCache[currentImageIdx];
        if (!img) return;
        const scale = displayScale();
        mainCanvas.style.width = Math.max(1, Math.round(img.width * scale)) + 'px';
        mainCanvas.style.height = Math.max(1, Math.round(img.height * scale)) + 'px';
        updateZoomLabel();
    }

    function setZoom(nextZoom, anchorClientX = null, anchorClientY = null) {
        const img = imageCache[currentImageIdx];
        if (!img || !canvasOuter) return;

        const previousZoom = zoomLevel;
        zoomLevel = clamp(nextZoom, MIN_ZOOM, MAX_ZOOM);
        if (Math.abs(previousZoom - zoomLevel) < 0.001) return;

        const outerRect = canvasOuter.getBoundingClientRect();
        const oldRect = mainCanvas.getBoundingClientRect();
        const anchorX = anchorClientX !== null && anchorClientX !== undefined ? anchorClientX : (outerRect.left + outerRect.width / 2);
        const anchorY = anchorClientY !== null && anchorClientY !== undefined ? anchorClientY : (outerRect.top + outerRect.height / 2);
        const relX = oldRect.width > 0 ? clamp((anchorX - oldRect.left) / oldRect.width, 0, 1) : 0.5;
        const relY = oldRect.height > 0 ? clamp((anchorY - oldRect.top) / oldRect.height, 0, 1) : 0.5;

        applyCanvasScale();

        requestAnimationFrame(() => {
            const newRect = mainCanvas.getBoundingClientRect();
            canvasOuter.scrollLeft += (newRect.left + newRect.width * relX) - anchorX;
            canvasOuter.scrollTop += (newRect.top + newRect.height * relY) - anchorY;
        });
    }

    function setActualSize() {
        const target = fitScale > 0 ? 1 / fitScale : 1;
        setZoom(target);
    }

    function requestDraw() {
        if (drawScheduled) return;
        drawScheduled = true;
        requestAnimationFrame(() => {
            drawScheduled = false;
            drawAll();
        });
    }

    function requestThumbnails() {
        if (thumbsScheduled) return;
        thumbsScheduled = true;
        requestAnimationFrame(() => {
            thumbsScheduled = false;
            updateThumbnails();
        });
    }

    // ---- Dirty state (F4) ----
    function markDirty(idx) {
        if (!(isPostrender || isStyleditor) || !images[idx]) return;
        images[idx].dirty = true;
        // V3: the page-local image (idx 0) IS the global page image.
        if (isStyleditor) {
            setDirtyBadge(globalImageIdx, true);
            saveDraftState();
        }
        requestDraw();
        requestThumbnails();
        updateFooterButtons();
    }

    // V3 F4: dirty badges survive page navigation via localStorage (P1 for the
    // whole session, P0 for the current image — spec 2.2).
    function dirtyBadgeKey() { return 'styleditor_dirty_' + sessionId; }
    function readDirtyBadges() {
        try {
            const raw = localStorage.getItem(dirtyBadgeKey());
            const obj = raw ? JSON.parse(raw) : {};
            return (obj && typeof obj === 'object') ? obj : {};
        } catch (_) { return {}; }
    }
    function writeDirtyBadges(obj) {
        try { localStorage.setItem(dirtyBadgeKey(), JSON.stringify(obj)); } catch (_) { /* private mode */ }
    }
    function setDirtyBadge(idx, dirty) {
        const obj = readDirtyBadges();
        const wasDirty = !!obj[idx];
        if (dirty) obj[idx] = true;
        else delete obj[idx];
        writeDirtyBadges(obj);
        if (wasDirty !== dirty) requestThumbnails();
    }
    function clearDirtyBadge(idx) { setDirtyBadge(idx, false); }

    // ---- V3 draft state persistence (chốt captain 8.3 / backend contract):
    // keep each image's un-rendered edits in sessionStorage so page navigation
    // never loses them, and "Lưu tất cả" can render EVERY dirty image
    // sequentially before redirecting. ----
    function draftKey(idx) { return 'styleditor_draft_' + sessionId + '_' + idx; }
    function readDraftState(idx) {
        try {
            const raw = sessionStorage.getItem(draftKey(idx));
            return raw ? JSON.parse(raw) : null;
        } catch (_) { return null; }
    }
    function removeDraftState(idx) {
        try { sessionStorage.removeItem(draftKey(idx)); } catch (_) { /* private mode */ }
    }
    function saveDraftState() {
        if (!isStyleditor) return;
        const img = images[currentImageIdx];
        if (!img) return;
        // Only un-rendered edits need a draft: a clean image must never be
        // re-marked dirty by a stale restore (chốt captain 8.3).
        if (!img.dirty) return;
        const imgEl = imageCache[currentImageIdx];
        const state = {
            blocks: (img.blocks || []).map(b => ({
                text: b.text || '', translated: b.translated || '',
                bbox: b.bbox ? [...b.bbox] : null,
                style: b.style ? { ...b.style } : null
            })),
            eraseRegions: (img.eraseRegions || []).map(r => [...r]),
            deletedRegions: (img.deletedRegions || []).map(r => [...r]),
            previewRects: (img.erasePreviewRects || []).map(r => [...r]),
            strokes: (img.eraseStrokes || []).map(s => ({
                points: (s.points || []).map(pt => [pt[0], pt[1]]), size: s.size, color: s.color
            })),
            strokesPreview: (img.eraseStrokesPreview || []).map(s => ({
                points: (s.points || []).map(pt => [pt[0], pt[1]]), size: s.size, color: s.color
            })),
            w: imgEl ? imgEl.width : 0,
            h: imgEl ? imgEl.height : 0
        };
        try { sessionStorage.setItem(draftKey(globalImageIdx), JSON.stringify(state)); } catch (_) { /* quota */ }
    }
    // A4.10 / §4.2 MERGE RULE: after a render, /styleditor returns
    // render_plan[i].erase_regions so the reloaded editor restores the erase
    // preview (flat fills, R3 approximation) and keeps the monotonic payload.
    // Tolerant: absent field (legacy backend) → empty lists, no regression.
    function loadServerEraseState() {
        const img = images[0];
        if (!img) return;
        const regions = Array.isArray(img.erase_regions) ? img.erase_regions : null;
        if (regions) {
            img.eraseRegions = regions.map(r => [r[0], r[1], r[2], r[3]]);
            img.erasePreviewRects = regions.map(r => [r[0], r[1], r[2], r[3]]);
        }
        // erase_mask (P1, spec 4.2): PNG b64 grayscale (white = erase) — the
        // server accumulates it monotonically, so the client never re-sends it;
        // here we rebuild the eraseLayer preview from it (flat fill sampled at
        // the mask bbox edges — R3 approximation, spec A4.10).
        if (typeof img.erase_mask === 'string' && img.erase_mask) {
            buildEraseMaskLayer(img.erase_mask);
        }
    }

    // Decode the persisted erase mask into an offscreen "eraseMaskLayer" that
    // redrawEraseLayer composites under the region/stroke previews. The layer
    // is static (monotonic server state) and survives undo (preview-only undo).
    function buildEraseMaskLayer(maskB64) {
        const img = imageCache[currentImageIdx];
        if (!img) return;
        const image = new Image();
        image.onload = () => {
            try {
                const maskCanvas = document.createElement('canvas');
                maskCanvas.width = img.width;
                maskCanvas.height = img.height;
                const mg = maskCanvas.getContext('2d');
                mg.drawImage(image, 0, 0, img.width, img.height);
                const data = mg.getImageData(0, 0, img.width, img.height).data;
                // bounding box of the white (erase) area, stride-sampled
                let minX = img.width, minY = img.height, maxX = -1, maxY = -1;
                for (let y = 0; y < img.height; y += 3) {
                    for (let x = 0; x < img.width; x += 3) {
                        const i = (y * img.width + x) * 4;
                        const lum = (data[i] + data[i + 1] + data[i + 2]) / 3;
                        if (lum > 128) {
                            if (x < minX) minX = x;
                            if (x > maxX) maxX = x;
                            if (y < minY) minY = y;
                            if (y > maxY) maxY = y;
                        }
                    }
                }
                if (maxX < minX || maxY < minY) return;
                // flat fill color sampled from the mask bbox edges (R3)
                const color = sampleEdgeColor([minX, minY, maxX + 2, maxY + 2]);
                // color-where-mask via 'destination-in' compositing
                const fillCanvas = document.createElement('canvas');
                fillCanvas.width = img.width;
                fillCanvas.height = img.height;
                const fg = fillCanvas.getContext('2d');
                fg.fillStyle = color;
                fg.fillRect(0, 0, img.width, img.height);
                fg.globalCompositeOperation = 'destination-in';
                fg.drawImage(maskCanvas, 0, 0);
                const layer = document.createElement('canvas');
                layer.width = img.width;
                layer.height = img.height;
                layer.getContext('2d').drawImage(fillCanvas, 0, 0);
                images[currentImageIdx].eraseMaskLayer = layer;
                redrawEraseLayer();
            } catch (_) { /* decode/preview failure — regions still cover P0 */ }
        };
        image.onerror = () => { /* ignore; regions cover the acceptance */ };
        image.src = 'data:image/png;base64,' + maskB64;
    }

    function restoreDraftState() {
        if (!isStyleditor) return;
        const img = images[0];
        const state = readDraftState(globalImageIdx);
        if (!state || !img) return;
        if (Array.isArray(state.blocks)) {
            img.blocks = state.blocks.map(b => ({
                text: b.text || '', translated: b.translated || '',
                bbox: b.bbox ? [...b.bbox] : null,
                style: normalizeStyle(b.style)
            }));
        }
        img.eraseRegions = Array.isArray(state.eraseRegions) ? state.eraseRegions.map(r => [...r]) : [];
        img.deletedRegions = Array.isArray(state.deletedRegions) ? state.deletedRegions.map(r => [...r]) : [];
        img.erasePreviewRects = Array.isArray(state.previewRects) ? state.previewRects.map(r => [...r]) : [];
        img.eraseStrokes = Array.isArray(state.strokes) ? state.strokes : [];
        img.eraseStrokesPreview = Array.isArray(state.strokesPreview) ? state.strokesPreview : [];
        img.dirty = true;
        setDirtyBadge(globalImageIdx, true);
    }

    // ---- Drawing ----
    function drawAll() {
        const img = imageCache[currentImageIdx];
        if (!img) return;

        const blocks = getCurrentBlocks();
        if (mainCanvas.width !== img.width) mainCanvas.width = img.width;
        if (mainCanvas.height !== img.height) mainCanvas.height = img.height;
        ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
        ctx.drawImage(img, 0, 0);
        // V3: user erase layer (rect fills + brush strokes) composites on top
        // of the erased background (spec F4).
        if (isStyleditor && eraseLayer.width > 0) ctx.drawImage(eraseLayer, 0, 0);

        blocks.forEach((block, i) => {
            const bbox = block.bbox;
            if (!bbox || bbox.length !== 4) return;
            const [x1, y1, x2, y2] = bbox;
            const dirty = (isPostrender || isStyleditor) && images[currentImageIdx].dirty;
            const isClean = (isPostrender || isStyleditor) && !dirty;
            const hasTranslated = !!(block.translated || '').trim();
            // V3 A1.6: block with empty translation renders as a dashed "—" chip.
            const isEmptyStyled = isStyleditor && !hasTranslated;

            if (i === selectedBlockIdx) {
                ctx.strokeStyle = '#00e676'; ctx.lineWidth = 2.5;
                ctx.fillStyle = 'rgba(0,230,118,0.15)';
            } else if (currentTool === 'delete' && i === hoveredDeleteIdx) {
                ctx.strokeStyle = '#ff1744'; ctx.lineWidth = 2.5;
                ctx.fillStyle = 'rgba(255,23,68,0.2)';
            } else if (isEmptyStyled) {
                // V3: empty translation = dashed neutral border, no fill.
                ctx.strokeStyle = '#b0a0bd'; ctx.lineWidth = 1.5;
                ctx.fillStyle = 'rgba(176,160,189,0.05)';
            } else if (dirty) {
                // post-render/styleditor: dirty = dashed orange (not rendered yet)
                ctx.strokeStyle = '#ff9100'; ctx.lineWidth = 1.5;
                ctx.fillStyle = 'rgba(255,145,0,0.08)';
            } else if (isClean) {
                // post-render/styleditor: clean = solid green (matches last render)
                ctx.strokeStyle = '#00e676'; ctx.lineWidth = 1.5;
                ctx.fillStyle = 'rgba(0,230,118,0.06)';
            } else {
                ctx.strokeStyle = '#ff9100'; ctx.lineWidth = 1.5;
                ctx.fillStyle = 'rgba(255,145,0,0.08)';
            }
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            if (dirty && i !== selectedBlockIdx && !(currentTool === 'delete' && i === hoveredDeleteIdx)) {
                ctx.setLineDash([5, 4]);
            }
            if (isEmptyStyled) ctx.setLineDash([5, 4]);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.setLineDash([]);

            // V3 F3.1/F6: draw the translated text LIVE with its style.
            if (isStyleditor && hasTranslated) drawStyledBlockText(block, i);

            const label = ((isPostrender || isStyleditor) ? (block.translated || block.text || '') : (block.text || '')).substring(0, 12);
            const labelText = label || (isStyleditor ? '—' : '');
            if (labelText) {
                ctx.font = 'bold 11px sans-serif';
                const lw = ctx.measureText(labelText).width + 8;
                const lh = 18;
                let ly = y1 - lh - 2; if (ly < 0) ly = y1 + 2;
                // Label chip: selected = green; dirty = orange; clean = dark
                // green; empty styled block = neutral; pre-render keeps the
                // incumbent orange for non-selected blocks (no visual regress).
                ctx.fillStyle = (i === selectedBlockIdx) ? '#00e676'
                    : (isEmptyStyled ? '#b0a0bd'
                        : (dirty ? '#ff9100' : ((isPostrender || isStyleditor) ? '#00a152' : '#ff9100')));
                ctx.fillRect(x1, ly, lw, lh);
                ctx.fillStyle = '#000';
                ctx.fillText(labelText, x1 + 4, ly + 13);
            }
            if (i === selectedBlockIdx) drawHandles(x1, y1, x2, y2);
        });

        if (isDrawing && drawStart && drawEnd) {
            const x = Math.min(drawStart.x, drawEnd.x);
            const y = Math.min(drawStart.y, drawEnd.y);
            const w = Math.abs(drawEnd.x - drawStart.x);
            const h = Math.abs(drawEnd.y - drawStart.y);
            if (currentTool === 'erase-rect') {
                // V3 F4.1: rect erase preview (dashed magenta — reads as "remove")
                ctx.strokeStyle = '#e91e63'; ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.fillStyle = 'rgba(233,30,99,0.08)';
            } else {
                ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.fillStyle = 'rgba(0,229,255,0.1)';
            }
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
        }

        // V3 F4: brush cursor ring shows the real brush diameter.
        if (isStyleditor && currentTool === 'erase-brush' && lastPointerPos) {
            const r = brushSize / 2;
            ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(lastPointerPos.x, lastPointerPos.y, r + 1, 0, Math.PI * 2); ctx.stroke();
            ctx.strokeStyle = '#5E1675'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(lastPointerPos.x, lastPointerPos.y, r, 0, Math.PI * 2); ctx.stroke();
        }
    }

    // F2: draw the 8 handles; the visual handle stays small (6px) but gets a
    // 1px white outline so it reads on dark manga panels.
    function drawHandles(x1, y1, x2, y2) {
        const s = 6;
        const points = [
            [x1, y1], [x2, y1], [x1, y2], [x2, y2],
            [(x1 + x2) / 2, y1], [(x1 + x2) / 2, y2],
            [x1, (y1 + y2) / 2], [x2, (y1 + y2) / 2]
        ];
        ctx.fillStyle = '#00e676';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        points.forEach(([cx, cy]) => {
            ctx.fillRect(cx - s / 2, cy - s / 2, s, s);
            ctx.strokeRect(cx - s / 2 - 0.5, cy - s / 2 - 0.5, s + 1, s + 1);
        });
    }

    // F2: screen-space position of a handle (CSS px relative to the canvas).
    function handleScreenPos(bbox, h) {
        const scale = displayScale();
        const w = bbox[2] - bbox[0];
        const hh = bbox[3] - bbox[1];
        return { x: (bbox[0] + w * h.fx) * scale, y: (bbox[1] + hh * h.fy) * scale };
    }

    // F2 hit-test. cx/cy are CSS px relative to the canvas element.
    // - mouse: distance ≤ HANDLE_MOUSE_RADIUS (zoom-independent).
    // - touch: invisible hit area ≥44×44 CSS px around each handle; when the
    //   areas overlap (tiny bbox), pick the handle whose centre is nearest and
    //   prefer corners over edges (captain decision, spec A2.6).
    function hitTestHandle(cx, cy, pointerType) {
        if (selectedBlockIdx < 0) return null;
        const bbox = getCurrentBlocks()[selectedBlockIdx].bbox;
        if (!bbox || bbox.length !== 4) return null;
        const rect = mainCanvas.getBoundingClientRect();
        const canvasW = rect.width, canvasH = rect.height;
        const isTouch = pointerType === 'touch' || pointerType === 'pen';

        if (!isTouch) {
            let best = null, bestDist = Infinity;
            for (const h of HANDLES) {
                const p = handleScreenPos(bbox, h);
                const d = Math.hypot(cx - p.x, cy - p.y);
                if (d <= HANDLE_MOUSE_RADIUS && d < bestDist) { bestDist = d; best = h; }
            }
            return best;
        }

        let best = null, bestDist = Infinity;
        for (const h of HANDLES) {
            const p = handleScreenPos(bbox, h);
            const half = HANDLE_TOUCH_SIZE / 2;
            const r = {
                left: clamp(p.x - half, 0, canvasW),
                top: clamp(p.y - half, 0, canvasH),
                right: clamp(p.x + half, 0, canvasW),
                bottom: clamp(p.y + half, 0, canvasH)
            };
            if (cx >= r.left && cx <= r.right && cy >= r.top && cy <= r.bottom) {
                const d = Math.hypot(cx - p.x, cy - p.y);
                // Corner-first: a corner beats an edge whenever both are
                // candidates; among equals, the nearest centre wins.
                if (h.corner) {
                    if (!best || !best.corner || d < bestDist - 0.001) { best = h; bestDist = d; }
                } else {
                    if (best && best.corner) continue;
                    if (!best || d < bestDist - 0.001) { best = h; bestDist = d; }
                }
            }
        }
        return best;
    }

    // F2: apply the resize for one pointer position (image coords).
    function applyResize(handleId, px, py) {
        const bbox = getCurrentBlocks()[selectedBlockIdx].bbox;
        const start = resizeStartBbox;
        if (!bbox || !start) return;
        const W = mainCanvas.width, H = mainCanvas.height;
        const MIN = MIN_BLOCK_SIZE;
        const [sx1, sy1, sx2, sy2] = start;
        const sw = sx2 - sx1, sh = sy2 - sy1;
        const keepAspect = resizeShiftHeld;

        let x1 = sx1, y1 = sy1, x2 = sx2, y2 = sy2;

        if (keepAspect && sw > 0 && sh > 0) {
            // Shift = keep aspect ratio (P1, spec F2.5): anchor the opposite
            // corner and size the box to fit the dragged corner's direction.
            const anchorX = handleId.indexOf('e') >= 0 ? sx1 : sx2;
            const anchorY = handleId.indexOf('s') >= 0 ? sy1 : sy2;
            const dx = px - anchorX, dy = py - anchorY;
            const aspect = sw / sh;
            let nw = Math.abs(dx), nh = Math.abs(dy);
            if (nw / nh > aspect) nh = nw / aspect; else nw = nh * aspect;
            nw = clamp(Math.round(nw), MIN, W);
            nh = clamp(Math.round(nh), MIN, H);
            const dirX = dx < 0 ? -1 : 1;
            const dirY = dy < 0 ? -1 : 1;
            if (handleId.indexOf('e') >= 0) x2 = clamp(anchorX + nw * dirX, anchorX + MIN, W);
            else x1 = clamp(anchorX - nw * dirX, 0, anchorX - MIN);
            if (handleId.indexOf('s') >= 0) y2 = clamp(anchorY + nh * dirY, anchorY + MIN, H);
            else y1 = clamp(anchorY - nh * dirY, 0, anchorY - MIN);
            // Keep the exact aspect once more on the final numbers.
            const fw = (x2 - x1), fh = (y2 - y1);
            if (fw / fh > aspect) { const t = Math.round(fw / aspect); y2 = y1 + t; }
            else { const t = Math.round(fh * aspect); x2 = x1 + t; }
        } else {
            if (handleId.indexOf('w') >= 0) x1 = clamp(Math.round(px), 0, x2 - MIN);
            if (handleId.indexOf('e') >= 0) x2 = clamp(Math.round(px), x1 + MIN, W);
            if (handleId.indexOf('n') >= 0) y1 = clamp(Math.round(py), 0, y2 - MIN);
            if (handleId.indexOf('s') >= 0) y2 = clamp(Math.round(py), y1 + MIN, H);
        }

        bbox[0] = x1; bbox[1] = y1; bbox[2] = x2; bbox[3] = y2;
        requestDraw();
    }

    function getCanvasCoords(e) {
        const rect = mainCanvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) * (mainCanvas.width / rect.width),
            y: (e.clientY - rect.top) * (mainCanvas.height / rect.height)
        };
    }

    function findBlockAt(x, y) {
        const blocks = getCurrentBlocks();
        for (let i = blocks.length - 1; i >= 0; i--) {
            const b = blocks[i].bbox;
            if (!b || b.length !== 4) continue;
            if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) return i;
        }
        return -1;
    }

    // ---- Auto OCR (pre-render only) ----
    function ocrNewBlock(idx, options = {}) {
        const recordUndo = options.recordUndo !== undefined ? options.recordUndo : true;
        const block = getCurrentBlocks()[idx];
        if (!block || !block.bbox || isOcrPending || isPostrender) return;
        isOcrPending = true;
        ocrStatus.innerHTML = '⏳ Đang OCR...';
        ocrStatus.style.display = 'block';

        const form = new FormData();
        form.append('session_id', sessionId);
        form.append('image_idx', currentImageIdx);
        form.append('x1', block.bbox[0]);
        form.append('y1', block.bbox[1]);
        form.append('x2', block.bbox[2]);
        form.append('y2', block.bbox[3]);

        fetch('/ocr-region', { method: 'POST', body: form })
            .then(r => r.json())
            .then(data => {
                if (data.text) {
                    const cleaned = cleanOcrText(data.text);
                    if (recordUndo && cleaned !== (block.text || '')) {
                        pushUndo();
                    }
                    block.text = cleaned;
                    updateBlockEditor(idx);
                    requestDraw();
                    requestThumbnails();
                    ocrStatus.innerHTML = '✅ ' + cleaned.substring(0, 30);
                } else {
                    ocrStatus.innerHTML = '⚠️ Không nhận được text';
                }
            })
            .catch(() => { ocrStatus.innerHTML = '❌ Lỗi OCR'; })
            .finally(() => {
                isOcrPending = false;
                setTimeout(() => { ocrStatus.style.display = 'none'; }, 3000);
            });
    }

    // ---- Pointer handling (mouse + touch via Pointer Events) ----
    mainCanvas.addEventListener('pointerdown', (e) => {
        if (isBusy) return;
        if (e.button === 1 || isSpaceDown) {
            startPan(e);
            return;
        }

        const rect = mainCanvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const pos = getCanvasCoords(e);

        if (currentTool === 'add') {
            isDrawing = true; drawStart = pos; drawEnd = pos;
            selectedBlockIdx = -1; updateBlockEditor(-1);
            return;
        }

        if (currentTool === 'delete') {
            const hitIdx = findBlockAt(pos.x, pos.y);
            // F1: remember the press; deletion only happens on a click mouseup.
            deleteDown = { clientX: e.clientX, clientY: e.clientY, hitIdx: hitIdx };
            return;
        }

        // V3 F4: background erase tools
        if (currentTool === 'erase-rect') {
            isDrawing = true; drawStart = pos; drawEnd = pos;
            selectedBlockIdx = -1; updateBlockEditor(-1);
            return;
        }
        if (currentTool === 'erase-brush') {
            isErasing = true;
            strokePoints = [pos];
            strokeColor = sampleColorAt(pos);
            paintBrushPoint(pos);
            lastPointerPos = pos;
            try { mainCanvas.setPointerCapture(e.pointerId); } catch (_) { /* noop */ }
            return;
        }

        // select tool — handle resize has priority over move (F3.1)
        if (selectedBlockIdx >= 0) {
            const handle = hitTestHandle(sx, sy, e.pointerType);
            if (handle) {
                const bbox = getCurrentBlocks()[selectedBlockIdx].bbox;
                resizeHandle = handle.id;
                resizeStartBbox = [...bbox];
                resizeStartSnapshot = snapshot();
                resizeShiftHeld = e.shiftKey;
                try { mainCanvas.setPointerCapture(e.pointerId); } catch (_) { /* noop */ }
                e.preventDefault();
                return;
            }
        }

        const hitIdx = findBlockAt(pos.x, pos.y);
        if (hitIdx >= 0) {
            selectedBlockIdx = hitIdx;
            isDragging = true; dragBlockIdx = hitIdx;
            const bbox = getCurrentBlocks()[hitIdx].bbox;
            dragStartBbox = [...bbox];
            dragStartSnapshot = snapshot();
            dragOffset = { x: pos.x - bbox[0], y: pos.y - bbox[1] };
            updateBlockEditor(hitIdx);
            try { mainCanvas.setPointerCapture(e.pointerId); } catch (_) { /* noop */ }
        } else {
            selectedBlockIdx = -1; updateBlockEditor(-1);
        }
        requestDraw();
    });

    mainCanvas.addEventListener('pointermove', (e) => {
        const rect = mainCanvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const pos = getCanvasCoords(e);

        if (resizeHandle && selectedBlockIdx >= 0) {
            resizeShiftHeld = e.shiftKey;
            applyResize(resizeHandle, pos.x, pos.y);
            return;
        }
        if (isDragging && dragBlockIdx >= 0) {
            const bbox = getCurrentBlocks()[dragBlockIdx].bbox;
            const w = bbox[2] - bbox[0], h = bbox[3] - bbox[1];
            let nx = Math.max(0, Math.min(pos.x - dragOffset.x, mainCanvas.width - w));
            let ny = Math.max(0, Math.min(pos.y - dragOffset.y, mainCanvas.height - h));
            bbox[0] = Math.round(nx); bbox[1] = Math.round(ny);
            bbox[2] = Math.round(nx + w); bbox[3] = Math.round(ny + h);
            requestDraw(); return;
        }
        if (isDrawing && drawStart) { drawEnd = pos; requestDraw(); }
        if (isErasing && currentTool === 'erase-brush') {
            strokePoints.push(pos);
            paintBrushPoint(pos);
            return;
        }
        if (currentTool === 'erase-rect' || currentTool === 'erase-brush') {
            // F4.3: live brush cursor ring (redraw only when it actually moved)
            if (!lastPointerPos || Math.hypot(pos.x - lastPointerPos.x, pos.y - lastPointerPos.y) >= 1) {
                lastPointerPos = pos;
                requestDraw();
            }
            mainCanvas.style.cursor = 'crosshair';
        } else if (currentTool === 'delete') {
            const hIdx = findBlockAt(pos.x, pos.y);
            if (hoveredDeleteIdx !== hIdx) { hoveredDeleteIdx = hIdx; requestDraw(); }
            mainCanvas.style.cursor = hIdx >= 0 ? 'pointer' : 'crosshair';
        } else if (currentTool === 'add') {
            mainCanvas.style.cursor = 'crosshair';
        } else if (selectedBlockIdx >= 0) {
            const handle = hitTestHandle(sx, sy, e.pointerType);
            mainCanvas.style.cursor = handle ? handle.cursor : (findBlockAt(pos.x, pos.y) >= 0 ? 'move' : 'default');
        } else {
            mainCanvas.style.cursor = findBlockAt(pos.x, pos.y) >= 0 ? 'move' : 'default';
        }
    });

    function finishDrag() {
        if (isDragging && dragBlockIdx >= 0 && dragStartBbox) {
            const currentBbox = getCurrentBlocks()[dragBlockIdx].bbox;
            if (!sameBbox(currentBbox, dragStartBbox)) {
                pushUndo(dragStartSnapshot || snapshot());
                if (dragBlockIdx === selectedBlockIdx) {
                    refreshBlockEditorValues();
                }
                markDirty(currentImageIdx);
                requestThumbnails();
            }
            dragStartBbox = null;
            dragStartSnapshot = null;
        }
        if (resizeHandle && selectedBlockIdx >= 0 && resizeStartBbox) {
            const currentBbox = getCurrentBlocks()[selectedBlockIdx].bbox;
            if (!sameBbox(currentBbox, resizeStartBbox)) {
                // F2.5/A2.5: one undo step for the whole resize gesture
                pushUndo(resizeStartSnapshot || snapshot());
                refreshBlockEditorValues();
                markDirty(currentImageIdx);
                requestThumbnails();
            }
            resizeStartBbox = null;
            resizeStartSnapshot = null;
        }
        resizeHandle = null;
    }

    mainCanvas.addEventListener('pointerup', (e) => {
        // F1: delete only on a real click (≤ 6px screen movement)
        if (currentTool === 'delete' && deleteDown) {
            const dx = e.clientX - deleteDown.clientX;
            const dy = e.clientY - deleteDown.clientY;
            if (deleteDown.hitIdx >= 0 && Math.hypot(dx, dy) <= DELETE_SLOP) {
                deleteBlockAt(deleteDown.hitIdx, { fromClick: true });
            }
            deleteDown = null;
        }

        // V3 F4.1/A4.1: rect erase — clamp to image edges, ignore < 4×4 px
        if (currentTool === 'erase-rect' && isDrawing && drawStart && drawEnd) {
            const W = mainCanvas.width, H = mainCanvas.height;
            const x1 = clamp(Math.round(Math.min(drawStart.x, drawEnd.x)), 0, W);
            const y1 = clamp(Math.round(Math.min(drawStart.y, drawEnd.y)), 0, H);
            const x2 = clamp(Math.round(Math.max(drawStart.x, drawEnd.x)), 0, W);
            const y2 = clamp(Math.round(Math.max(drawStart.y, drawEnd.y)), 0, H);
            if (x2 - x1 >= 4 && y2 - y1 >= 4) {
                pushUndo();
                const img = images[currentImageIdx];
                const region = [x1, y1, x2, y2];
                img.eraseRegions.push(region);
                img.erasePreviewRects.push(region);
                markDirty(currentImageIdx);
                redrawEraseLayer();
            } else {
                // A4.9 (P1): tiny rects never add an erase region.
                showToast('Vùng xoá quá nhỏ (tối thiểu 4×4 px)', { variant: 'error', duration: 2500 });
            }
            isDrawing = false; drawStart = null; drawEnd = null;
        }

        // V3 F4.2/A4.2: brush stroke — bbox of the stroke joins eraseRegions
        if (currentTool === 'erase-brush' && isErasing) {
            isErasing = false;
            const bbox = strokeBBox();
            if (bbox && (bbox[2] - bbox[0]) >= 2 && (bbox[3] - bbox[1]) >= 2) {
                pushUndo();
                const img = images[currentImageIdx];
                const pts = strokePoints.map(pt => [Math.round(pt.x), Math.round(pt.y)]);
                img.eraseRegions.push(bbox);
                img.eraseStrokes.push({ points: pts, size: brushSize, color: strokeColor });
                img.eraseStrokesPreview.push({ points: pts, size: brushSize, color: strokeColor });
                markDirty(currentImageIdx);
                requestDraw();
            } else {
                // A4.6: strokes < 2px are dropped; rebuild the layer from the
                // committed preview lists (removes the live-painted dots).
                redrawEraseLayer();
            }
            strokePoints = []; strokeColor = null;
        }

        finishDrag();
        if (isDrawing && drawStart && drawEnd) {
            const x1 = Math.round(Math.min(drawStart.x, drawEnd.x));
            const y1 = Math.round(Math.min(drawStart.y, drawEnd.y));
            const x2 = Math.round(Math.max(drawStart.x, drawEnd.x));
            const y2 = Math.round(Math.max(drawStart.y, drawEnd.y));
            if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
                pushUndo();
                const blocks = getCurrentBlocks();
                const newBlock = { text: '', translated: '', bbox: [x1, y1, x2, y2] };
                blocks.push(newBlock);
                setBlocks(currentImageIdx, blocks);
                const newIdx = blocks.length - 1;
                selectedBlockIdx = newIdx;
                updateBlockEditor(newIdx);
                ocrNewBlock(newIdx, { recordUndo: false });
            }
        }
        isDrawing = false; drawStart = null; drawEnd = null;
        isDragging = false; dragBlockIdx = -1; dragOffset = null;
        try { mainCanvas.releasePointerCapture(e.pointerId); } catch (_) { /* noop */ }
        requestDraw(); requestThumbnails();
    });

    mainCanvas.addEventListener('pointercancel', () => {
        deleteDown = null;
        if (isErasing && currentTool === 'erase-brush') {
            isErasing = false;
            strokePoints = []; strokeColor = null;
            redrawEraseLayer();
        }
        finishDrag();
        isDrawing = false; drawStart = null; drawEnd = null;
        isDragging = false; dragBlockIdx = -1; dragOffset = null;
        resizeHandle = null;
        requestDraw(); requestThumbnails();
    });

    mainCanvas.addEventListener('pointerleave', (e) => {
        if (e.pointerType === 'mouse') {
            hoveredDeleteIdx = -1;
            if (!isDragging && !resizeHandle) requestDraw();
        }
    });

    function startPan(e) {
        if (!canvasOuter) return;
        isPanning = true;
        panStart = {
            x: e.clientX,
            y: e.clientY,
            scrollLeft: canvasOuter.scrollLeft,
            scrollTop: canvasOuter.scrollTop
        };
        mainCanvas.classList.add('panning');
        e.preventDefault();
    }

    function stopPan() {
        isPanning = false;
        panStart = null;
        mainCanvas.classList.remove('panning');
    }

    window.addEventListener('pointermove', (e) => {
        if (!isPanning || !panStart || !canvasOuter) return;
        canvasOuter.scrollLeft = panStart.scrollLeft - (e.clientX - panStart.x);
        canvasOuter.scrollTop = panStart.scrollTop - (e.clientY - panStart.y);
    });

    window.addEventListener('pointerup', stopPan);

    if (canvasOuter) {
        canvasOuter.addEventListener('wheel', (e) => {
            if (!(e.ctrlKey || e.metaKey)) return;
            e.preventDefault();
            const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
            setZoom(zoomLevel * factor, e.clientX, e.clientY);
        }, { passive: false });
    }

    // ---- Nudge (F3, P1-but-cheap) ----
    function nudgeBlock(dx, dy, recordUndo) {
        if (selectedBlockIdx < 0 || isBusy) return;
        const block = getCurrentBlocks()[selectedBlockIdx];
        const bbox = block.bbox;
        if (!bbox || bbox.length !== 4) return;
        if (recordUndo) pushUndo();
        const W = mainCanvas.width, H = mainCanvas.height;
        const w = bbox[2] - bbox[0], h = bbox[3] - bbox[1];
        bbox[0] = clamp(bbox[0] + dx, 0, W - w);
        bbox[1] = clamp(bbox[1] + dy, 0, H - h);
        bbox[2] = bbox[0] + w;
        bbox[3] = bbox[1] + h;
        requestDraw(); requestThumbnails();
        refreshBlockEditorValues();
        markDirty(currentImageIdx);
    }

    // ---- Keyboard (F8) ----
    document.addEventListener('keydown', (e) => {
        const tag = e.target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target.isContentEditable) return;

        if (e.code === 'Space') {
            e.preventDefault();
            isSpaceDown = true;
            mainCanvas.style.cursor = 'grab';
            return;
        }

        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
            e.preventDefault();
            if (e.shiftKey) { redo(); } else { undo(); }
            return;
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
            e.preventDefault(); redo(); return;
        }

        const hasSelection = selectedBlockIdx >= 0;
        // F3/A3.3: Shift+Arrow nudges ±10 px, plain arrow ±1 px.
        const nudgeStep = e.shiftKey ? 10 : 1;
        switch (e.key.toLowerCase()) {
            case '+':
            case '=': e.preventDefault(); setZoom(zoomLevel * ZOOM_STEP); break;
            case '-': e.preventDefault(); setZoom(zoomLevel / ZOOM_STEP); break;
            case '0':
            case 'f': e.preventDefault(); zoomLevel = 1; fitCanvas(); break;
            case '1': e.preventDefault(); setActualSize(); break;
            case 's': setTool('select'); break;
            case 'a': if (!isPostrender && !isStyleditor) setTool('add'); break;
            case 'd': setTool('delete'); break;
            case 'e':
                // V3 F11: E cycles select → erase-rect → erase-brush → select
                if (isStyleditor) {
                    setTool(currentTool === 'select' ? 'erase-rect'
                        : (currentTool === 'erase-rect' ? 'erase-brush' : 'select'));
                }
                break;
            case 'b':
                if (isStyleditor && hasSelection) toggleStyle('bold');
                break;
            case 'i':
                if (isStyleditor && hasSelection) toggleStyle('italic');
                break;
            case 'l':
                if (isStyleditor && hasSelection) setBlockStyle({ align: 'left' });
                break;
            case 'c':
                if (isStyleditor && hasSelection) setBlockStyle({ align: 'center' });
                break;
            case 'r':
                if (isStyleditor && hasSelection) setBlockStyle({ align: 'right' });
                break;
            case 'arrowleft':
                e.preventDefault();
                if (hasSelection) nudgeBlock(-nudgeStep, 0, !e.repeat);
                else if (navImageIdx() > 0) switchImage(navImageIdx() - 1);
                break;
            case 'arrowright':
                e.preventDefault();
                if (hasSelection) nudgeBlock(nudgeStep, 0, !e.repeat);
                else if (navImageIdx() < allImages.length - 1) switchImage(navImageIdx() + 1);
                break;
            case 'arrowup':
                e.preventDefault();
                if (hasSelection) nudgeBlock(0, -nudgeStep, !e.repeat);
                break;
            case 'arrowdown':
                e.preventDefault();
                if (hasSelection) nudgeBlock(0, nudgeStep, !e.repeat);
                break;
            case 'escape':
                if (isStyleditor && (currentTool === 'erase-rect' || currentTool === 'erase-brush')) {
                    // V3 F11: Esc exits the erase tool back to Select
                    if (isErasing) { isErasing = false; strokePoints = []; strokeColor = null; redrawEraseLayer(); }
                    isDrawing = false; drawStart = null; drawEnd = null;
                    setTool('select');
                    break;
                }
                if (currentTool === 'delete') { setTool('select'); break; }
                selectedBlockIdx = -1; updateBlockEditor(-1);
                isDrawing = false; drawStart = null; drawEnd = null;
                requestDraw();
                break;
            case 'delete':
                // F8/A1.5: ONLY Delete deletes the selected block. Backspace is
                // deliberately excluded — in Firefox it also triggers
                // browser-back, which would lose work (reviewer P1-1).
                if (selectedBlockIdx >= 0 && currentTool === 'select') { pushUndo(); deleteBlockAt(selectedBlockIdx); }
                break;
            case '[':
                e.preventDefault(); cycleBlock(-1); break;
            case ']':
                e.preventDefault(); cycleBlock(1); break;
        }
    });

    document.addEventListener('keyup', (e) => {
        if (e.code !== 'Space') return;
        isSpaceDown = false;
        if (!isPanning) {
            mainCanvas.style.cursor = (currentTool === 'add' || currentTool === 'delete') ? 'crosshair' : 'default';
        }
    });

    function cycleBlock(dir) {
        const blocks = getCurrentBlocks();
        if (!blocks.length) return;
        const next = selectedBlockIdx < 0
            ? (dir > 0 ? 0 : blocks.length - 1)
            : (selectedBlockIdx + dir + blocks.length) % blocks.length;
        selectedBlockIdx = next;
        updateBlockEditor(next);
        requestDraw();
    }

    // ---- Delete (F1) ----
    function deleteBlockAt(idx, opts) {
        const blocks = getCurrentBlocks();
        if (idx < 0 || idx >= blocks.length) return;
        pushUndo();
        const bbox = blocks[idx].bbox ? [...blocks[idx].bbox] : null;
        blocks.splice(idx, 1);
        setBlocks(currentImageIdx, blocks);
        if (selectedBlockIdx === idx) { selectedBlockIdx = -1; updateBlockEditor(-1); }
        else if (selectedBlockIdx > idx) { selectedBlockIdx--; updateBlockEditor(selectedBlockIdx); }
        // F1.6/post-render: remember the erased region so re-render inpaints it.
        if (isPostrender && bbox) {
            images[currentImageIdx].deletedRegions.push(bbox);
            markDirty(currentImageIdx);
        }
        // V3 F4.7: delete merges into the monotonic eraseRegions list too, so
        // the region stays erased across re-renders (spec F4.5).
        if (isStyleditor && bbox) {
            const img = images[currentImageIdx];
            img.deletedRegions.push(bbox);
            img.eraseRegions.push(bbox);
            img.erasePreviewRects.push(bbox);
            markDirty(currentImageIdx);
            redrawEraseLayer();
        }
        requestDraw(); updateThumbnails(); updateFooterButtons();

        showToast('🗑️ Đã xoá bóng thoại', {
            variant: 'undo',
            duration: 4000,
            actionLabel: 'Hoàn tác',
            onAction: function () { undo(); }
        });
    }

    function setTool(tool) {
        currentTool = tool;
        document.querySelectorAll('.tool-btn').forEach(b => {
            b.classList.toggle('active', b.id === 'tool-' + tool);
            if (b.hasAttribute('aria-pressed')) b.setAttribute('aria-pressed', String(b.id === 'tool-' + tool));
        });
        if (tool !== 'select') { selectedBlockIdx = -1; updateBlockEditor(-1); }
        const eraseTool = (tool === 'erase-rect' || tool === 'erase-brush');
        if (!eraseTool) lastPointerPos = null;
        mainCanvas.style.cursor = (tool === 'add' || tool === 'delete' || eraseTool) ? 'crosshair' : 'default';
        hoveredDeleteIdx = -1;
        updateHints();
        requestDraw();
    }

    // ---- Hint bar (F1.4, F3.2) ----
    function updateHints() {
        if (!hintsBar) return;
        let msg = '';
        if (currentTool === 'erase-rect') {
            msg = '▭ Kéo trên ảnh để xoá vùng hình chữ nhật (text gốc/SFX còn sót) · vùng nhỏ hơn 4×4px bị bỏ qua · ↩ Undo chỉ khôi phục hiển thị, vùng xoá vẫn render sạch · Esc để thoát';
        } else if (currentTool === 'erase-brush') {
            msg = '🖌 Vẽ tự do để xoá text gốc/SFX còn sót · Cỡ cọ ' + brushSize + 'px (đổi ở toolbar) · ↩ Undo chỉ khôi phục hiển thị, vùng xoá vẫn render sạch · Esc để thoát';
        } else if (currentTool === 'delete') {
            msg = isPostrender
                ? '🖱️ Nhấp vào bóng thoại để xoá (vùng đó sẽ được xoá nền, không render chữ) · kéo lướt qua sẽ không xoá · Esc để thoát'
                : '🖱️ Nhấp vào bóng thoại để xoá (text gốc sẽ giữ nguyên trên ảnh kết quả) · kéo lướt qua sẽ không xoá · Esc để thoát';
        } else if (currentTool === 'select' && selectedBlockIdx >= 0) {
            msg = '⌨️ Dùng ←→↑↓ để di chuyển 1px · Giữ Shift = 10px · Kéo cạnh/góc để resize';
        } else if (currentTool === 'add') {
            msg = '✏️ Kéo trên ảnh để vẽ bóng thoại mới · Esc để huỷ';
        }
        hintsBar.classList.toggle('show', !!msg);
        hintsBar.textContent = msg;
    }

    // ---- Toast (F1.3, F7) ----
    function showToast(msg, opts) {
        if (!toastEl) return;
        const o = opts || {};
        const variant = o.variant || '';
        const duration = o.duration || 2500;
        const actionLabel = o.actionLabel || '';
        const onAction = o.onAction || null;
        toastEl.className = 'toast show' + (variant ? ' toast--' + variant : '');
        toastEl.setAttribute('role', variant === 'error' ? 'alert' : 'status');
        toastEl.setAttribute('aria-live', variant === 'error' ? 'assertive' : 'polite');
        toastEl.innerHTML = '';
        const span = document.createElement('span');
        span.textContent = msg;
        toastEl.appendChild(span);
        if (actionLabel && onAction) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'toast-action';
            btn.textContent = actionLabel;
            btn.addEventListener('click', function () {
                clearTimeout(toastEl._t);
                toastEl.classList.remove('show');
                onAction();
            });
            toastEl.appendChild(btn);
        }
        clearTimeout(toastEl._t);
        toastEl._t = setTimeout(() => toastEl.classList.remove('show'), duration);
    }

    // ---- Re-render (F5, F7) ----
    function currentBlocksPayload() {
        const img = images[currentImageIdx];
        return (img.blocks || []).map(b => {
            const item = {
                text: b.text || '',
                translated: b.translated || '',
                bbox: b.bbox ? [...b.bbox] : null
            };
            // V3: style rides along (spec 4.6/P1 pass-through for postrender too).
            if (b.style) item.style = { ...b.style };
            return item;
        });
    }

    function setBusy(busy, busyLabel) {
        isBusy = busy;
        const one = document.getElementById('btn-rerender-one');
        if (one) {
            if (busy && busyLabel) {
                one.dataset.origLabel = one.dataset.origLabel || one.textContent;
                one.textContent = busyLabel;
            } else if (!busy && one.dataset.origLabel) {
                one.textContent = one.dataset.origLabel;
            }
        }
        updateFooterButtons();
        const eraseTool = (currentTool === 'erase-rect' || currentTool === 'erase-brush');
        mainCanvas.style.cursor = busy ? 'wait' : ((currentTool === 'add' || currentTool === 'delete' || eraseTool) ? 'crosshair' : 'default');
    }

    function updateFooterButtons() {
        if (!isPostrender && !isStyleditor) return;
        const one = document.getElementById('btn-rerender-one');
        const all = document.getElementById('btn-save-all');
        const img = images[currentImageIdx];
        if (one) one.disabled = isBusy || !img || !img.dirty;
        if (all) all.disabled = isBusy;
    }

    function rerenderCurrentImage(opts) {
        const o = opts || {};
        const navigateAfter = !!o.navigateAfter;
        const busyLabel = o.busyLabel || '⏳ Đang render…';
        if (isBusy) return;
        const img = images[currentImageIdx];
        if (!img) return;

        const payload = {
            session_id: sessionId,
            image_idx: String(globalImageIdx),
            blocks_json: JSON.stringify(currentBlocksPayload()),
            deleted_regions_json: JSON.stringify((img.deletedRegions || []).map(r => [...r]))
        };
        if (isStyleditor) {
            // V3 F7: canonical erase payload — monotonic eraseRegions + mask.
            payload.erase_regions_json = JSON.stringify((img.eraseRegions || []).map(r => [...r]));
            const mask = buildEraseMask();
            if (mask) payload.erase_mask = mask;
        }

        const run = () => {
            setBusy(true, busyLabel);
            const form = new FormData();
            for (const k in payload) form.append(k, payload[k]);
            fetch('/re-render-image', { method: 'POST', body: form })
                .then(async r => {
                    const data = await r.json().catch(() => ({}));
                    if (!r.ok) {
                        const err = new Error(data.error || ('HTTP ' + r.status));
                        err.status = r.status;
                        throw err;
                    }
                    return data;
                })
                .then(data => {
                    if (isStyleditor) {
                        // V3 1.5: editor ALWAYS keeps the erased background +
                        // live text layer (never switches to the baked image).
                        // Adopt server-normalized blocks (bbox/style), clear dirty.
                        img.blocks = (data.blocks || []).map(b => ({
                            text: b.text || '',
                            translated: b.translated || '',
                            bbox: b.bbox ? [...b.bbox] : null,
                            style: b.style ? { ...b.style } : null
                        }));
                        img.dirty = false;
                        clearDirtyBadge(globalImageIdx);
                        removeDraftState(globalImageIdx);
                        selectedBlockIdx = -1;
                        updateBlockEditor(-1);
                        requestDraw(); updateThumbnails(); updateFooterButtons();
                    } else {
                        img.data = data.data;
                        delete imageCache[currentImageIdx];
                        img.blocks = (data.blocks || []).map(b => ({
                            text: b.text || '',
                            translated: b.translated || '',
                            bbox: b.bbox ? [...b.bbox] : null
                        }));
                        img.deletedRegions = [];
                        img.dirty = false;
                        selectedBlockIdx = -1;
                        updateBlockEditor(-1);
                        loadImage(currentImageIdx).then(() => {
                            fitCanvas(); requestDraw(); updateThumbnails(); updateNavButtons(); updateFooterButtons();
                        });
                    }
                    if (navigateAfter) {
                        window.location.href = '/translate-result/' + sessionId;
                        return;
                    }
                    showToast('✅ Đã render lại ảnh', { variant: 'success', duration: 2500 });
                })
                .catch(err => {
                    if (err.status === 404) {
                        showToast('Phiên đã hết hạn', { variant: 'error', duration: 3000 });
                        setTimeout(() => { window.location.href = '/'; }, 2000);
                    } else if (err.status === 422) {
                        showToast('Toạ độ bbox không hợp lệ', { variant: 'error', duration: 3000 });
                        document.querySelectorAll('.coord-input').forEach(inp => inp.classList.add('error'));
                        setTimeout(() => document.querySelectorAll('.coord-input').forEach(inp => inp.classList.remove('error')), 3000);
                    } else {
                        showToast('Không thể render lại ảnh', { variant: 'error', duration: 5000, actionLabel: 'Thử lại', onAction: run });
                    }
                })
                .finally(() => {
                    if (!navigateAfter) setBusy(false);
                });
        };
        run();
    }

    // V3 (chốt captain 8.3 + backend contract): "Lưu tất cả" renders EVERY
    // dirty image sequentially via /re-render-image (current page uses live
    // state; other pages replay their persisted draft state), showing
    // "⏳ Đang render ảnh i/n" on a locked button, clearing each image's dirty
    // badge as it finishes, then redirects to the results page.
    function saveAll() {
        if (isBusy) return;
        if (isStyleditor) { saveAllStyleditor(); return; }
        const img = images[currentImageIdx];
        if (img && img.dirty) {
            const label = '⏳ Đang render…';
            showToast(label, { variant: 'info', duration: 6000 });
            rerenderCurrentImage({ navigateAfter: true, busyLabel: label });
        } else {
            window.location.href = '/translate-result/' + sessionId;
        }
    }

    function buildSaveAllPayload(idx) {
        let state = null;
        if (idx === globalImageIdx && images[0]) {
            const cur = images[0];
            const imgEl = imageCache[currentImageIdx];
            state = {
                blocks: (cur.blocks || []).map(b => ({
                    text: b.text || '', translated: b.translated || '',
                    bbox: b.bbox ? [...b.bbox] : null,
                    style: b.style ? { ...b.style } : null
                })),
                eraseRegions: (cur.eraseRegions || []).map(r => [...r]),
                deletedRegions: (cur.deletedRegions || []).map(r => [...r]),
                strokes: (cur.eraseStrokes || []).map(s => ({
                    points: (s.points || []).map(pt => [pt[0], pt[1]]), size: s.size, color: s.color
                })),
                w: imgEl ? imgEl.width : 0,
                h: imgEl ? imgEl.height : 0
            };
        } else {
            state = readDraftState(idx);
        }
        if (!state) {
            // No local draft (image never opened): send an empty block list and
            // let the server fall back to v3_draft (translated + default style).
            return {
                session_id: sessionId,
                image_idx: String(idx),
                idx: idx,
                blocks_json: '[]',
                erase_regions_json: '[]',
                deleted_regions_json: '[]'
            };
        }
        const payload = {
            session_id: sessionId,
            image_idx: String(idx),
            idx: idx,
            blocks_json: JSON.stringify(state.blocks || []),
            erase_regions_json: JSON.stringify((state.eraseRegions || []).map(r => [...r])),
            deleted_regions_json: JSON.stringify((state.deletedRegions || []).map(r => [...r]))
        };
        const mask = buildEraseMaskFor(state.strokes || [], state.w, state.h);
        if (mask) payload.erase_mask = mask;
        return payload;
    }

    function saveAllStyleditor() {
        const curImg = images[currentImageIdx];
        const badges = readDirtyBadges();
        // Render EVERY image of the session: images the user edited carry a
        // local draft (or live state), untouched images fall back to the
        // server-side v3_draft (translated text + default styles). Rendering
        // only dirty images would leave the results page showing the raw
        // original (with untranslated source text) for every other image.
        const dirty = [];
        for (let i = 0; i < allImages.length; i++) {
            if (i === globalImageIdx) {
                if (curImg && curImg.dirty) dirty.push(i);
            } else if (badges[i] && readDraftState(i)) {
                dirty.push(i);
            } else {
                dirty.push(i);
            }
        }
        if (!dirty.length) {
            window.location.href = '/translate-result/' + sessionId;
            return;
        }
        dirty.sort((a, b) => a - b);
        const payloads = dirty.map(buildSaveAllPayload).filter(Boolean);
        if (!payloads.length) {
            window.location.href = '/translate-result/' + sessionId;
            return;
        }
        let i = 0;
        const runNext = () => {
            if (i >= payloads.length) {
                window.location.href = '/translate-result/' + sessionId;
                return;
            }
            const p = payloads[i];
            const label = '⏳ Đang render ảnh ' + (i + 1) + '/' + payloads.length;
            setBusy(true, label);
            showToast(label, { variant: 'info', duration: 6000 });
            const form = new FormData();
            for (const k in p) form.append(k, p[k]);
            fetch('/re-render-image', { method: 'POST', body: form })
                .then(async r => {
                    const data = await r.json().catch(() => ({}));
                    if (!r.ok) {
                        const err = new Error(data.error || ('HTTP ' + r.status));
                        err.status = r.status;
                        throw err;
                    }
                    return data;
                })
                .then(data => {
                    if (p.idx === globalImageIdx && images[0]) {
                        // adopt server-normalized blocks for the visible image
                        images[0].blocks = (data.blocks || []).map(b => ({
                            text: b.text || '', translated: b.translated || '',
                            bbox: b.bbox ? [...b.bbox] : null,
                            style: b.style ? { ...b.style } : null
                        }));
                        images[0].dirty = false;
                        updateBlockEditor(-1);
                        requestDraw();
                    }
                    clearDirtyBadge(p.idx);
                    removeDraftState(p.idx);
                    i++;
                    runNext();
                })
                .catch(err => {
                    setBusy(false);
                    if (err.status === 404) {
                        showToast('Phiên đã hết hạn', { variant: 'error', duration: 3000 });
                        setTimeout(() => { window.location.href = '/'; }, 2000);
                    } else if (err.status === 422) {
                        showToast('Toạ độ bbox không hợp lệ ở ảnh ' + (p.idx + 1), { variant: 'error', duration: 4000 });
                    } else {
                        showToast('Không thể render ảnh ' + (p.idx + 1) + '/' + payloads.length, {
                            variant: 'error', duration: 6000, actionLabel: 'Thử lại',
                            onAction: runNext
                        });
                    }
                });
        };
        runNext();
    }

    // ---- Buttons ----
    const toolSelect = document.getElementById('tool-select');
    const toolAdd = document.getElementById('tool-add');
    const toolDelete = document.getElementById('tool-delete');
    const toolUndo = document.getElementById('tool-undo');
    const toolRedo = document.getElementById('tool-redo');
    if (toolSelect) toolSelect.addEventListener('click', () => setTool('select'));
    if (toolAdd) toolAdd.addEventListener('click', () => setTool('add'));
    if (toolDelete) toolDelete.addEventListener('click', () => setTool('delete'));
    if (toolUndo) toolUndo.addEventListener('click', undo);
    if (toolRedo) toolRedo.addEventListener('click', redo);
    const toolEraseRect = document.getElementById('tool-erase-rect');
    const toolEraseBrush = document.getElementById('tool-erase-brush');
    const brushSizeSel = document.getElementById('brush-size');
    if (toolEraseRect) toolEraseRect.addEventListener('click', () => setTool('erase-rect'));
    if (toolEraseBrush) toolEraseBrush.addEventListener('click', () => setTool('erase-brush'));
    if (brushSizeSel) brushSizeSel.addEventListener('change', () => {
        brushSize = parseInt(brushSizeSel.value, 10) || 12;
        updateHints();
        requestDraw();
    });
    document.getElementById('zoom-out').addEventListener('click', () => setZoom(zoomLevel / ZOOM_STEP));
    document.getElementById('zoom-in').addEventListener('click', () => setZoom(zoomLevel * ZOOM_STEP));
    document.getElementById('zoom-fit').addEventListener('click', () => { zoomLevel = 1; fitCanvas(); });
    document.getElementById('zoom-actual').addEventListener('click', setActualSize);
    const toolReset = document.getElementById('tool-reset');
    if (toolReset) toolReset.addEventListener('click', () => {
        if (confirm('Reset tất cả bóng thoại về kết quả OCR gốc?')) {
            undoStack = []; redoStack = [];
            images.forEach((img) => {
                img.blocks = img._originalBlocks ? img._originalBlocks.map(b => ({
                    text: b.text || '', translated: b.translated || '', bbox: b.bbox ? [...b.bbox] : null
                })) : [];
                img.deletedRegions = [];
            });
            selectedBlockIdx = -1; updateBlockEditor(-1); requestDraw(); updateThumbnails();
            showToast('Đã reset về OCR gốc');
        }
    });

    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    if (btnPrev) btnPrev.addEventListener('click', () => { if (currentImageIdx > 0) switchImage(currentImageIdx - 1); });
    if (btnNext) btnNext.addEventListener('click', () => { if (currentImageIdx < images.length - 1) switchImage(currentImageIdx + 1); });

    const btnRerenderOne = document.getElementById('btn-rerender-one');
    const btnSaveAll = document.getElementById('btn-save-all');
    if (btnRerenderOne) btnRerenderOne.addEventListener('click', () => rerenderCurrentImage());
    if (btnSaveAll) btnSaveAll.addEventListener('click', saveAll);

    const btnContinue = document.getElementById('btn-continue');
    if (btnContinue) btnContinue.addEventListener('click', () => {
        btnContinue.disabled = true;
        btnContinue.textContent = '⏳ Đang chuẩn bị…';
        const allBlocks = images.map((img, idx) => ({ image_idx: idx, blocks: img.blocks }));
        modifiedBlocksInput.value = JSON.stringify(allBlocks);
        document.getElementById('continue-form').submit();
    });

    // Editor drawer / bottom-sheet toggle (F9)
    const editorToggle = document.getElementById('btn-toggle-editor');
    const editorClose = document.getElementById('btn-close-editor');
    const corrBody = document.querySelector('.corr-body');
    function setEditorOpen(open) {
        if (!corrBody || !editorToggle) return;
        corrBody.classList.toggle('editor-open', open);
        editorToggle.setAttribute('aria-expanded', String(open));
        if (open) {
            const closeBtn = document.getElementById('btn-close-editor');
            if (closeBtn) closeBtn.focus({ preventScroll: true });
        }
    }
    if (editorToggle) editorToggle.addEventListener('click', () => setEditorOpen(!corrBody.classList.contains('editor-open')));
    if (editorClose) editorClose.addEventListener('click', () => setEditorOpen(false));

    function switchImage(idx) {
        if (isStyleditor) {
            // V3 2.2: each editor page loads ONE image; switching reloads the
            // page with ?img=<idx> (server-side per-image state).
            if (idx < 0 || idx >= allImages.length) return;
            window.location.href = '/styleditor/' + sessionId + '?img=' + idx;
            return;
        }
        currentImageIdx = idx; selectedBlockIdx = -1; updateBlockEditor(-1);
        loadImage(idx).then(() => { fitCanvas({ resetZoom: true }); requestDraw(); updateThumbnails(); updateNavButtons(); updateFooterButtons(); });
    }

    // V3: global page index used for navigation in styleditor mode.
    function navImageIdx() { return isStyleditor ? globalImageIdx : currentImageIdx; }

    function fitCanvas(options) {
        const o = options || {};
        const resetZoom = !!o.resetZoom;
        const img = imageCache[currentImageIdx];
        const wrap = canvasOuter;
        if (!img || !wrap) return;
        const wrapRect = wrap.getBoundingClientRect();
        const maxW = Math.max(120, wrapRect.width - 28);
        const maxH = Math.max(120, wrapRect.height - 28);
        fitScale = Math.min(1, maxW / img.width, maxH / img.height);
        if (resetZoom) zoomLevel = 1;
        applyCanvasScale();
        requestAnimationFrame(() => {
            wrap.scrollLeft = Math.max(0, (mainCanvas.offsetWidth - wrap.clientWidth) / 2);
            wrap.scrollTop = Math.max(0, (mainCanvas.offsetHeight - wrap.clientHeight) / 2);
        });
    }

    function updateNavButtons() {
        if (isPostrender) return;
        const idx = navImageIdx();
        if (isStyleditor) {
            const img = allImages[idx] || images[0];
            if (btnPrev) btnPrev.disabled = idx === 0;
            if (btnNext) btnNext.disabled = idx >= allImages.length - 1;
            currentImageLabel.textContent = (img ? img.name : '') + ' (' + (idx + 1) + '/' + allImages.length + ')';
            updateCanvasAria();
            return;
        }
        if (btnPrev) btnPrev.disabled = currentImageIdx === 0;
        if (btnNext) btnNext.disabled = currentImageIdx === images.length - 1;
        currentImageLabel.textContent = images[currentImageIdx].name + ' (' + (currentImageIdx + 1) + '/' + images.length + ')';
        updateCanvasAria();
    }

    function updateCanvasAria() {
        const img = images[currentImageIdx];
        if (!img) return;
        const n = (img.blocks || []).length;
        const dirty = (isPostrender || isStyleditor) && img.dirty;
        const label = isStyleditor ? 'Ảnh đã xoá text và vẽ chữ dịch ' : (isPostrender ? 'Ảnh đã dịch ' : 'Ảnh ');
        mainCanvas.setAttribute('role', 'img');
        mainCanvas.setAttribute('aria-label',
            label + (img.name || '') +
            ' — ' + n + ' bóng thoại' + (dirty ? ', có thay đổi chưa render' : ''));
    }

    // ---- Thumbnails ----
    function updateThumbnails() {
        const items = thumbnails();
        items.forEach(el => el.classList.remove('active'));
        const activeIdx = isStyleditor ? navImageIdx() : currentImageIdx;
        const active = document.querySelector('.thumb-item[data-index="' + activeIdx + '"]');
        if (active) {
            active.classList.add('active');
            active.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        }
        items.forEach(el => {
            const idx = parseInt(el.dataset.index);
            const img = images[idx];
            const countEl = el.querySelector('.thumb-count');
            if (countEl) countEl.textContent = img ? (img.blocks.length + ' blocks') : '—';
            // F4: dirty badge on thumbs with un-rendered changes (P0 current
            // image; P1 whole session via localStorage — spec 2.2).
            if (isStyleditor) {
                const badges = readDirtyBadges();
                const badged = !!badges[idx] || (idx === globalImageIdx && !!images[0] && images[0].dirty);
                el.classList.toggle('dirty', badged);
            } else {
                el.classList.toggle('dirty', isPostrender && !!images[idx].dirty);
            }
            renderThumbnail(idx);
        });
        document.getElementById('total-blocks').textContent = images.reduce((s, i) => s + i.blocks.length, 0);
        updateCanvasAria();
    }

    function renderThumbnail(idx) {
        loadThumbImage(idx).then(img => {
            const el = document.querySelector('.thumb-item[data-index="' + idx + '"]');
            if (!el) return;
            const tc = el.querySelector('.thumb-canvas');
            if (!tc) return;
            if (!img) {
                // V3: remote images (other pages) have no b64 payload — show a
                // neutral placeholder (spec 4.2: all_images carries names only).
                const rect = tc.getBoundingClientRect();
                const thumbW = Math.max(80, Math.round(rect.width || 150));
                const thumbH = Math.max(60, Math.round(rect.height || 102));
                const key = 'empty:' + thumbW + 'x' + thumbH;
                if (thumbnailKeys[idx] === key) return;
                thumbnailKeys[idx] = key;
                const tctx = tc.getContext('2d');
                const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
                tc.width = Math.round(thumbW * dpr);
                tc.height = Math.round(thumbH * dpr);
                tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                tctx.clearRect(0, 0, thumbW, thumbH);
                tctx.fillStyle = '#e7dfee';
                tctx.fillRect(0, 0, thumbW, thumbH);
                tctx.fillStyle = '#a891bb';
                tctx.font = '600 22px "Exo 2", sans-serif';
                tctx.textAlign = 'center';
                tctx.textBaseline = 'middle';
                tctx.fillText('🖼', thumbW / 2, thumbH / 2 - 10);
                tctx.font = '500 11px "Exo 2", sans-serif';
                const name = (allImages[idx] || {}).name || '';
                tctx.fillText(name.substring(0, 14), thumbW / 2, thumbH / 2 + 16);
                tctx.textAlign = 'start';
                return;
            }
            const blocks = getBlocks(idx);
            const rect = tc.getBoundingClientRect();
            const thumbW = Math.max(80, Math.round(rect.width || 150));
            const thumbH = Math.max(60, Math.round(rect.height || 102));
            const key = thumbW + 'x' + thumbH + ':' + blocks.map(b => (b.bbox || []).join(',')).join('|') + ':' + blocks.length;
            if (thumbnailKeys[idx] === key) return;
            thumbnailKeys[idx] = key;

            const tctx = tc.getContext('2d');
            const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
            tc.width = Math.round(thumbW * dpr);
            tc.height = Math.round(thumbH * dpr);
            tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            tctx.clearRect(0, 0, thumbW, thumbH);
            tctx.fillStyle = '#f2edf5';
            tctx.fillRect(0, 0, thumbW, thumbH);

            const s = Math.min(thumbW / img.width, thumbH / img.height);
            const drawW = img.width * s;
            const drawH = img.height * s;
            const dx = (thumbW - drawW) / 2;
            const dy = (thumbH - drawH) / 2;
            tctx.drawImage(img, dx, dy, drawW, drawH);
            blocks.forEach(b => {
                if (!b.bbox) return;
                tctx.strokeStyle = 'rgba(255,145,0,0.75)';
                tctx.lineWidth = 1;
                tctx.strokeRect(
                    dx + b.bbox[0] * s,
                    dy + b.bbox[1] * s,
                    (b.bbox[2] - b.bbox[0]) * s,
                    (b.bbox[3] - b.bbox[1]) * s
                );
            });
        });
    }

    thumbnails().forEach(el => {
        el.addEventListener('click', () => switchImage(parseInt(el.dataset.index)));
        // P2-1/F10: keyboard activation for the thumbnail buttons
        el.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') {
                ev.preventDefault();
                switchImage(parseInt(el.dataset.index));
            }
        });
    });

    // ═══════════════════════════════════════════════════════════════════
    // V3 Style editor engine (spec F3-F9): fonts, live WYSIWYG text,
    // background erase (rect/brush + monotonic regions + mask), style panel.
    // ═══════════════════════════════════════════════════════════════════

    function clampInt(v, lo, hi) {
        const n = parseInt(v, 10);
        return isNaN(n) ? lo : Math.max(lo, Math.min(hi, n));
    }

    // ---- Fonts (F8): /api/fonts + /font-file + FontFace ----
    function fontFamilyFor(name) {
        const n = String(name || DEFAULT_STYLE.font).replace(/"/g, '');
        return '"' + n + '"';
    }
    function ensureFontLoaded(name) {
        const family = String(name || '').replace(/"/g, '');
        if (!family) return Promise.resolve();
        if (fontFailures[family]) return Promise.resolve();
        if (fontLoads[family]) return fontLoads[family];
        try {
            if (document.fonts && document.fonts.check('16px "' + family + '"')) return Promise.resolve();
        } catch (_) { /* older browsers */ }
        fontLoads[family] = new Promise((resolve) => {
            let ff;
            try {
                ff = new FontFace(family, 'url(/font-file/' + encodeURIComponent(family) + ')');
            } catch (err) {
                fontFailures[family] = true;
                delete fontLoads[family];
                showToast('⚠️ Không tải được font ' + family, { variant: 'error', duration: 4000 });
                resolve();
                return;
            }
            ff.load().then(f => {
                document.fonts.add(f);
                fontReady[family] = true;
                resolve();
            }).catch(() => {
                fontFailures[family] = true;
                delete fontLoads[family];
                showToast('⚠️ Không tải được font ' + family + ' — dùng font mặc định', { variant: 'error', duration: 4000 });
                resolve();
            });
        });
        return fontLoads[family];
    }
    function initFonts() {
        fetch('/api/fonts')
            .then(r => r.json())
            .then(d => {
                fontList = Array.isArray(d.fonts) ? d.fonts : [];
                populateFontOptions();
            })
            .catch(() => {
                const cur = (images[0] && images[0].blocks && images[0].blocks[0] && images[0].blocks[0].style)
                    ? images[0].blocks[0].style.font : DEFAULT_STYLE.font;
                fontList = [{ name: cur, label: cur }];
                populateFontOptions();
                showToast('⚠️ Không tải được danh sách phông chữ', { variant: 'error', duration: 4000 });
            });
    }
    function populateFontOptions() {
        const sel = document.getElementById('style-font');
        if (!sel) return;
        const cur = sel.value || '';
        const seen = {};
        const options = [];
        fontList.forEach(f => {
            const name = String(f.name || f.label || '');
            if (!name || seen[name]) return;
            seen[name] = true;
            options.push('<option value="' + escapeHtml(name) + '">' + escapeHtml(String(f.label || name)) + '</option>');
        });
        if (cur && !seen[cur]) options.unshift('<option value="' + escapeHtml(cur) + '">' + escapeHtml(cur) + '</option>');
        sel.innerHTML = options.join('');
        if (cur) sel.value = cur;
        else if (fontList.length) sel.value = fontList[0].name;
    }

    // ---- WYSIWYG text layout: client mirror of add_text.py (spec F6.2) ----
    function cleanStyledText(text) {
        if (!text) return '';
        return String(text)
            .replace(/\s*\n\s*/g, ' ')
            .replace(/[ \t]{2,}/g, ' ')
            .trim();
    }
    function wrapTextAtSize(text, size, usableW, useCharWrap, family) {
        const lines = [];
        const measure = (s) => {
            ctx.font = size + 'px ' + family;
            return ctx.measureText(s).width;
        };
        if (useCharWrap) {
            let current = '';
            for (const ch of text) {
                if (ch === '\n') {
                    if (current) { lines.push(current); current = ''; }
                    continue;
                }
                const test = current + ch;
                if (measure(test) > usableW && current) { lines.push(current); current = ch; }
                else current = test;
            }
            if (current) lines.push(current);
        } else {
            const words = text.split(' ');
            let current = '';
            for (const word of words) {
                if (!word) continue;
                const sep = current ? ' ' : '';
                const test = current + sep + word;
                if (measure(test) > usableW) {
                    if (current) lines.push(current);
                    current = word;
                    if (measure(word) > usableW) {
                        const charLines = [];
                        let chunk = '';
                        for (const ch of word) {
                            const tc = chunk + ch;
                            if (measure(tc) > usableW && chunk) { charLines.push(chunk); chunk = ch; }
                            else chunk = tc;
                        }
                        if (chunk) charLines.push(chunk);
                        lines.push.apply(lines, charLines.slice(0, -1));
                        if (charLines.length === 1 && lines.length === 0) {
                            lines.push(charLines[0]);
                            current = '';
                        } else {
                            current = charLines.length ? charLines[charLines.length - 1] : '';
                        }
                    }
                } else current = test;
            }
            if (current) lines.push(current);
        }
        return lines.length ? lines : [text];
    }
    function linesFitAtSize(lines, size, usableW, family) {
        ctx.font = size + 'px ' + family;
        for (const line of lines) {
            if (ctx.measureText(line).width > usableW) return false;
        }
        return true;
    }
    function computeStyledLayout(block, blockIdx) {
        const textRaw = cleanStyledText(block.translated);
        if (!textRaw) return null;
        const style = normalizeStyle(block.style);
        const family = fontFamilyFor(style.font);
        const bbox = block.bbox;
        if (!bbox || bbox.length !== 4) return null;
        const w = bbox[2] - bbox[0], h = bbox[3] - bbox[1];
        if (w <= 0 || h <= 0) return null;
        let usableW = Math.floor(w * (1 - 2 * 0.12));
        let usableH = Math.floor(h * (1 - 2 * 0.12));
        if (usableW <= 0 || usableH <= 0) { usableW = w; usableH = h; }
        const hasCJK = /[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(textRaw);
        const useCharWrap = hasCJK && textRaw.indexOf(' ') < 0;

        const fixed = clampInt(style.font_size, 0, 120);
        if (fixed) {
            if (fixed <= 12) {
                const lines = wrapTextAtSize(textRaw, fixed, usableW, useCharWrap, family);
                return { family, size: fixed, lines, lineHeight: Math.floor(fixed * 1.3) };
            }
            let best = 0, bestLines = null;
            let lo = 12, hi = fixed;
            while (lo <= hi) {
                const mid = (lo + hi) >> 1;
                const lh = Math.floor(mid * 1.3);
                const lines = wrapTextAtSize(textRaw, mid, usableW, useCharWrap, family);
                if (lines.length * lh <= usableH && linesFitAtSize(lines, mid, usableW, family)) {
                    best = mid; bestLines = lines; lo = mid + 1;
                } else hi = mid - 1;
            }
            if (best) return { family, size: best, lines: bestLines, lineHeight: Math.floor(best * 1.3) };
            // Nothing ≥ 12 fits → fall through to the auto search (server mirror).
        }

        const area = usableW * usableH;
        const chars = Math.max(textRaw.length, 1);
        const estimated = Math.round(Math.sqrt(area / (chars * 0.8)));
        const guess = Math.max(12, Math.min(60, estimated));
        let best = 0, bestLines = null;
        let lo = 12, hi = guess;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            const lh = Math.floor(mid * 1.3);
            const lines = wrapTextAtSize(textRaw, mid, usableW, useCharWrap, family);
            if (lines.length * lh <= usableH && linesFitAtSize(lines, mid, usableW, family)) {
                best = mid; bestLines = lines; lo = mid + 1;
            } else hi = mid - 1;
        }
        const size = best || 12;
        const lines = bestLines || wrapTextAtSize(textRaw, size, usableW, useCharWrap, family);
        return { family, size, lines, lineHeight: Math.floor(size * 1.3) };
    }
    function getStyledLayout(block, blockIdx) {
        const key = currentImageIdx + ':' + blockIdx + ':' + (block.bbox || []).join(',') + ':' +
            cleanStyledText(block.translated) + ':' + JSON.stringify(normalizeStyle(block.style));
        if (layoutCache[key]) return layoutCache[key];
        const layout = computeStyledLayout(block, blockIdx);
        layoutCache[key] = layout;
        return layout;
    }

    // ---- Auto text color (F5.5): luminance of the erased bg inside bbox ----
    function getAutoTextColor(block, blockIdx) {
        const key = currentImageIdx + ':' + blockIdx + ':' + (block.bbox || []).join(',');
        if (autoColorCache[key]) return autoColorCache[key];
        let color = '#000000';
        const img = imageCache[currentImageIdx];
        const bbox = block.bbox;
        if (img && bbox && bgCanvas.width) {
            try {
                const X1 = clamp(Math.round(bbox[0]), 0, img.width - 1);
                const Y1 = clamp(Math.round(bbox[1]), 0, img.height - 1);
                const X2 = clamp(Math.round(bbox[2]), X1 + 1, img.width);
                const Y2 = clamp(Math.round(bbox[3]), Y1 + 1, img.height);
                const g = bgCanvas.getContext('2d');
                const data = g.getImageData(X1, Y1, X2 - X1, Y2 - Y1).data;
                let lum = 0, n = 0;
                for (let i = 0; i < data.length; i += 16) {
                    lum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                    n++;
                }
                if (n > 0) { lum /= n; color = lum < 128 ? '#ffffff' : '#000000'; }
            } catch (_) { /* sampling fallback */ }
        }
        autoColorCache[key] = color;
        return color;
    }

    // ---- Live styled text drawing (F3.1/F6): mirror of _draw_text_on_pil ----
    function drawStyledBlockText(block, blockIdx) {
        const style = normalizeStyle(block.style);
        const layout = getStyledLayout(block, blockIdx);
        if (!layout) return;
        const color = style.text_color ? style.text_color : getAutoTextColor(block, blockIdx);
        const bold = !!style.bold;
        const italic = !!style.italic;
        const align = (style.align === 'left' || style.align === 'right') ? style.align : 'center';
        const bbox = block.bbox;
        const [x1, y1, x2, y2] = bbox;
        const bw = x2 - x1, bh = y2 - y1;
        const totalH = layout.lines.length * layout.lineHeight;
        const textY = y1 + Math.floor((bh - totalH) / 2);
        const padding = Math.round(bw * 0.12);
        ctx.save();
        ctx.font = layout.size + 'px ' + layout.family;
        ctx.textBaseline = 'top';
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;      // bold = synthetic stroke, PIL stroke_width=2 (F5.6)
        ctx.lineJoin = 'round';
        for (let i = 0; i < layout.lines.length; i++) {
            const line = layout.lines[i];
            ctx.font = layout.size + 'px ' + layout.family;
            const lw = ctx.measureText(line).width;
            let textX;
            if (align === 'left') textX = x1 + padding;
            else if (align === 'right') textX = x2 - padding - lw;
            else textX = x1 + Math.floor((bw - lw) / 2);
            const yLine = textY + i * layout.lineHeight;
            ctx.save();
            if (italic) ctx.transform(1, 0.18, 0, 1, textX, yLine);  // shear 0.18 (F5.7)
            const dx = italic ? 0 : textX;
            const dy = italic ? 0 : yLine;
            if (bold) ctx.strokeText(line, dx, dy);
            ctx.fillText(line, dx, dy);
            ctx.restore();
        }
        ctx.restore();
        // Kick a font load so the first paint (sans-serif) re-renders in the
        // real TTF as soon as it arrives (F8.3). Guarded by fontReady so a
        // resolved promise never re-schedules draws (no infinite loop).
        const fam = String(style.font || '').replace(/"/g, '');
        if (fam && !fontReady[fam] && !fontFailures[fam]) {
            ensureFontLoaded(style.font).then(requestDraw);
        }
    }

    // ---- Erase layer (F4): bg sampling + rect fills + brush strokes ----
    function initBgCanvas() {
        const img = imageCache[currentImageIdx];
        if (!img) return;
        bgCanvas.width = img.width;
        bgCanvas.height = img.height;
        bgCanvas.getContext('2d').drawImage(img, 0, 0);
    }
    function sampleColorAt(pos) {
        const img = imageCache[currentImageIdx];
        if (!img || !bgCanvas.width) return '#ffffff';
        const r = 2;
        const x = clamp(Math.round(pos.x) - r, 0, Math.max(0, img.width - 1));
        const y = clamp(Math.round(pos.y) - r, 0, Math.max(0, img.height - 1));
        const w = Math.min(2 * r + 1, img.width - x);
        const h = Math.min(2 * r + 1, img.height - y);
        if (w <= 0 || h <= 0) return '#ffffff';
        try {
            const g = bgCanvas.getContext('2d');
            const data = g.getImageData(x, y, w, h).data;
            let rs = 0, gs = 0, bs = 0, n = 0;
            for (let i = 0; i < data.length; i += 4) { rs += data[i]; gs += data[i + 1]; bs += data[i + 2]; n++; }
            return n ? 'rgb(' + Math.round(rs / n) + ',' + Math.round(gs / n) + ',' + Math.round(bs / n) + ')' : '#ffffff';
        } catch (_) { return '#ffffff'; }
    }
    function sampleEdgeColor(region) {
        const img = imageCache[currentImageIdx];
        if (!img || !bgCanvas.width) return '#ffffff';
        const key = currentImageIdx + ':' + region.join(',');
        if (edgeColorCache[key]) return edgeColorCache[key];
        const W = img.width, H = img.height;
        const X1 = clamp(Math.round(region[0]), 0, W - 1);
        const Y1 = clamp(Math.round(region[1]), 0, H - 1);
        const X2 = clamp(Math.round(region[2]), 0, W);
        const Y2 = clamp(Math.round(region[3]), 0, H);
        const rw = X2 - X1 + 1, rh = Y2 - Y1 + 1;
        let color = '#ffffff';
        try {
            const g = bgCanvas.getContext('2d');
            const data = g.getImageData(X1, Y1, rw, rh).data;
            const at = (x, y) => {
                const i = ((y - Y1) * rw + (x - X1)) * 4;
                return [data[i], data[i + 1], data[i + 2]];
            };
            let rs = 0, gs = 0, bs = 0, n = 0;
            const addPx = (p) => { rs += p[0]; gs += p[1]; bs += p[2]; n++; };
            for (let x = X1; x <= X2; x += 3) { addPx(at(x, Y1)); addPx(at(x, Math.min(Y2, H - 1))); }
            for (let y = Y1; y <= Y2; y += 3) { addPx(at(X1, y)); addPx(at(Math.min(X2, W - 1), y)); }
            if (n > 0) color = 'rgb(' + Math.round(rs / n) + ',' + Math.round(gs / n) + ',' + Math.round(bs / n) + ')';
        } catch (_) { /* fallback */ }
        edgeColorCache[key] = color;
        return color;
    }
    function redrawEraseLayer() {
        const img = imageCache[currentImageIdx];
        if (!img) return;
        eraseLayer.width = img.width;
        eraseLayer.height = img.height;
        const lctx = eraseLayer.getContext('2d');
        lctx.clearRect(0, 0, img.width, img.height);
        const cur = images[currentImageIdx];
        if (!cur) return;
        // P1: persisted erase mask composites first (monotonic, survives undo).
        if (cur.eraseMaskLayer) lctx.drawImage(cur.eraseMaskLayer, 0, 0);
        (cur.erasePreviewRects || []).forEach(region => {
            lctx.fillStyle = sampleEdgeColor(region);
            lctx.fillRect(region[0], region[1], region[2] - region[0], region[3] - region[1]);
        });
        (cur.eraseStrokesPreview || []).forEach(s => {
            const pts = s.points || [];
            if (!pts.length) return;
            lctx.strokeStyle = s.color;
            lctx.fillStyle = s.color;
            lctx.lineWidth = s.size;
            lctx.lineCap = 'round';
            lctx.lineJoin = 'round';
            if (pts.length === 1) {
                lctx.beginPath();
                lctx.arc(pts[0][0], pts[0][1], s.size / 2, 0, Math.PI * 2);
                lctx.fill();
                return;
            }
            lctx.beginPath();
            lctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) lctx.lineTo(pts[i][0], pts[i][1]);
            lctx.stroke();
        });
    }
    function paintBrushPoint(pos) {
        const img = imageCache[currentImageIdx];
        if (!img || eraseLayer.width !== img.width) return;
        if (!strokeColor) strokeColor = sampleColorAt(pos);
        const lctx = eraseLayer.getContext('2d');
        lctx.strokeStyle = strokeColor;
        lctx.fillStyle = strokeColor;
        lctx.lineWidth = brushSize;
        lctx.lineCap = 'round';
        lctx.lineJoin = 'round';
        if (strokePoints.length <= 1) {
            lctx.beginPath();
            lctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
            lctx.fill();
            return;
        }
        const prev = strokePoints[strokePoints.length - 2];
        lctx.beginPath();
        lctx.moveTo(prev.x, prev.y);
        lctx.lineTo(pos.x, pos.y);
        lctx.stroke();
    }
    function strokeBBox() {
        if (!strokePoints.length) return null;
        const W = mainCanvas.width, H = mainCanvas.height;
        let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
        strokePoints.forEach(p => {
            x1 = Math.min(x1, p.x); y1 = Math.min(y1, p.y);
            x2 = Math.max(x2, p.x); y2 = Math.max(y2, p.y);
        });
        return [
            clamp(Math.round(x1), 0, W),
            clamp(Math.round(y1), 0, H),
            clamp(Math.round(x2), 0, W),
            clamp(Math.round(y2), 0, H)
        ];
    }
    // F4.8 (P1): PNG b64 grayscale mask (white = erase) from ALL strokes —
    // monotonic, so undone strokes still render clean (F4.5). Long side ≤ 2048.
    function buildEraseMask() {
        const cur = images[currentImageIdx];
        const strokes = (cur && cur.eraseStrokes) || [];
        const img = imageCache[currentImageIdx];
        if (!strokes.length || !img) return '';
        return buildEraseMaskFor(strokes, img.width, img.height);
    }
    function buildEraseMaskFor(strokes, imgW, imgH) {
        if (!strokes || !strokes.length || !imgW || !imgH) return '';
        const scale = Math.min(1, 2048 / Math.max(imgW, imgH));
        const w = Math.max(1, Math.round(imgW * scale));
        const h = Math.max(1, Math.round(imgH * scale));
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const g = c.getContext('2d');
        g.fillStyle = '#000';
        g.fillRect(0, 0, w, h);
        g.strokeStyle = '#fff';
        g.fillStyle = '#fff';
        g.lineCap = 'round';
        g.lineJoin = 'round';
        strokes.forEach(s => {
            g.lineWidth = Math.max(1, s.size * scale);
            const pts = (s.points || []).map(p => [p[0] * scale, p[1] * scale]);
            if (!pts.length) return;
            if (pts.length === 1) {
                g.beginPath();
                g.arc(pts[0][0], pts[0][1], g.lineWidth / 2, 0, Math.PI * 2);
                g.fill();
                return;
            }
            g.beginPath();
            g.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
            g.stroke();
        });
        return c.toDataURL('image/png').split(',')[1] || '';
    }

    // ---- Per-block style (F5) ----
    function normalizeStyle(raw) {
        const s = Object.assign({}, DEFAULT_STYLE, raw || {});
        if (!s.font) s.font = DEFAULT_STYLE.font;
        s.font_size = clampInt(s.font_size, 0, 120);
        s.text_color = (typeof s.text_color === 'string' && /^#[0-9a-fA-F]{6}$/.test(s.text_color)) ? s.text_color.toLowerCase() : null;
        s.bold = !!s.bold;
        s.italic = !!s.italic;
        s.align = (s.align === 'left' || s.align === 'right') ? s.align : 'center';
        return s;
    }
    function styleEqual(a, b) {
        if (!a || !b) return a === b;
        return (a.font || '') === (b.font || '') &&
            (a.font_size || 0) === (b.font_size || 0) &&
            (a.text_color || null) === (b.text_color || null) &&
            !!a.bold === !!b.bold && !!a.italic === !!b.italic &&
            (a.align || 'center') === (b.align || 'center');
    }
    function styleGroupHtml(block) {
        const style = normalizeStyle(block.style);
        const auto = !style.font_size;
        return '<div class="prop-group style-group" id="style-group">' +
            '<p class="style-group-head">Kiểu chữ</p>' +
            '<label for="style-font">Phông chữ</label>' +
            '<select id="style-font" class="style-font" aria-label="Phông chữ">' +
                fontList.map(f => '<option value="' + escapeHtml(String(f.name || '')) + '">' + escapeHtml(String(f.label || f.name || '')) + '</option>').join('') +
            '</select>' +
            '<div class="style-size-row">' +
                '<label for="style-size">Cỡ chữ</label>' +
                '<input type="number" id="style-size" class="style-size" min="8" max="120" step="1" value="' + (style.font_size || '') + '" ' + (auto ? 'disabled' : '') + ' aria-label="Cỡ chữ tính bằng px">' +
                '<label class="style-auto"><input type="checkbox" id="style-size-auto" ' + (auto ? 'checked' : '') + '> Tự động</label>' +
            '</div>' +
            '<p class="style-size-warn" id="style-size-warn">⚠️ Cỡ chữ sẽ được thu nhỏ cho vừa khung</p>' +
            '<div class="style-color-row">' +
                '<span class="style-color-label" id="style-color-label">Màu chữ</span>' +
                '<div class="style-swatches" id="style-swatches" role="group" aria-label="Màu chữ mẫu">' +
                    '<button type="button" class="swatch swatch-auto' + (!style.text_color ? ' swatch-active' : '') + '" data-color="" title="Tự động: đen/trắng theo nền" aria-label="Màu tự động">A</button>' +
                    STYLE_SWATCHES.map(c => '<button type="button" class="swatch' + (style.text_color === c ? ' swatch-active' : '') + '" data-color="' + c + '" style="background:' + c + '" title="' + c + '" aria-label="Màu ' + c + '"></button>').join('') +
                '</div>' +
                '<input type="color" id="style-color" class="style-color" value="' + (style.text_color || '#000000') + '" aria-label="Chọn màu chữ tuỳ chỉnh">' +
            '</div>' +
            '<div class="style-toggles" role="group" aria-label="Kiểu đậm nghiêng">' +
                '<button type="button" id="style-bold" class="style-btn" aria-pressed="' + (style.bold ? 'true' : 'false') + '" title="In đậm (phím B)"><b>B</b></button>' +
                '<button type="button" id="style-italic" class="style-btn" aria-pressed="' + (style.italic ? 'true' : 'false') + '" title="In nghiêng (phím I)"><i>I</i></button>' +
            '</div>' +
            '<div class="style-align" role="group" aria-label="Căn lề chữ">' +
                '<button type="button" id="align-left" class="style-btn align-btn" aria-pressed="' + (style.align === 'left' ? 'true' : 'false') + '" title="Căn trái (phím L)">⬅ Trái</button>' +
                '<button type="button" id="align-center" class="style-btn align-btn" aria-pressed="' + (style.align === 'center' ? 'true' : 'false') + '" title="Căn giữa (phím C)">↔ Giữa</button>' +
                '<button type="button" id="align-right" class="style-btn align-btn" aria-pressed="' + (style.align === 'right' ? 'true' : 'false') + '" title="Căn phải (phím R)">➡ Phải</button>' +
            '</div>' +
            '<button type="button" id="style-apply-all" class="style-apply-all">📋 Áp dụng cho tất cả block ảnh này</button>' +
        '</div>';
    }
    function setBlockStyle(patch) {
        const block = getCurrentBlocks()[selectedBlockIdx];
        if (!block || isBusy) return;
        const cur = normalizeStyle(block.style);
        const merged = Object.assign({}, cur, patch);
        if (styleEqual(cur, merged)) return;
        pushUndo();
        block.style = merged;
        requestDraw();
        markDirty(currentImageIdx);
        refreshStylePanel();
    }
    function toggleStyle(key) {
        const block = getCurrentBlocks()[selectedBlockIdx];
        if (!block) return;
        setBlockStyle({ [key]: !(block.style && block.style[key]) });
    }
    function applyStyleToAll() {
        const block = getCurrentBlocks()[selectedBlockIdx];
        if (!block || isBusy) return;
        pushUndo();
        const style = normalizeStyle(block.style);
        getCurrentBlocks().forEach(b => { b.style = Object.assign({}, DEFAULT_STYLE, style); });
        requestDraw();
        markDirty(currentImageIdx);
        updateThumbnails();
        showToast('📋 Đã áp dụng kiểu cho tất cả block ảnh này', { duration: 2500 });
    }
    // P1-1 (t5): sync the editor panel's VALUES without rebuilding the DOM —
    // used by coord blur / nudge / drag / resize so keyboard focus survives
    // (Tab traversal A13.2).
    function refreshBlockEditorValues() {
        const block = getCurrentBlocks()[selectedBlockIdx];
        if (!block) return;
        const bbox = block.bbox;
        if (bbox && bbox.length === 4) {
            const setVal = (id, v) => {
                const el = document.getElementById(id);
                if (el) el.value = v;
            };
            setVal('edit-x1', bbox[0]); setVal('edit-y1', bbox[1]);
            setVal('edit-x2', bbox[2]); setVal('edit-y2', bbox[3]);
            const sizeEl = document.getElementById('prop-size');
            if (sizeEl) sizeEl.textContent = (bbox[2] - bbox[0]) + '×' + (bbox[3] - bbox[1]) + ' px';
        }
        refreshStylePanel();
    }

    function refreshStylePanel() {
        const block = getCurrentBlocks()[selectedBlockIdx];
        if (!block) return;
        const style = normalizeStyle(block.style);
        const setPressed = (id, on) => {
            const el = document.getElementById(id);
            if (el) el.setAttribute('aria-pressed', on ? 'true' : 'false');
        };
        setPressed('style-bold', style.bold);
        setPressed('style-italic', style.italic);
        setPressed('align-left', style.align === 'left');
        setPressed('align-center', style.align === 'center');
        setPressed('align-right', style.align === 'right');
        const autoBox = document.getElementById('style-size-auto');
        const sizeInput = document.getElementById('style-size');
        if (autoBox) autoBox.checked = !style.font_size;
        if (sizeInput) { sizeInput.disabled = !!style.font_size; sizeInput.value = style.font_size || ''; }
        const colorInput = document.getElementById('style-color');
        if (colorInput) colorInput.value = style.text_color || '#000000';
        document.querySelectorAll('#style-swatches .swatch').forEach(btn => {
            const active = btn.dataset.color ? (btn.dataset.color === style.text_color) : (!style.text_color);
            btn.classList.toggle('swatch-active', active);
        });
        const warn = document.getElementById('style-size-warn');
        if (warn && style.font_size) {
            const layout = computeStyledLayout(block, selectedBlockIdx);
            warn.classList.toggle('show', !!(layout && layout.size < style.font_size));
        } else if (warn) warn.classList.remove('show');
    }
    function bindStylePanel(block) {
        const style = normalizeStyle(block.style);
        const fontSel = document.getElementById('style-font');
        const sizeInput = document.getElementById('style-size');
        const autoBox = document.getElementById('style-size-auto');
        const colorInput = document.getElementById('style-color');
        const warn = document.getElementById('style-size-warn');
        if (fontSel) {
            fontSel.value = style.font;
            populateFontOptions();
            fontSel.addEventListener('change', () => {
                const font = fontSel.value;
                setBlockStyle({ font: font });
                ensureFontLoaded(font).then(requestDraw);
            });
        }
        if (sizeInput) {
            sizeInput.addEventListener('change', () => {
                let v = clampInt(sizeInput.value, 8, 120);
                sizeInput.value = v;
                setBlockStyle({ font_size: v });
                if (warn) {
                    const layout = computeStyledLayout(getCurrentBlocks()[selectedBlockIdx], selectedBlockIdx);
                    warn.classList.toggle('show', !!(layout && layout.size < v));
                }
            });
        }
        if (autoBox) {
            autoBox.addEventListener('change', () => {
                if (autoBox.checked) {
                    setBlockStyle({ font_size: 0 });
                    if (sizeInput) sizeInput.disabled = true;
                } else {
                    if (sizeInput) {
                        sizeInput.disabled = false;
                        if (!sizeInput.value) sizeInput.value = 20;
                        setBlockStyle({ font_size: clampInt(sizeInput.value, 8, 120) });
                    }
                }
                refreshStylePanel();
            });
        }
        if (colorInput) {
            colorInput.value = style.text_color || '#000000';
            colorInput.addEventListener('input', () => setBlockStyle({ text_color: colorInput.value }));
        }
        document.querySelectorAll('#style-swatches .swatch').forEach(btn => {
            btn.addEventListener('click', () => setBlockStyle({ text_color: btn.dataset.color ? btn.dataset.color : null }));
        });
        const boldBtn = document.getElementById('style-bold');
        const italicBtn = document.getElementById('style-italic');
        if (boldBtn) boldBtn.addEventListener('click', () => toggleStyle('bold'));
        if (italicBtn) italicBtn.addEventListener('click', () => toggleStyle('italic'));
        document.querySelectorAll('.align-btn').forEach(btn => {
            btn.addEventListener('click', () => setBlockStyle({
                align: btn.id === 'align-left' ? 'left' : (btn.id === 'align-right' ? 'right' : 'center')
            }));
        });
        const applyAll = document.getElementById('style-apply-all');
        if (applyAll) applyAll.addEventListener('click', applyStyleToAll);
    }

    // ---- Editor panel ----
    function updateBlockEditor(idx) {
        if (idx < 0) {
            blockProperties.innerHTML = isStyleditor
                ? '<p class="no-sel">Chọn bóng thoại để chỉnh sửa — bấm <kbd>[</kbd>/<kbd>]</kbd> để chọn</p>'
                : '<p class="no-sel">Chọn bóng thoại để chỉnh sửa</p>';
            updateHints();
            return;
        }
        const block = getCurrentBlocks()[idx];
        if (!block) return;

        let editStartSnapshot = null;
        let editStartValue = null;
        let editStartBbox = null;
        const w = block.bbox ? (block.bbox[2] - block.bbox[0]) : 0;
        const h = block.bbox ? (block.bbox[3] - block.bbox[1]) : 0;

        const textareaLabel = (isPostrender || isStyleditor) ? 'Bản dịch' : 'Nội dung text';
        const textareaValue = escapeHtml((isPostrender || isStyleditor) ? (block.translated || '') : (block.text || ''));
        const originalLine = (isPostrender || isStyleditor) && block.text
            ? '<p class="prop-original">Gốc: ' + escapeHtml(block.text.substring(0, 60)) + '</p>' : '';

        blockProperties.innerHTML =
            '<div class="prop-group">' +
                '<label>' + textareaLabel + '</label>' +
                '<textarea id="edit-text" rows="3" class="prop-textarea" aria-label="' + textareaLabel + '">' + textareaValue + '</textarea>' +
                originalLine +
            '</div>' +
            '<div class="prop-group">' +
                '<label>Vị trí (x1,y1,x2,y2)</label>' +
                '<div class="prop-coords">' +
                    '<input type="number" id="edit-x1" value="' + (block.bbox ? (block.bbox[0] || 0) : 0) + '" class="coord-input" placeholder="x1" aria-label="x1">' +
                    '<input type="number" id="edit-y1" value="' + (block.bbox ? (block.bbox[1] || 0) : 0) + '" class="coord-input" placeholder="y1" aria-label="y1">' +
                    '<input type="number" id="edit-x2" value="' + (block.bbox ? (block.bbox[2] || 0) : 0) + '" class="coord-input" placeholder="x2" aria-label="x2">' +
                    '<input type="number" id="edit-y2" value="' + (block.bbox ? (block.bbox[3] || 0) : 0) + '" class="coord-input" placeholder="y2" aria-label="y2">' +
                '</div>' +
            '</div>' +
            '<div class="prop-group prop-meta">' +
                '<span class="prop-size" id="prop-size">' + w + '×' + h + ' px</span>' +
                '<div class="prop-nudge" role="group" aria-label="Di chuyển bóng thoại">' +
                    '<button type="button" class="nudge-btn" data-dx="0" data-dy="-1" title="Lên 1px">▲</button>' +
                    '<button type="button" class="nudge-btn" data-dx="-1" data-dy="0" title="Trái 1px">◀</button>' +
                    '<button type="button" class="nudge-btn" data-dx="1" data-dy="0" title="Phải 1px">▶</button>' +
                    '<button type="button" class="nudge-btn" data-dx="0" data-dy="1" title="Xuống 1px">▼</button>' +
                '</div>' +
            '</div>' +
            (isStyleditor ? styleGroupHtml(block) : '') +
            '<div class="prop-actions">' +
                '<button id="btn-delete-block" class="btn-delete">🗑️ Xoá bóng thoại</button>' +
            '</div>';

        document.getElementById('edit-text').addEventListener('focus', function () {
            editStartSnapshot = snapshot();
            editStartValue = this.value;
        });
        document.getElementById('edit-text').addEventListener('blur', function () {
            if (editStartSnapshot && this.value !== editStartValue) {
                pushUndo(editStartSnapshot);
            }
            editStartSnapshot = null;
            editStartValue = null;
        });
        document.getElementById('edit-text').addEventListener('input', function () {
            if (isPostrender || isStyleditor) {
                block.translated = this.value;
                markDirty(currentImageIdx);
            } else {
                block.text = this.value;
            }
            requestDraw();
        });

        document.querySelectorAll('.coord-input').forEach(inp => {
            inp.addEventListener('focus', function () {
                editStartSnapshot = snapshot();
                editStartValue = this.value;
                editStartBbox = block.bbox ? [...block.bbox] : null;
            });
            inp.addEventListener('blur', function () {
                const bboxChanged = editStartBbox ? !sameBbox(block.bbox, editStartBbox) : false;
                if (editStartSnapshot && (this.value !== editStartValue || bboxChanged)) {
                    pushUndo(editStartSnapshot);
                }
                editStartSnapshot = null;
                editStartValue = null;
                editStartBbox = null;
                // F3.4/A3.5: round + clamp on blur
                const img = imageCache[currentImageIdx];
                const nb = normalizeBbox([
                    parseInt(document.getElementById('edit-x1').value) || 0,
                    parseInt(document.getElementById('edit-y1').value) || 0,
                    parseInt(document.getElementById('edit-x2').value) || 0,
                    parseInt(document.getElementById('edit-y2').value) || 0
                ], img ? img.width : mainCanvas.width, img ? img.height : mainCanvas.height);
                if (nb) {
                    block.bbox = nb;
                    inp.classList.remove('error');
                } else {
                    inp.classList.add('error');
                    showToast('Bbox không hợp lệ (x2 phải > x1, y2 > y1)', { variant: 'error', duration: 3000 });
                }
                // P1-1 (t5): light refresh WITHOUT rebuilding the panel — the
                // focused input survives, so Tab keeps moving through the
                // coord fields → nudge → style controls → delete → footer (A13.2).
                refreshBlockEditorValues();
                requestDraw();
                requestThumbnails();
                // P2-1 (t5): only mark dirty when the bbox actually changed.
                if ((isPostrender || isStyleditor) && bboxChanged) markDirty(currentImageIdx);
            });
            inp.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); inp.blur(); }  // F3: Enter commits
            });
            inp.addEventListener('input', () => {
                // Robustness: if the field was mutated without a focus first
                // (synthetic/edge input), snapshot the pre-change bbox so the
                // blur handler can still detect the change (P2-1).
                if (!editStartBbox) editStartBbox = block.bbox ? [...block.bbox] : null;
                block.bbox = [
                    parseInt(document.getElementById('edit-x1').value) || 0,
                    parseInt(document.getElementById('edit-y1').value) || 0,
                    parseInt(document.getElementById('edit-x2').value) || 0,
                    parseInt(document.getElementById('edit-y2').value) || 0
                ];
                requestDraw();
                requestThumbnails();
            });
        });
        document.getElementById('btn-delete-block').addEventListener('click', () => { pushUndo(); deleteBlockAt(selectedBlockIdx); });

        document.querySelectorAll('.nudge-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                nudgeBlock(parseInt(btn.dataset.dx), parseInt(btn.dataset.dy), true);
            });
        });

        if (isStyleditor) bindStylePanel(block);

        updateHints();
        // F9: on small screens, selecting a block opens the editor drawer.
        if (window.innerWidth < 1100 && editorToggle) setEditorOpen(true);
    }

    function escapeHtml(str) {
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    // ---- Init ----
    images.forEach((img) => {
        img._originalBlocks = (img.blocks || []).map(b => ({
            text: b.text || '', translated: b.translated || '', bbox: b.bbox ? [...b.bbox] : null
        }));
        if (!img.deletedRegions) img.deletedRegions = [];
        img.dirty = false;
        if (isStyleditor) {
            // V3 F4/F9: erase state per image — eraseRegions/eraseStrokes are
            // MONOTONIC (payload), the preview lists drive eraseLayer and undo.
            if (!img.eraseRegions) img.eraseRegions = [];
            if (!img.erasePreviewRects) img.erasePreviewRects = [];
            if (!img.eraseStrokes) img.eraseStrokes = [];
            if (!img.eraseStrokesPreview) img.eraseStrokesPreview = [];
            (img.blocks || []).forEach(b => { b.style = normalizeStyle(b.style); });
        }
    });

    // a11y base state (F10)
    document.querySelectorAll('.tool-btn').forEach(b => {
        if (!b.hasAttribute('aria-pressed')) b.setAttribute('aria-pressed', String(b.id === 'tool-select'));
        if (!b.hasAttribute('aria-label')) b.setAttribute('aria-label', b.textContent.trim());
    });
    updateHints();

    window.addEventListener('resize', () => {
        fitCanvas();
        requestThumbnails();
    });
    loadImage(0).then(() => {
        if (isStyleditor) {
            // A4.10: reload regions persisted by previous renders (server
            // truth). Runs BEFORE the sessionStorage draft so an un-rendered
            // draft (if any) overrides it.
            loadServerEraseState();
            // Restore this image's un-rendered edits from a previous page visit
            // (chốt captain 8.3: navigating must not lose edits).
            restoreDraftState();
            initBgCanvas();
            redrawEraseLayer();
            initFonts();
            // F1.5/A10.3: translator failure still opens the editor — warn once.
            if (DATA.warning) showToast('⚠️ ' + DATA.warning, { duration: 6000 });
        }
        fitCanvas({ resetZoom: true });
        requestDraw();
        updateNavButtons();
        updateThumbnails();
        updateFooterButtons();
    });

    thumbnails().forEach(el => {
        const idx = parseInt(el.dataset.index);
        renderThumbnail(idx);
    });

    // Safety net: flush the current draft on navigation (markDirty already
    // saves on every edit; this covers any edge path).
    window.addEventListener('beforeunload', () => {
        if (isStyleditor) saveDraftState();
    });
})();