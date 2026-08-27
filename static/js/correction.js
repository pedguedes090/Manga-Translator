(function () {
    const DATA = window.CORRECTION_DATA || {};
    const images = DATA.images || [];
    const sessionId = DATA.sessionId || '';
    // mode === 'postrender' → post-render editor (spec F4): canvas shows the
    // RENDERED image, blocks edit translated text + bbox, re-render wired to
    // POST /re-render-image (spec F5).
    const isPostrender = DATA.mode === 'postrender';
    // Global image index in the session (the API needs it even though the
    // post-render editor only ever loads ONE image per page).
    const globalImageIdx = (typeof DATA.postrenderImageIdx === 'number' && DATA.postrenderImageIdx >= 0) ? DATA.postrenderImageIdx : 0;

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
    if (isPostrender) {
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
            bbox: b.bbox ? [...b.bbox] : null
        }));
    }

    function snapshot() {
        return {
            imageIdx: currentImageIdx,
            images: images.map(img => ({
                blocks: cloneBlocks(img.blocks || []),
                deletedRegions: [...(img.deletedRegions || [])]
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
            // Undo returns the block state that is NOT rendered yet → dirty.
            if (isPostrender && idx === snap.imageIdx) images[idx].dirty = true;
        });
        selectedBlockIdx = -1;
        updateBlockEditor(-1);
        loadImage(currentImageIdx).then(() => { fitCanvas(); requestDraw(); updateThumbnails(); updateNavButtons(); updateFooterButtons(); });
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
        if (!isPostrender || !images[idx]) return;
        images[idx].dirty = true;
        requestDraw();
        requestThumbnails();
        updateFooterButtons();
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

        blocks.forEach((block, i) => {
            const bbox = block.bbox;
            if (!bbox || bbox.length !== 4) return;
            const [x1, y1, x2, y2] = bbox;
            const dirty = isPostrender && images[currentImageIdx].dirty;
            const isClean = isPostrender && !dirty;

            if (i === selectedBlockIdx) {
                ctx.strokeStyle = '#00e676'; ctx.lineWidth = 2.5;
                ctx.fillStyle = 'rgba(0,230,118,0.15)';
            } else if (currentTool === 'delete' && i === hoveredDeleteIdx) {
                ctx.strokeStyle = '#ff1744'; ctx.lineWidth = 2.5;
                ctx.fillStyle = 'rgba(255,23,68,0.2)';
            } else if (dirty) {
                // post-render: dirty = dashed orange (not re-rendered yet)
                ctx.strokeStyle = '#ff9100'; ctx.lineWidth = 1.5;
                ctx.fillStyle = 'rgba(255,145,0,0.08)';
            } else if (isClean) {
                // post-render: clean = solid green (matches the last render)
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
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.setLineDash([]);

            const label = (isPostrender ? (block.translated || block.text || '') : (block.text || '')).substring(0, 12);
            if (label) {
                ctx.font = 'bold 11px sans-serif';
                const lw = ctx.measureText(label).width + 8;
                const lh = 18;
                let ly = y1 - lh - 2; if (ly < 0) ly = y1 + 2;
                // Label chip: selected = green; post-render dirty = orange;
                // post-render clean = dark green; pre-render keeps the
                // incumbent orange for non-selected blocks (no visual regress).
                ctx.fillStyle = (i === selectedBlockIdx) ? '#00e676' : (dirty ? '#ff9100' : (isPostrender ? '#00a152' : '#ff9100'));
                ctx.fillRect(x1, ly, lw, lh);
                ctx.fillStyle = '#000';
                ctx.fillText(label, x1 + 4, ly + 13);
            }
            if (i === selectedBlockIdx) drawHandles(x1, y1, x2, y2);
        });

        if (isDrawing && drawStart && drawEnd) {
            const x = Math.min(drawStart.x, drawEnd.x);
            const y = Math.min(drawStart.y, drawEnd.y);
            const w = Math.abs(drawEnd.x - drawStart.x);
            const h = Math.abs(drawEnd.y - drawStart.y);
            ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.fillStyle = 'rgba(0,229,255,0.1)';
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
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
        if (currentTool === 'delete') {
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
                    updateBlockEditor(selectedBlockIdx);
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
                updateBlockEditor(selectedBlockIdx);
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
        updateBlockEditor(selectedBlockIdx);
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
            case 'a': if (!isPostrender) setTool('add'); break;
            case 'd': setTool('delete'); break;
            case 'arrowleft':
                e.preventDefault();
                if (hasSelection) nudgeBlock(-nudgeStep, 0, !e.repeat);
                else if (currentImageIdx > 0) switchImage(currentImageIdx - 1);
                break;
            case 'arrowright':
                e.preventDefault();
                if (hasSelection) nudgeBlock(nudgeStep, 0, !e.repeat);
                else if (currentImageIdx < images.length - 1) switchImage(currentImageIdx + 1);
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
        mainCanvas.style.cursor = (tool === 'add' || tool === 'delete') ? 'crosshair' : 'default';
        hoveredDeleteIdx = -1;
        updateHints();
        requestDraw();
    }

    // ---- Hint bar (F1.4, F3.2) ----
    function updateHints() {
        if (!hintsBar) return;
        let msg = '';
        if (currentTool === 'delete') {
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
        return (img.blocks || []).map(b => ({
            text: b.text || '',
            translated: b.translated || '',
            bbox: b.bbox ? [...b.bbox] : null
        }));
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
        mainCanvas.style.cursor = busy ? 'wait' : ((currentTool === 'add' || currentTool === 'delete') ? 'crosshair' : 'default');
    }

    function updateFooterButtons() {
        if (!isPostrender) return;
        const one = document.getElementById('btn-rerender-one');
        const all = document.getElementById('btn-save-all');
        const img = images[currentImageIdx];
        if (one) one.disabled = isBusy || !img || !img.dirty;
        if (all) all.disabled = isBusy;
    }

    function rerenderCurrentImage(opts) {
        const o = opts || {};
        const navigateAfter = !!o.navigateAfter;
        if (isBusy) return;
        const img = images[currentImageIdx];
        if (!img) return;

        const payload = {
            session_id: sessionId,
            image_idx: String(globalImageIdx),
            blocks_json: JSON.stringify(currentBlocksPayload()),
            deleted_regions_json: JSON.stringify((img.deletedRegions || []).map(r => [...r]))
        };

        const run = () => {
            setBusy(true, '⏳ Đang render…');
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

    function saveAll() {
        if (isBusy) return;
        const img = images[currentImageIdx];
        if (img && img.dirty) {
            // Persist this image through the single-image endpoint (the
            // /re-render-all endpoint renders from the server's persisted plan,
            // so edits must be saved first), then open the results page.
            rerenderCurrentImage({ navigateAfter: true });
        } else {
            window.location.href = '/translate-result/' + sessionId;
        }
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
        currentImageIdx = idx; selectedBlockIdx = -1; updateBlockEditor(-1);
        loadImage(idx).then(() => { fitCanvas({ resetZoom: true }); requestDraw(); updateThumbnails(); updateNavButtons(); updateFooterButtons(); });
    }

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
        if (btnPrev) btnPrev.disabled = currentImageIdx === 0;
        if (btnNext) btnNext.disabled = currentImageIdx === images.length - 1;
        currentImageLabel.textContent = images[currentImageIdx].name + ' (' + (currentImageIdx + 1) + '/' + images.length + ')';
        updateCanvasAria();
    }

    function updateCanvasAria() {
        const img = images[currentImageIdx];
        if (!img) return;
        const n = (img.blocks || []).length;
        const dirty = isPostrender && img.dirty;
        mainCanvas.setAttribute('role', 'img');
        mainCanvas.setAttribute('aria-label',
            (isPostrender ? 'Ảnh đã dịch ' : 'Ảnh ') + (img.name || '') +
            ' — ' + n + ' bóng thoại' + (dirty ? ', có thay đổi chưa render' : ''));
    }

    // ---- Thumbnails ----
    function updateThumbnails() {
        const items = thumbnails();
        items.forEach(el => el.classList.remove('active'));
        const active = document.querySelector('.thumb-item[data-index="' + currentImageIdx + '"]');
        if (active) {
            active.classList.add('active');
            active.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        }
        items.forEach(el => {
            const idx = parseInt(el.dataset.index);
            const countEl = el.querySelector('.thumb-count');
            if (countEl) countEl.textContent = images[idx].blocks.length + ' blocks';
            // F4: dirty badge on thumbs with un-rendered changes (P0)
            el.classList.toggle('dirty', isPostrender && !!images[idx].dirty);
            renderThumbnail(idx);
        });
        document.getElementById('total-blocks').textContent = images.reduce((s, i) => s + i.blocks.length, 0);
        updateCanvasAria();
    }

    function renderThumbnail(idx) {
        loadImage(idx).then(img => {
            if (!img) return;
            const el = document.querySelector('.thumb-item[data-index="' + idx + '"]');
            if (!el) return;
            const tc = el.querySelector('.thumb-canvas');
            if (!tc) return;
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

    // ---- Editor panel ----
    function updateBlockEditor(idx) {
        if (idx < 0) {
            blockProperties.innerHTML = '<p class="no-sel">Chọn bóng thoại để chỉnh sửa</p>';
            updateHints();
            return;
        }
        const block = getCurrentBlocks()[idx];
        if (!block) return;

        let editStartSnapshot = null;
        let editStartValue = null;
        const w = block.bbox ? (block.bbox[2] - block.bbox[0]) : 0;
        const h = block.bbox ? (block.bbox[3] - block.bbox[1]) : 0;

        const textareaLabel = isPostrender ? 'Bản dịch' : 'Nội dung text';
        const textareaValue = escapeHtml(isPostrender ? (block.translated || '') : (block.text || ''));
        const originalLine = isPostrender && block.text
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
            if (isPostrender) {
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
            });
            inp.addEventListener('blur', function () {
                if (editStartSnapshot && this.value !== editStartValue) {
                    pushUndo(editStartSnapshot);
                }
                editStartSnapshot = null;
                editStartValue = null;
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
                updateBlockEditor(selectedBlockIdx);
                requestDraw();
                requestThumbnails();
                if (isPostrender) markDirty(currentImageIdx);
            });
            inp.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); inp.blur(); }  // F3: Enter commits
            });
            inp.addEventListener('input', () => {
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
})();