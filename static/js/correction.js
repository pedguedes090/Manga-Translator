(function() {
    const DATA = window.CORRECTION_DATA;
    const images = DATA.images;
    const sessionId = DATA.sessionId;

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
    const MAX_UNDO = 60;
    const MIN_ZOOM = 0.25;
    const MAX_ZOOM = 8;
    const ZOOM_STEP = 1.2;

    const mainCanvas = document.getElementById('main-canvas');
    const ctx = mainCanvas.getContext('2d');
    const canvasOuter = document.querySelector('.canvas-outer');
    const currentImageLabel = document.getElementById('current-image-label');
    const thumbnails = () => document.querySelectorAll('.thumb-item');
    const blockProperties = document.getElementById('block-properties');
    const modifiedBlocksInput = document.getElementById('modified-blocks-input');
    const ocrStatus = document.getElementById('ocr-status');
    const zoomLabel = document.getElementById('zoom-label');

    const imageCache = {};
    const thumbnailKeys = {};

    // ---- Clean OCR text ----
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

    // ---- Undo/Redo ----
    function cloneBlocks(blocks) {
        return blocks.map(b => ({ text: b.text || '', bbox: b.bbox ? [...b.bbox] : null }));
    }

    function snapshot() {
        return {
            imageIdx: currentImageIdx,
            images: images.map(img => ({ blocks: cloneBlocks(img.blocks || []) }))
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
        });
        selectedBlockIdx = -1;
        updateBlockEditor(-1);
        loadImage(currentImageIdx).then(() => { fitCanvas(); requestDraw(); updateThumbnails(); updateNavButtons(); });
    }

    function undo() {
        if (undoStack.length === 0) return;
        redoStack.push(snapshot());
        restoreSnapshot(undoStack.pop());
        showToast('Đã hoàn tác ↩');
    }

    function redo() {
        if (redoStack.length === 0) return;
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
            img.src = 'data:image/jpeg;base64,' + images[idx].data;
        });
    }

    function getBlocks(idx) { return images[idx].blocks; }
    function setBlocks(idx, blocks) { images[idx].blocks = blocks; }
    function getCurrentBlocks() { return getBlocks(currentImageIdx); }

    function sameBbox(a, b) {
        return a && b && a.length === 4 && b.length === 4 &&
            a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
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
        const anchorX = anchorClientX ?? (outerRect.left + outerRect.width / 2);
        const anchorY = anchorClientY ?? (outerRect.top + outerRect.height / 2);
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

            if (i === selectedBlockIdx) {
                ctx.strokeStyle = '#00e676'; ctx.lineWidth = 2.5;
                ctx.fillStyle = 'rgba(0,230,118,0.15)';
            } else if (currentTool === 'delete' && i === hoveredDeleteIdx) {
                ctx.strokeStyle = '#ff1744'; ctx.lineWidth = 2.5;
                ctx.fillStyle = 'rgba(255,23,68,0.2)';
            } else {
                ctx.strokeStyle = '#ff9100'; ctx.lineWidth = 1.5;
                ctx.fillStyle = 'rgba(255,145,0,0.08)';
            }
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            const label = (block.text || '').substring(0, 12);
            if (label) {
                ctx.font = 'bold 11px sans-serif';
                const lw = ctx.measureText(label).width + 8;
                const lh = 18;
                let ly = y1 - lh - 2; if (ly < 0) ly = y1 + 2;
                ctx.fillStyle = i === selectedBlockIdx ? '#00e676' : '#ff9100';
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

    function drawHandles(x1, y1, x2, y2) {
        const s = 6;
        const corners = [[x1,y1],[x2,y1],[x1,y2],[x2,y2],[(x1+x2)/2,y1],[(x1+x2)/2,y2],[x1,(y1+y2)/2],[x2,(y1+y2)/2]];
        ctx.fillStyle = '#00e676';
        corners.forEach(([cx,cy]) => ctx.fillRect(cx-s/2, cy-s/2, s, s));
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

    let hoveredDeleteIdx = -1;

    // ---- Auto OCR ----
    function ocrNewBlock(idx, options = {}) {
        const { recordUndo = true } = options;
        const block = getCurrentBlocks()[idx];
        if (!block || !block.bbox || isOcrPending) return;
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

    // ---- Mouse ----
    mainCanvas.addEventListener('mousedown', (e) => {
        if (e.button === 1 || isSpaceDown) {
            startPan(e);
            return;
        }

        const pos = getCanvasCoords(e);

        if (currentTool === 'add') {
            isDrawing = true; drawStart = pos; drawEnd = pos;
            selectedBlockIdx = -1; updateBlockEditor(-1);
            return;
        }

        if (currentTool === 'delete') {
            const hitIdx = findBlockAt(pos.x, pos.y);
            if (hitIdx >= 0) { pushUndo(); removeBlock(hitIdx); }
            return;
        }

        // select tool
        const hitIdx = findBlockAt(pos.x, pos.y);
        if (hitIdx >= 0) {
            selectedBlockIdx = hitIdx;
            isDragging = true; dragBlockIdx = hitIdx;
            const bbox = getCurrentBlocks()[hitIdx].bbox;
            dragStartBbox = [...bbox];
            dragStartSnapshot = snapshot();
            dragOffset = { x: pos.x - bbox[0], y: pos.y - bbox[1] };
            updateBlockEditor(hitIdx);
        } else {
            selectedBlockIdx = -1; updateBlockEditor(-1);
        }
        requestDraw();
    });

    mainCanvas.addEventListener('mousemove', (e) => {
        const pos = getCanvasCoords(e);
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
                requestThumbnails();
            }
            dragStartBbox = null;
            dragStartSnapshot = null;
        }
    }

    mainCanvas.addEventListener('mouseup', () => {
        finishDrag();
        if (isDrawing && drawStart && drawEnd) {
            const x1 = Math.round(Math.min(drawStart.x, drawEnd.x));
            const y1 = Math.round(Math.min(drawStart.y, drawEnd.y));
            const x2 = Math.round(Math.max(drawStart.x, drawEnd.x));
            const y2 = Math.round(Math.max(drawStart.y, drawEnd.y));
            if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
                pushUndo();
                const blocks = getCurrentBlocks();
                const newBlock = { text: '', bbox: [x1, y1, x2, y2] };
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
        requestDraw(); requestThumbnails();
    });

    mainCanvas.addEventListener('mouseleave', () => {
        finishDrag();
        isDragging = false; dragBlockIdx = -1; dragOffset = null;
        isDrawing = false; drawStart = null; drawEnd = null;
        hoveredDeleteIdx = -1; requestDraw(); requestThumbnails();
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

    window.addEventListener('mousemove', (e) => {
        if (!isPanning || !panStart || !canvasOuter) return;
        canvasOuter.scrollLeft = panStart.scrollLeft - (e.clientX - panStart.x);
        canvasOuter.scrollTop = panStart.scrollTop - (e.clientY - panStart.y);
    });

    window.addEventListener('mouseup', stopPan);

    if (canvasOuter) {
        canvasOuter.addEventListener('wheel', (e) => {
            if (!(e.ctrlKey || e.metaKey)) return;
            e.preventDefault();
            const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
            setZoom(zoomLevel * factor, e.clientX, e.clientY);
        }, { passive: false });
    }

    // ---- Keyboard ----
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

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

        switch (e.key.toLowerCase()) {
            case '+':
            case '=': e.preventDefault(); setZoom(zoomLevel * ZOOM_STEP); break;
            case '-': e.preventDefault(); setZoom(zoomLevel / ZOOM_STEP); break;
            case '0': e.preventDefault(); zoomLevel = 1; fitCanvas(); break;
            case '1': e.preventDefault(); setActualSize(); break;
            case 's': setTool('select'); break;
            case 'a': setTool('add'); break;
            case 'd': setTool('delete'); break;
            case 'arrowleft': e.preventDefault(); if (currentImageIdx > 0) switchImage(currentImageIdx - 1); break;
            case 'arrowright': e.preventDefault(); if (currentImageIdx < images.length - 1) switchImage(currentImageIdx + 1); break;
            case 'escape': selectedBlockIdx = -1; updateBlockEditor(-1); isDrawing = false; drawStart = null; drawEnd = null; requestDraw(); break;
            case 'delete': if (selectedBlockIdx >= 0 && currentTool === 'select') { pushUndo(); removeBlock(selectedBlockIdx); } break;
        }
    });

    document.addEventListener('keyup', (e) => {
        if (e.code !== 'Space') return;
        isSpaceDown = false;
        if (!isPanning) {
            mainCanvas.style.cursor = currentTool === 'add' || currentTool === 'delete' ? 'crosshair' : 'default';
        }
    });

    function removeBlock(idx) {
        const blocks = getCurrentBlocks();
        if (idx < 0 || idx >= blocks.length) return;
        blocks.splice(idx, 1);
        setBlocks(currentImageIdx, blocks);
        if (selectedBlockIdx === idx) { selectedBlockIdx = -1; updateBlockEditor(-1); }
        else if (selectedBlockIdx > idx) { selectedBlockIdx--; updateBlockEditor(selectedBlockIdx); }
        requestDraw(); updateThumbnails(); showToast('Đã xoá bóng thoại');
    }

    function setTool(tool) {
        currentTool = tool;
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        const btn = document.getElementById('tool-' + tool);
        if (btn) btn.classList.add('active');
        if (tool !== 'select') { selectedBlockIdx = -1; updateBlockEditor(-1); }
        mainCanvas.style.cursor = (tool === 'add' || tool === 'delete') ? 'crosshair' : 'default';
        hoveredDeleteIdx = -1; requestDraw();
    }

    // ---- Buttons ----
    document.getElementById('tool-select').addEventListener('click', () => setTool('select'));
    document.getElementById('tool-add').addEventListener('click', () => setTool('add'));
    document.getElementById('tool-delete').addEventListener('click', () => setTool('delete'));
    document.getElementById('tool-undo').addEventListener('click', undo);
    document.getElementById('tool-redo').addEventListener('click', redo);
    document.getElementById('zoom-out').addEventListener('click', () => setZoom(zoomLevel / ZOOM_STEP));
    document.getElementById('zoom-in').addEventListener('click', () => setZoom(zoomLevel * ZOOM_STEP));
    document.getElementById('zoom-fit').addEventListener('click', () => { zoomLevel = 1; fitCanvas(); });
    document.getElementById('zoom-actual').addEventListener('click', setActualSize);
    document.getElementById('tool-reset').addEventListener('click', () => {
        if (confirm('Reset tất cả bóng thoại về kết quả OCR gốc?')) {
            undoStack = []; redoStack = [];
            images.forEach((img) => {
                img.blocks = img._originalBlocks ? img._originalBlocks.map(b => ({
                    text: b.text || '', bbox: b.bbox ? [...b.bbox] : null
                })) : [];
            });
            selectedBlockIdx = -1; updateBlockEditor(-1); requestDraw(); updateThumbnails();
            showToast('Đã reset về OCR gốc');
        }
    });

    document.getElementById('btn-prev').addEventListener('click', () => { if (currentImageIdx > 0) switchImage(currentImageIdx - 1); });
    document.getElementById('btn-next').addEventListener('click', () => { if (currentImageIdx < images.length - 1) switchImage(currentImageIdx + 1); });

    function switchImage(idx) {
        currentImageIdx = idx; selectedBlockIdx = -1; updateBlockEditor(-1);
        loadImage(idx).then(() => { fitCanvas({ resetZoom: true }); requestDraw(); updateThumbnails(); updateNavButtons(); });
    }

    function fitCanvas(options = {}) {
        const { resetZoom = false } = options;
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
        document.getElementById('btn-prev').disabled = currentImageIdx === 0;
        document.getElementById('btn-next').disabled = currentImageIdx === images.length - 1;
        currentImageLabel.textContent = images[currentImageIdx].name + ' (' + (currentImageIdx + 1) + '/' + images.length + ')';
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
            renderThumbnail(idx);
        });
        document.getElementById('total-blocks').textContent = images.reduce((s, i) => s + i.blocks.length, 0);
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
    });

    // ---- Editor ----
    function updateBlockEditor(idx) {
        if (idx < 0) {
            blockProperties.innerHTML = '<p class="no-sel">Chọn bóng thoại để chỉnh sửa</p>';
            return;
        }
        const block = getCurrentBlocks()[idx];
        if (!block) return;

        let editStartSnapshot = null;
        let editStartValue = null;
        blockProperties.innerHTML = `
            <div class="prop-group">
                <label>Nội dung text</label>
                <textarea id="edit-text" rows="3" class="prop-textarea">${escapeHtml(block.text || '')}</textarea>
            </div>
            <div class="prop-group">
                <label>Vị trí (x1,y1,x2,y2)</label>
                <div class="prop-coords">
                    <input type="number" id="edit-x1" value="${block.bbox?.[0]||0}" class="coord-input" placeholder="x1">
                    <input type="number" id="edit-y1" value="${block.bbox?.[1]||0}" class="coord-input" placeholder="y1">
                    <input type="number" id="edit-x2" value="${block.bbox?.[2]||0}" class="coord-input" placeholder="x2">
                    <input type="number" id="edit-y2" value="${block.bbox?.[3]||0}" class="coord-input" placeholder="y2">
                </div>
            </div>
            <button id="btn-clean" class="btn-reocr">🧹 Clean text</button>
            <button id="btn-reocr" class="btn-reocr">🔍 OCR lại</button>
            <button id="btn-delete-block" class="btn-delete">🗑️ Xoá</button>
        `;

        document.getElementById('edit-text').addEventListener('focus', function() {
            editStartSnapshot = snapshot();
            editStartValue = this.value;
        });
        document.getElementById('edit-text').addEventListener('blur', function() {
            if (editStartSnapshot && this.value !== editStartValue) {
                pushUndo(editStartSnapshot);
            }
            editStartSnapshot = null;
            editStartValue = null;
        });
        document.getElementById('edit-text').addEventListener('input', function() { block.text = this.value; requestDraw(); });

        document.querySelectorAll('.coord-input').forEach(inp => {
            inp.addEventListener('focus', function() {
                editStartSnapshot = snapshot();
                editStartValue = this.value;
            });
            inp.addEventListener('blur', function() {
                if (editStartSnapshot && this.value !== editStartValue) {
                    pushUndo(editStartSnapshot);
                }
                editStartSnapshot = null;
                editStartValue = null;
            });
            inp.addEventListener('input', () => {
                block.bbox = [
                    parseInt(document.getElementById('edit-x1').value)||0,
                    parseInt(document.getElementById('edit-y1').value)||0,
                    parseInt(document.getElementById('edit-x2').value)||0,
                    parseInt(document.getElementById('edit-y2').value)||0
                ];
                requestDraw();
                requestThumbnails();
            });
        });
        document.getElementById('btn-delete-block').addEventListener('click', () => { pushUndo(); removeBlock(selectedBlockIdx); });
        document.getElementById('btn-reocr').addEventListener('click', () => ocrNewBlock(selectedBlockIdx));
        document.getElementById('btn-clean').addEventListener('click', () => {
            pushUndo();
            block.text = cleanOcrText(block.text);
            updateBlockEditor(selectedBlockIdx);
            requestDraw();
            showToast('Đã clean text');
        });
    }

    function escapeHtml(str) {
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    // ---- Continue ----
    document.getElementById('btn-continue').addEventListener('click', () => {
        const allBlocks = images.map((img, idx) => ({ image_idx: idx, blocks: img.blocks }));
        modifiedBlocksInput.value = JSON.stringify(allBlocks);
        document.getElementById('continue-form').submit();
    });

    function showToast(msg) {
        const t = document.getElementById('toast');
        t.textContent = msg; t.classList.add('show');
        clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove('show'), 2000);
    }

    // ---- Init ----
    images.forEach((img) => {
        img._originalBlocks = img.blocks.map(b => ({
            text: b.text || '', bbox: b.bbox ? [...b.bbox] : null
        }));
    });

    window.addEventListener('resize', () => {
        fitCanvas();
        requestThumbnails();
    });
    loadImage(0).then(() => { fitCanvas({ resetZoom: true }); requestDraw(); updateNavButtons(); updateThumbnails(); });

    thumbnails().forEach(el => {
        const idx = parseInt(el.dataset.index);
        renderThumbnail(idx);
    });
})();
