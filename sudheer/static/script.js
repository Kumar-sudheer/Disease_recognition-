// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('panel-' + btn.dataset.crop).classList.add('active');
    });
});

// ── Generic upload panel wiring (multi-image) ─────────────────────────────────
function wirePanel({ prefix, apiUrl, renderResult, extraFormData }) {
    const uploadArea       = document.getElementById(`${prefix}-uploadArea`);
    const fileInput        = document.getElementById(`${prefix}-fileInput`);
    const previewContainer = document.getElementById(`${prefix}-previewContainer`);
    const analyzeBtn       = document.getElementById(`${prefix}-analyzeBtn`);
    const results          = document.getElementById(`${prefix}-results`);
    const report           = document.getElementById(`${prefix}-report`);

    let selectedFiles = [];

    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', e => handleFiles(Array.from(e.target.files)));

    uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(Array.from(e.dataTransfer.files));
    });

    analyzeBtn.addEventListener('click', analyze);

    function handleFiles(files) {
        if (!files.length) return;
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
        const valid = files.filter(f => validTypes.includes(f.type) && f.size <= 16 * 1024 * 1024);
        const skipped = files.length - valid.length;
        if (!valid.length) { alert('No valid images found. Use PNG, JPG, JPEG, BMP, TIFF up to 16MB each.'); return; }
        if (skipped > 0) alert(`${skipped} file(s) skipped — invalid type or exceeded 16MB.`);
        if (valid.length > 4) { alert('Maximum 4 images allowed at a time. Only the first 4 will be used.'); valid.splice(4); }

        selectedFiles = valid;

        // Render thumbnail grid
        previewContainer.innerHTML = '';
        valid.forEach(file => {
            const reader = new FileReader();
            reader.onload = e => {
                const wrap = document.createElement('div');
                wrap.className = 'multi-preview-thumb-wrap';
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'multi-preview-thumb';
                img.title = file.name;
                const label = document.createElement('span');
                label.className = 'multi-preview-label';
                label.textContent = file.name.length > 18 ? file.name.slice(0, 15) + '…' : file.name;
                wrap.appendChild(img);
                wrap.appendChild(label);
                previewContainer.appendChild(wrap);
            };
            reader.readAsDataURL(file);
        });
        previewContainer.style.display = 'flex';
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';

        const countLabel = uploadArea.querySelector('.upload-count-label');
        if (countLabel) countLabel.textContent = `${valid.length} image${valid.length > 1 ? 's' : ''} selected`;

        analyzeBtn.disabled = false;
        results.style.display = 'none';
    }

    async function analyze() {
        if (!selectedFiles.length) return;
        setLoading(true);

        const form = new FormData();
        selectedFiles.forEach(f => form.append('files', f));
        if (extraFormData) extraFormData(form);

        try {
            const res  = await fetch(apiUrl, { method: 'POST', body: form });
            const data = await res.json();
            if (data.success) {
                report.innerHTML = data.results.map((r, i) => {
                    const name = r.filename || selectedFiles[i]?.name || `Image ${i + 1}`;
                    if (!r.success) {
                        return `<div class="batch-result-card batch-result-error">
                            <div class="batch-result-header"><span class="batch-filename">⚠ ${escHtml(name)}</span></div>
                            <p style="color:var(--danger);margin-top:.5rem">Error: ${escHtml(r.error)}</p>
                        </div>`;
                    }
                    return `<div class="batch-result-card">
                        <div class="batch-result-header">
                            <img src="${r.image}" class="batch-result-thumb" alt="${escHtml(name)}">
                            <span class="batch-filename">${escHtml(name)}</span>
                        </div>
                        <div class="batch-result-body">${renderResult(r.analysis)}</div>
                    </div>`;
                }).join('');
                results.style.display = 'block';
                results.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                alert('Error: ' + data.error);
            }
        } catch (err) {
            alert('Request failed: ' + err.message);
        } finally {
            setLoading(false);
        }
    }

    function setLoading(loading) {
        analyzeBtn.disabled = loading;
        analyzeBtn.querySelector('.btn-text').style.display   = loading ? 'none' : 'inline';
        analyzeBtn.querySelector('.btn-loader').style.display = loading ? 'flex'  : 'none';
    }
}

function escHtml(str) {
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function statusBadge(status) {
    const labels = { healthy: '✓ Healthy', warning: '⚠ Warning', critical: '⚠ Critical' };
    return `<div class="status-badge status-${status}">${labels[status] || status}</div>`;
}

function recommendationsList(recs) {
    if (!recs || !recs.length) return '';
    return `<h4 style="margin:1.25rem 0 .6rem;color:var(--gray-900)">Recommended Actions</h4>
        <ul class="recommendations-list">${recs.map(r => `<li>${r}</li>`).join('')}</ul>`;
}

function top3List(top3) {
    if (!top3 || !top3.length) return '';
    return `<h4 style="margin:1.25rem 0 .5rem;color:var(--gray-900)">Top Predictions</h4>
        <ul class="top3-list">${top3.map(t => `<li><span>${t.name}</span><span>${t.confidence}%</span></li>`).join('')}</ul>`;
}

function pesticideCards(pesticides) {
    if (!pesticides || !pesticides.length) return '';
    let html = `<h4 style="margin:1.25rem 0 .6rem;color:var(--gray-900)">🧪 Recommended Pesticides</h4>`;
    pesticides.forEach(p => {
        html += `<div class="pesticide-card">
            <div class="pesticide-header">
                <span class="pesticide-product">${p.product}</span>
                <span class="pesticide-group">${p.chemical_group}</span>
            </div>
            <div class="pesticide-detail"><strong>Active Ingredient:</strong> ${p.active_ingredient}</div>
            <div class="pesticide-detail"><strong>Dosage:</strong> ${p.dosage}</div>
            <div class="pesticide-detail"><strong>Application:</strong> ${p.application}</div>
            <div class="pesticide-detail pesticide-moa"><strong>Mode of Action:</strong> ${p.mode_of_action}</div>
        </div>`;
    });
    return html;
}

// ── Sugarcane renderer ────────────────────────────────────────────────────────
function renderSugarcane(a) {
    if (!a) return '<p>No data returned.</p>';
    let html = statusBadge(a.status);
    html += `<p style="margin-bottom:1rem;font-size:1rem"><strong>${a.message}</strong></p>`;

    if (a.detections && a.detections.length) {
        html += `<h4 style="margin-bottom:.75rem;color:var(--gray-900)">Detection Details</h4>`;
        a.detections.forEach(d => {
            html += `<div class="detection-item" style="border-left-color:${d.color}">
                <div class="detection-header">
                    <span class="detection-class" style="color:${d.color}">${d.icon} ${d.class}</span>
                    <span class="detection-confidence">${d.confidence}% confident</span>
                </div>
                <p class="detection-description"><strong>Count:</strong> ${d.count} | <strong>Severity:</strong> ${d.severity}</p>
                <p class="detection-description">${d.description}</p>
                <div class="detection-recommendation"><strong>Recommendation:</strong> ${d.recommendation}</div>
                ${pesticideCards(d.pesticides)}
            </div>`;
        });
    }

    html += recommendationsList(a.recommendations);
    html += `<p style="margin-top:1.25rem;padding-top:1.25rem;border-top:1px solid var(--gray-200);color:var(--gray-600);font-size:.8rem">
        <strong>Analysis type:</strong> ${a.model_type === 'detection' ? 'Object Detection' : 'Instance Segmentation'}</p>`;
    return html;
}

// ── Rice renderer ─────────────────────────────────────────────────────────────
function renderRice(a) {
    if (!a) return '<p>No data returned.</p>';
    const sevColor = a.severity_pct === 0 ? '#10b981' : a.severity_pct <= 25 ? '#f59e0b' : '#ef4444';
    let html = statusBadge(a.status);
    html += `<p style="margin-bottom:.75rem"><strong>Disease:</strong> ${a.disease || a.class}</p>`;
    html += `<p style="margin-bottom:.75rem"><strong>Severity Level:</strong> ${a.severity_label || 'Unknown'}</p>`;
    html += `<p style="margin-bottom:.75rem"><strong>Confidence:</strong> ${a.confidence}%</p>`;
    html += `<div class="severity-bar-wrap">
        <p style="margin-bottom:.35rem;font-size:.88rem"><strong>Severity:</strong> ${a.severity_pct}%</p>
        <div class="severity-bar-bg">
            <div class="severity-bar-fill" style="width:${a.severity_pct}%;background:${sevColor}"></div>
        </div>
    </div>`;
    html += recommendationsList(a.recommendations);
    html += top3List(a.top3);
    html += pesticideCards(a.pesticides);
    return html;
}

// ── Wheat renderer ────────────────────────────────────────────────────────────
function renderWheat(a) {
    if (!a) return '<p>No data returned.</p>';
    let html = statusBadge(a.status);
    html += `<p style="margin-bottom:.75rem"><strong>Diagnosis:</strong> 
        <span style="color:${a.color};font-weight:700">${a.class}</span></p>`;
    html += `<p style="margin-bottom:.75rem"><strong>Confidence:</strong> ${a.confidence}%</p>`;
    html += recommendationsList(a.recommendations);
    html += top3List(a.top3);
    html += pesticideCards(a.pesticides);
    return html;
}

// ── Wire all three panels ─────────────────────────────────────────────────────
wirePanel({
    prefix: 'sc',
    apiUrl: '/api/analyze/sugarcane',
    renderResult: renderSugarcane,
    extraFormData: null
});

wirePanel({
    prefix: 'ri',
    apiUrl: '/api/analyze/rice',
    renderResult: renderRice,
    extraFormData: null
});

wirePanel({
    prefix: 'wh',
    apiUrl: '/api/analyze/wheat',
    renderResult: renderWheat,
    extraFormData: null
});


