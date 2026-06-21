/**
 * RxPilot — Frontend Application
 *
 * Vanilla JS handling file upload, API calls, DOM updates,
 * drag-and-drop, and real-time status checks.
 */

// ── API Base URL ──
const API_BASE = window.location.hostname === 'localhost' && window.location.port === '3000'
    ? 'http://localhost:8000'  // Dev: frontend on 3000, backend on 8000
    : '';                       // Docker: nginx proxies /v1/ to backend

// ── DOM References ──
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dropZone = $('#drop-zone');
const fileInput = $('#file-input');
const browseBtn = $('#browse-btn');
const clearBtn = $('#clear-btn');
const extractBtn = $('#extract-btn');
const retryBtn = $('#retry-btn');
const refreshBillsBtn = $('#refresh-bills-btn');

const imagePreview = $('#image-preview');
const previewImage = $('#preview-image');
const previewFilename = $('#preview-filename');
const dropZoneContent = $('#drop-zone-content');

const uploadCard = $('#upload-card');
const processingCard = $('#processing-card');
const resultsCard = $('#results-card');
const errorCard = $('#error-card');
const welcomeCard = $('#welcome-card');

const resultMeta = $('#result-meta');
const resultsBody = $('#results-body');
const errorMessage = $('#error-message');
const historyBody = $('#history-body');

const statusDot = $('.status-dot');
const statusText = $('.status-text');

// ── State ──
let selectedFile = null;

// ── Health Check ──
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();

        statusDot.className = 'status-dot';
        if (data.status === 'healthy') {
            statusDot.classList.add('healthy');
            statusText.textContent = 'All Systems Go';
        } else {
            statusDot.classList.add('degraded');
            statusText.textContent = 'Degraded';
        }
    } catch {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Backend Offline';
    }
}

// ── File Selection ──
function handleFileSelect(file) {
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('Please select a valid image file (JPG, PNG, GIF, or WebP)');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('File is too large. Maximum size is 10MB.');
        return;
    }

    selectedFile = file;
    showPreview(file);
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewFilename.textContent = file.name;
        imagePreview.classList.remove('hidden');
        dropZone.classList.add('hidden');
        extractBtn.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    imagePreview.classList.add('hidden');
    dropZone.classList.remove('hidden');
    extractBtn.classList.add('hidden');
}

// ── Drag & Drop ──
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFileSelect(fileInput.files[0]);
});
clearBtn.addEventListener('click', clearSelection);

// ── Extraction ──
extractBtn.addEventListener('click', runExtraction);
retryBtn.addEventListener('click', () => {
    hideCard(errorCard);
    showCard(welcomeCard);
    clearSelection();
});

async function runExtraction() {
    if (!selectedFile) return;

    // Show processing state
    hideCard(welcomeCard);
    hideCard(resultsCard);
    hideCard(errorCard);
    showCard(processingCard);
    extractBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const res = await fetch(`${API_BASE}/v1/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `Upload failed: ${res.status}`);
        }

        const data = await res.json();
        hideCard(processingCard);

        if (data.status === 'failed' || data.error) {
            showError(data.error || 'Extraction failed');
        } else {
            showResults(data);
        }

        // Refresh history
        loadBills();

    } catch (err) {
        hideCard(processingCard);
        showError(err.message);
    } finally {
        extractBtn.disabled = false;
    }
}

// ── Display Results ──
function showResults(data) {
    hideCard(welcomeCard);
    hideCard(errorCard);

    // Meta tags
    resultMeta.innerHTML = `
        <span class="meta-tag count">📦 ${data.items_count} items</span>
        <span class="meta-tag time">⏱ ${data.processing_time_ms.toFixed(0)}ms</span>
        <span class="meta-tag cost">💰 $${data.estimated_cost_usd.toFixed(4)}</span>
    `;

    // Item cards
    if (data.items && data.items.length > 0) {
        resultsBody.innerHTML = data.items.map((item, idx) => `
            <div class="item-card" style="animation-delay: ${idx * 0.05}s">
                <div class="item-name">
                    <span class="pill-icon">💊</span>
                    ${escapeHtml(item.medicine_name)}
                </div>
                <div class="item-fields">
                    ${renderField('Batch', item.batch_number)}
                    ${renderField('Expiry', item.expiry_date, 'expiry')}
                    ${renderField('Mfg Date', item.manufacture_date)}
                    ${renderField('Quantity', item.quantity ? `${item.quantity} ${item.unit || ''}`.trim() : null)}
                    ${renderField('Supplier', item.supplier_name)}
                    ${renderField('Price', item.price != null ? `${item.currency || 'INR'} ${item.price.toFixed(2)}` : null, 'price')}
                </div>
            </div>
        `).join('');
    } else {
        resultsBody.innerHTML = `
            <div class="empty-text">
                No medicine items were found on this bill.
                Try uploading a clearer image.
            </div>
        `;
    }

    showCard(resultsCard);
}

function renderField(label, value, className = '') {
    const displayValue = value != null && value !== ''
        ? `<span class="field-value ${className}">${escapeHtml(String(value))}</span>`
        : `<span class="field-value empty">—</span>`;

    return `
        <div class="field">
            <span class="field-label">${label}</span>
            ${displayValue}
        </div>
    `;
}

function showError(message) {
    errorMessage.textContent = message;
    hideCard(welcomeCard);
    hideCard(resultsCard);
    showCard(errorCard);
}

// ── History ──
async function loadBills() {
    try {
        const res = await fetch(`${API_BASE}/v1/bills`);
        if (!res.ok) return;

        const data = await res.json();
        const bills = data.bills || [];

        if (bills.length === 0) {
            historyBody.innerHTML = `
                <p class="empty-text">No bills processed yet. Upload one above to get started!</p>
            `;
            return;
        }

        historyBody.innerHTML = `
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Items</th>
                        <th>Processing</th>
                        <th>Cost</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
                    ${bills.map(bill => `
                        <tr>
                            <td>
                                <span class="status-badge ${bill.status}">
                                    ${bill.status === 'completed' ? '✓' : bill.status === 'failed' ? '✕' : '◌'}
                                    ${bill.status}
                                </span>
                            </td>
                            <td>${bill.items_count} items</td>
                            <td style="font-family: var(--font-mono); font-size: 0.78rem;">
                                ${bill.processing_time_ms ? bill.processing_time_ms.toFixed(0) + 'ms' : '—'}
                            </td>
                            <td style="font-family: var(--font-mono); font-size: 0.78rem;">
                                ${bill.estimated_cost_usd ? '$' + bill.estimated_cost_usd.toFixed(4) : '—'}
                            </td>
                            <td style="font-size: 0.78rem; color: var(--text-muted);">
                                ${formatDate(bill.created_at)}
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch {
        // Silent fail — history is non-critical
    }
}

refreshBillsBtn.addEventListener('click', loadBills);

// ── Utilities ──
function showCard(el) { el.classList.remove('hidden'); }
function hideCard(el) { el.classList.add('hidden'); }

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function formatDate(dateStr) {
    if (!dateStr) return '—';
    try {
        const d = new Date(dateStr);
        return d.toLocaleString('en-IN', {
            day: '2-digit', month: 'short', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
        });
    } catch {
        return dateStr;
    }
}

// ── Init ──
checkHealth();
loadBills();
// Poll health every 30s
setInterval(checkHealth, 30000);
