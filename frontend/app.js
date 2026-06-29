/**
 * RxPilot — Frontend Application (Phase 2)
 *
 * Vanilla JS handling file upload, API calls, DOM updates,
 * drag-and-drop, and real-time status checks.
 * Phase 2 adds: validation warnings + safety alert rendering.
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
const validationCard = $('#validation-card');
const safetyCard = $('#safety-card');
const errorCard = $('#error-card');
const welcomeCard = $('#welcome-card');

const resultMeta = $('#result-meta');
const resultsBody = $('#results-body');
const validationBody = $('#validation-body');
const validationCount = $('#validation-count');
const safetyBody = $('#safety-body');
const safetyCount = $('#safety-count');
const errorMessage = $('#error-message');
const historyBody = $('#history-body');

const statusDot = $('.status-dot');
const statusText = $('.status-text');

// ── Pipeline Step refs ──
const stepValidate = $('#step-validate');
const stepSafety = $('#step-safety');
const stepStore = $('#step-store');

// ── State ──
let selectedFile = null;
let pipelineStepTimer = null;

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

// ── Pipeline Step Animation ──
function animatePipelineSteps() {
    // Simulate step progression during API call
    // Step 1: upload (already active on show)
    let step = 0;
    const steps = [
        { el: stepValidate, label: 'Validating output', delay: 1800 },
        { el: stepSafety,   label: 'Checking drug safety', delay: 3500 },
        { el: stepStore,    label: 'Storing results', delay: 5500 },
    ];

    steps.forEach(({ el, delay }) => {
        setTimeout(() => {
            if (el) {
                el.classList.add('active');
                const icon = el.querySelector('.step-icon');
                if (icon) icon.classList.add('spinner-sm');
            }
        }, delay);
    });
}

function resetPipelineSteps() {
    [stepValidate, stepSafety, stepStore].forEach(el => {
        if (el) {
            el.classList.remove('active', 'done');
            const icon = el.querySelector('.step-icon');
            if (icon) {
                icon.classList.remove('spinner-sm');
                icon.textContent = '○';
            }
        }
    });
}

// ── Extraction ──
extractBtn.addEventListener('click', runExtraction);
retryBtn.addEventListener('click', () => {
    hideCard(errorCard);
    hideCard(validationCard);
    hideCard(safetyCard);
    showCard(welcomeCard);
    clearSelection();
});

async function runExtraction() {
    if (!selectedFile) return;

    // Reset and show processing state
    hideCard(welcomeCard);
    hideCard(resultsCard);
    hideCard(validationCard);
    hideCard(safetyCard);
    hideCard(errorCard);
    showCard(processingCard);
    resetPipelineSteps();
    animatePipelineSteps();
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
    const warningCount = (data.validation_warnings || []).length;
    const alertCount = (data.safety_alerts || []).length;
    const warningBadge = warningCount > 0
        ? `<span class="meta-tag" style="background:rgba(245,158,11,0.1);color:var(--accent-amber);border:1px solid rgba(245,158,11,0.2);">⚠️ ${warningCount} warning${warningCount > 1 ? 's' : ''}</span>`
        : '';
    const alertBadge = alertCount > 0
        ? `<span class="meta-tag" style="background:rgba(244,63,94,0.1);color:var(--accent-rose);border:1px solid rgba(244,63,94,0.2);">🛡️ ${alertCount} alert${alertCount > 1 ? 's' : ''}</span>`
        : '';

    resultMeta.innerHTML = `
        <span class="meta-tag count">📦 ${data.items_count} items</span>
        <span class="meta-tag time">⏱ ${data.processing_time_ms.toFixed(0)}ms</span>
        <span class="meta-tag cost">💰 $${data.estimated_cost_usd.toFixed(4)}</span>
        ${warningBadge}
        ${alertBadge}
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

    // Render validation warnings
    renderValidationWarnings(data.validation_warnings || []);

    // Render safety alerts
    renderSafetyAlerts(data.safety_alerts || []);
}

// ── Validation Warnings ──
function renderValidationWarnings(warnings) {
    if (!warnings || warnings.length === 0) {
        hideCard(validationCard);
        return;
    }

    validationCount.textContent = `${warnings.length} warning${warnings.length > 1 ? 's' : ''}`;

    const icons = {
        'duplicate_batch': '🔁',
        'expired': '📅',
        'price_anomaly': '💲',
        'date_inconsistency': '📆',
        'missing_fields': '📋',
    };

    validationBody.innerHTML = warnings.map((w, idx) => {
        const flagType = w.flag.split(':')[0];
        const icon = icons[flagType] || '⚠️';
        return `
            <div class="validation-item" style="animation-delay: ${idx * 0.06}s">
                <div class="validation-item-icon">${icon}</div>
                <div class="validation-item-content">
                    <div class="validation-item-flag">${escapeHtml(w.flag)}</div>
                    <div class="validation-item-message">${escapeHtml(w.message)}</div>
                </div>
            </div>
        `;
    }).join('');

    showCard(validationCard);
}

// ── Safety Alerts ──
function renderSafetyAlerts(alerts) {
    if (!alerts || alerts.length === 0) {
        hideCard(safetyCard);
        return;
    }

    safetyCount.textContent = `${alerts.length} alert${alerts.length > 1 ? 's' : ''}`;

    // Determine highest severity for card border
    const severityOrder = ['critical', 'high', 'moderate', 'low'];
    const highestSeverity = severityOrder.find(s => alerts.some(a => a.severity === s)) || 'low';
    safetyCard.className = `glass-card safety-card has-${highestSeverity}`;

    safetyBody.innerHTML = alerts.map((alert, idx) => {
        const [drugA, drugB] = alert.drug_pair || ['Drug A', 'Drug B'];
        const severityEmoji = {
            critical: '🚨',
            high: '⛔',
            moderate: '⚠️',
            low: 'ℹ️',
        }[alert.severity] || '⚠️';

        return `
            <div class="safety-alert-item severity-${alert.severity}" style="animation-delay: ${idx * 0.08}s">
                <div class="safety-alert-header">
                    <div class="safety-drug-pair">
                        ${severityEmoji}
                        <span>${escapeHtml(drugA)}</span>
                        <span class="drug-pair-separator">+</span>
                        <span>${escapeHtml(drugB)}</span>
                    </div>
                    <span class="severity-badge ${alert.severity}">${alert.severity.toUpperCase()}</span>
                </div>
                <div class="safety-alert-description">${escapeHtml(alert.description)}</div>
                <div class="safety-alert-source">Source: ${escapeHtml(alert.source)}</div>
            </div>
        `;
    }).join('');

    showCard(safetyCard);
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
    hideCard(validationCard);
    hideCard(safetyCard);
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
