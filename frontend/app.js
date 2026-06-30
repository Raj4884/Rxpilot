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
const voiceCard = $('#voice-card');
const voiceBody = $('#voice-body');
const voiceStatus = $('#voice-status');
const errorMessage = $('#error-message');
const historyBody = $('#history-body');

const statusDot = $('.status-dot');
const statusText = $('.status-text');

// ── Pipeline Step refs ──
const stepValidate = $('#step-validate');
const stepSafety = $('#step-safety');
const stepStore = $('#step-store');

// ── Voice refs ──
const voiceBtn = $('#voice-btn');
const voiceBtnText = $('#voice-btn-text');
const voiceBtnIcon = $('#voice-btn-icon');

// ── State ──
let selectedFile = null;
let pipelineStepTimer = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

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

// ── Voice Recording ──
const RECORDING_MAX_MS = 10000; // auto-stop after 10s
let recordingTimer = null;

if (voiceBtn) {
    voiceBtn.addEventListener('click', toggleRecording);
}

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            await submitVoiceQuery(blob);
        };

        mediaRecorder.start();
        isRecording = true;
        setVoiceRecordingUI(true);

        // Auto-stop after 10s
        recordingTimer = setTimeout(stopRecording, RECORDING_MAX_MS);

    } catch (err) {
        if (err.name === 'NotAllowedError') {
            alert('Microphone access denied. Please allow microphone access to use voice queries.');
        } else {
            alert(`Could not start recording: ${err.message}`);
        }
    }
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    clearTimeout(recordingTimer);
    mediaRecorder.stop();
    isRecording = false;
    setVoiceRecordingUI(false);
    setVoiceRecordingUI(false, true); // Show processing state
}

function setVoiceRecordingUI(recording, processing = false) {
    if (!voiceBtn) return;

    if (recording) {
        voiceBtn.classList.add('recording');
        voiceBtnText.textContent = 'Stop Recording';
        voiceBtnIcon.innerHTML = `
            <div class="waveform">
                ${Array(5).fill('<div class="waveform-bar"></div>').join('')}
            </div>
        `;
    } else if (processing) {
        voiceBtn.disabled = true;
        voiceBtnText.textContent = 'Processing...';
        voiceBtnIcon.innerHTML = '<span class="spinner-sm"></span>';
    } else {
        voiceBtn.classList.remove('recording');
        voiceBtn.disabled = false;
        voiceBtnText.textContent = 'Ask with Voice';
        voiceBtnIcon.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="22"/>
                <line x1="8" y1="22" x2="16" y2="22"/>
            </svg>
        `;
    }
}

async function submitVoiceQuery(audioBlob) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'query.webm');

    hideCard(voiceCard);

    try {
        const res = await fetch(`${API_BASE}/v1/voice`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `Voice query failed: ${res.status}`);
        }

        const data = await res.json();
        renderVoiceResult(data);

    } catch (err) {
        voiceStatus.textContent = 'Error';
        voiceBody.innerHTML = `
            <div class="empty-text" style="color: var(--accent-rose);">
                Voice query failed: ${escapeHtml(err.message)}
            </div>
        `;
        showCard(voiceCard);
    } finally {
        setVoiceRecordingUI(false);
    }
}

function renderVoiceResult(data) {
    const intentLabels = {
        stock_query: '📦 Stock Query',
        expiry_query: '📅 Expiry Query',
        interaction_query: '🛡️ Interaction Query',
        general_query: '💬 General Query',
    };

    const sourceLabel = data.answer_source || 'unknown';
    const confidence = (data.answer_confidence * 100).toFixed(0);
    const intentLabel = intentLabels[data.intent] || data.intent;

    voiceStatus.textContent = `${data.processing_time_ms.toFixed(0)}ms`;

    voiceBody.innerHTML = `
        <div class="voice-transcript-block">
            <div class="voice-transcript-label">Transcript</div>
            <div class="voice-transcript-text">
                "${escapeHtml(data.transcript || '(no transcript)')}"
            </div>
        </div>

        <div class="voice-intent-row">
            <span class="voice-intent-badge">${intentLabel}</span>
            ${data.drug_name ? `<span class="voice-drug-tag">💊 ${escapeHtml(data.drug_name)}</span>` : ''}
        </div>

        <div class="voice-answer-block">
            <div class="voice-answer-label">🤖 Answer</div>
            <div class="voice-answer-text">${escapeHtml(data.answer)}</div>
            <div class="voice-answer-meta">
                <span class="voice-source-tag">source: ${escapeHtml(sourceLabel)}</span>
                <span class="voice-confidence">${confidence}% confidence</span>
            </div>
        </div>
    `;

    showCard(voiceCard);
}
