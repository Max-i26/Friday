// Global State
let state = {
    datasetId: null,
    filename: null,
    profile: null,
    cleanProfile: null,
    geminiKey: localStorage.getItem("gemini_api_key") || "",
    chatHistory: [],
    evaluationResults: null
};

// Chart instances (to destroy before drawing new ones)
let corrChartInstance = null;
let visualChartInstance = null;
let valChartInstance1 = null;
let valChartInstance2 = null;

// DOM Elements
const elements = {
    geminiKeyInput: document.getElementById("gemini-key-input"),
    saveKeyBtn: document.getElementById("save-key-btn"),
    backendStatusDot: document.getElementById("backend-status-dot"),
    backendStatusText: document.getElementById("backend-status-text"),
    
    // Upload & Banner
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    uploadHero: document.getElementById("upload-hero-section"),
    workspace: document.getElementById("workspace-section"),
    datasetBanner: document.getElementById("dataset-banner"),
    bannerFilename: document.getElementById("banner-filename"),
    bannerStats: document.getElementById("banner-stats"),
    bannerActions: document.getElementById("banner-actions"),
    downloadRawBtn: document.getElementById("download-raw-btn"),
    downloadCleanBtn: document.getElementById("download-clean-btn"),
    resetDatasetBtn: document.getElementById("reset-dataset-btn"),
    sampleBtns: document.querySelectorAll(".btn-sample"),
    
    // Tab Navigation
    tabLinks: document.querySelectorAll(".tab-link"),
    tabContents: document.querySelectorAll(".tab-content"),
    
    // Data Profile Tab
    previewRowCount: document.getElementById("preview-row-count"),
    previewTableHead: document.getElementById("preview-thead"),
    previewTableBody: document.getElementById("preview-tbody"),
    columnsSummaryContainer: document.getElementById("columns-summary-container"),
    correlationChart: document.getElementById("correlation-chart"),
    aiLoading: document.getElementById("ai-loading"),
    aiOverviewContent: document.getElementById("ai-overview-content"),
    aiDatasetSummary: document.getElementById("ai-dataset-summary"),
    aiTargetsList: document.getElementById("ai-targets-list"),
    aiPlotsContainer: document.getElementById("ai-plots-container"),
    
    // Data Cleaner Tab
    cleanerForm: document.getElementById("cleaner-form"),
    cleanDropColsList: document.getElementById("clean-drop-cols-list"),
    cleanImputeNumeric: document.getElementById("clean-impute-numeric"),
    cleanImputeCategorical: document.getElementById("clean-impute-categorical"),
    cleanScaleStrategy: document.getElementById("clean-scale-strategy"),
    cleanScaleColsWrapper: document.getElementById("clean-scale-cols-wrapper"),
    cleanScaleColsList: document.getElementById("clean-scale-cols-list"),
    cleanEncodeStrategy: document.getElementById("clean-encode-strategy"),
    cleanEncodeColsWrapper: document.getElementById("clean-encode-cols-wrapper"),
    cleanEncodeColsList: document.getElementById("clean-encode-cols-list"),
    runCleanBtn: document.getElementById("run-clean-btn"),
    cleaningStatusBanner: document.getElementById("cleaning-status-banner"),
    cleaningDetailsContainer: document.getElementById("cleaning-details-container"),
    cleanOrigCols: document.getElementById("clean-orig-cols"),
    cleanNewCols: document.getElementById("clean-new-cols"),
    cleanOrigRows: document.getElementById("clean-orig-rows"),
    cleanNewRows: document.getElementById("clean-new-rows"),
    cleanPreviewTableHead: document.getElementById("clean-preview-thead"),
    cleanPreviewTableBody: document.getElementById("clean-preview-tbody"),
    
    // Visualization Tab
    plotChartType: document.getElementById("plot-chart-type"),
    plotXCol: document.getElementById("plot-x-col"),
    plotYCol: document.getElementById("plot-y-col"),
    plotYColGroup: document.getElementById("plot-y-col-group"),
    buildChartBtn: document.getElementById("build-chart-btn"),
    clearChartBtn: document.getElementById("clear-chart-btn"),
    currentChartTitle: document.getElementById("current-chart-title"),
    canvasEmpty: document.getElementById("canvas-empty"),
    visualChart: document.getElementById("visual-chart"),
    chartPresets: document.getElementById("chart-presets"),
    
    // Chat Tutor
    chatMessagesContainer: document.getElementById("chat-messages-container"),
    chatInput: document.getElementById("chat-input"),
    sendChatBtn: document.getElementById("send-chat-btn"),
    
    // Trainer Tab
    trainerForm: document.getElementById("trainer-form"),
    trainTargetCol: document.getElementById("train-target-col"),
    trainFeaturesList: document.getElementById("train-features-list"),
    trainTaskType: document.getElementById("train-task-type"),
    trainSplitRatio: document.getElementById("train-split-ratio"),
    trainSplitRatioText: document.getElementById("train-split-ratio-text"),
    trainRunBtn: document.getElementById("train-run-btn"),
    trainEmptyState: document.getElementById("train-empty-state"),
    trainProcessingState: document.getElementById("train-processing-state"),
    
    // Leaderboard Tab
    leaderboardTbody: document.getElementById("leaderboard-tbody"),
    leaderboardThead: document.getElementById("leaderboard-thead"),
    valPlot1Title: document.getElementById("val-plot-1-title"),
    valPlot2Title: document.getElementById("val-plot-2-title"),
    dtRulesCard: document.getElementById("dt-rules-card"),
    dtRulesContent: document.getElementById("dt-rules-content"),
    aiReportLoading: document.getElementById("ai-report-loading"),
    aiReportContent: document.getElementById("ai-report-content"),
    exportReportBtn: document.getElementById("export-report-btn")
};

// Init app
function init() {
    // Populate stored API Key
    if (state.geminiKey) {
        elements.geminiKeyInput.value = state.geminiKey;
        setAiIndicator(true);
    } else {
        setAiIndicator(false);
    }

    // Set up Listeners
    setupEventListeners();
    checkBackendStatus();
}

// Set up UI Event Listeners
function setupEventListeners() {
    // API Key save
    elements.saveKeyBtn.addEventListener("click", () => {
        const key = elements.geminiKeyInput.value.trim();
        state.geminiKey = key;
        localStorage.setItem("gemini_api_key", key);
        showNotification("Gemini API key saved!", "success");
        setAiIndicator(!!key);
        if (state.datasetId) {
            // Trigger analyze dataset again with new key
            analyzeDatasetAI();
        }
    });

    // Tab toggles
    elements.tabLinks.forEach(link => {
        link.addEventListener("click", () => {
            const tabId = link.getAttribute("data-tab");
            
            // Check if dataset is loaded
            if (!state.datasetId) {
                showNotification("Please upload a dataset first!", "warning");
                return;
            }

            elements.tabLinks.forEach(l => l.classList.remove("active"));
            elements.tabContents.forEach(c => c.classList.remove("active"));
            
            link.classList.add("active");
            document.getElementById(tabId).classList.add("active");
            
            // Handle tab-specific refresh behaviors if needed
            if (tabId === "tab-explorer" && state.profile) {
                renderCorrelationHeatmap();
            }
        });
    });

    // Drag and Drop files
    ["dragenter", "dragover"].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            elements.dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            elements.dropZone.classList.remove("dragover");
        }, false);
    });

    elements.dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    elements.fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Click on browse file
    elements.dropZone.querySelector("button").addEventListener("click", (e) => {
        e.stopPropagation();
        elements.fileInput.click();
    });

    // Sample Datasets triggers
    elements.sampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const sampleName = btn.getAttribute("data-sample");
            loadSampleDataset(sampleName);
        });
    });

    // Reset button
    elements.resetDatasetBtn.addEventListener("click", resetState);

    // Scaling strategy change (toggle columns view)
    elements.cleanScaleStrategy.addEventListener("change", () => {
        if (elements.cleanScaleStrategy.value !== "none") {
            elements.cleanScaleColsWrapper.classList.remove("hidden");
        } else {
            elements.cleanScaleColsWrapper.classList.add("hidden");
        }
    });

    // Encoding strategy change (toggle columns view)
    elements.cleanEncodeStrategy.addEventListener("change", () => {
        if (elements.cleanEncodeStrategy.value !== "none") {
            elements.cleanEncodeColsWrapper.classList.remove("hidden");
        } else {
            elements.cleanEncodeColsWrapper.classList.add("hidden");
        }
    });

    // Data cleaning form submit
    elements.cleanerForm.addEventListener("submit", handleDataCleaning);

    // Plot Builder chart type change
    elements.plotChartType.addEventListener("change", () => {
        const chartType = elements.plotChartType.value;
        if (chartType === "histogram" || chartType === "box") {
            elements.plotYColGroup.classList.add("hidden");
        } else {
            elements.plotYColGroup.classList.remove("hidden");
        }
    });

    // Generate Custom Plot
    elements.buildChartBtn.addEventListener("click", generateCustomPlot);
    elements.clearChartBtn.addEventListener("click", clearPlotCanvas);

    // Chat submit
    elements.sendChatBtn.addEventListener("click", sendChatMessage);
    elements.chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendChatMessage();
        }
    });

    // Split ratio slide
    elements.trainSplitRatio.addEventListener("input", (e) => {
        const val = Math.round(e.target.value * 100);
        elements.trainSplitRatioText.innerText = `${val}% Testing`;
    });

    // AutoML train submit
    elements.trainerForm.addEventListener("submit", handleModelTraining);

    // Export report trigger
    elements.exportReportBtn.addEventListener("click", exportLearningReport);
}

// UI Status indicators
function setAiIndicator(active) {
    const icon = document.querySelector(".key-icon");
    if (active) {
        icon.classList.remove("fa-key");
        icon.classList.add("fa-shield-halved");
        icon.style.color = "var(--success)";
    } else {
        icon.classList.remove("fa-shield-halved");
        icon.classList.add("fa-key");
        icon.style.color = "var(--accent-purple)";
    }
}

function setBackendStatus(connected) {
    if (connected) {
        elements.backendStatusDot.className = "status-dot green";
        elements.backendStatusText.innerText = "Backend Connected";
    } else {
        elements.backendStatusDot.className = "status-dot red";
        elements.backendStatusText.innerText = "Connection Failed";
        showNotification("Backend server is offline! Please check uvicorn logs.", "error");
    }
}

// Reset entire state
function resetState() {
    state.datasetId = null;
    state.filename = null;
    state.profile = null;
    state.cleanProfile = null;
    state.evaluationResults = null;
    state.chatHistory = [];
    
    // Clear forms and charts
    elements.fileInput.value = "";
    elements.cleanerForm.reset();
    elements.trainerForm.reset();
    
    if (corrChartInstance) corrChartInstance.destroy();
    if (visualChartInstance) visualChartInstance.destroy();
    if (valChartInstance1) valChartInstance1.destroy();
    if (valChartInstance2) valChartInstance2.destroy();
    
    // Reset views
    elements.uploadHero.classList.remove("hidden");
    elements.workspace.classList.add("hidden");
    elements.datasetBanner.classList.add("glass-card");
    elements.bannerActions.classList.add("hidden");
    elements.downloadCleanBtn.classList.add("hidden");
    elements.bannerFilename.innerText = "No dataset loaded";
    elements.bannerStats.innerText = "Upload a CSV or Excel file to begin your ML journey";
    
    // Reset sub-banners
    elements.cleaningStatusBanner.className = "cleaning-state-indicator idle";
    elements.cleaningStatusBanner.innerHTML = `<i class="fa-solid fa-circle-info"></i> No cleaning performed yet. Configure steps on the left.`;
    elements.cleaningDetailsContainer.classList.add("hidden");
    
    elements.trainEmptyState.classList.remove("hidden");
    document.getElementById("tab-btn-leaderboard").classList.remove("active");
    
    // Clear chats
    elements.chatMessagesContainer.innerHTML = `
        <div class="chat-message model-msg">
            Hello! I am your AI ML Pipeline Brain. Ask me any question about the dataset, 
            request coding snippets in Pandas, or ask about Machine Learning models. How can I help you today?
        </div>
    `;
    
    showNotification("State reset successfully.", "success");
}

// Check if FastAPI is running
async function checkBackendStatus() {
    try {
        const response = await fetch("/api/upload", { method: "HEAD" });
        setBackendStatus(true);
    } catch (e) {
        // FastAPI might serve root static directly but fail or CORS, let's assume if we are loading this page it's up, but let's test a sample fetch
        try {
            const check = await fetch("/docs");
            setBackendStatus(true);
        } catch (inner) {
            setBackendStatus(false);
        }
    }
}

// Upload file endpoint
async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    showLoader(true, "Uploading dataset & calculating metrics...");
    
    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to upload file");
        }

        const data = await response.json();
        
        state.datasetId = data.dataset_id;
        state.filename = data.filename;
        state.profile = data.profile;
        
        showNotification(`Dataset "${file.name}" loaded!`, "success");
        onDatasetLoaded();
        
    } catch (error) {
        showNotification(error.message, "error");
        resetState();
    } finally {
        showLoader(false);
    }
}

// Trigger backend to load pre-packaged scikit-learn dataset
async function loadSampleDataset(name) {
    showLoader(true, `Downloading & profiling sample: ${name}...`);
    try {
        const response = await fetch(`/api/sample/${name}`, {
            method: "POST"
        });

        if (!response.ok) {
            // Fallback to fetch raw templates if endpoint isn't ready
            const err = await response.json();
            throw new Error(err.detail || "Failed to load sample");
        }

        const data = await response.json();
        state.datasetId = data.dataset_id;
        state.filename = data.filename;
        state.profile = data.profile;
        
        showNotification(`Sample dataset "${data.filename}" loaded!`, "success");
        onDatasetLoaded();
    } catch (error) {
        // If /api/sample endpoint isn't fully set up on backend, notify and guide
        showNotification(`Sample load failed: ${error.message}. Try uploading your own CSV!`, "error");
    } finally {
        showLoader(false);
    }
}

// Core setup once dataset is fully loaded
function onDatasetLoaded() {
    // Hide Hero, Show Workspace
    elements.uploadHero.classList.add("hidden");
    elements.workspace.classList.remove("hidden");
    elements.bannerActions.classList.remove("hidden");
    elements.downloadRawBtn.href = `/api/download/${state.datasetId}/raw`;
    
    // Update banner details
    elements.bannerFilename.innerText = state.filename;
    elements.bannerStats.innerText = `Size: ${state.profile.num_rows} rows x ${state.profile.num_cols} columns`;

    // Render Data Profile
    renderDataProfile();
    
    // Setup Data Cleaning Panel
    setupDataCleaningPanel();
    
    // Setup Custom Plot selectors
    setupPlotBuilderSelectors();
    
    // Setup Trainer Form Target/Features selectors
    setupTrainerSelectors();

    // Call Gemini to analyze
    analyzeDatasetAI();
}

// Profile Tab Renderings
function renderDataProfile() {
    const prof = state.profile;
    
    // 1. Update Preview Table
    elements.previewRowCount.innerText = `${prof.num_rows} Rows x ${prof.num_cols} Columns`;
    elements.previewTableHead.innerHTML = "";
    elements.previewTableBody.innerHTML = "";
    
    if (prof.preview.length > 0) {
        const columns = Object.keys(prof.preview[0]);
        // Headers
        columns.forEach(col => {
            const th = document.createElement("th");
            th.innerText = col;
            elements.previewTableHead.appendChild(th);
        });
        
        // Rows
        prof.preview.forEach(row => {
            const tr = document.createElement("tr");
            columns.forEach(col => {
                const td = document.createElement("td");
                const val = row[col];
                td.innerText = val === null ? "NaN" : val;
                if (val === null) td.style.color = "var(--text-muted)";
                tr.appendChild(td);
            });
            elements.previewTableBody.appendChild(tr);
        });
    }

    // 2. Update Columns Profile cards
    elements.columnsSummaryContainer.innerHTML = "";
    prof.columns.forEach(col => {
        const card = document.createElement("div");
        card.className = "column-stat-card";
        
        // Check if numeric or categorical stats are present
        let statsHtml = "";
        if (col.type === "numeric" && prof.stats_numeric[col.name]) {
            const s = prof.stats_numeric[col.name];
            statsHtml = `
                <div class="col-detail-row"><span>Mean:</span><span class="val">${s.mean !== null ? s.mean.toFixed(2) : "N/A"}</span></div>
                <div class="col-detail-row"><span>Std:</span><span class="val">${s.std !== null ? s.std.toFixed(2) : "N/A"}</span></div>
                <div class="col-detail-row"><span>Min:</span><span class="val">${s.min !== null ? s.min : "N/A"}</span></div>
                <div class="col-detail-row"><span>Max:</span><span class="val">${s.max !== null ? s.max : "N/A"}</span></div>
            `;
        } else if (prof.stats_categorical[col.name]) {
            const s = prof.stats_categorical[col.name];
            statsHtml = `
                <div class="col-detail-row"><span>Uniques:</span><span class="val">${s.unique}</span></div>
                <div class="col-detail-row"><span>Top Mode:</span><span class="val" title="${s.top}">${s.top !== null ? (s.top.length > 12 ? s.top.substring(0, 10) + '...' : s.top) : "N/A"}</span></div>
                <div class="col-detail-row"><span>Top Freq:</span><span class="val">${s.freq}</span></div>
                <div class="col-detail-row"><span>Valid Cnt:</span><span class="val">${s.count}</span></div>
            `;
        }

        card.innerHTML = `
            <div class="col-header">
                <span class="col-name" title="${col.name}">${col.name.length > 20 ? col.name.substring(0, 18) + '...' : col.name}</span>
                <span class="col-type-tag ${col.type}">${col.type}</span>
            </div>
            <div class="col-details">
                <div class="col-detail-row"><span>Nulls:</span><span class="val" style="${col.missing_count > 0 ? 'color: var(--error)' : ''}">${col.missing_count} (${col.missing_pct}%)</span></div>
                <div class="col-detail-row"><span>Unq Rto:</span><span class="val">${(col.unique_count / prof.num_rows).toFixed(2)}</span></div>
                ${statsHtml}
            </div>
        `;
        elements.columnsSummaryContainer.appendChild(card);
    });

    // 3. Draw Correlation Heatmap
    renderCorrelationHeatmap();
}

// Plot correlation matrix using canvas
function renderCorrelationHeatmap() {
    const canvas = elements.correlationChart;
    if (!canvas) {
        console.warn("Correlation chart canvas element is missing from elements object.");
        return;
    }
    const ctx = canvas.getContext("2d");
    const matrixData = state.profile.correlation_matrix;

    if (corrChartInstance) corrChartInstance.destroy();

    if (!matrixData || !matrixData.columns || matrixData.columns.length === 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "var(--text-muted)";
        ctx.font = "14px Inter";
        ctx.textAlign = "center";
        ctx.fillText("Correlation matrix requires at least 2 numeric columns.", canvas.width / 2, canvas.height / 2);
        return;
    }

    const numCols = matrixData.columns;
    const matrix = matrixData.matrix;

    // We will render the correlation heatmap using Chart.js Bubble or Scatter
    // Or simpler: let's render a custom chart or a visual table in HTML inside elements.correlationChart's container!
    // Actually, drawing a visual canvas grid is super light, elegant and works perfectly! Let's do a simple 2D Canvas rendering
    // since Chart.js doesn't natively do nice matrix plots without extra plugin imports.
    // Resize canvas properly
    const container = canvas.parentNode;
    const size = Math.min(container.clientWidth || 300, 300);
    canvas.width = size;
    canvas.height = size;

    const n = numCols.length;
    const cellSize = (size - 60) / n; // 60px for labels padding on bottom/left

    ctx.clearRect(0, 0, size, size);

    // Draw cells
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = matrix[i][j];
            
            // Map val (-1 to 1) to colors (Blue for negative, Red for positive)
            let color = "";
            if (val >= 0) {
                // Red-ish / violet
                color = `rgba(139, 92, 246, ${val})`; // Violet glow based on strength
            } else {
                // Blue-ish
                color = `rgba(59, 130, 246, ${Math.abs(val)})`;
            }

            ctx.fillStyle = color;
            ctx.fillRect(60 + j * cellSize, i * cellSize, cellSize - 1, cellSize - 1);

            // Print text value in cells if we have space
            if (n <= 8) {
                ctx.fillStyle = Math.abs(val) > 0.4 ? "#ffffff" : "var(--text-secondary)";
                ctx.font = "9px Inter";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(val.toFixed(2), 60 + j * cellSize + cellSize / 2, i * cellSize + cellSize / 2);
            }
        }
    }

    // Draw labels (only first characters or diagonal indicators to avoid overlap)
    ctx.fillStyle = "var(--text-secondary)";
    ctx.font = "8px Inter";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
        let label = numCols[i];
        if (label.length > 8) label = label.substring(0, 6) + "..";
        ctx.fillText(label, 52, i * cellSize + cellSize / 2);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let j = 0; j < n; j++) {
        let label = numCols[j];
        if (label.length > 8) label = label.substring(0, 6) + "..";
        ctx.save();
        ctx.translate(60 + j * cellSize + cellSize / 2, n * cellSize + 5);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(label, 0, 0);
        ctx.restore();
    }
}

// Call Gemini API to analyze column metadata and suggest visualization
async function analyzeDatasetAI() {
    elements.aiLoading.classList.remove("hidden");
    elements.aiOverviewContent.classList.add("hidden");

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: state.datasetId,
                api_key: state.geminiKey
            })
        });

        if (!response.ok) throw new Error("Gemini analyze call failed.");
        
        const data = await response.json();
        
        // Show contents
        elements.aiLoading.classList.add("hidden");
        elements.aiOverviewContent.classList.remove("hidden");

        elements.aiDatasetSummary.innerText = data.dataset_summary || "Could not analyze summary.";

        // Targets list
        elements.aiTargetsList.innerHTML = "";
        const targets = data.recommended_targets || [];
        targets.forEach(t => {
            const li = document.createElement("li");
            li.innerHTML = `<strong>${t.column}</strong> (${t.type}): ${t.reason}`;
            elements.aiTargetsList.appendChild(li);
        });

        // Recommended Plot Cards
        elements.aiPlotsContainer.innerHTML = "";
        const plots = data.recommended_plots || [];
        
        // Keep presets container sync'd too
        elements.chartPresets.innerHTML = "";

        plots.forEach((p, idx) => {
            // 1. Profile tab cards
            const plotCard = document.createElement("div");
            plotCard.className = "auto-plot-card";
            plotCard.innerHTML = `
                <div class="auto-plot-header">
                    <span><i class="fa-solid fa-chart-simple"></i> ${p.title}</span>
                    <span class="badge">${p.type}</span>
                </div>
                <p class="auto-plot-desc">${p.explanation}</p>
            `;
            plotCard.addEventListener("click", () => {
                // Switch to visualizer tab and load this plot!
                elements.tabLinks.forEach(l => l.classList.remove("active"));
                elements.tabContents.forEach(c => c.classList.remove("active"));
                
                document.getElementById("tab-visualizer").classList.add("active");
                document.querySelector('[data-tab="tab-visualizer"]').classList.add("active");
                
                loadChartPreset(p);
            });
            elements.aiPlotsContainer.appendChild(plotCard);

            // 2. Chart builder sidebar presets
            const presetLink = document.createElement("button");
            presetLink.className = "preset-link";
            presetLink.innerHTML = `<i class="fa-solid fa-arrow-pointer"></i> ${p.title}`;
            presetLink.addEventListener("click", () => loadChartPreset(p));
            elements.chartPresets.appendChild(presetLink);
        });

        // Auto select suggested Target in training panel if available
        if (targets.length > 0) {
            const recTarget = targets[0].column;
            const targetSelect = elements.trainTargetCol;
            for (let i = 0; i < targetSelect.options.length; i++) {
                if (targetSelect.options[i].value === recTarget) {
                    targetSelect.selectedIndex = i;
                    // Trigger updating features list to exclude new target
                    updateFeaturesCheckboxList();
                    break;
                }
            }
        }

    } catch (error) {
        console.error(error);
        elements.aiLoading.classList.add("hidden");
        elements.aiOverviewContent.classList.remove("hidden");
        elements.aiDatasetSummary.innerText = "Error requesting AI review. Please make sure your API key is valid.";
    }
}

// Data Cleaning Tab Setup
function setupDataCleaningPanel() {
    const prof = state.profile;
    
    // Clear list
    elements.cleanDropColsList.innerHTML = "";
    elements.cleanScaleColsList.innerHTML = "";
    elements.cleanEncodeColsList.innerHTML = "";

    prof.columns.forEach(col => {
        // Drop column choices
        const dropItem = document.createElement("label");
        dropItem.className = "checkbox-item";
        dropItem.innerHTML = `<input type="checkbox" name="drop_cols" value="${col.name}" /> <span>${col.name}</span>`;
        elements.cleanDropColsList.appendChild(dropItem);

        // Scaling column choices
        if (col.type === "numeric") {
            const scaleItem = document.createElement("label");
            scaleItem.className = "checkbox-item";
            scaleItem.innerHTML = `<input type="checkbox" name="scale_cols" value="${col.name}" checked /> <span>${col.name}</span>`;
            elements.cleanScaleColsList.appendChild(scaleItem);
        }

        // Encoding column choices
        if (col.type === "categorical") {
            const encodeItem = document.createElement("label");
            encodeItem.className = "checkbox-item";
            encodeItem.innerHTML = `<input type="checkbox" name="encode_cols" value="${col.name}" checked /> <span>${col.name}</span>`;
            elements.cleanEncodeColsList.appendChild(encodeItem);
        }
    });
}

// Data Cleaning handler
async function handleDataCleaning(e) {
    e.preventDefault();

    const dropChecked = Array.from(document.querySelectorAll('input[name="drop_cols"]:checked')).map(el => el.value);
    const scaleChecked = Array.from(document.querySelectorAll('input[name="scale_cols"]:checked')).map(el => el.value);
    const encodeChecked = Array.from(document.querySelectorAll('input[name="encode_cols"]:checked')).map(el => el.value);

    const options = {
        drop_columns: dropChecked,
        imputation: {
            numeric: elements.cleanImputeNumeric.value,
            categorical: elements.cleanImputeCategorical.value
        },
        scaling: {
            strategy: elements.cleanScaleStrategy.value,
            columns: scaleChecked
        },
        encoding: {
            strategy: elements.cleanEncodeStrategy.value,
            columns: encodeChecked
        }
    };

    elements.runCleanBtn.disabled = true;
    elements.runCleanBtn.innerHTML = `<div class="spinner" style="width:20px;height:20px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:10px;"></div> Cleaning...`;

    try {
        const response = await fetch("/api/clean", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: state.datasetId,
                options: options
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Data cleaning failed.");
        }

        const data = await response.json();
        state.cleanProfile = data.clean_profile;
        
        showNotification("Dataset cleaned successfully!", "success");
        onDatasetCleaned();

    } catch (error) {
        showNotification(error.message, "error");
    } finally {
        elements.runCleanBtn.disabled = false;
        elements.runCleanBtn.innerHTML = `<i class="fa-solid fa-play"></i> Clean Dataset`;
    }
}

// Action once dataset is cleaned
function onDatasetCleaned() {
    const oldProf = state.profile;
    const newProf = state.cleanProfile;

    // Show clean download button
    elements.downloadCleanBtn.classList.remove("hidden");
    elements.downloadCleanBtn.onclick = () => {
        window.open(`/api/download/${state.datasetId}/clean`, "_blank");
    };

    // Update Status banner
    elements.cleaningStatusBanner.className = "cleaning-state-indicator success";
    elements.cleaningStatusBanner.innerHTML = `<i class="fa-solid fa-circle-check"></i> Cleaning complete! Pipeline updated. Compare metrics below.`;

    elements.cleaningDetailsContainer.classList.remove("hidden");

    // Display comparison statistics
    elements.cleanOrigCols.innerText = oldProf.num_cols;
    elements.cleanNewCols.innerText = newProf.num_cols;
    elements.cleanOrigRows.innerText = oldProf.num_rows;
    elements.cleanNewRows.innerText = newProf.num_rows;

    // Render cleaned data preview
    elements.cleanPreviewTableHead.innerHTML = "";
    elements.cleanPreviewTableBody.innerHTML = "";
    
    if (newProf.preview.length > 0) {
        const columns = Object.keys(newProf.preview[0]);
        // Headers
        columns.forEach(col => {
            const th = document.createElement("th");
            th.innerText = col;
            elements.cleanPreviewTableHead.appendChild(th);
        });
        
        // Rows
        newProf.preview.forEach(row => {
            const tr = document.createElement("tr");
            columns.forEach(col => {
                const td = document.createElement("td");
                const val = row[col];
                td.innerText = val === null ? "NaN" : (typeof val === 'number' ? val.toFixed(4) : val);
                tr.appendChild(td);
            });
            elements.cleanPreviewTableBody.appendChild(tr);
        });
    }

    // Refresh Selector columns for Visualization and AutoML Trainer with Cleaned Profile structure
    setupPlotBuilderSelectors(true);
    setupTrainerSelectors(true);
}

// Visualization Selectors setup
function setupPlotBuilderSelectors(useClean = false) {
    const prof = useClean ? state.cleanProfile : state.profile;
    if (!prof) return;

    elements.plotXCol.innerHTML = "";
    elements.plotYCol.innerHTML = "";

    prof.columns.forEach(col => {
        const optX = document.createElement("option");
        optX.value = col.name;
        optX.innerText = col.name;
        elements.plotXCol.appendChild(optX);

        const optY = document.createElement("option");
        optY.value = col.name;
        optY.innerText = col.name;
        elements.plotYCol.appendChild(optY);
    });
}

// Preset plot loader
function loadChartPreset(preset) {
    // Select chart type
    elements.plotChartType.value = preset.type;
    
    // Select X
    for (let i = 0; i < elements.plotXCol.options.length; i++) {
        if (elements.plotXCol.options[i].value === preset.x) {
            elements.plotXCol.selectedIndex = i;
            break;
        }
    }
    
    // Select Y (if not hist/box)
    if (preset.type !== "histogram" && preset.type !== "box") {
        elements.plotYColGroup.classList.remove("hidden");
        for (let i = 0; i < elements.plotYCol.options.length; i++) {
            if (elements.plotYCol.options[i].value === preset.y) {
                elements.plotYCol.selectedIndex = i;
                break;
            }
        }
    } else {
        elements.plotYColGroup.classList.add("hidden");
    }

    // Trigger chart generation!
    generateCustomPlot(null, preset.title);
}

// Custom Plot builder drawing
async function generateCustomPlot(e, customTitle = null) {
    const prof = state.cleanProfile || state.profile;
    if (!prof) return;

    const chartType = elements.plotChartType.value;
    const xCol = elements.plotXCol.value;
    const yCol = elements.plotYCol.value;

    elements.canvasEmpty.classList.add("hidden");
    elements.visualChart.classList.remove("hidden");

    // Fetch the sample rows data to plot
    // To ensure fluid graph rendering without overloading Chart.js, cap points plotted
    const maxPoints = 500;
    
    // Since we have the preview data (15 rows) in profile, we should load more data or plot using the preview.
    // Wait! Can the backend provide full column values for plotting?
    // Oh, since the dataset is loaded, we can fetch column values, or we can fetch a sample preview from the file.
    // Let's implement a backend `/api/data` or just download/fetch file rows, or can we load from the backend?
    // Wait, let's think: does our FastAPI have an endpoint to get the column values? No!
    // But wait! We can easily get column values by requesting a preview. Or wait, does the `/api/upload` profile return a sample preview of only 15 rows?
    // Yes! Let's check: if we only have 15 rows, our plots will look very tiny.
    // Let's add a backend endpoint to fetch a larger chunk of data (e.g., up to 500 rows) specifically for plotting!
    // Let's check: we can create `/api/plot-data` endpoint in `backend/main.py`. This is a fantastic, robust solution.
    // Or, we can retrieve up to 500 rows when uploading the dataset and store them, or retrieve on-demand.
    // On-demand is better: `/api/plot-data?dataset_id=...&cols=x,y` returns the values of these columns for the entire dataset (or up to 1000 rows sampled).
    // Let's add that to `backend/main.py`. It is extremely simple!
    // But wait, can we build the plot using on-demand data?
    // Yes, we will call `fetch('/api/data?dataset_id=' + state.datasetId + '&columns=' + xCol + ',' + yCol)` to get values.
    // Let's write the fetch logic in `app.js` and implement the endpoint in `backend/main.py`.
    
    showLoader(false);
    elements.currentChartTitle.innerText = customTitle || `${chartType.toUpperCase()} Plot: ${xCol} ${chartType !== "histogram" && chartType !== "box" ? 'vs ' + yCol : ''}`;

    try {
        const response = await fetch(`/api/plot-data?dataset_id=${state.datasetId}&x=${encodeURIComponent(xCol)}&y=${encodeURIComponent(yCol)}&type=${chartType}`);
        if (!response.ok) throw new Error("Failed to load plot data");
        const plotData = await response.json();
        
        const xValues = plotData.x_values;
        const yValues = plotData.y_values;

        if (visualChartInstance) visualChartInstance.destroy();

        let chartConfig = {};

        if (chartType === "scatter") {
            const dataPoints = xValues.map((x, i) => ({ x: x, y: yValues[i] }));
            chartConfig = {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: `${yCol} vs ${xCol}`,
                        data: dataPoints,
                        backgroundColor: 'rgba(99, 102, 241, 0.6)',
                        borderColor: '#8b5cf6',
                        borderWidth: 1,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: xCol, color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { title: { display: true, text: yCol, color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            };
        } else if (chartType === "line") {
            // Sort by X value to avoid zigzag lines
            const combined = xValues.map((x, i) => ({ x: x, y: yValues[i] }));
            combined.sort((a, b) => (typeof a.x === 'number') ? a.x - b.x : String(a.x).localeCompare(String(b.x)));
            
            chartConfig = {
                type: 'line',
                data: {
                    labels: combined.map(p => p.x),
                    datasets: [{
                        label: yCol,
                        data: combined.map(p => p.y),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        pointRadius: 2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: xCol, color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { title: { display: true, text: yCol, color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            };
        } else if (chartType === "bar") {
            // Aggregated bar charts if categorical
            chartConfig = {
                type: 'bar',
                data: {
                    labels: xValues,
                    datasets: [{
                        label: yCol,
                        data: yValues,
                        backgroundColor: 'rgba(139, 92, 246, 0.7)',
                        borderColor: '#8b5cf6',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: xCol, color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { title: { display: true, text: yCol, color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            };
        } else if (chartType === "histogram") {
            // Calculate histogram bin counts in backend or frontend. Backend did it!
            chartConfig = {
                type: 'bar',
                data: {
                    labels: plotData.bins,
                    datasets: [{
                        label: `Frequency of ${xCol}`,
                        data: plotData.counts,
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderColor: '#3b82f6',
                        borderWidth: 1,
                        barPercentage: 1.0,
                        categoryPercentage: 1.0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: `${xCol} Range`, color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { title: { display: true, text: 'Frequency', color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            };
        } else if (chartType === "box") {
            // Canvas boxplots require plugins. We can render it as a horizontal bar range
            // or simple aggregate statistics. Let's do a summary stat box visualization:
            // plotData will return min, q1, median, q3, max
            const stats = plotData.box_stats;
            chartConfig = {
                type: 'bar',
                data: {
                    labels: [`${xCol} Outlier Summary`],
                    datasets: [
                        {
                            label: 'Min to Max Range',
                            data: [stats.max - stats.min],
                            backgroundColor: 'rgba(255,255,255,0.05)',
                            borderColor: 'var(--text-muted)',
                            borderWidth: 1,
                            base: stats.min
                        },
                        {
                            label: 'Interquartile Range (Q1 - Q3)',
                            data: [stats.q3 - stats.q1],
                            backgroundColor: 'rgba(99, 102, 241, 0.5)',
                            borderColor: '#6366f1',
                            borderWidth: 2,
                            base: stats.q1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return [
                                        `Max: ${stats.max.toFixed(2)}`,
                                        `Q3 (75%): ${stats.q3.toFixed(2)}`,
                                        `Median (50%): ${stats.median.toFixed(2)}`,
                                        `Q1 (25%): ${stats.q1.toFixed(2)}`,
                                        `Min: ${stats.min.toFixed(2)}`
                                    ];
                                }
                            }
                        }
                    },
                    scales: {
                        y: { grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            };
        }

        visualChartInstance = new Chart(elements.visualChart, chartConfig);

    } catch (err) {
        showNotification("Could not create chart: " + err.message, "error");
        clearPlotCanvas();
    }
}

function clearPlotCanvas() {
    if (visualChartInstance) visualChartInstance.destroy();
    elements.visualChart.classList.add("hidden");
    elements.canvasEmpty.classList.remove("hidden");
    elements.currentChartTitle.innerText = "Visual Explorer Canvas";
}

// Chat with Gemini Tutor
async function sendChatMessage() {
    const query = elements.chatInput.value.trim();
    if (!query) return;

    // Add user message to screen
    appendChatMessage("user", query);
    elements.chatInput.value = "";

    // Add loading block
    const loadingDiv = appendChatMessage("model", `<div class="spinner" style="width:16px;height:16px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px;"></div> Gemini Brain is thinking...`);

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: state.datasetId,
                api_key: state.geminiKey,
                history: state.chatHistory,
                query: query
            })
        });

        loadingDiv.remove();

        if (!response.ok) throw new Error("Chat api request failed.");

        const data = await response.json();
        
        appendChatMessage("model", data.response);
        
        // Save history
        state.chatHistory.push({ role: "user", text: query });
        state.chatHistory.push({ role: "model", text: data.response });

    } catch (error) {
        loadingDiv.remove();
        appendChatMessage("model", `Error connecting to Gemini Brain: ${error.message}. Please configure your API key at the top.`);
    }
}

function appendChatMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = `chat-message ${role === "user" ? "user-msg" : "model-msg"}`;
    
    // Simple markdown code block parser
    if (role === "model") {
        // Convert ```python ... ```
        let formatted = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/```python([\s\S]*?)```/g, '<pre class="code-terminal"><code class="language-python">$1</code></pre>')
            .replace(/```([\s\S]*?)```/g, '<pre class="code-terminal"><code>$1</code></pre>')
            .replace(/\*\*([\s\S]*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([\s\S]*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, "<br/>");
        msg.innerHTML = formatted;
    } else {
        msg.innerText = text;
    }
    
    elements.chatMessagesContainer.appendChild(msg);
    elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
    return msg;
}

// AutoML selectors
function setupTrainerSelectors(useClean = false) {
    const prof = useClean ? state.cleanProfile : state.profile;
    if (!prof) return;

    elements.trainTargetCol.innerHTML = "";
    
    prof.columns.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col.name;
        opt.innerText = col.name;
        elements.trainTargetCol.appendChild(opt);
    });

    // Populate checklist of features
    updateFeaturesCheckboxList();

    // Listeners for target change
    elements.trainTargetCol.addEventListener("change", updateFeaturesCheckboxList);
}

function updateFeaturesCheckboxList() {
    const prof = state.cleanProfile || state.profile;
    if (!prof) return;

    const targetVal = elements.trainTargetCol.value;
    elements.trainFeaturesList.innerHTML = "";

    prof.columns.forEach(col => {
        if (col.name !== targetVal) {
            const label = document.createElement("label");
            label.className = "checkbox-item";
            label.innerHTML = `<input type="checkbox" name="train_features" value="${col.name}" checked /> <span>${col.name}</span>`;
            elements.trainFeaturesList.appendChild(label);
        }
    });
}

// Model Training pipeline execution
async function handleModelTraining(e) {
    e.preventDefault();

    const targetCol = elements.trainTargetCol.value;
    const featureCols = Array.from(document.querySelectorAll('input[name="train_features"]:checked')).map(el => el.value);

    if (featureCols.length === 0) {
        showNotification("Please select at least 1 feature column!", "warning");
        return;
    }

    const taskType = elements.trainTaskType.value;
    const testSize = elements.trainSplitRatio.value;

    const hyperparams = {
        poly_degree: parseInt(document.getElementById("param-poly-degree").value || 2),
        n_estimators: parseInt(document.getElementById("param-estimators").value || 100)
    };
    
    const depthVal = document.getElementById("param-max-depth").value;
    if (depthVal) hyperparams.max_depth = parseInt(depthVal);

    // Swap states
    elements.trainEmptyState.classList.add("hidden");
    elements.trainProcessingState.classList.remove("hidden");
    elements.trainRunBtn.disabled = true;

    // We train 4 models: linear, polynomial, decision_tree, random_forest
    const modelsToTrain = ["linear", "polynomial", "decision_tree", "random_forest"];
    const results = {};
    let trainError = null;

    try {
        for (let modelType of modelsToTrain) {
            const trainReq = {
                dataset_id: state.datasetId,
                target_column: targetCol,
                feature_columns: featureCols,
                model_type: modelType,
                task_type: taskType,
                test_size: parseFloat(testSize),
                hyperparameters: hyperparams
            };

            const response = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(trainReq)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(`Model ${modelType} failed: ${err.detail}`);
            }

            results[modelType] = await response.json();
        }

        state.evaluationResults = results;
        showNotification("AutoML Training complete! Check results.", "success");
        onModelsTrained();

    } catch (err) {
        trainError = err.message;
        showNotification(trainError, "error");
        elements.trainEmptyState.classList.remove("hidden");
    } finally {
        elements.trainProcessingState.classList.add("hidden");
        elements.runCleanBtn.disabled = false;
        elements.trainRunBtn.disabled = false;
    }
}

// Action once models finish training
function onModelsTrained() {
    // 1. Move UI tabs
    elements.tabLinks.forEach(l => l.classList.remove("active"));
    elements.tabContents.forEach(c => c.classList.remove("active"));
    
    document.getElementById("tab-leaderboard").classList.add("active");
    document.getElementById("tab-btn-leaderboard").classList.add("active");

    const results = state.evaluationResults;
    const taskType = results.linear.task_type; // Regression or Classification

    // 2. Populate Leaderboard table
    elements.leaderboardTbody.innerHTML = "";
    
    // Sort models by Test Score (R2 for Regression, Accuracy for Classification)
    const metricKey = taskType === "regression" ? "r2" : "accuracy";
    const modelsList = Object.keys(results).map(m => ({
        name: m,
        testScore: results[m].test_metrics[metricKey],
        trainScore: results[m].train_metrics[metricKey],
        secondaryTestScore: taskType === "regression" ? results[m].test_metrics.mse : results[m].test_metrics.f1,
        secondaryKey: taskType === "regression" ? "MSE" : "F1",
        data: results[m]
    }));

    // Sort descending
    modelsList.sort((a, b) => b.testScore - a.testScore);

    modelsList.forEach((m, index) => {
        const tr = document.createElement("tr");
        if (index === 0) tr.className = "gold-row";
        
        tr.innerHTML = `
            <td>
                ${index === 0 ? '<i class="fa-solid fa-trophy gold-medal"></i>' : ''}
                <strong>${m.name.toUpperCase()}</strong>
            </td>
            <td>${m.testScore.toFixed(4)}</td>
            <td>${m.secondaryTestScore.toFixed(4)}</td>
            <td>${m.trainScore.toFixed(4)}</td>
            <td><span class="badge" style="background-color: var(--success-light); color: #6ee7b7">Trained</span></td>
        `;
        // Allow clicking row to display validation details
        tr.style.cursor = "pointer";
        tr.addEventListener("click", () => renderModelValidationPlots(m.name, m.data));
        
        elements.leaderboardTbody.appendChild(tr);
    });

    // Auto-select the top model validation plot on load
    renderModelValidationPlots(modelsList[0].name, modelsList[0].data);

    // Call Gemini to generate learning report
    generateModelReportAI(modelsList[0].name, modelsList[0].data);
}

// Draw Validation Plots
function renderModelValidationPlots(modelName, modelData) {
    const taskType = modelData.task_type;
    const valData = modelData.visualization_data;

    // Highlight active selected row on leaderboard
    const rows = elements.leaderboardTbody.querySelectorAll("tr");
    rows.forEach(r => {
        const rowName = r.querySelector("strong").innerText.toLowerCase();
        if (rowName === modelName) {
            r.style.border = "1px solid var(--accent-indigo)";
        } else {
            r.style.border = "none";
        }
    });

    // Destroy charts
    if (valChartInstance1) valChartInstance1.destroy();
    if (valChartInstance2) valChartInstance2.destroy();

    // Reset Decision Tree rules view
    elements.dtRulesCard.classList.add("hidden");

    const canvas1 = document.getElementById("validation-plot-1");
    const cmGrid = document.getElementById("confusion-matrix-grid");
    const canvas2 = document.getElementById("validation-plot-2");
    const noImportanceBanner = document.getElementById("no-importance-banner");

    if (taskType === "regression") {
        elements.valPlot1Title.innerHTML = `<i class="fa-solid fa-chart-line"></i> ${modelName.toUpperCase()} - Actual vs Predicted`;
        elements.valPlot2Title.innerHTML = `<i class="fa-solid fa-circle-nodes"></i> ${modelName.toUpperCase()} - Residual Plot`;

        // Toggle visibility
        canvas1.classList.remove("hidden");
        cmGrid.classList.add("hidden");
        canvas2.classList.remove("hidden");
        if (noImportanceBanner) noImportanceBanner.classList.add("hidden");

        // 1. Actual vs Predicted Scatter
        const actVal = valData.actual_vs_predicted.actual;
        const predVal = valData.actual_vs_predicted.predicted;
        
        // Find min/max for drawing diagonal line
        const minVal = Math.min(...actVal, ...predVal);
        const maxVal = Math.max(...actVal, ...predVal);

        valChartInstance1 = new Chart(canvas1, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Predictions',
                        data: actVal.map((v, i) => ({ x: v, y: predVal[i] })),
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: '#8b5cf6',
                        pointRadius: 4
                    },
                    {
                        label: 'Perfect Fit (y = x)',
                        data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
                        type: 'line',
                        borderColor: 'rgba(239, 68, 68, 0.6)',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Actual Value', color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Predicted Value', color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });

        // 2. Residuals plot (Predicted vs Residual)
        const predRes = valData.residuals.predicted;
        const resVal = valData.residuals.residuals;

        valChartInstance2 = new Chart(canvas2, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Residual Error',
                        data: predRes.map((v, i) => ({ x: v, y: resVal[i] })),
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderColor: '#3b82f6',
                        pointRadius: 4
                    },
                    {
                        label: 'Zero Error Line',
                        data: [{ x: Math.min(...predRes), y: 0 }, { x: Math.max(...predRes), y: 0 }],
                        type: 'line',
                        borderColor: 'rgba(255,255,255,0.2)',
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Predicted Value', color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Residual (Actual - Pred)', color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });

    } else { // Classification
        elements.valPlot1Title.innerHTML = `<i class="fa-solid fa-table-cells"></i> ${modelName.toUpperCase()} - Confusion Matrix`;
        elements.valPlot2Title.innerHTML = `<i class="fa-solid fa-chart-bar"></i> Feature Importance`;

        // Toggle visibility
        canvas1.classList.add("hidden");
        cmGrid.classList.remove("hidden");

        // 1. Confusion matrix (we render using bubble chart or custom HTML blocks inside container, custom block is easier!)
        const cm = valData.confusion_matrix;
        const classes = cm.classes;
        const matrix = cm.matrix;

        // Custom HTML confusion matrix grid
        cmGrid.innerHTML = "";
        cmGrid.style.display = "grid";
        cmGrid.style.gridTemplateColumns = `repeat(${classes.length + 1}, 1fr)`;
        cmGrid.style.gap = "4px";
        cmGrid.style.width = "100%";
        cmGrid.style.height = "100%";

        // Empty corner
        const corner = document.createElement("div");
        corner.className = "cm-label corner";
        corner.innerText = "Act\\Pred";
        cmGrid.appendChild(corner);

        // Header labels
        classes.forEach(c => {
            const h = document.createElement("div");
            h.className = "cm-label header";
            h.innerText = c;
            cmGrid.appendChild(h);
        });

        // Rows
        for (let i = 0; i < classes.length; i++) {
            const rLabel = document.createElement("div");
            rLabel.className = "cm-label row-lbl";
            rLabel.innerText = classes[i];
            cmGrid.appendChild(rLabel);

            for (let j = 0; j < classes.length; j++) {
                const cell = document.createElement("div");
                const val = matrix[i][j];
                cell.className = "cm-cell";
                cell.innerText = val;
                
                // Color cell based on correctness (diagonal elements)
                if (i === j) {
                    cell.style.backgroundColor = `rgba(16, 185, 129, ${val > 0 ? 0.3 + (val / Math.max(...matrix.flat())) * 0.7 : 0.1})`;
                    cell.style.border = "1px solid var(--success)";
                } else {
                    cell.style.backgroundColor = `rgba(239, 68, 68, ${val > 0 ? 0.2 + (val / Math.max(...matrix.flat())) * 0.8 : 0})`;
                    if (val > 0) cell.style.border = "1px solid var(--error)";
                }
                cmGrid.appendChild(cell);
            }
        }

        // 2. Feature Importance Chart
        if (valData.feature_importance) {
            const feats = valData.feature_importance.features;
            const scores = valData.feature_importance.scores;

            // Toggle visibility
            canvas2.classList.remove("hidden");
            if (noImportanceBanner) noImportanceBanner.classList.add("hidden");

            valChartInstance2 = new Chart(canvas2, {
                type: 'bar',
                data: {
                    labels: feats.slice(0, 10),
                    datasets: [{
                        label: 'Importance Score',
                        data: scores.slice(0, 10),
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: '#6366f1',
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Score', color: 'var(--text-secondary)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { grid: { display: false } }
                    }
                }
            });
        } else {
            // Toggle visibility
            canvas2.classList.add("hidden");
            if (noImportanceBanner) noImportanceBanner.classList.remove("hidden");
        }
    }

    // Display rules if Decision Tree
    if (modelName === "decision_tree" && valData.tree_rules) {
        elements.dtRulesCard.classList.remove("hidden");
        elements.dtRulesContent.innerText = valData.tree_rules;
    }
}

// Request Gemini explanation of results
async function generateModelReportAI(topModelName, topModelData) {
    elements.aiReportLoading.classList.remove("hidden");
    elements.aiReportContent.innerHTML = "";
    elements.exportReportBtn.classList.add("hidden");

    try {
        const response = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: state.datasetId,
                api_key: state.geminiKey,
                evaluation_results: topModelData
            })
        });

        if (!response.ok) throw new Error("API call for evaluation explanation failed.");
        
        const data = await response.json();
        
        elements.aiReportLoading.classList.add("hidden");
        
        // Parse simple markdown
        let formatted = data.explanation
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/### (.*?)\n/g, '<h3>$1</h3>')
            .replace(/## (.*?)\n/g, '<h2>$1</h2>')
            .replace(/# (.*?)\n/g, '<h1>$1</h1>')
            .replace(/\*\*([\s\S]*?)\*\*/g, '<strong>$1</strong>')
            .replace(/- (.*?)\n/g, '<li>$1</li>')
            .replace(/\n/g, "<br/>");
            
        elements.aiReportContent.innerHTML = `<div class="ai-report-box">${formatted}</div>`;
        elements.exportReportBtn.classList.remove("hidden");

    } catch (error) {
        elements.aiReportLoading.classList.add("hidden");
        elements.aiReportContent.innerHTML = `
            <div class="canvas-empty-state">
                <i class="fa-solid fa-triangle-exclamation" style="color:var(--error)"></i>
                <p>AI Explanation Report failed to load. Please configure your API key at the top.</p>
            </div>
        `;
    }
}

// Export learning report
function exportLearningReport() {
    const reportText = elements.aiReportContent.innerText;
    if (!reportText) return;

    const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `ML_Brain_Report_${state.filename.split('.')[0]}.txt`;
    link.click();
    showNotification("Report downloaded as TXT file!", "success");
}

// UI Loading utility
function showLoader(show, text = "Loading...") {
    // We can use a global overlay loader or toast
    // For now, write a clean loading overlay in HTML dynamically
    let loader = document.getElementById("global-loader-overlay");
    if (show) {
        if (!loader) {
            loader = document.createElement("div");
            loader.id = "global-loader-overlay";
            loader.style.position = "fixed";
            loader.style.top = "0";
            loader.style.left = "0";
            loader.style.width = "100%";
            loader.style.height = "100%";
            loader.style.backgroundColor = "rgba(10,14,23,0.85)";
            loader.style.backdropFilter = "blur(12px)";
            loader.style.zIndex = "9999";
            loader.style.display = "flex";
            loader.style.flexDirection = "column";
            loader.style.alignItems = "center";
            loader.style.justifyContent = "center";
            loader.style.color = "white";
            loader.innerHTML = `
                <div class="spinner-giant"></div>
                <h3 id="global-loader-text" style="margin-top:20px; font-family:var(--font-heading)">${text}</h3>
            `;
            document.body.appendChild(loader);
        } else {
            document.getElementById("global-loader-text").innerText = text;
            loader.style.display = "flex";
        }
    } else {
        if (loader) loader.style.display = "none";
    }
}

// Notification system
function showNotification(msg, type = "success") {
    // Standard beautiful toast
    const container = document.getElementById("toast-container") || createToastContainer();
    
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    let icon = "fa-circle-check";
    if (type === "error") icon = "fa-circle-xmark";
    if (type === "warning") icon = "fa-triangle-exclamation";
    
    toast.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <span>${msg}</span>
    `;
    
    container.appendChild(toast);
    
    // Animate in
    setTimeout(() => toast.classList.add("show"), 10);
    
    // Animate out
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function createToastContainer() {
    const c = document.createElement("div");
    c.id = "toast-container";
    c.style.position = "fixed";
    c.style.bottom = "20px";
    c.style.right = "20px";
    c.style.zIndex = "10000";
    c.style.display = "flex";
    c.style.flexDirection = "column";
    c.style.gap = "10px";
    
    // Add CSS dynamically for toast
    const style = document.createElement("style");
    style.innerHTML = `
        .toast {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.8rem 1.2rem;
            border-radius: 8px;
            color: white;
            font-size: 0.88rem;
            font-family: var(--font-body);
            font-weight: 500;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.08);
            transform: translateX(120%);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
        }
        .toast.show {
            transform: translateX(0);
        }
        .toast.success { background-color: rgba(16, 185, 129, 0.9); border-color: rgba(16, 185, 129, 0.3); }
        .toast.error { background-color: rgba(239, 68, 68, 0.9); border-color: rgba(239, 68, 68, 0.3); }
        .toast.warning { background-color: rgba(245, 158, 11, 0.9); border-color: rgba(245, 158, 11, 0.3); }
        
        /* Confusion matrix specific styling */
        .cm-label {
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.75rem;
            color: var(--text-secondary);
            padding: 8px;
            text-align: center;
        }
        .cm-label.header { border-bottom: 2px solid var(--border-glass); }
        .cm-label.row-lbl { border-right: 2px solid var(--border-glass); text-align: right; justify-content: flex-end; }
        .cm-cell {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            font-weight: 800;
            color: white;
            border-radius: 6px;
            min-height: 50px;
        }
        
        .ai-report-box {
            line-height: 1.7;
            padding: 5px;
        }
        .ai-report-box h2 {
            font-size: 1.3rem;
            margin-top: 15px;
            border-bottom: 1px solid var(--border-glass);
            padding-bottom: 5px;
            color: var(--accent-purple);
        }
        .ai-report-box h3 {
            font-size: 1.05rem;
            margin-top: 10px;
            color: var(--accent-blue);
        }
    `;
    document.head.appendChild(style);
    document.body.appendChild(c);
    return c;
}

// Window load init
window.addEventListener("DOMContentLoaded", init);
