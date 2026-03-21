// Global variables
let codeEditor;
let currentCompilationData = null;

// API Base URL - Update this with your actual Render backend URL
// Set this to true when deploying to Vercel!
const IS_PRODUCTION = true; 
const API_BASE_URL = IS_PRODUCTION 
  ? "https://rl-optimised-compiler.onrender.com" 
  : "http://localhost:5000";

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  initializeEditor();
  setupEventListeners();
  setupDocumentationNav();
});

// Initialize CodeMirror editor
function initializeEditor() {
  const textarea = document.getElementById("codeEditor");
  codeEditor = CodeMirror.fromTextArea(textarea, {
    mode: "text/x-csrc",
    theme: "default", // Changed to Default Theme for Professional UI
    lineNumbers: true,
    indentUnit: 4,
    tabSize: 4,
    indentWithTabs: false,
    lineWrapping: true,
    matchBrackets: true,
    autoCloseBrackets: true,
    styleActiveLine: true,
    foldGutter: true,
    gutters: ["CodeMirror-linenumbers", "CodeMirror-foldgutter"],
  });

  // Load saved code on startup
  const savedCode = localStorage.getItem("compiler_code");
  if (savedCode) {
    codeEditor.setValue(savedCode);
  }

  // Auto-save functionality
  let autoSaveTimeout;
  codeEditor.on("change", () => {
    clearTimeout(autoSaveTimeout);
    autoSaveTimeout = setTimeout(() => {
      const code = codeEditor.getValue();
      localStorage.setItem("compiler_code", code);
    }, 1000);
  });
}

// Setup event listeners
function setupEventListeners() {
  // Run button
  document.getElementById("runBtn").addEventListener("click", compileAndRun);

  // Copy button
  document.getElementById("copyBtn").addEventListener("click", copyCode);

  // Tab switching
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", (e) => {
      const tabName = e.currentTarget.getAttribute("data-tab");
      switchTab(tabName);
    });
  });

  // Stage toggles
  document.querySelectorAll(".stage-header").forEach((header) => {
    header.addEventListener("click", () => {
      const stageId = header.getAttribute("data-stage");
      toggleStage(stageId);
    });
  });

  // Expand/Collapse all buttons
  document.getElementById("expandAllBtn").addEventListener("click", expandAllStages);
  document.getElementById("collapseAllBtn").addEventListener("click", collapseAllStages);

  // Keyboard shortcuts
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      compileAndRun();
    }
    if ((event.ctrlKey || event.metaKey) && event.key === "/") {
      event.preventDefault();
      toggleComment();
    }
  });
}

// Setup documentation navigation
function setupDocumentationNav() {
  const docLinks = document.querySelectorAll(".doc-nav a");

  docLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      docLinks.forEach((l) => l.classList.remove("active"));
      link.classList.add("active");

      const targetId = link.getAttribute("href").substring(1);
      const targetSection = document.getElementById(targetId);

      if (targetSection) {
        targetSection.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
}

// Tab switching
function switchTab(tabName) {
  document.querySelectorAll(".tab-content").forEach((content) => content.classList.remove("active"));
  document.querySelectorAll(".tab-button").forEach((button) => button.classList.remove("active"));
  document.getElementById(tabName).classList.add("active");
  document.querySelector(`.tab-button[data-tab="${tabName}"]`).classList.add("active");
}

// Toggle compilation stage visibility
function toggleStage(stageId) {
  const content = document.getElementById(stageId);
  const stage = content.closest(".stage");
  stage.classList.toggle("collapsed");
}

// Expand all stages
function expandAllStages() {
  document.querySelectorAll(".stage").forEach((stage) => stage.classList.remove("collapsed"));
}

// Collapse all stages
function collapseAllStages() {
  document.querySelectorAll(".stage").forEach((stage) => stage.classList.add("collapsed"));
}

// Copy code to clipboard
function copyCode() {
  const code = codeEditor.getValue();
  navigator.clipboard.writeText(code)
    .then(() => {
      const btn = document.getElementById("copyBtn");
      const originalText = btn.innerHTML;
      btn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
        Copied!
      `;
      btn.classList.add("success");
      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.classList.remove("success");
      }, 2000);
    })
    .catch((err) => {
      console.error("Failed to copy code:", err);
      showToast("Failed to copy code to clipboard", "error");
    });
}

// Show toast notification
function showToast(message, type = "info") {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.className = `toast ${type}`;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3000);
}

// Async delay utility
const delay = ms => new Promise(res => setTimeout(res, ms));

// Main compilation and execution function
async function compileAndRun() {
  const runBtn = document.getElementById("runBtn");
  const originalText = runBtn.innerHTML;

  runBtn.innerHTML = `<span class="loading"></span> Compiling...`;
  runBtn.disabled = true;

  updateExecutionStatus("Compiling to Source...", "info");
  
  // Show UI Bar and reset
  const pipelineProgress = document.getElementById("pipelineProgress");
  pipelineProgress.style.display = "flex";
  document.getElementById("progressBar").style.width = "0%";
  for (let i = 1; i <= 7; i++) {
    const node = document.getElementById(`node-${i}`);
    node.className = "pipeline-node";
  }

  try {
    const sourceCode = codeEditor.getValue();
    const inputData = document.getElementById("inputData").value;

    if (!sourceCode.trim()) throw new Error("Please enter some source code");

    const response = await fetch(`${API_BASE_URL}/api/compile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_code: sourceCode, input_data: inputData }),
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const result = await response.json();
    if (!result.success) throw new Error(result.error || "Compilation failed");

    currentCompilationData = result.data;
    
    // Animate successful compilation stages staggeringly!
    await updateCompilationStagesAnimated(result.data);

    const executionStage = document.getElementById("execution").closest(".stage");
    executionStage.classList.remove("collapsed");
    executionStage.scrollIntoView({ behavior: "smooth" });

    if (result.data.success) {
      updateExecutionStatus("Executed Successfully", "success");
    } else {
      updateExecutionStatus("Completed with Errors", "warning");
    }
  } catch (error) {
    console.error("Compilation error:", error);
    updateExecutionStatus("Compilation Failed", "error");

    const errorSection = document.getElementById("errorSection");
    const errorResult = document.getElementById("errorResult");
    errorSection.style.display = "block";

    if (error.message.includes("Failed to fetch") || error.message.includes("HTTP error")) {
      errorResult.textContent = `Connection Error to Backend: ${API_BASE_URL}`;
    } else {
      errorResult.textContent = error.message;
    }
    
    // Set error on progress bar
    document.getElementById("node-1").classList.add("error");

    clearCompilationStages();
  } finally {
    runBtn.innerHTML = originalText;
    runBtn.disabled = false;
  }
}

function updateExecutionStatus(message, type) {
  const statusElement = document.getElementById("executionStatus");
  statusElement.textContent = `Status: ${message}`;
  statusElement.className = type;
  if (currentCompilationData && currentCompilationData.execution_time) {
    document.getElementById("executionTime").textContent = `Execution time: ${(currentCompilationData.execution_time * 1000).toFixed(2)}ms`;
  }
}

// Mark node progress
function setPipelineProgress(stageIndex, status = "active") {
    const node = document.getElementById(`node-${stageIndex}`);
    if (node) node.classList.add(status);
    
    const maxStages = 7;
    const progressPercent = ((stageIndex) / (maxStages - 1)) * 100;
    document.getElementById("progressBar").style.width = `${progressPercent}%`;
}

async function typeWriterOutput(element, htmlContent, speed = 2) {
    element.innerHTML = "";
    element.classList.add('typewriter');
    
    // To support syntax highlighted HTML nodes being typed, we insert directly
    // since typing raw HTML chars looks bad. We simulate wait.
    element.innerHTML = htmlContent;
    
    // Wait for dramatic effect
    await delay(300);
    element.classList.remove('typewriter');
}

// Function to safely encode HTML except for our injected span tags
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// Animated updates
async function updateCompilationStagesAnimated(data) {
  // Clear and Collapse everything for dramatic effect
  collapseAllStages();
  
  // 1. Lexical Analysis
  setPipelineProgress(1, "active");
  document.getElementById("tokens").closest(".stage").classList.remove("collapsed");
  await delay(400);
  const tokenHTML = formatTokensOutput(data.tokens);
  await typeWriterOutput(document.getElementById("tokensOutput"), tokenHTML);
  setPipelineProgress(1, "success");

  // 2. AST
  setPipelineProgress(2, "active");
  document.getElementById("ast").closest(".stage").classList.remove("collapsed");
  await delay(300);
  const astHTML = formatASTOutput(data.ast);
  await typeWriterOutput(document.getElementById("astOutput"), astHTML);
  setPipelineProgress(2, "success");

  // 3. Semantic Analysis
  setPipelineProgress(3, "active");
  await delay(200);
  if (data.semantic_errors && data.semantic_errors.length > 0) {
      document.getElementById("semanticOutput").textContent = `Semantic Errors Found:\n${data.semantic_errors.join("\n")}`;
      document.getElementById("semanticOutput").classList.add("error-output");
      setPipelineProgress(3, "error");
  } else {
      document.getElementById("semanticOutput").textContent = "No semantic errors found ✓";
      document.getElementById("semanticOutput").classList.remove("error-output");
      setPipelineProgress(3, "success");
  }

  // 4. Intermediate Code
  setPipelineProgress(4, "active");
  document.getElementById("intermediate").closest(".stage").classList.remove("collapsed");
  await delay(200);
  const intHTML = formatIntermediateOutput(data.intermediate_code);
  await typeWriterOutput(document.getElementById("intermediateOutput"), intHTML);
  setPipelineProgress(4, "success");

  // 5. Optimization
  setPipelineProgress(5, "active");
  document.getElementById("optimization").closest(".stage").classList.remove("collapsed");
  await delay(300);
  updateOptimizationOutput(data.optimized_code, data.optimization_log);
  setPipelineProgress(5, "success");

  // 6. Python Code
  setPipelineProgress(6, "active");
  document.getElementById("python").closest(".stage").classList.remove("collapsed");
  await delay(200);
  await typeWriterOutput(document.getElementById("pythonOutput"), escapeHTML(data.python_code || "No Python code generated"));
  setPipelineProgress(6, "success");

  // 7. Execution Results
  setPipelineProgress(7, "active");
  await delay(300);
  updateExecutionOutput(data.output, data.errors, data.success);
  setPipelineProgress(7, data.success ? "success" : "error");
}

function formatTokensOutput(tokens) {
    if (!tokens || tokens.length === 0) return "No tokens generated";
    let html = `Type            | Value           | Position\n${"─".repeat(60)}\n`;
    
    tokens.forEach(token => {
        const displayValue = escapeHTML(token.type === "NEWLINE" ? "\\n" : token.value);
        const displayType = token.type === "RBRACE" ? "RPAREN" : token.type;
        
        let colorClass = "token-identifier";
        if (["KEYWORD", "TYPE", "CONTROL"].includes(displayType)) colorClass = "token-keyword";
        if (displayType === "NUMBER") colorClass = "token-number";
        if (displayType === "STRING") colorClass = "token-string";
        if (["OPERATOR", "ASSIGN"].includes(displayType)) colorClass = "token-operator";
        
        html += `${displayType.padEnd(15)} | <span class="${colorClass}">${displayValue.padEnd(15)}</span> | Line ${token.line}, Col ${token.column}\n`;
    });
    return html;
}

function formatASTOutput(ast) {
    if (!ast) return "No AST generated";
    // We stringify the JSON but color the keys
    let str = JSON.stringify(ast, null, 2);
    str = escapeHTML(str);
    // Rough regex highlight for JSON AST representation
    str = str.replace(/"type":\s*&quot;([^&]+)&quot;/g, '"type": &quot;<span class="ast-node">$1</span>&quot;');
    str = str.replace(/"(value|operation|name)":\s*&quot;([^&]+)&quot;/g, '"$1": &quot;<span class="token-string">$2</span>&quot;');
    return str;
}

function formatIntermediateOutput(code) {
    if (!code || code.length === 0) return "No intermediate code generated";
    let str = code.join("\n");
    str = escapeHTML(str);
    str = str.replace(/([a-zA-Z0-9_]+) =/g, '<span class="token-identifier">$1</span> =');
    str = str.replace(/GOTO|IF_FALSE|PRINT|SCAN/g, '<span class="token-keyword">$&</span>');
    str = str.replace(/([0-9.]+)/g, '<span class="token-number">$1</span>');
    return str;
}

function updateOptimizationOutput(optimizedCode, optimizationLog) {
  const logElement = document.getElementById("optimizationLog");
  const codeElement = document.getElementById("optimizedOutput");

  if (optimizationLog && optimizationLog.length > 0) {
    logElement.innerHTML = "=== RL Optimization Agent Logs ===<br>" + optimizationLog.join("<br>");
  } else {
    logElement.innerHTML = "No optimizations applied by Agent";
  }

  if (optimizedCode && optimizedCode.length > 0) {
    const intHTML = formatIntermediateOutput(optimizedCode);
    codeElement.innerHTML = intHTML;
  } else {
    codeElement.textContent = "No optimized code generated";
  }
}

function updateExecutionOutput(output, errors, success) {
  const outputElement = document.getElementById("outputResult");
  const errorElement = document.getElementById("errorResult");
  const errorSection = document.getElementById("errorSection");

  if (output) {
    // Add typewriter effect
    outputElement.classList.add('success');
    outputElement.textContent = output;
  } else {
    outputElement.textContent = "No output generated";
  }

  if (errors && errors.trim()) {
    errorSection.style.display = "block";
    errorElement.textContent = errors;
  } else {
    errorSection.style.display = "none";
  }
}

function clearCompilationStages() {
  document.getElementById("tokensOutput").textContent = "Run compilation to see tokens...";
  document.getElementById("astOutput").textContent = "Run compilation to see AST...";
  document.getElementById("semanticOutput").textContent = "Run compilation to see semantic analysis...";
  document.getElementById("intermediateOutput").textContent = "Run compilation to see intermediate code...";
  document.getElementById("optimizationLog").textContent = "Run compilation to see optimization log...";
  document.getElementById("optimizedOutput").textContent = "Optimized code will appear here...";
  document.getElementById("pythonOutput").textContent = "Run compilation to see generated Python code...";
  document.getElementById("outputResult").textContent = "Run compilation to see output...";
  document.getElementById("executionTime").textContent = "Execution time: --";
  document.getElementById("pipelineProgress").style.display = "none";
}

function toggleComment() {
  const cursor = codeEditor.getCursor();
  const line = codeEditor.getLine(cursor.line);

  if (line.trim().startsWith("//")) {
    const newLine = line.replace(/^\s*\/\/\s?/, "");
    codeEditor.replaceRange(newLine, { line: cursor.line, ch: 0 }, { line: cursor.line, ch: line.length });
  } else {
    const indentMatch = line.match(/^(\s*)/);
    const indent = indentMatch ? indentMatch[1] : "";
    const newLine = indent + "// " + line.trim();
    codeEditor.replaceRange(newLine, { line: cursor.line, ch: 0 }, { line: cursor.line, ch: line.length });
  }
}
