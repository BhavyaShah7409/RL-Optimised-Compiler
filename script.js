// Global variables
let codeEditor
let currentCompilationData = null
const allStagesExpanded = true

// API Base URL - Update this with your actual Render backend URL
const API_BASE_URL = "https://rl-optimised-compiler.onrender.com"

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  initializeEditor()
  setupEventListeners()
  setupDocumentationNav()
})

// Initialize CodeMirror editor
function initializeEditor() {
  const textarea = document.getElementById("codeEditor")
  codeEditor = CodeMirror.fromTextArea(textarea, {
    mode: "text/x-csrc",
    theme: "default", // Light theme
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
  })

  // Load saved code on startup
  const savedCode = localStorage.getItem("compiler_code")
  if (savedCode) {
    codeEditor.setValue(savedCode)
  }

  // Auto-save functionality
  let autoSaveTimeout
  codeEditor.on("change", () => {
    clearTimeout(autoSaveTimeout)
    autoSaveTimeout = setTimeout(() => {
      const code = codeEditor.getValue()
      localStorage.setItem("compiler_code", code)
    }, 1000)
  })
}

// Setup event listeners
function setupEventListeners() {
  // Run button
  document.getElementById("runBtn").addEventListener("click", compileAndRun)

  // Copy button
  document.getElementById("copyBtn").addEventListener("click", copyCode)

  // Tab switching
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", (e) => {
      const tabName = e.currentTarget.getAttribute("data-tab")
      switchTab(tabName)
    })
  })

  // Stage toggles
  document.querySelectorAll(".stage-header").forEach((header) => {
    header.addEventListener("click", () => {
      const stageId = header.getAttribute("data-stage")
      toggleStage(stageId)
    })
  })

  // Expand/Collapse all buttons
  document.getElementById("expandAllBtn").addEventListener("click", expandAllStages)
  document.getElementById("collapseAllBtn").addEventListener("click", collapseAllStages)

  // Keyboard shortcuts
  document.addEventListener("keydown", (event) => {
    // Ctrl+Enter or Cmd+Enter to compile and run
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault()
      compileAndRun()
    }

    // Ctrl+/ or Cmd+/ to toggle comments (basic implementation)
    if ((event.ctrlKey || event.metaKey) && event.key === "/") {
      event.preventDefault()
      toggleComment()
    }
  })
}

// Setup documentation navigation
function setupDocumentationNav() {
  const docLinks = document.querySelectorAll(".doc-nav a")

  docLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault()

      // Remove active class from all links
      docLinks.forEach((l) => l.classList.remove("active"))

      // Add active class to clicked link
      link.classList.add("active")

      // Scroll to section
      const targetId = link.getAttribute("href").substring(1)
      const targetSection = document.getElementById(targetId)

      if (targetSection) {
        targetSection.scrollIntoView({ behavior: "smooth" })
      }
    })
  })
}

// Tab switching
function switchTab(tabName) {
  // Hide all tab contents
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.remove("active")
  })

  // Remove active class from all tab buttons
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.remove("active")
  })

  // Show selected tab content
  document.getElementById(tabName).classList.add("active")

  // Add active class to clicked tab button
  document.querySelector(`.tab-button[data-tab="${tabName}"]`).classList.add("active")
}

// Toggle compilation stage visibility
function toggleStage(stageId) {
  const content = document.getElementById(stageId)
  const stage = content.closest(".stage")

  stage.classList.toggle("collapsed")
}

// Expand all stages
function expandAllStages() {
  document.querySelectorAll(".stage").forEach((stage) => {
    stage.classList.remove("collapsed")
  })
}

// Collapse all stages
function collapseAllStages() {
  document.querySelectorAll(".stage").forEach((stage) => {
    stage.classList.add("collapsed")
  })
}

// Copy code to clipboard
function copyCode() {
  const code = codeEditor.getValue()
  navigator.clipboard
    .writeText(code)
    .then(() => {
      const btn = document.getElementById("copyBtn")
      const originalText = btn.innerHTML
      btn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
        Copied!
      `
      btn.classList.add("success")

      setTimeout(() => {
        btn.innerHTML = originalText
        btn.classList.remove("success")
      }, 2000)
    })
    .catch((err) => {
      console.error("Failed to copy code:", err)
      showToast("Failed to copy code to clipboard", "error")
    })
}

// Show toast notification
function showToast(message, type = "info") {
  // Create toast element if it doesn't exist
  let toast = document.querySelector(".toast")
  if (!toast) {
    toast = document.createElement("div")
    toast.className = "toast"
    document.body.appendChild(toast)
  }

  // Set message and type
  toast.textContent = message
  toast.className = `toast ${type}`

  // Show toast
  toast.classList.add("show")

  // Hide toast after 3 seconds
  setTimeout(() => {
    toast.classList.remove("show")
  }, 3000)
}

// Main compilation and execution function
async function compileAndRun() {
  const runBtn = document.getElementById("runBtn")
  const originalText = runBtn.innerHTML

  // Show loading state
  runBtn.innerHTML = `
    <span class="loading"></span>
    Compiling...
  `
  runBtn.disabled = true

  // Update status
  updateExecutionStatus("Compiling...", "info")

  try {
    const sourceCode = codeEditor.getValue()
    const inputData = document.getElementById("inputData").value

    if (!sourceCode.trim()) {
      throw new Error("Please enter some source code")
    }

    // Send compilation request to Render backend
    const response = await fetch(`${API_BASE_URL}/api/compile`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        source_code: sourceCode,
        input_data: inputData,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()

    if (!result.success) {
      throw new Error(result.error || "Compilation failed")
    }

    // Store compilation data
    currentCompilationData = result.data

    // Update all compilation stages
    updateCompilationStages(result.data)

    // Expand the execution results stage
    const executionStage = document.getElementById("execution").closest(".stage")
    executionStage.classList.remove("collapsed")

    // Scroll to execution results
    executionStage.scrollIntoView({ behavior: "smooth" })

    // Update execution status
    if (result.data.success) {
      updateExecutionStatus("Completed Successfully", "success")
    } else {
      updateExecutionStatus("Completed with Errors", "error")
    }
  } catch (error) {
    console.error("Compilation error:", error)
    updateExecutionStatus("Compilation Failed", "error")

    // Show error in the error section
    const errorSection = document.getElementById("errorSection")
    const errorResult = document.getElementById("errorResult")
    errorSection.style.display = "block"

    // Show more detailed error message
    if (error.message.includes("Failed to fetch") || error.message.includes("HTTP error")) {
      errorResult.textContent = `Connection Error: Unable to connect to the backend server. Please check if the backend is running at ${API_BASE_URL}`
    } else {
      errorResult.textContent = error.message
    }

    // Clear other outputs
    clearCompilationStages()
  } finally {
    // Restore button state
    runBtn.innerHTML = originalText
    runBtn.disabled = false
  }
}

// Update execution status
function updateExecutionStatus(message, type) {
  const statusElement = document.getElementById("executionStatus")
  statusElement.textContent = `Status: ${message}`
  statusElement.className = type

  if (currentCompilationData && currentCompilationData.execution_time) {
    document.getElementById("executionTime").textContent =
      `Execution time: ${(currentCompilationData.execution_time * 1000).toFixed(2)}ms`
  }
}

// Update all compilation stages with data
function updateCompilationStages(data) {
  // 1. Tokens
  updateTokensOutput(data.tokens)

  // 2. AST
  updateASTOutput(data.ast)

  // 3. Semantic Analysis
  updateSemanticOutput(data.semantic_errors)

  // 4. Intermediate Code
  updateIntermediateOutput(data.intermediate_code)

  // 5. Optimization
  updateOptimizationOutput(data.optimized_code, data.optimization_log)

  // 6. Python Code
  updatePythonOutput(data.python_code)

  // 7. Execution Results
  updateExecutionOutput(data.output, data.errors, data.success)
}

// Update tokens output
function updateTokensOutput(tokens) {
  const output = document.getElementById("tokensOutput")
  if (tokens && tokens.length > 0) {
    // Format each token with proper padding and handle newlines
    const tokenList = tokens
      .map((token) => {
        // Replace newline values with visible representation
        const displayValue = token.type === "NEWLINE" ? "\\n" : token.value

        // Fix token type display (RBRACE should be RPAREN)
        const displayType = token.type === "RBRACE" ? "RPAREN" : token.type

        // Ensure consistent padding
        return `${displayType.padEnd(15)} | ${displayValue.padEnd(15)} | Line ${token.line}, Col ${token.column}`
      })
      .join("\n")

    output.textContent = `Type            | Value           | Position\n${"─".repeat(60)}\n${tokenList}`
  } else {
    output.textContent = "No tokens generated"
  }
}

// Update AST output
function updateASTOutput(ast) {
  const output = document.getElementById("astOutput")
  if (ast) {
    output.textContent = JSON.stringify(ast, null, 2)
  } else {
    output.textContent = "No AST generated"
  }
}

// Update semantic analysis output
function updateSemanticOutput(errors) {
  const output = document.getElementById("semanticOutput")
  if (errors && errors.length > 0) {
    output.textContent = `Semantic Errors Found:\n${errors.join("\n")}`
    output.classList.add("error-output")
  } else {
    output.textContent = "No semantic errors found"
    output.classList.remove("error-output")
  }
}

// Update intermediate code output
function updateIntermediateOutput(code) {
  const output = document.getElementById("intermediateOutput")
  if (code && code.length > 0) {
    output.textContent = code.join("\n")
  } else {
    output.textContent = "No intermediate code generated"
  }
}

// Update optimization output
function updateOptimizationOutput(optimizedCode, optimizationLog) {
  const logElement = document.getElementById("optimizationLog")
  const codeElement = document.getElementById("optimizedOutput")

  if (optimizationLog && optimizationLog.length > 0) {
    logElement.innerHTML = optimizationLog.join("<br>")
  } else {
    logElement.innerHTML = "No optimizations applied"
  }

  if (optimizedCode && optimizedCode.length > 0) {
    codeElement.textContent = optimizedCode.join("\n")
  } else {
    codeElement.textContent = "No optimized code generated"
  }
}

// Update Python code output
function updatePythonOutput(code) {
  const output = document.getElementById("pythonOutput")
  if (code) {
    output.textContent = code
  } else {
    output.textContent = "No Python code generated"
  }
}

// Update execution output
function updateExecutionOutput(output, errors, success) {
  const outputElement = document.getElementById("outputResult")
  const errorElement = document.getElementById("errorResult")
  const errorSection = document.getElementById("errorSection")

  // Update output
  if (output) {
    outputElement.textContent = output
  } else {
    outputElement.textContent = "No output generated"
  }

  // Update errors
  if (errors && errors.trim()) {
    errorSection.style.display = "block"
    errorElement.textContent = errors
  } else {
    errorSection.style.display = "none"
  }
}

// Clear all compilation stages
function clearCompilationStages() {
  document.getElementById("tokensOutput").textContent = "Run compilation to see tokens..."
  document.getElementById("astOutput").textContent = "Run compilation to see AST..."
  document.getElementById("semanticOutput").textContent = "Run compilation to see semantic analysis..."
  document.getElementById("intermediateOutput").textContent = "Run compilation to see intermediate code..."
  document.getElementById("optimizationLog").textContent = "Run compilation to see optimization log..."
  document.getElementById("optimizedOutput").textContent = "Optimized code will appear here..."
  document.getElementById("pythonOutput").textContent = "Run compilation to see generated Python code..."
  document.getElementById("outputResult").textContent = "Run compilation to see output..."
  document.getElementById("executionTime").textContent = "Execution time: --"
}

// Basic comment toggling
function toggleComment() {
  const cursor = codeEditor.getCursor()
  const line = codeEditor.getLine(cursor.line)

  if (line.trim().startsWith("//")) {
    // Remove comment
    const newLine = line.replace(/^\s*\/\/\s?/, "")
    codeEditor.replaceRange(newLine, { line: cursor.line, ch: 0 }, { line: cursor.line, ch: line.length })
  } else {
    // Add comment
    const indentMatch = line.match(/^(\s*)/)
    const indent = indentMatch ? indentMatch[1] : ""
    const newLine = indent + "// " + line.trim()
    codeEditor.replaceRange(newLine, { line: cursor.line, ch: 0 }, { line: cursor.line, ch: line.length })
  }
}

// Add CSS for toast notifications
const toastStyle = document.createElement("style")
toastStyle.textContent = `
.toast {
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 12px 20px;
  background-color: var(--color-bg);
  color: var(--color-text);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  opacity: 0;
  transform: translateY(20px);
  transition: opacity var(--transition-normal), transform var(--transition-normal);
  z-index: 1000;
  max-width: 300px;
}

.toast.show {
  opacity: 1;
  transform: translateY(0);
}

.toast.info {
  border-left: 4px solid var(--color-primary);
}

.toast.success {
  border-left: 4px solid var(--color-success);
}

.toast.error {
  border-left: 4px solid var(--color-error);
}

.toast.warning {
  border-left: 4px solid var(--color-warning);
}
`
document.head.appendChild(toastStyle)
