// Global variables
let codeEditor
let currentCompilationData = null

const API_BASE_URL = "https://rl-optimised-compiler.onrender.com"

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  initializeEditor()
  setupEventListeners()
})

// Initialize CodeMirror editor
function initializeEditor() {
  const textarea = document.getElementById("codeEditor")
  codeEditor = CodeMirror.fromTextArea(textarea, {
    mode: "text/x-csrc",
    theme: "default",
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

  codeEditor.setSize(null, 300)
}

// Setup event listeners
function setupEventListeners() {
  // Run button
  document.getElementById("runBtn").addEventListener("click", compileAndRun)

  // Copy button
  document.getElementById("copyBtn").addEventListener("click", copyCode)

  // Stage toggles
  document.querySelectorAll(".stage-header").forEach((header) => {
    header.addEventListener("click", function () {
      const stageId = this.getAttribute("onclick").match(/'([^']+)'/)[1]
      toggleStage(stageId)
    })
  })
}

// Tab switching
function showTab(tabName) {
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
  event.target.classList.add("active")
}

// Toggle compilation stage visibility
function toggleStage(stageId) {
  const content = document.getElementById(stageId)
  const header = content.previousElementSibling
  const icon = header.querySelector(".toggle-icon")

  if (content.classList.contains("collapsed")) {
    content.classList.remove("collapsed")
    icon.textContent = "-"
  } else {
    content.classList.add("collapsed")
    icon.textContent = "+"
  }
}

// Copy code to clipboard
function copyCode() {
  const code = codeEditor.getValue()
  navigator.clipboard
    .writeText(code)
    .then(() => {
      const btn = document.getElementById("copyBtn")
      const originalText = btn.textContent
      btn.textContent = "Copied!"
      btn.style.backgroundColor = "#28a745"
      btn.style.color = "white"

      setTimeout(() => {
        btn.textContent = originalText
        btn.style.backgroundColor = ""
        btn.style.color = ""
      }, 2000)
    })
    .catch((err) => {
      console.error("Failed to copy code:", err)
      alert("Failed to copy code to clipboard")
    })
}

// Main compilation and execution function
async function compileAndRun() {
  const runBtn = document.getElementById("runBtn")
  const originalText = runBtn.innerHTML

  // Show loading state
  runBtn.innerHTML = '<span class="loading"></span>Compiling...'
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

    const result = await response.json()

    if (!result.success) {
      throw new Error(result.error || "Compilation failed")
    }

    // Store compilation data
    currentCompilationData = result.data

    // Update all compilation stages
    updateCompilationStages(result.data)

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
    errorResult.textContent = error.message

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
    const tokenList = tokens
      .map((token) => `${token.type.padEnd(15)} | ${token.value.padEnd(10)} | Line ${token.line}, Col ${token.column}`)
      .join("\n")
    output.textContent = `Type            | Value      | Position\n${"─".repeat(50)}\n${tokenList}`
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
    output.style.color = "#dc3545"
  } else {
    output.textContent = "No semantic errors found"
    output.style.color = "#28a745"
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
    logElement.innerHTML = `<strong>Optimizations Applied:</strong><br>${optimizationLog.join("<br>")}`
  } else {
    logElement.innerHTML = "<strong>No optimizations applied</strong>"
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

// Auto-save functionality (optional)
let autoSaveTimeout
if (typeof codeEditor !== "undefined") {
  codeEditor.on("change", () => {
    clearTimeout(autoSaveTimeout)
    autoSaveTimeout = setTimeout(() => {
      const code = codeEditor.getValue()
      localStorage.setItem("compiler_code", code)
    }, 1000)
  })

  // Load saved code on startup
  const savedCode = localStorage.getItem("compiler_code")
  if (savedCode) {
    codeEditor.setValue(savedCode)
  }
}
