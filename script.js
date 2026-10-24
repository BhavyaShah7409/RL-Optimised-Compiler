/* ═══════════════════════════════════════════════════════════════
   RL-Optimized Source to Source Compiler — script.js
   Fixes: #17 (RBRACE→RPAREN masking removed), #18 (number regex)
═══════════════════════════════════════════════════════════════ */

"use strict";

// ── Config ───────────────────────────────────────────────────────
const API_BASE_URL = (() => {
  if (typeof window !== "undefined") {
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      return window.location.port === "5000" ? "" : "http://localhost:5000";
    }
    if (window.location.protocol === "file:") {
      return "http://localhost:5000";
    }
  }
  return "https://rl-optimised-compiler.onrender.com";
})();

// ── State ────────────────────────────────────────────────────────
let editor = null;
let compilationData = null;

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initEditor();
  bindEvents();
  bindDocNav();
});

// ── Theme Management (Default: Light) ─────────────────────────────
function initTheme() {
  const saved = localStorage.getItem("rl_compiler_theme");
  const theme = saved || "light"; // default-light
  setTheme(theme);
}

function setTheme(theme) {
  const isDark = theme === "dark";
  document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
  localStorage.setItem("rl_compiler_theme", isDark ? "dark" : "light");
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  setTheme(current === "dark" ? "light" : "dark");
}

// ── Editor setup ─────────────────────────────────────────────────
function initEditor() {
  const ta = document.getElementById("codeEditor");
  editor = CodeMirror.fromTextArea(ta, {
    mode:             "text/x-csrc",
    theme:            "default",
    lineNumbers:      true,
    indentUnit:       4,
    tabSize:          4,
    indentWithTabs:   false,
    lineWrapping:     true,
    matchBrackets:    true,
    autoCloseBrackets:true,
    styleActiveLine:  true,
  });

  const saved = localStorage.getItem("rl_compiler_code");
  if (saved) editor.setValue(saved);

  let debounce;
  editor.on("change", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      localStorage.setItem("rl_compiler_code", editor.getValue());
    }, 800);
  });
}

// ── Pipeline Navigation State ─────────────────────────────────────
let currentStageIndex = 7;
let currentViewMode = "focused"; // "focused" | "all"

// ── Event bindings ────────────────────────────────────────────────
function bindEvents() {
  document.getElementById("themeToggleBtn")?.addEventListener("click", toggleTheme);

  document.getElementById("runBtn").addEventListener("click", compileAndRun);

  document.getElementById("copyBtn").addEventListener("click", () => {
    const code = editor.getValue();
    navigator.clipboard.writeText(code).then(() => {
      showToast("Copied to clipboard", "success");
    }).catch(() => showToast("Copy failed", "error"));
  });

  document.getElementById("clearBtn").addEventListener("click", () => {
    editor.setValue("");
    localStorage.removeItem("rl_compiler_code");
    clearStages();
    showToast("Editor cleared");
  });

  // Tab switching (Compiler vs Docs)
  document.querySelectorAll(".tab-button").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Pipeline Stepper Cards
  document.querySelectorAll(".step-card").forEach(card => {
    card.addEventListener("click", () => {
      const stageIdx = parseInt(card.dataset.stage, 10);
      setActiveStage(stageIdx);
    });
  });

  // View Mode Toggle (Focused vs All Stages)
  document.getElementById("viewFocusedBtn")?.addEventListener("click", () => setViewMode("focused"));
  document.getElementById("viewAllBtn")?.addEventListener("click", () => setViewMode("all"));

  // Prev / Next stage buttons
  document.getElementById("prevStageBtn")?.addEventListener("click", () => {
    if (currentStageIndex > 1) setActiveStage(currentStageIndex - 1);
  });
  document.getElementById("nextStageBtn")?.addEventListener("click", () => {
    if (currentStageIndex < 7) setActiveStage(currentStageIndex + 1);
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); compileAndRun(); }
    if ((e.ctrlKey || e.metaKey) && e.key === "/")     { e.preventDefault(); toggleComment(); }
    if (e.altKey && e.key === "ArrowLeft")             { e.preventDefault(); if (currentStageIndex > 1) setActiveStage(currentStageIndex - 1); }
    if (e.altKey && e.key === "ArrowRight")            { e.preventDefault(); if (currentStageIndex < 7) setActiveStage(currentStageIndex + 1); }
  });

  // Bind interactive AST Tree zoom/pan controls
  bindAstTreeEvents();

  window.addEventListener("resize", () => {
    if (currentStageIndex === 2 && currentAstRaw && astViewMode === "tree") {
      fitAstTreeToScreen();
    }
  });
}

// ── Stage Switching & View Modes ──────────────────────────────────
function setActiveStage(idx) {
  currentStageIndex = idx;

  // Update Stepper Cards
  document.querySelectorAll(".step-card").forEach(c => {
    const isTarget = parseInt(c.dataset.stage, 10) === idx;
    c.classList.toggle("active", isTarget);
    c.setAttribute("aria-selected", String(isTarget));
  });

  // Update Viewports
  document.querySelectorAll(".stage-view").forEach(v => {
    const isTarget = v.id === `stageView-${idx}`;
    v.classList.toggle("active", isTarget);
  });

  // If entering AST stage, ensure tree fits viewport nicely
  if (idx === 2 && currentAstRaw && astViewMode === "tree") {
    requestAnimationFrame(() => fitAstTreeToScreen());
  }

  // If in all-stages mode, scroll to the stage
  if (currentViewMode === "all") {
    const target = document.getElementById(`stageView-${idx}`);
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function setViewMode(mode) {
  currentViewMode = mode;
  const container = document.getElementById("pipelineViewsContainer");
  const focusedBtn = document.getElementById("viewFocusedBtn");
  const allBtn = document.getElementById("viewAllBtn");

  if (mode === "all") {
    container?.classList.add("all-stages-mode");
    allBtn?.classList.add("active");
    focusedBtn?.classList.remove("active");
  } else {
    container?.classList.remove("all-stages-mode");
    focusedBtn?.classList.add("active");
    allBtn?.classList.remove("active");
    setActiveStage(currentStageIndex);
  }
}

// Global copy helper for stage action buttons
window.copyStageContent = function(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const text = el.innerText || el.textContent;
  navigator.clipboard.writeText(text).then(() => {
    showToast("Copied to clipboard", "success");
  }).catch(() => showToast("Copy failed", "error"));
};

// ── Tab switching ─────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  document.querySelectorAll(".tab-button").forEach(b => {
    b.classList.remove("active");
    b.setAttribute("aria-selected", "false");
  });
  document.getElementById(name)?.classList.add("active");
  const btn = document.querySelector(`.tab-button[data-tab="${name}"]`);
  if (btn) { btn.classList.add("active"); btn.setAttribute("aria-selected", "true"); }
}

// ── Documentation sidebar & Dynamic Scrollspy ─────────────────────
function bindDocNav() {
  const links = Array.from(document.querySelectorAll(".doc-nav a"));
  const sections = Array.from(document.querySelectorAll(".doc-section"));
  if (!links.length || !sections.length) return;

  function setActiveLink(targetId) {
    links.forEach(l => {
      const isMatch = l.getAttribute("href") === `#${targetId}`;
      l.classList.toggle("active", isMatch);
    });
  }

  // Click smooth scrolling
  links.forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      const targetId = link.getAttribute("href").slice(1);
      const target = document.getElementById(targetId);
      if (target) {
        setActiveLink(targetId);
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  // Dynamic Scrollspy with IntersectionObserver
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      // Find the topmost intersecting section
      const visibleEntries = entries.filter(e => e.isIntersecting);
      if (visibleEntries.length) {
        visibleEntries.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        setActiveLink(visibleEntries[0].target.id);
      }
    }, {
      root: null,
      rootMargin: "-80px 0px -50% 0px",
      threshold: [0, 0.25, 0.5, 0.75, 1.0]
    });

    sections.forEach(sec => observer.observe(sec));
  }

  // Secondary window scroll tracking for rapid wheel scrolling
  window.addEventListener("scroll", () => {
    const docTab = document.getElementById("documentation");
    if (!docTab || !docTab.classList.contains("active")) return;

    let currentId = sections[0].id;
    const scrollPos = window.scrollY + 120;

    for (let i = 0; i < sections.length; i++) {
      const sec = sections[i];
      if (sec.offsetTop <= scrollPos) {
        currentId = sec.id;
      }
    }
    setActiveLink(currentId);
  }, { passive: true });
}

// ── Toast ─────────────────────────────────────────────────────────
function showToast(msg, type = "info") {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.className   = `toast ${type} show`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 2800);
}

// ── Async helpers ─────────────────────────────────────────────────
const wait = ms => new Promise(r => setTimeout(r, ms));

// ── Main compile ──────────────────────────────────────────────────
async function compileAndRun() {
  const runBtn = document.getElementById("runBtn");
  runBtn.innerHTML = `<span class="spinner"></span> Compiling…`;
  runBtn.disabled = true;

  const dot = document.getElementById("statusPulseDot");
  const statusText = document.getElementById("pipelineStatusText");
  if (dot) dot.className = "status-pulse-dot running";
  if (statusText) statusText.textContent = "Processing";

  // Reset card states
  for (let i = 1; i <= 7; i++) {
    const card = document.getElementById(`stepCard-${i}`);
    const stateEl = document.getElementById(`stepState-${i}`);
    if (card) {
      card.classList.remove("running", "success", "error");
    }
    if (stateEl) stateEl.textContent = "WAIT";
  }

  try {
    const source = editor.getValue().trim();
    const input  = document.getElementById("inputData").value;
    if (!source) throw new Error("Source code is empty.");

    const res = await fetch(`${API_BASE_URL}/api/compile`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ source_code: source, input_data: input }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status} from server.`);

    const json = await res.json();
    if (!json.success) throw new Error(json.error || "Unknown compilation failure.");

    compilationData = json.data;
    await animateStages(json.data);

    if (dot) dot.className = json.data.success ? "status-pulse-dot success" : "status-pulse-dot error";
    if (statusText) statusText.textContent = json.data.success ? "Completed" : "Diagnostics Alert";

  } catch (err) {
    console.error(err);
    if (dot) dot.className = "status-pulse-dot error";
    if (statusText) statusText.textContent = "Failed";

    const errSection = document.getElementById("errorSection");
    const errOut     = document.getElementById("errorResult");
    if (errSection) errSection.style.display = "block";
    if (errOut) {
      errOut.textContent = err.message.includes("fetch") || err.message.includes("HTTP")
        ? `Cannot reach backend at ${API_BASE_URL}\n\nMake sure the local server is running on port 5000.`
        : err.message;
    }
    setActiveStage(7);

  } finally {
    runBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run`;
    runBtn.disabled = false;
  }
}

// ── Pipeline Stepper Animation ────────────────────────────────────
async function animateStages(data) {
  // Helper to update card state
  function updateCard(idx, state, metric = null, isRunning = false, isSuccess = false, isError = false) {
    const card = document.getElementById(`stepCard-${idx}`);
    const stateEl = document.getElementById(`stepState-${idx}`);
    const metricEl = document.getElementById(`stepMetric-${idx}`);
    if (stateEl) stateEl.textContent = state;
    if (metric && metricEl) metricEl.textContent = metric;
    if (card) {
      card.classList.toggle("running", isRunning);
      card.classList.toggle("success", isSuccess);
      card.classList.toggle("error", isError);
    }
  }

  // Stage 1: Lexer
  updateCard(1, "RUN", null, true);
  setActiveStage(1);
  await wait(180);
  const tbody = document.getElementById("tokensTableBody");
  if (tbody) {
    tbody.innerHTML = renderTokens(data.tokens);
  }
  document.getElementById("tokenCountChip").textContent = `${data.tokens?.length || 0} Tokens`;
  updateCard(1, "PASS", `${data.tokens?.length || 0} tok`, false, true);

  // Stage 2: AST
  updateCard(2, "RUN", null, true);
  setActiveStage(2);
  await wait(180);
  document.getElementById("astOutput").innerHTML = renderAST(data.ast);
  updateCard(2, "PASS", "Valid", false, true);

  // Stage 3: Semantic Analysis
  updateCard(3, "RUN", null, true);
  setActiveStage(3);
  await wait(160);
  const semCard = document.getElementById("semanticCard");
  const semIcon = document.getElementById("semanticIcon");
  const semTitle = document.getElementById("semanticTitle");
  const semSubtitle = document.getElementById("semanticSubtitle");
  const semOut = document.getElementById("semanticOutput");

  if (data.semantic_errors?.length) {
    semCard.classList.add("error");
    semIcon.textContent = "✕";
    semTitle.textContent = "Semantic Violations Identified";
    semSubtitle.textContent = `${data.semantic_errors.length} error(s) prevented target code generation.`;
    semOut.textContent = data.semantic_errors.join("\n");
    updateCard(3, "ERR", `${data.semantic_errors.length} err`, false, false, true);
  } else {
    semCard.classList.remove("error");
    semIcon.textContent = "✓";
    semTitle.textContent = "Scope & Reference Validation Passed";
    semSubtitle.textContent = "All identifiers, assignments, and nested blocks are structurally sound.";
    semOut.textContent = "✓ All symbols resolved successfully in symbol table.\n✓ Scope hierarchy validated.\n✓ Zero undeclared variables.";
    updateCard(3, "PASS", "0 err", false, true);
  }

  // Stage 4: TAC IR
  updateCard(4, "RUN", null, true);
  setActiveStage(4);
  await wait(160);
  document.getElementById("intermediateOutput").innerHTML = renderTAC(data.intermediate_code);
  document.getElementById("tacCountChip").textContent = `${data.intermediate_code?.length || 0} Instructions`;
  updateCard(4, "PASS", `${data.intermediate_code?.length || 0} TAC`, false, true);

  // Stage 5: RL Optimizer
  updateCard(5, "RUN", null, true);
  setActiveStage(5);
  await wait(200);
  renderOptimisation(data.optimized_code, data.optimization_log, data.intermediate_code);
  const optCount = data.optimization_log?.length || 0;
  updateCard(5, "PASS", `${optCount} opts`, false, true);

  // Stage 6: Python Code Gen
  updateCard(6, "RUN", null, true);
  setActiveStage(6);
  await wait(160);
  document.getElementById("pythonOutput").textContent = data.python_code || "# No target Python generated.";
  updateCard(6, "PASS", "Python 3.10", false, true);

  // Stage 7: Execution Environment
  updateCard(7, "RUN", null, true);
  setActiveStage(7);
  await wait(220);
  renderExecution(data.output, data.errors, data.success, data.execution_time);
  const runtimeMs = data.execution_time != null ? `${(data.execution_time * 1000).toFixed(1)}ms` : "Done";
  updateCard(7, data.success ? "PASS" : "ERR", runtimeMs, false, data.success, !data.success);
}

// ── Renderers ─────────────────────────────────────────────────────

function esc(str) {
  return String(str).replace(/[&<>"']/g, c => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":'&#39;',
  }[c]));
}

// Tabular Lexer Token stream renderer (Clean, simple, unboxed)
function renderTokens(tokens) {
  if (!tokens?.length) {
    return `<tr><td colspan="4" class="empty-state">No tokens found in stream.</td></tr>`;
  }

  const kwSet = new Set(["PRINT", "SCAN", "WHILE", "IF", "ELSE"]);
  const opSet = new Set(["PLUS", "MINUS", "MULTIPLY", "DIVIDE", "ASSIGN", "PLUS_ASSIGN", "MINUS_ASSIGN", "MULT_ASSIGN", "DIV_ASSIGN", "EQ", "NEQ", "LT", "GT", "LTE", "GTE", "NOT", "AND", "OR"]);
  const punctSet = new Set(["SEMICOLON", "LPAREN", "RPAREN", "LBRACE", "RBRACE", "COMMA"]);

  return tokens.map((tok, idx) => {
    let typeClass = "type-id";
    if (kwSet.has(tok.type)) {
      typeClass = "type-kw";
    } else if (tok.type === "NUMBER") {
      typeClass = "type-num";
    } else if (tok.type === "STRING") {
      typeClass = "type-str";
    } else if (opSet.has(tok.type) || tok.type.includes("ASSIGN") || tok.type.includes("OP")) {
      typeClass = "type-op";
    } else if (punctSet.has(tok.type)) {
      typeClass = "type-punct";
    } else if (["NEWLINE", "EOF"].includes(tok.type)) {
      typeClass = "type-ctrl";
    }

    const valDisplay = tok.type === "NEWLINE" ? "\\n" : (tok.value === "" ? "—" : esc(tok.value));

    return `<tr>
      <td class="col-num">${idx + 1}</td>
      <td class="col-type"><span class="tok-type-text ${typeClass}">${esc(tok.type)}</span></td>
      <td class="col-val"><span class="tok-val-text">${valDisplay}</span></td>
      <td class="col-pos">${tok.line}:${tok.column}</td>
    </tr>`;
  }).join("");
}

// ── Interactive AST Tree State & Logic ─────────────────────────────
let astZoom = 1.0;
let astPanX = 0;
let astPanY = 0;
let isAstPanning = false;
let astStartPan = { x: 0, y: 0 };
let astTreeBounds = null;
let currentAstRaw = null;
let astViewMode = "tree"; // "tree" | "json"

function buildAstTreeNode(node, edgeLabel = "") {
  if (!node) return null;
  const t = node.type || "Node";
  let badge = "NODE", name = t, sub = "", category = "cat-stmt", children = [];

  if (t === "Program") {
    badge = "ROOT";
    name = "Program";
    const stmtCount = (node.statements || []).length;
    sub = `${stmtCount} stmt${stmtCount === 1 ? '' : 's'}`;
    category = "cat-program";
    children = (node.statements || []).map((s, i) => buildAstTreeNode(s, `stmt[${i + 1}]`));
  } else if (t === "AssignmentStatement") {
    badge = "ASSIGN";
    name = `${node.target} ${node.operator || '='}`;
    category = "cat-stmt";
    if (node.value) children.push(buildAstTreeNode(node.value, "val"));
  } else if (t === "WhileStatement") {
    badge = "WHILE";
    name = "while (...)";
    category = "cat-stmt";
    if (node.condition) children.push(buildAstTreeNode(node.condition, "cond"));
    (node.body || []).forEach((s, i) => children.push(buildAstTreeNode(s, `body[${i + 1}]`)));
  } else if (t === "IfStatement") {
    badge = "IF";
    name = "if (...)";
    category = "cat-stmt";
    if (node.condition) children.push(buildAstTreeNode(node.condition, "cond"));
    (node.then_body || []).forEach((s, i) => children.push(buildAstTreeNode(s, `then[${i + 1}]`)));
    if (node.else_body && node.else_body.length) {
      node.else_body.forEach((s, i) => children.push(buildAstTreeNode(s, `else[${i + 1}]`)));
    }
  } else if (t === "PrintStatement") {
    badge = "PRINT";
    name = "print(...)";
    category = "cat-stmt";
    if (node.expression) children.push(buildAstTreeNode(node.expression, "expr"));
  } else if (t === "ScanStatement") {
    badge = "SCAN";
    name = `scan(${node.target})`;
    category = "cat-stmt";
  } else if (t === "BinaryOperation") {
    badge = "BIN_OP";
    name = `Op: ${node.operator}`;
    category = "cat-expr";
    if (node.left) children.push(buildAstTreeNode(node.left, "left"));
    if (node.right) children.push(buildAstTreeNode(node.right, "right"));
  } else if (t === "UnaryOperation") {
    badge = "UN_OP";
    name = `Op: ${node.operator}`;
    category = "cat-expr";
    if (node.operand) children.push(buildAstTreeNode(node.operand, "operand"));
  } else if (t === "NumberLiteral") {
    badge = "NUM";
    let numVal = node.value;
    if (typeof numVal === "number" && Number.isInteger(numVal)) {
      numVal = String(numVal);
    } else {
      numVal = String(numVal);
    }
    name = numVal;
    category = "cat-leaf";
  } else if (t === "StringLiteral") {
    badge = "STR";
    name = `"${node.value}"`;
    category = "cat-leaf";
  } else if (t === "Identifier") {
    badge = "ID";
    name = String(node.name || "");
    category = "cat-leaf";
  } else {
    badge = t.slice(0, 6).toUpperCase();
    name = t;
    category = "cat-stmt";
    for (const key of Object.keys(node)) {
      if (key !== "type" && typeof node[key] === "object" && node[key] !== null) {
        if (Array.isArray(node[key])) {
          node[key].forEach((item, i) => {
            if (item && typeof item === "object") children.push(buildAstTreeNode(item, `${key}[${i + 1}]`));
          });
        } else {
          children.push(buildAstTreeNode(node[key], key));
        }
      }
    }
  }

  return {
    type: t,
    badge,
    name,
    sub,
    category,
    edgeLabel,
    children: children.filter(Boolean)
  };
}

function layoutTree(root) {
  const NODE_W = 144;
  const GAP_X = 26;
  const GAP_Y = 96;

  function calcSubtreeWidth(node) {
    if (!node.children || node.children.length === 0) {
      node.subtreeWidth = NODE_W;
      return;
    }
    let total = 0;
    node.children.forEach(c => {
      calcSubtreeWidth(c);
      total += c.subtreeWidth;
    });
    total += (node.children.length - 1) * GAP_X;
    node.subtreeWidth = Math.max(NODE_W, total);
  }
  calcSubtreeWidth(root);

  function assignCoords(node, leftX, depth) {
    node.depth = depth;
    node.y = depth * GAP_Y + 40;
    if (!node.children || node.children.length === 0) {
      node.x = leftX + (node.subtreeWidth - NODE_W) / 2;
      return;
    }
    let curX = leftX;
    node.children.forEach(c => {
      assignCoords(c, curX, depth + 1);
      curX += c.subtreeWidth + GAP_X;
    });
    const firstChildX = node.children[0].x;
    const lastChildX = node.children[node.children.length - 1].x;
    node.x = (firstChildX + lastChildX) / 2;
  }
  assignCoords(root, 40, 0);
}

function getTreeBoundsAndStats(root) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  let nodeCount = 0, maxDepth = 0;

  function traverse(n) {
    nodeCount++;
    if (n.depth > maxDepth) maxDepth = n.depth;
    minX = Math.min(minX, n.x);
    maxX = Math.max(maxX, n.x + 144);
    minY = Math.min(minY, n.y);
    maxY = Math.max(maxY, n.y + 50);
    (n.children || []).forEach(traverse);
  }
  traverse(root);

  return {
    minX, maxX, minY, maxY,
    width: maxX - minX,
    height: maxY - minY,
    nodeCount,
    maxDepth: maxDepth + 1
  };
}

function renderTreeLinks(node, svgLinksArr) {
  const NODE_W = 144;
  const NODE_H = 50;
  const px = node.x + NODE_W / 2;
  const py = node.y + NODE_H;

  (node.children || []).forEach(child => {
    const cx = child.x + NODE_W / 2;
    const cy = child.y;
    const midY = (py + cy) / 2;
    const d = `M ${px} ${py} C ${px} ${midY}, ${cx} ${midY}, ${cx} ${cy}`;

    let labelHtml = "";
    if (child.edgeLabel) {
      const lx = (px + cx) / 2;
      const ly = midY;
      const lw = Math.max(34, child.edgeLabel.length * 6.5 + 10);
      labelHtml = `
        <g class="ast-edge-group">
          <rect class="ast-edge-rect" x="${lx - lw/2}" y="${ly - 9}" width="${lw}" height="18" rx="4" ry="4"></rect>
          <text class="ast-edge-text" x="${lx}" y="${ly}">${esc(child.edgeLabel)}</text>
        </g>
      `;
    }

    svgLinksArr.push(`
      <g class="ast-link-group">
        <path class="ast-tree-link" d="${d}" />
        ${labelHtml}
      </g>
    `);

    renderTreeLinks(child, svgLinksArr);
  });
}

function renderNodeCards(node, svgNodesArr) {
  const NODE_W = 144;
  const NODE_H = 50;
  const x = node.x;
  const y = node.y;

  let displayName = node.name;
  if (displayName.length > 15) {
    displayName = displayName.slice(0, 14) + "…";
  }

  const badgeText = node.badge || "NODE";
  const badgeW = Math.max(34, badgeText.length * 6.5 + 10);

  svgNodesArr.push(`
    <g class="ast-node-group ${node.category}" transform="translate(${x}, ${y})" data-title="${esc(node.name)}" data-type="${esc(node.type)}">
      <rect class="ast-node-card-bg" x="0" y="0" width="${NODE_W}" height="${NODE_H}" rx="7" ry="7"></rect>
      <g transform="translate(10, 8)">
        <rect class="ast-badge-bg" x="0" y="0" width="${badgeW}" height="15" rx="3.5" ry="3.5"></rect>
        <text class="ast-badge-text" x="${badgeW / 2}" y="8" text-anchor="middle">${esc(badgeText)}</text>
      </g>
      ${node.sub ? `<text class="ast-node-sub" x="${NODE_W - 10}" y="15" text-anchor="end">${esc(node.sub)}</text>` : ''}
      <text class="ast-node-title" x="10" y="34">${esc(displayName)}</text>
      <title>${esc(node.type)}: ${esc(node.name)}</title>
    </g>
  `);

  (node.children || []).forEach(child => renderNodeCards(child, svgNodesArr));
}

function renderAST(ast) {
  currentAstRaw = ast;
  const jsonEl = document.getElementById("astOutput");
  const emptyEl = document.getElementById("astEmptyState");
  const viewport = document.getElementById("astTreeViewport");
  const chip = document.getElementById("astDepthChip");

  if (!ast) {
    if (jsonEl) jsonEl.textContent = "No AST generated.";
    if (emptyEl) emptyEl.classList.remove("hidden");
    if (viewport) viewport.innerHTML = "";
    if (chip) chip.textContent = "0 Nodes";
    astTreeBounds = null;
    return "No AST generated.";
  }

  // 1. Format raw JSON for the JSON view tab
  let str = esc(JSON.stringify(ast, null, 2));
  str = str.replace(/"type":\s*&quot;([^&]+)&quot;/g,
    '"type": &quot;<span class="ast-node">$1</span>&quot;');
  if (jsonEl) jsonEl.innerHTML = str;

  // 2. Hide empty state placeholder
  if (emptyEl) emptyEl.classList.add("hidden");

  // 3. Build tree and run layout
  const treeRoot = buildAstTreeNode(ast);
  if (!treeRoot) return str;

  layoutTree(treeRoot);

  const stats = getTreeBoundsAndStats(treeRoot);
  astTreeBounds = stats;

  if (chip) {
    chip.textContent = `${stats.nodeCount} Nodes • Depth ${stats.maxDepth}`;
  }

  // 4. Render SVG links and node cards
  const linksArr = [];
  const nodesArr = [];
  renderTreeLinks(treeRoot, linksArr);
  renderNodeCards(treeRoot, nodesArr);

  if (viewport) {
    viewport.innerHTML = `
      <g class="ast-links-layer">${linksArr.join("")}</g>
      <g class="ast-nodes-layer">${nodesArr.join("")}</g>
    `;
  }

  // 5. Fit to screen automatically
  requestAnimationFrame(() => {
    fitAstTreeToScreen();
  });

  return str;
}

function applyAstTransform(animate = false) {
  const viewport = document.getElementById("astTreeViewport");
  const zoomText = document.getElementById("astZoomLevelText");
  if (viewport) {
    viewport.style.transition = animate ? "transform 0.25s cubic-bezier(0.16, 1, 0.3, 1)" : "none";
    viewport.setAttribute("transform", `translate(${astPanX}, ${astPanY}) scale(${astZoom})`);
  }
  if (zoomText) {
    zoomText.textContent = `${Math.round(astZoom * 100)}%`;
  }
}

function zoomAst(factor, clientX, clientY) {
  const wrapper = document.getElementById("astTreeWrapper");
  if (!wrapper) return;
  const rect = wrapper.getBoundingClientRect();
  const cx = clientX !== undefined ? clientX - rect.left : wrapper.clientWidth / 2;
  const cy = clientY !== undefined ? clientY - rect.top : wrapper.clientHeight / 2;

  const newZoom = Math.max(0.12, Math.min(3.5, astZoom * factor));
  if (newZoom === astZoom) return;

  // Zoom centered at (cx, cy)
  astPanX = cx - (cx - astPanX) * (newZoom / astZoom);
  astPanY = cy - (cy - astPanY) * (newZoom / astZoom);
  astZoom = newZoom;
  applyAstTransform();
}

function fitAstTreeToScreen() {
  const wrapper = document.getElementById("astTreeWrapper");
  if (!wrapper || !astTreeBounds || astTreeBounds.width === 0) return;

  const containerW = wrapper.clientWidth || 600;
  const containerH = wrapper.clientHeight || 460;
  const padding = 45;

  const scaleX = (containerW - padding * 2) / astTreeBounds.width;
  const scaleY = (containerH - padding * 2) / astTreeBounds.height;
  let targetScale = Math.min(scaleX, scaleY);

  targetScale = Math.max(0.15, Math.min(1.05, targetScale));

  astZoom = targetScale;
  astPanX = (containerW - astTreeBounds.width * astZoom) / 2 - astTreeBounds.minX * astZoom;
  astPanY = (containerH - astTreeBounds.height * astZoom) / 2 - astTreeBounds.minY * astZoom;

  applyAstTransform(true);
}

function resetAstZoom() {
  const wrapper = document.getElementById("astTreeWrapper");
  if (!wrapper) return;
  astZoom = 1.0;
  if (astTreeBounds) {
    astPanX = (wrapper.clientWidth - astTreeBounds.width) / 2 - astTreeBounds.minX;
    astPanY = 40 - astTreeBounds.minY;
  } else {
    astPanX = 0;
    astPanY = 0;
  }
  applyAstTransform(true);
}

function setAstViewMode(mode) {
  astViewMode = mode;
  const treeBtn = document.getElementById("astViewTreeBtn");
  const jsonBtn = document.getElementById("astViewJsonBtn");
  const treeWrapper = document.getElementById("astTreeWrapper");
  const jsonOutput = document.getElementById("astOutput");
  const zoomControls = document.getElementById("astZoomControls");

  if (mode === "json") {
    jsonBtn?.classList.add("active");
    treeBtn?.classList.remove("active");
    if (treeWrapper) treeWrapper.style.display = "none";
    if (jsonOutput) jsonOutput.style.display = "block";
    if (zoomControls) zoomControls.style.display = "none";
  } else {
    treeBtn?.classList.add("active");
    jsonBtn?.classList.remove("active");
    if (treeWrapper) treeWrapper.style.display = "block";
    if (jsonOutput) jsonOutput.style.display = "none";
    if (zoomControls) zoomControls.style.display = "inline-flex";
    fitAstTreeToScreen();
  }
}

function bindAstTreeEvents() {
  const wrapper = document.getElementById("astTreeWrapper");
  if (!wrapper) return;

  // Mouse wheel zoom centered at cursor
  wrapper.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    zoomAst(factor, e.clientX, e.clientY);
  }, { passive: false });

  // Mouse drag panning
  wrapper.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    isAstPanning = true;
    wrapper.classList.add("is-panning");
    astStartPan = { x: e.clientX - astPanX, y: e.clientY - astPanY };
  });

  window.addEventListener("mousemove", (e) => {
    if (!isAstPanning) return;
    astPanX = e.clientX - astStartPan.x;
    astPanY = e.clientY - astStartPan.y;
    applyAstTransform();
  });

  window.addEventListener("mouseup", () => {
    if (isAstPanning) {
      isAstPanning = false;
      wrapper.classList.remove("is-panning");
    }
  });

  // Double click resets to fit
  wrapper.addEventListener("dblclick", () => {
    fitAstTreeToScreen();
  });

  // Touch panning
  let touchStart = { x: 0, y: 0 };
  let touchStartPan = { x: 0, y: 0 };
  wrapper.addEventListener("touchstart", (e) => {
    if (e.touches.length === 1) {
      touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      touchStartPan = { x: astPanX, y: astPanY };
      isAstPanning = true;
      wrapper.classList.add("is-panning");
    }
  }, { passive: true });

  wrapper.addEventListener("touchmove", (e) => {
    if (isAstPanning && e.touches.length === 1) {
      astPanX = touchStartPan.x + (e.touches[0].clientX - touchStart.x);
      astPanY = touchStartPan.y + (e.touches[0].clientY - touchStart.y);
      applyAstTransform();
    }
  }, { passive: true });

  wrapper.addEventListener("touchend", () => {
    isAstPanning = false;
    wrapper.classList.remove("is-panning");
  });

  // Stage 2 Toolbar Buttons
  document.getElementById("astZoomInBtn")?.addEventListener("click", () => zoomAst(1.25));
  document.getElementById("astZoomOutBtn")?.addEventListener("click", () => zoomAst(0.8));
  document.getElementById("astZoomResetBtn")?.addEventListener("click", resetAstZoom);
  document.getElementById("astZoomFitBtn")?.addEventListener("click", fitAstTreeToScreen);

  // View Mode Toggle (Tree View vs JSON)
  document.getElementById("astViewTreeBtn")?.addEventListener("click", () => setAstViewMode("tree"));
  document.getElementById("astViewJsonBtn")?.addEventListener("click", () => setAstViewMode("json"));
}

// Token-based TAC syntax highlighter — avoids corrupting injected HTML tags
function renderTAC(code) {
  if (!code?.length) return "No IR generated.";

  const keywords = new Set(["GOTO", "IF_FALSE", "PRINT", "SCAN", "ASSIGN", "LABEL"]);

  return code.map((rawLine, idx) => {
    const lineNum = `<span class="tac-lineno">${String(idx + 1).padStart(3, '0')}</span> `;
    const line = esc(rawLine);

    // If it's a label instruction line e.g. "L1:"
    if (/^[A-Za-z0-9_]+:$/.test(line.trim())) {
      return lineNum + `<span class="tok-label">${line}</span>`;
    }

    // Tokenize line cleanly by whitespace, operators, and delimiters
    const parts = line.split(/(\s+|[=+\-*/><!&|]+|[(),:])/);
    const formatted = parts.map(part => {
      if (!part || /^\s+$/.test(part)) return part;

      if (keywords.has(part)) {
        return `<span class="tok-tac">${part}</span>`;
      }
      if (/^-?\d+(\.\d+)?$/.test(part)) {
        return `<span class="tok-num">${part}</span>`;
      }
      if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(part)) {
        if (/^t\d+$/.test(part)) {
          return `<span class="tok-temp">${part}</span>`;
        }
        if (/^L\d+$/.test(part)) {
          return `<span class="tok-label">${part}</span>`;
        }
        return `<span class="tok-id">${part}</span>`;
      }
      if (/^[=+\-*/><!&|]+$/.test(part)) {
        return `<span class="tok-op">${part}</span>`;
      }
      return part;
    }).join("");

    return lineNum + formatted;
  }).join("\n");
}

function renderOptimisation(code, log, origCode) {
  const logEl  = document.getElementById("optimizationLog");
  const codeEl = document.getElementById("optimizedOutput");
  const origLen = origCode?.length || 0;
  const optLen  = code?.length || 0;
  const actionsCount = log?.length || 0;
  const reduction = origLen > 0 ? (((origLen - optLen) / origLen) * 100).toFixed(1) : "0.0";

  document.getElementById("optMetricOrig").textContent = String(origLen);
  document.getElementById("optMetricOpt").textContent = String(optLen);
  document.getElementById("optMetricRed").textContent = `${reduction}%`;
  document.getElementById("optMetricCount").textContent = String(actionsCount);
  document.getElementById("optActionsChip").textContent = `${actionsCount} Agent Actions`;

  if (log?.length) {
    const formattedLog = log.map((entry, idx) => {
      return `[Pass #${String(idx + 1).padStart(2, '0')}] ${esc(entry)}`;
    }).join("\n");
    logEl.textContent = formattedLog;
  } else {
    logEl.textContent = "✓ No optimisations needed (intermediate code already optimal).";
  }

  codeEl.innerHTML = code?.length ? renderTAC(code) : "No optimised IR.";
}

function renderExecution(output, errors, success, execTime) {
  const outEl = document.getElementById("outputResult");
  const errSec = document.getElementById("errorSection");
  const errEl  = document.getElementById("errorResult");
  const exitTag = document.getElementById("terminalExitTag");
  const timeChip = document.getElementById("executionTimeChip");
  const statusChip = document.getElementById("executionStatusChip");

  const formattedTime = execTime != null ? `${(execTime * 1000).toFixed(1)} ms` : "0.0 ms";
  if (timeChip) timeChip.textContent = formattedTime;
  if (statusChip) statusChip.textContent = success ? "Success (0)" : "Error";

  if (exitTag) {
    exitTag.textContent = success ? "EXIT 0" : "EXIT 1";
    exitTag.classList.toggle("error", !success);
  }

  outEl.textContent = output?.trim() || "(Process finished with no standard output)";

  if (errors?.trim()) {
    errSec.style.display = "block";
    errEl.textContent = errors.trim();
  } else {
    errSec.style.display = "none";
  }
}

// ── Clear all stage outputs ───────────────────────────────────────
function clearStages() {
  const tokensBody = document.getElementById("tokensTableBody");
  if (tokensBody) {
    tokensBody.innerHTML = `<tr><td colspan="4" class="empty-state">Run pipeline to inspect lexical token stream…</td></tr>`;
  }
  document.getElementById("astOutput").textContent = "Run pipeline to view parsed AST hierarchy…";
  const emptyState = document.getElementById("astEmptyState");
  if (emptyState) emptyState.classList.remove("hidden");
  const astViewport = document.getElementById("astTreeViewport");
  if (astViewport) astViewport.innerHTML = "";
  const zoomText = document.getElementById("astZoomLevelText");
  if (zoomText) zoomText.textContent = "100%";
  const depthChip = document.getElementById("astDepthChip");
  if (depthChip) depthChip.textContent = "Hierarchical Syntax";
  currentAstRaw = null;
  astTreeBounds = null;

  document.getElementById("semanticOutput").textContent = "Run pipeline to run semantic validation…";
  document.getElementById("intermediateOutput").textContent = "Run pipeline to generate Three-Address Code…";
  document.getElementById("optimizationLog").textContent = "Run pipeline to inspect agent policy actions…";
  document.getElementById("optimizedOutput").textContent = "Optimized IR will display here…";
  document.getElementById("pythonOutput").textContent = "Run pipeline to generate executable Python…";
  document.getElementById("outputResult").textContent = "Ready. Press 'Run' (Ctrl+Enter) to compile and execute.";

  for (let i = 1; i <= 7; i++) {
    const card = document.getElementById(`stepCard-${i}`);
    const state = document.getElementById(`stepState-${i}`);
    const metric = document.getElementById(`stepMetric-${i}`);
    if (card) card.classList.remove("running", "success", "error");
    if (state) state.textContent = "IDLE";
    if (metric) metric.textContent = "—";
  }

  const dot = document.getElementById("statusPulseDot");
  const statusText = document.getElementById("pipelineStatusText");
  if (dot) dot.className = "status-pulse-dot";
  if (statusText) statusText.textContent = "Ready";

  const errSection = document.getElementById("errorSection");
  if (errSection) errSection.style.display = "none";

  setActiveStage(7);
}

// ── Comment toggle (Ctrl+/) ───────────────────────────────────────
function toggleComment() {
  if (!editor) return;
  const cursor = editor.getCursor();
  const line   = editor.getLine(cursor.line);
  if (line.trim().startsWith("//")) {
    editor.replaceRange(line.replace(/^\s*\/\/\s?/, ""),
      { line: cursor.line, ch: 0 }, { line: cursor.line, ch: line.length });
  } else {
    const indent = (line.match(/^(\s*)/) || ["",""])[1];
    editor.replaceRange(`${indent}// ${line.trim()}`,
      { line: cursor.line, ch: 0 }, { line: cursor.line, ch: line.length });
  }
}
