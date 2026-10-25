# RL-Optimized Source-to-Source Compiler

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Architecture: 7--Stage%20Pipeline](https://img.shields.io/badge/Architecture-7--Stage%20Pipeline-emerald.svg)](#compiler-architecture)
[![Optimization: Q--Learning%20MDP](https://img.shields.io/badge/Optimization-Q--Learning%20MDP-purple.svg)](#reinforcement-learning-optimization-engine)

A self-optimizing imperative compiler framework that compiles a custom procedural programming language into optimized Python 3 source code. The compilation pipeline integrates an autonomous reinforcement learning agent (tabular Q-Learning) operating over Three-Address Code (TAC) intermediate representations to dynamically explore, discover, and apply optimal transformation sequences.

---

## Table of Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [Compiler Architecture](#compiler-architecture)
- [Reinforcement Learning Optimization Engine](#reinforcement-learning-optimization-engine)
  - [MDP Formulation](#mdp-formulation)
  - [Action Space](#action-space)
  - [Reward Formulation](#reward-formulation)
  - [Q-Value Update Rule](#q-value-update-rule)
- [Language Specification](#language-specification)
  - [Data Types](#data-types)
  - [Reserved Keywords](#reserved-keywords)
  - [Operator Precedence and Semantics](#operator-precedence-and-semantics)
  - [Delimiters and Punctuation](#delimiters-and-punctuation)
  - [Token Classification Table](#token-classification-table)
- [Security and Runtime Sandboxing](#security-and-runtime-sandboxing)
- [Code Example and Transformation Walkthrough](#code-example-and-transformation-walkthrough)
- [Installation and Local Setup](#installation-and-local-setup)
- [Automated Testing and Verification](#automated-testing-and-verification)
- [Deployment Architecture](#deployment-architecture)

---

## Overview

Traditional optimizing compilers rely on hand-tuned static heuristics and fixed phase orderings. While effective for common code structures, fixed optimization sequences often suffer from phase-ordering conflicts, where one transformation pass inadvertently obscures optimization opportunities for subsequent passes.

This project reframes code optimization as a **sequential decision-making problem** formulated as a **Markov Decision Process (MDP)**. A tabular Q-Learning agent observes local instruction contexts and variable liveness within the Three-Address Code intermediate representation, iteratively selecting actions such as constant folding, algebraic simplification, common subexpression elimination, dead code elimination, and loop-invariant code motion. The agent receives positive reinforcement for code reductions and folded operations, learning adaptive optimization strategies tailored to the program structure.

---

## Key Capabilities

- **Autonomous Transformation Sequencing:** Tabular Q-learning eliminates rigid, hardcoded optimization sequences by selecting transformation actions based on contextual state fingerprints.
- **Full Source-to-Source Pipeline:** Complete implementation covering lexical analysis, recursive descent parsing, scoped semantic analysis, intermediate representation generation, reinforcement learning passes, target Python decompilation, and sandboxed execution.
- **Defensive Execution Sandboxing:** Sandboxed execution environment employing daemon thread watchdog timers, memory isolation, non-zero arithmetic guards, and strict Python builtins whitelisting.
- **Interactive Developer Dashboard:** High-performance web interface built with vanilla HTML5, CSS3, and JavaScript, featuring real-time interactive AST pan/zoom visualization, step-by-step pipeline execution inspection, and live intermediate code diffing.

---

## Compiler Architecture

The compilation pipeline processes source code through seven distinct, decoupled phases:

```
+---------------------------------------------------------------------------------+
|                                COMPILATION PIPELINE                             |
+---------------------------------------------------------------------------------+
|                                                                                 |
|   Source Code (.src)                                                            |
|          |                                                                      |
|          v                                                                      |
|   [Stage 01] Lexical Analysis             --> Token Stream (32 Token Types)     |
|          |                                                                      |
|          v                                                                      |
|   [Stage 02] Syntax Parsing               --> Abstract Syntax Tree (AST)        |
|          |                                                                      |
|          v                                                                      |
|   [Stage 03] Semantic Analysis            --> Scoped Symbol Table Verification  |
|          |                                                                      |
|          v                                                                      |
|   [Stage 04] Intermediate Representation  --> Linear Three-Address Code (TAC)   |
|          |                                                                      |
|          v                                                                      |
|   [Stage 05] RL Optimization Engine       --> Optimized TAC (Q-Learning)        |
|          |                                                                      |
|          v                                                                      |
|   [Stage 06] Target Code Generation       --> Idiomatic Python 3 Source         |
|          |                                                                      |
|          v                                                                      |
|   [Stage 07] Sandboxed Execution          --> Verified Output Buffer & Metrics  |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

### Pipeline Stages

1. **Stage 01: Lexical Analysis (`api/lexer.py`)**  
   Deterministic, coordinate-tracked character scanner converting UTF-8 source into classified tokens. Handles numeric literals (integers and floats), string literals (supporting single and double quotes), identifiers, delimiters, comments (`//`), and stream markers.
2. **Stage 02: Syntax Parsing (`api/parser.py`)**  
   Recursive descent parser with precedence climbing enforcing the formal LL(1) grammar. Constructs a strongly typed, hierarchical Abstract Syntax Tree (AST).
3. **Stage 03: Semantic Analysis (`api/semantic_analyzer.py`)**  
   Validates lexical scoping hierarchies, enforces declaration-before-use invariants, verifies compound assignment targets, and executes static compile-time division and modulo by zero detection.
4. **Stage 04: Intermediate Code Generation (`api/intermediate_code.py`)**  
   Translates AST nodes into linear Three-Address Code quadruples (`operation`, `arg1`, `arg2`, `result`) using synthetic temporaries (`t1`, `t2`, ...) and branch targets (`L1`, `L2`, ...).
5. **Stage 05: RL Optimization Engine (`api/rl_optimizer.py`)**  
   Tabular Q-learning agent executing optimization passes over the TAC stream, using context fingerprints and variable live-range analysis to select reduction actions.
6. **Stage 06: Target Code Generation (`api/code_generator.py`)**  
   Decompiles the optimized Three-Address Code into readable, structured Python 3 code with safe arithmetic guards for zero denominators and control-flow translation.
7. **Stage 07: Sandboxed Execution (`api/executor.py`)**  
   Executes the emitted target Python code inside an isolated daemon thread equipped with a 5.0-second timeout, input feeding, and captured stdout/stderr buffers.

---

## Reinforcement Learning Optimization Engine

### MDP Formulation

The program optimization problem is modeled as a Markov Decision Process:

- **State Space ($S$):** The state represents a localized context window of position $\pm 1$ surrounding the active instruction, combined with the live-range status of the target register:
  $$\text{State}(i) = \langle \text{op}_{i-1},\; \text{op}_i,\; \text{op}_{i+1},\; \text{is\_live}(\text{result}_i) \rangle$$
- **Action Space ($A$):** Transformation operations valid for the instruction pattern at position $i$.
- **Transition Function ($T$):** Deterministic code rewrites transforming the TAC instruction list $C \to C'$.

### Action Space

| Action Name | Description | Preconditions |
|:---|:---|:---|
| `constant_folding` | Evaluates arithmetic operations between known constants at compile time. | Operation in `{+, -, *, /, %}`; both operands are numeric constants. |
| `dead_code_elimination` | Removes dead stores to variables that are overwritten or never read downstream. | `result` is not read subsequently within the current scope or basic block. |
| `common_subexpression` | Replaces redundant computations with previously computed temporary registers. | Identical operation and operands exist earlier within the same basic block without intervening writes. |
| `algebraic_simplification` | Applies algebraic identities (`x + 0 -> x`, `x * 1 -> x`, `x * 0 -> 0`, `x % 1 -> 0`, `x % x -> 0`, `x * 2 -> x + x`). | Operation matches identity signatures. |
| `loop_invariant_motion` | Hoists loop-independent computations outside the loop preheader. | Instruction resides within a loop; operands are not modified within the loop body; no loop-carried dependency. |

### Reward Formulation

The reward function incentivizes code size reduction and constant evaluation while heavily penalizing code growth:

$$\text{Reward} = (+10 \times \Delta_{\text{removed}}) + (+5 \times \Delta_{\text{folded}}) - (20 \times \Delta_{\text{growth}})$$

Where:
- $\Delta_{\text{removed}}$ = Number of eliminated Three-Address Code instructions.
- $\Delta_{\text{folded}}$ = Number of precomputed constant operations.
- $\Delta_{\text{growth}}$ = Increase in instruction count (if any).

### Q-Value Update Rule

State-action values are updated using the standard temporal difference formulation:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

- Learning Rate ($\alpha$): $0.1$
- Discount Factor ($\gamma$): $0.9$
- Exploration Strategy: $\epsilon$-greedy exploration ($\epsilon = 0.1$) with exponential decay factor ($\times 0.99$ per iteration).

---

## Language Specification

The compiler targets a clean, procedural, statically validated language.

### Data Types

| Data Type | Definition | Formatting / Target Mapping |
|:---|:---|:---|
| **`Number`** | 64-bit IEEE 754 floating-point numbers. Supports integer (`42`, `-10`) and fractional forms (`3.14159`). | Formatted cleanly on output (whole numbers emitted without trailing decimals, e.g., `10` instead of `10.0`). |
| **`String`** | Immutable UTF-8 text sequences enclosed in single (`'...'`) or double (`"..."`) quotes. | Python `str` mapping; escape sequences preserved (`\n`, `\t`, `\"`, `\'`, `\\`). |

*Boolean Logic:* Booleans are represented numerically: `1` denotes True and `0` denotes False. Conditional statements evaluate truthiness using non-zero value semantics.

### Reserved Keywords

The language defines five reserved lowercase keywords:

| Keyword | Category | Syntax | Semantic Description |
|:---|:---|:---|:---|
| `print` | I/O | `print(expression);` | Evaluates expression and writes string representation with a trailing newline to stdout. |
| `scan` | I/O | `scan(identifier);` | Reads next line from standard input, parses as a floating-point value, and defines or updates variable. |
| `while` | Control Flow | `while (cond) { ... }` | Evaluates loop condition before each iteration; executes block while condition evaluates non-zero. |
| `if` | Control Flow | `if (cond) { ... }` | Evaluates condition; executes then-block if condition evaluates non-zero. |
| `else` | Control Flow | `else { ... }` | Optional alternate branch paired with an `if` construct; executes when condition evaluates to zero. |

### Operator Precedence and Semantics

Operators are listed in order of binding strength from lowest to highest:

| Precedence | Category | Operators | Associativity | Semantics & Behavior |
|:---:|:---|:---|:---:|:---|
| 1 (Lowest) | Assignment | `=`, `+=`, `-=` | Right-to-left | Direct assignment and compound accumulation. Variable must be defined prior to `+=` or `-=`. |
| 2 | Logical OR | `\|\|` | Left-to-right | Short-circuit logical disjunction. Evaluates to `1` or `0`. |
| 3 | Logical AND | `&&` | Left-to-right | Short-circuit logical conjunction. Evaluates to `1` or `0`. |
| 4 | Equality | `==`, `!=` | Left-to-right | Structural value comparison. Evaluates to `1` or `0`. |
| 5 | Relational | `<`, `<=`, `>`, `>=` | Left-to-right | Scalar numerical ordering comparison. Evaluates to `1` or `0`. |
| 6 | Additive | `+`, `-` | Left-to-right | Binary arithmetic addition and subtraction. |
| 7 | Multiplicative | `*`, `/`, `%` | Left-to-right | Multiplication, guarded division, and guarded modulo remainder. |
| 8 (Highest) | Unary | `!`, `-`, `+` | Right-to-left | Logical negation, arithmetic negation, identity positive. |

*Zero-Denominator Protection:* Both division (`/`) and modulo (`%`) operations are automatically wrapped in safe runtime conditionals:
- Division: `(a / b if b != 0 else 0)`
- Modulo: `(a % b if b != 0 else 0)`

### Delimiters and Punctuation

| Symbol | Name | Usage Description |
|:---:|:---|:---|
| `(` | `LPAREN` | Opens grouped expressions, condition envelopes, and statement argument lists. |
| `)` | `RPAREN` | Closes grouped expressions, condition envelopes, and statement argument lists. |
| `{` | `LBRACE` | Opens an imperative statement block scope. |
| `}` | `RBRACE` | Closes an imperative statement block scope. |
| `;` | `SEMICOLON` | Mandatory terminator for assignment, `print`, and `scan` statements. |

### Token Classification Table

The lexical analyzer classifies source input into 32 distinct token types organized across 8 categories:

| Category | Token Count | Enumeration Variants |
|:---|:---:|:---|
| **Literals** | 3 | `NUMBER`, `STRING`, `IDENTIFIER` |
| **Keywords** | 5 | `PRINT`, `SCAN`, `WHILE`, `IF`, `ELSE` |
| **Arithmetic Operators** | 5 | `PLUS`, `MINUS`, `MULTIPLY`, `DIVIDE`, `MODULO` |
| **Assignment Operators** | 3 | `ASSIGN`, `PLUS_ASSIGN`, `MINUS_ASSIGN` |
| **Comparison Operators** | 6 | `EQUAL`, `NOT_EQUAL`, `GREATER`, `LESS`, `GREATER_EQUAL`, `LESS_EQUAL` |
| **Logical Operators** | 3 | `AND`, `OR`, `NOT` |
| **Delimiters** | 5 | `LPAREN`, `RPAREN`, `LBRACE`, `RBRACE`, `SEMICOLON` |
| **Stream Control** | 2 | `NEWLINE`, `EOF` |

---

## Security and Runtime Sandboxing

The execution engine runs target Python code inside an isolated environment designed to prevent denial-of-service and unhandled crash conditions:

1. **Daemon Thread Execution Isolation:** Target scripts execute on a dedicated daemon thread. If execution exceeds the 5.0-second watchdog limit (e.g., infinite loop), the thread is abandoned and a clean timeout error is returned.
2. **Whitelisted Builtins Environment:** The global execution scope restricts available functions to a hardened whitelist (`len`, `range`, `min`, `max`, `abs`, etc.). Destructive namespaces including `os`, `sys`, `subprocess`, `open`, `eval`, and `__import__` are stripped.
3. **Division and Modulo Safety:** Denominators evaluated as zero produce `0` rather than triggering uncaught `ZeroDivisionError` exceptions.
4. **Recursion Depth Protection:** Recursion limits are raised locally during Abstract Syntax Tree traversal and restored upon exit to prevent process-level stack overflows.

---

## Code Example and Transformation Walkthrough

### Source Input

```c
// Variable initialization & arithmetic
a = 10;
b = 20;
c = 2 + 3;
d = a + b;
e = a + b;       // Common subexpression candidate
f = c * 1;       // Algebraic identity candidate
mod = 17 % 5;    // Constant folding candidate
unused = 999;    // Dead store candidate

// Loop with invariant computation
i = 0;
while (i < 3) {
    result = a * b; // Invariant computation candidate
    print(result);
    i += 1;
}

print(d);
print(f);
print(mod);
```

### Compiler Transformations Applied

1. **Constant Folding:** `c = 2 + 3` folds to `5`; `mod = 17 % 5` folds to `2`.
2. **Algebraic Simplification:** `f = c * 1` rewrites to direct assignment `f = c`.
3. **Common Subexpression Elimination:** `e = a + b` reuses the register previously computed for `d`.
4. **Loop-Invariant Code Motion:** `result = a * b` is hoisted outside the `while` loop preheader.
5. **Dead Code Elimination:** `unused = 999` is removed as the register is never referenced.

### Target Emitted Python

```python
# Generated Python code (from RL-Optimized TAC)
def custom_input():
    global input_index
    if input_index < len(input_values):
        val = input_values[input_index]
        input_index += 1
        return val
    return '0'

def custom_print(value):
    output_buffer.write(str(value) + '\n')

# --- Main execution ---
a = 10
b = 20
c = 5
d = (a + b)
e = d
f = c
mod = 2
result = (a * b)

i = 0
while (i < 3):
    custom_print(result)
    i = (i + 1)

custom_print(d)
custom_print(f)
custom_print(mod)

result = output_buffer.getvalue()
```

---

## Installation and Local Setup

### Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Clone and Install

```bash
git clone https://github.com/BhavyaShah7409/RL-Optimised-Compiler.git
cd RL-Optimised-Compiler

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will start locally at **`http://localhost:5000`**, serving both the REST API and the interactive browser frontend.

---

## Automated Testing and Verification

The repository includes a comprehensive regression suite and an adversarial fuzzer:

```bash
# 1. Execute canonical test suites (Fibonacci, Control Flow, Dead Code, Modulo)
python -c "from api.compiler import SourceToSourceCompiler; c = SourceToSourceCompiler(); print(c.compile_and_execute('x = 10 % 3; print(x);')['output'])"

# 2. Run adversarial property-based fuzzer (30 random iterations)
python adversarial_tester.py 30
```

---

## Deployment Architecture

- **Backend API (Render):** Production deployment managed through `render.yaml` running `gunicorn app:app`.
  - *Cold Start Notice:* The free-tier instance on Render spins down after inactivity. Initial requests may take approximately 15 seconds to wake up the service. The web frontend includes an interactive **API Live / Wake API** header button and background health check to wake the container proactively.
- **Frontend (Vercel):** Static edge delivery managed via `vercel.json` serving HTML, CSS, and client-side JavaScript.

---

## License

This project is licensed under the [MIT License](LICENSE).
