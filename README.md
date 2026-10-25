# 🧠 RL-Optimized Source-to-Source Compiler ⚡

An intelligent, end-to-end source-to-source compiler translating a custom C-like procedural language into highly optimized Python 3 code, leveraging Reinforcement Learning (tabular Q-Learning) to adaptively optimize intermediate representations.

Unlike traditional compilers that rely on static heuristics with fixed phase ordering, this system models program optimization as a Markov Decision Process (MDP) over Three-Address Code (TAC), dynamically selecting and composing code transformations to minimize instruction count and runtime overhead.

---

## 🚀 Key Highlights

- 🔁 **Self-Optimizing RL Engine:** Tabular Q-Learning agent evaluates instruction contexts to select optimal transformation actions.
- ⚡ **Multi-Pass Transformations:** Constant folding, dead code elimination, common subexpression elimination, algebraic simplification, and loop-invariant code motion.
- 🧩 **Complete 7-Stage Pipeline:** Lexical Analysis → Syntax Parsing (AST) → Semantic Analysis → Three-Address Code (TAC) → RL Optimization → Python Decompilation → Sandboxed Execution.
- 🛡️ **Defensive Runtime Sandbox:** Thread-isolated execution environment with restricted builtins, zero-division guards, recursion limits, and a strict 5.0-second watchdog timer.
- 🌐 **Interactive Web IDE:** Clean Vanilla JS + CSS dashboard featuring live AST pan/zoom visualization, step-by-step pipeline animations, TAC inspectors, and detailed language documentation.

---

## 📖 Language Specification & Inventory

A detailed breakdown of all language constructs implemented across the compiler:

### 1. Data Types (2 Supported Types)
| Data Type | Description | Examples | Representation |
|:---|:---|:---|:---|
| **`Number`** | 64-bit IEEE 754 floating-point numbers; integers and decimals are unified. Clean integer formatting (`10` vs `10.5`) on output. | `0`, `42`, `-15`, `3.14159`, `0.001` | Python `float` / `int` |
| **`String`** | Immutable UTF-8 text sequences enclosed in double (`"..."`) or single (`'...'`) quotes. | `"Hello World"`, `'Total: '` | Python `str` |

> *Note:* Boolean truths are represented numerically: `1` (True) and `0` (False) in relational evaluations and conditional branches.

---

### 2. Keywords (5 Built-in Keywords)
All keywords are lowercase, reserved identifiers:
| Keyword | Category | Usage & Syntax | Description |
|:---|:---|:---|:---|
| **`print`** | I/O | `print(expr);` | Evaluates `expr` and writes the string representation followed by a newline to stdout. |
| **`scan`** | I/O | `scan(var);` | Consumes the next input line from stdin, parses it as a float, and defines/updates `var`. |
| **`while`** | Control Flow | `while (cond) { ... }` | Repeatedly executes the enclosed block as long as `cond` evaluates truthy (`!= 0`). |
| **`if`** | Control Flow | `if (cond) { ... }` | Executes the then-block if `cond` evaluates truthy (`!= 0`). |
| **`else`** | Control Flow | `else { ... }` | Optional alternate branch paired with an `if` block when `cond` evaluates falsy (`== 0`). |

---

### 3. Operators (17 Total Operators across 4 Categories)

#### A. Arithmetic Operators (5)
| Operator | Name | Arity | Example | Semantic Behavior |
|:---:|:---|:---:|:---|:---|
| `+` | Addition / Positive | Binary / Unary | `a + b`, `+a` | Numerical addition or identity unary positive. |
| `-` | Subtraction / Negation | Binary / Unary | `a - b`, `-a` | Numerical subtraction or arithmetic negation. |
| `*` | Multiplication | Binary | `a * b` | Numerical multiplication. |
| `/` | Safe Division | Binary | `a / b` | Floating-point division with zero-check protection (`(a / b if b != 0 else 0)`). |
| `%` | Safe Modulo | Binary | `a % b` | Modulo remainder with zero-check protection (`(a % b if b != 0 else 0)`). |

#### B. Relational & Comparison Operators (6)
| Operator | Name | Example | Semantic Behavior |
|:---:|:---|:---|:---|
| `==` | Equality | `a == b` | Yields `1` if operands are equal, else `0`. |
| `!=` | Inequality | `a != b` | Yields `1` if operands differ, else `0`. |
| `<` | Less Than | `a < b` | Yields `1` if left is strictly less than right. |
| `<=` | Less Than or Equal | `a <= b` | Yields `1` if left is less than or equal to right. |
| `>` | Greater Than | `a > b` | Yields `1` if left is strictly greater than right. |
| `>=` | Greater Than or Equal | `a >= b` | Yields `1` if left is greater than or equal to right. |

#### C. Logical Operators (3)
| Operator | Name | Example | Semantic Behavior |
|:---:|:---|:---|:---|
| `&&` | Logical AND | `a && b` | Yields `1` if both operands are truthy (`!= 0`), short-circuiting. |
| `\|\|` | Logical OR | `a \|\| b` | Yields `1` if either operand is truthy (`!= 0`), short-circuiting. |
| `!` | Logical NOT | `!a` | Unary negation; yields `1` if operand is `0`, else `0`. |

#### D. Assignment Operators (3)
| Operator | Name | Example | Semantic Behavior |
|:---:|:---|:---|:---|
| `=` | Simple Assignment | `x = 10;` | Assigns value of expression to `x`. Implicitly defines `x` if new. |
| `+=` | Addition Assignment | `x += 5;` | In-place addition `x = x + 5`. Variable `x` must be previously defined. |
| `-=` | Subtraction Assignment | `x -= 2;` | In-place subtraction `x = x - 2`. Variable `x` must be previously defined. |

---

### 4. Delimiters & Punctuation (5 Symbols)
| Delimiter | Token Type | Usage |
|:---:|:---|:---|
| `(` | `LPAREN` | Left parenthesis for condition envelopes, function calls, and expression grouping. |
| `)` | `RPAREN` | Right parenthesis closing grouped expressions and argument lists. |
| `{` | `LBRACE` | Left brace opening a block scope. |
| `}` | `RBRACE` | Right brace terminating a block scope. |
| `;` | `SEMICOLON` | Statement terminator (mandatory after assignments, `print()`, and `scan()`). |

---

### 5. Token Categories (8 Categories / 32 Unique Token Types)
The lexical analyzer ([`lexer.py`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/api/lexer.py)) identifies and classifies tokens into 32 distinct `TokenType` variants across 8 categories:

1. **Literals (3):** `NUMBER`, `STRING`, `IDENTIFIER`
2. **Keywords (5):** `PRINT`, `SCAN`, `WHILE`, `IF`, `ELSE`
3. **Arithmetic Operators (5):** `PLUS`, `MINUS`, `MULTIPLY`, `DIVIDE`, `MODULO`
4. **Assignment Operators (3):** `ASSIGN`, `PLUS_ASSIGN`, `MINUS_ASSIGN`
5. **Comparison Operators (6):** `EQUAL`, `NOT_EQUAL`, `GREATER`, `LESS`, `GREATER_EQUAL`, `LESS_EQUAL`
6. **Logical Operators (3):** `AND`, `OR`, `NOT`
7. **Delimiters (5):** `LPAREN`, `RPAREN`, `LBRACE`, `RBRACE`, `SEMICOLON`
8. **Stream Control (2):** `NEWLINE`, `EOF`

---

## 🏗️ 7-Stage Compiler Architecture

```
Source Code
    │
    ▼ [Stage 1: Lexical Analysis]
Token Stream (Lexer)
    │
    ▼ [Stage 2: Syntax Analysis]
Abstract Syntax Tree (LL(1) Recursive Descent Parser)
    │
    ▼ [Stage 3: Semantic Analysis]
Symbol Table & Scope Verification
    │
    ▼ [Stage 4: Intermediate Code Generation]
Three-Address Code (TAC Linear Quadruples)
    │
    ▼ [Stage 5: RL Optimization]
Optimized TAC (Q-Learning Agent)
    │
    ▼ [Stage 6: Target Code Generation]
Python 3 Source (TACToPythonGenerator)
    │
    ▼ [Stage 7: Sandboxed Execution]
Output Buffer & Runtime Verification
```

### Stage Details:
1. **Lexical Analysis ([`lexer.py`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/api/lexer.py)):** Single-pass deterministic tokenizer handling whitespace skipping, comment stripping (`//`), single/double quoted strings, decimal float recognition, and coordinate tracking (line and column).
2. **Syntax Analysis ([`parser.py`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/api/parser.py)):** LL(1) recursive descent parser implementing precedence climbing across logical OR → logical AND → equality → comparison → additive → multiplicative (`*`, `/`, `%`) → unary (`!`, `-`) → primary.
3. **Semantic Analysis ([`semantic_analyzer.py`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/api/semantic_analyzer.py)):** Enforces lexical scoping hierarchies, validates that variables are defined before reference, checks compound assignment targets, and performs static compile-time division/modulo by zero detection on constants.
4. **Intermediate Representation ([`intermediate_code.py`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/api/intermediate_code.py)):** Flattens the hierarchical AST into linear Three-Address Code (TAC) with synthetic temporaries (`t1`, `t2`, ...) and branch labels (`L1`, `L2`, ...).
5. **RL Optimization Engine ([`rl_optimizer.py`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/api/rl_optimizer.py)):** Tabular Q-learning agent evaluating local opcode context windows (position &plusmn; 1) and variable live ranges.
6. **Target Code Generation ([`code_generator.py`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/api/code_generator.py)):** Translates the optimized TAC back into executable Python 3 code with safe arithmetic wrappers and structured control flow.
7. **Sandboxed Execution ([`executor.py`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/api/executor.py)):** Runs the generated Python code inside an isolated daemon thread with a 5.0-second timeout, restricted `SAFE_BUILTINS` whitelist, and captured I/O streams.

---

## 🤖 RL Optimization Engine Specifications

The optimization engine models compiler pass selection as a Markov Decision Process (MDP):

- **State Representation:** Fingerprint of surrounding instruction opcodes within a window of position &plusmn; 1, paired with variable live-range status:
  `State = f"{prev_op}_{current_op}_{next_op}_live_{live_status}"`
- **Action Space:**
  - `constant_folding`: Precomputes arithmetic expressions (`+`, `-`, `*`, `/`, `%`) between known constants.
  - `dead_code_elimination`: Removes assignments to variables that are never read subsequently or overwritten before use (dead store elimination).
  - `common_subexpression`: Deduplicates identical subexpressions within basic blocks.
  - `algebraic_simplification`: Rewrites identities (`x + 0 → x`, `x * 1 → x`, `x * 0 → 0`, `x % 1 → 0`, `x % x → 0`, `x * 2 → x + x`).
  - `loop_invariant_motion`: Detects loop-invariant expressions and hoists them before the loop header.
- **Reward Function:**
  $$\text{Reward} = (+10 \times \text{removed\_instructions}) + (+5 \times \text{folded\_constants}) - (20 \times \text{code\_growth})$$
- **Policy:** $\epsilon$-greedy exploration with exponential decay ($\times 0.99$ per iteration), balancing discovery of novel transformation sequences with exploitation of high-value paths.

---

## 🧪 Valid Language Sample

```c
// Variable initialization & arithmetic
a = 10;
b = 20;
c = 2 + 3;
d = a + b;
e = a + b;       // Common subexpression (CSE candidate)
f = c * 1;       // Algebraic simplification (f = c)
mod = 17 % 5;    // Constant folded to 2
unused = 999;    // Dead store eliminated

// Iterative while loop
i = 0;
while (i < 3) {
    result = a * b; // Loop-invariant hoisted outside loop
    print(result);
    i += 1;
}

print(d);
print(f);
print(mod);
```

---

## ⚙️ Local Development & Setup

### 🔧 1. Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.13)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### 🚀 2. Run the Compiler Backend & Frontend
```bash
# Clone the repository
git clone https://github.com/BhavyaShah7409/RL-Optimised-Compiler.git
cd RL-Optimised-Compiler

# Install backend dependencies
pip install -r requirements.txt

# Start the Flask server (serves API and static frontend UI)
python app.py
```
Open your browser at: **`http://localhost:5000`**

### 🧪 3. Run Automated Tests
```bash
# Run canonical compiler test cases (8 suites)
python -c "from api.compiler import SourceToSourceCompiler; c = SourceToSourceCompiler(); print(c.compile_and_execute('x = 10; print(x);')['output'])"

# Run adversarial fuzzing suite (30 iterations)
python adversarial_tester.py 30
```

---

## ☁️ Deployment Architecture

- **Backend (Render):** Deployed using [`render.yaml`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/render.yaml) with `gunicorn app:app`.
- **Frontend (Vercel):** Static edge deployment configured via [`vercel.json`](file:///c:/Users/bhavy/OneDrive/Desktop/RL-Optimised-Compiler/vercel.json).
