# 🧠 RL-Optimized Source-to-Source Compiler ⚡

An intelligent, end-to-end compiler that translates a custom high-level language into highly optimized Python code, leveraging Reinforcement Learning (Q-Learning) to dynamically improve optimization strategies.

Unlike traditional compilers that rely on static optimization heuristics, this system models code optimization as a sequential decision-making problem, enabling adaptive and performance-driven transformations on intermediate representations.

---

## 🚀 Highlights

- 🔁 Self-Optimizing Compiler using Q-Learning  
- ⚡ Dynamic Code Optimization based on execution feedback  
- 🧩 Full compilation pipeline: Lex → AST → TAC → Optimized Code  
- 🛡️ Secure sandboxed execution for adversarial inputs  
- 🌐 Interactive frontend dashboard (Vanilla JS)  

---

## 🏗️ Compiler Architecture

### 🔹 Multi-Stage Compilation Pipeline

- **Lexical Analysis**  
  Converts source code into tokens (identifiers, operators, literals)

- **Syntax Analysis**  
  Builds a hierarchical Abstract Syntax Tree (AST)

- **Semantic Analysis**  
  Ensures correctness (type checking, scope resolution, variable validation)

- **Intermediate Code Generation**  
  Translates AST into Three-Address Code (TAC)

---

## 🤖 Reinforcement Learning Optimization Engine

At the core lies a Q-Learning agent that learns optimal transformation strategies over time.

### 🎯 Key Components

- **State Space:** TAC representation of the program  

- **Action Space:**  
  - Constant Folding  
  - Loop Invariant Code Motion  
  - Common Subexpression Elimination (CSE)  
  - Dead Code Elimination  

- **Policy:**  
  Epsilon-Greedy (balances exploration vs exploitation)

- **Reward Function:**  
  - Reduced execution time  
  - Smaller code size  
  - Improved runtime efficiency  

The agent continuously updates its Q-table, improving optimization decisions across executions.

---

## 🛡️ Secure Execution Environment

- Multi-process isolation to handle infinite loops  
- Execution timeouts for non-terminating programs  
- Safe arithmetic handling (e.g., division by zero protection)  
- AST depth control to prevent recursion overflow  

---

## 🌐 Frontend (Developer Experience)

- Step-by-step animated compilation pipeline  
- Real-time syntax highlighting  
- REST API integration with backend  
- Lightweight, high-performance Vanilla JS interface  

---

## ⚙️ Local Setup

### 🔧 Backend (Flask API)

```bash
pip install -r requirements.txt
python app.py
```

Runs at: http://localhost:5000

---

### 🌐 Frontend

```bash
python -m http.server 8000
```

Open: http://localhost:8000

---

## ☁️ Deployment

### 🔹 Backend – Render

- Uses render.yaml for automated deployment  
- Auto-detects gunicorn app:app  
- Provides scalable hosted API  

---

### 🔹 Frontend – Vercel

- Static deployment using vercel.json  
- Configure backend URL in script.js  
- Optimized edge delivery  

---

## 🧪 Sample Language Syntax

```javascript
// Variable Declaration
var counter = 0;
var max_limit = 100;

// Loop
while (counter < max_limit) {
    counter = counter + 1;

    // Conditional
    if (counter == 50) {
        print("Halfway there!");
    } else {
        print(counter);
    }
}
```
