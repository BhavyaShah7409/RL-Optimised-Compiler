# RL-Optimised-Compiler
![Image](https://github.com/user-attachments/assets/7b32dae2-390f-4ef0-be0d-eba0c3f24fe2)
This project is a web-based, RL-optimized source-to-source compiler that translates code written in a custom high-level language into Python. It demonstrates the complete compilation process including lexical analysis, parsing, semantic checking, intermediate code generation, and AI-powered optimization. The compiler uses reinforcement learning to intelligently apply transformations like constant folding, dead code elimination, and loop optimizations, ultimately generating efficient Python code. The interactive UI provides a visual breakdown of each compilation stage, making it both educational and practical.

## Tech Stack
Frontend: HTML, CSS, JavaScript, CodeMirror
<br>
Backend: Python, Flask, Flask-CORS
<br>
Deployment: Vercel (Frontend), Render (Backend)

## Compiler Features
<ul> <li><strong>Lexical Analysis:</strong> Scans the input source code and converts it into a stream of tokens, identifying keywords, identifiers, operators, and symbols.</li> <li><strong>Parsing:</strong> Constructs an Abstract Syntax Tree (AST) from the token stream to represent the syntactic structure of the program.</li> <li><strong>Semantic Analysis:</strong> Validates the AST by checking for semantic errors like undeclared variables, type mismatches, and improper control flow.</li> <li><strong>Intermediate Code Generation:</strong> Translates the AST into a lower-level representation (three-address code), suitable for analysis and optimization.</li> <li><strong>RL Optimization:</strong> Applies reinforcement learning techniques to enhance code by performing transformations like constant folding, dead code elimination, strength reduction, loop invariant motion, and common subexpression reuse.</li> <li><strong>Code Generation:</strong> Converts the optimized intermediate code into clean and executable Python code that preserves the original program logic.</li> <li><strong>Secure Execution:</strong> Safely runs the generated Python code in a sandboxed environment and displays real-time output or error messages to the user.</li> </ul>
