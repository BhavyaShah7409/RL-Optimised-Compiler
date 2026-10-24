from typing import List, Dict, Any
from .ast_nodes import *


class DirectASTToPythonGenerator(ASTVisitor):
    """Generate Python directly from the AST — correct control-flow, no goto/label issues."""

    def __init__(self):
        self.code_lines: List[str] = []
        self.indent_level = 0

    def indent(self) -> str:
        return "    " * self.indent_level

    def emit(self, line: str):
        self.code_lines.append(self.indent() + line)

    def generate(self, ast: 'Program') -> str:
        self.code_lines = []
        self.indent_level = 0

        # Preamble — modules are injected by executor
        self.emit("# Generated Python code")
        self.emit("# Execution environment provides: output_buffer, input_values, input_index")
        self.emit("")
        self.emit("def custom_input():")
        self.indent_level += 1
        self.emit("global input_index")
        self.emit("if input_index < len(input_values):")
        self.indent_level += 1
        self.emit("val = input_values[input_index]")
        self.emit("input_index += 1")
        self.emit("return val")
        self.indent_level -= 1
        self.emit("else:")
        self.indent_level += 1
        self.emit("return '0'")
        self.indent_level -= 2
        self.emit("")
        self.emit("def custom_print(value):")
        self.indent_level += 1
        self.emit("output_buffer.write(str(value) + '\\n')")
        self.indent_level -= 1
        self.emit("")
        self.emit("# --- Main execution ---")

        accept_visitor(ast, self)

        self.emit("")
        self.emit("result = output_buffer.getvalue()")
        return "\n".join(self.code_lines)

    # ---- Expression visitors (return a Python expression string) ----

    def visit_number_literal(self, node: NumberLiteral) -> str:
        v = node.value
        # Emit integers cleanly — 10.0 → 10
        if v == int(v):
            return str(int(v))
        return str(v)

    def visit_string_literal(self, node: StringLiteral) -> str:
        # Escape backslashes and double-quotes for embedding in a Python string
        escaped = node.value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'

    def visit_identifier(self, node: Identifier) -> str:
        return node.name

    def visit_binary_operation(self, node: BinaryOperation) -> str:
        left = accept_visitor(node.left, self)
        right = accept_visitor(node.right, self)

        if node.operator == '&&':
            return f"({left} and {right})"
        elif node.operator == '||':
            return f"({left} or {right})"
        elif node.operator == '/':
            return f"({left} / {right} if ({right}) != 0 else 0)"
        else:
            return f"({left} {node.operator} {right})"

    def visit_unary_operation(self, node: UnaryOperation) -> str:
        operand = accept_visitor(node.operand, self)
        if node.operator == '!':
            return f"(not {operand})"
        return f"({node.operator}{operand})"

    # ---- Statement visitors (emit lines, return None) ----

    def visit_assignment_statement(self, node: AssignmentStatement) -> Any:
        value_expr = accept_visitor(node.value, self)
        if node.operator == '=':
            self.emit(f"{node.target} = {value_expr}")
        elif node.operator == '+=':
            self.emit(f"{node.target} += {value_expr}")
        elif node.operator == '-=':
            self.emit(f"{node.target} -= {value_expr}")
        return None

    def visit_print_statement(self, node: PrintStatement) -> Any:
        expr = accept_visitor(node.expression, self)
        self.emit(f"custom_print({expr})")
        return None

    def visit_scan_statement(self, node: ScanStatement) -> Any:
        self.emit(f"{node.target} = float(custom_input())")
        return None

    def visit_while_statement(self, node: WhileStatement) -> Any:
        condition = accept_visitor(node.condition, self)
        self.emit(f"while {condition}:")
        self.indent_level += 1
        for stmt in node.body:
            accept_visitor(stmt, self)
        self.indent_level -= 1
        return None

    def visit_if_statement(self, node: IfStatement) -> Any:
        condition = accept_visitor(node.condition, self)
        self.emit(f"if {condition}:")
        self.indent_level += 1
        for stmt in node.then_body:
            accept_visitor(stmt, self)
        self.indent_level -= 1
        if node.else_body:
            self.emit("else:")
            self.indent_level += 1
            for stmt in node.else_body:
                accept_visitor(stmt, self)
            self.indent_level -= 1
        return None

    def visit_program(self, node: 'Program') -> Any:
        for stmt in node.statements:
            accept_visitor(stmt, self)
        return None


# BUG FIX #5: PythonCodeGenerator (TAC-based) is kept for reference but the
# compiler uses DirectASTToPythonGenerator which correctly handles all control flow.
# The broken TAC-based generator is removed to avoid confusion.
PythonCodeGenerator = DirectASTToPythonGenerator
