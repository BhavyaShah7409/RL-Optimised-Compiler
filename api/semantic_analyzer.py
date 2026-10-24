from typing import Dict, Set, List, Any, Optional
from .ast_nodes import *


class SemanticError(Exception):
    pass


class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.symbols: Dict[str, Any] = {}
        self.parent = parent

    def define(self, name: str, value: Any = None):
        self.symbols[name] = value

    def lookup(self, name: str) -> bool:
        if name in self.symbols:
            return True
        if self.parent:
            return self.parent.lookup(name)
        return False

    def get(self, name: str) -> Any:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.get(name)
        return None


class SemanticAnalyzer(ASTVisitor):
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[str] = []

    def analyze(self, ast: Program) -> List[str]:
        self.errors = []
        self.symbol_table = SymbolTable()
        accept_visitor(ast, self)
        return self.errors

    def visit_number_literal(self, node: NumberLiteral) -> Any:
        return node.value

    def visit_string_literal(self, node: StringLiteral) -> Any:
        return node.value

    def visit_identifier(self, node: Identifier) -> Any:
        if not self.symbol_table.lookup(node.name):
            self.errors.append(f"Undefined variable '{node.name}'")
        return None

    def visit_binary_operation(self, node: BinaryOperation) -> Any:
        accept_visitor(node.left, self)
        accept_visitor(node.right, self)

        # Static division-by-zero check (literal only)
        if node.operator in ['/', '%'] and isinstance(node.right, NumberLiteral) and node.right.value == 0:
            self.errors.append("Division by zero detected")

        return None

    def visit_unary_operation(self, node: UnaryOperation) -> Any:
        accept_visitor(node.operand, self)
        return None

    def visit_assignment_statement(self, node: AssignmentStatement) -> Any:
        # For compound assignments (+=, -=) the variable must already exist
        if node.operator in ('+=', '-='):
            if not self.symbol_table.lookup(node.target):
                self.errors.append(f"Undefined variable '{node.target}' used in compound assignment")

        accept_visitor(node.value, self)
        # Define in the CURRENT (innermost) scope
        self.symbol_table.define(node.target)
        return None

    def visit_print_statement(self, node: PrintStatement) -> Any:
        accept_visitor(node.expression, self)
        return None

    def visit_scan_statement(self, node: ScanStatement) -> Any:
        # scan() defines the variable it reads into
        self.symbol_table.define(node.target)
        return None

    def visit_while_statement(self, node: WhileStatement) -> Any:
        accept_visitor(node.condition, self)

        # BUG FIX #7: push a new scope for the loop body
        outer = self.symbol_table
        self.symbol_table = SymbolTable(parent=outer)
        for stmt in node.body:
            accept_visitor(stmt, self)
        self.symbol_table = outer
        return None

    def visit_if_statement(self, node: IfStatement) -> Any:
        accept_visitor(node.condition, self)

        # BUG FIX #7: push a new scope for then/else bodies
        outer = self.symbol_table

        self.symbol_table = SymbolTable(parent=outer)
        for stmt in node.then_body:
            accept_visitor(stmt, self)
        self.symbol_table = outer

        if node.else_body:
            self.symbol_table = SymbolTable(parent=outer)
            for stmt in node.else_body:
                accept_visitor(stmt, self)
            self.symbol_table = outer

        return None

    def visit_program(self, node: Program) -> Any:
        for stmt in node.statements:
            accept_visitor(stmt, self)
        return None
