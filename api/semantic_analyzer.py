from typing import Dict, Set, List, Any
from .ast_nodes import *

class SemanticError(Exception):
    pass

class SymbolTable:
    def __init__(self):
        self.symbols: Dict[str, Any] = {}
        self.parent: Optional['SymbolTable'] = None
    
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
        
        # Type checking could be added here
        if node.operator in ['/', '%'] and isinstance(node.right, NumberLiteral) and node.right.value == 0:
            self.errors.append("Division by zero")
        
        return None
    
    def visit_unary_operation(self, node: UnaryOperation) -> Any:
        accept_visitor(node.operand, self)
        return None
    
    def visit_assignment_statement(self, node: AssignmentStatement) -> Any:
        accept_visitor(node.value, self)
        self.symbol_table.define(node.target)
        return None
    
    def visit_print_statement(self, node: PrintStatement) -> Any:
        accept_visitor(node.expression, self)
        return None
    
    def visit_scan_statement(self, node: ScanStatement) -> Any:
        self.symbol_table.define(node.target)
        return None
    
    def visit_while_statement(self, node: WhileStatement) -> Any:
        accept_visitor(node.condition, self)
        for stmt in node.body:
            accept_visitor(stmt, self)
        return None
    
    def visit_if_statement(self, node: IfStatement) -> Any:
        accept_visitor(node.condition, self)
        for stmt in node.then_body:
            accept_visitor(stmt, self)
        if node.else_body:
            for stmt in node.else_body:
                accept_visitor(stmt, self)
        return None
    
    def visit_program(self, node: Program) -> Any:
        for stmt in node.statements:
            accept_visitor(stmt, self)
        return None
