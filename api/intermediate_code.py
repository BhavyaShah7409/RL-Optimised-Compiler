from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .ast_nodes import *

@dataclass
class ThreeAddressCode:
    operation: str
    arg1: Optional[str] = None
    arg2: Optional[str] = None
    result: Optional[str] = None
    
    def __str__(self):
        if self.operation in ['ASSIGN']:
            return f"{self.result} = {self.arg1}"
        elif self.operation in ['PRINT']:
            return f"PRINT {self.arg1}"
        elif self.operation in ['SCAN']:
            return f"SCAN {self.result}"
        elif self.operation in ['LABEL']:
            return f"{self.arg1}:"
        elif self.operation in ['GOTO']:
            return f"GOTO {self.arg1}"
        elif self.operation in ['IF_FALSE']:
            return f"IF_FALSE {self.arg1} GOTO {self.arg2}"
        elif self.arg2 is None:
            return f"{self.result} = {self.operation} {self.arg1}"
        else:
            return f"{self.result} = {self.arg1} {self.operation} {self.arg2}"

class IntermediateCodeGenerator(ASTVisitor):
    def __init__(self):
        self.code: List[ThreeAddressCode] = []
        self.temp_counter = 0
        self.label_counter = 0
    
    def new_temp(self) -> str:
        self.temp_counter += 1
        return f"t{self.temp_counter}"
    
    def new_label(self) -> str:
        self.label_counter += 1
        return f"L{self.label_counter}"
    
    def emit(self, operation: str, arg1: str = None, arg2: str = None, result: str = None):
        self.code.append(ThreeAddressCode(operation, arg1, arg2, result))
    
    def generate(self, ast: Program) -> List[ThreeAddressCode]:
        self.code = []
        self.temp_counter = 0
        self.label_counter = 0
        accept_visitor(ast, self)
        return self.code
    
    def visit_number_literal(self, node: NumberLiteral) -> str:
        return str(node.value)
    
    def visit_string_literal(self, node: StringLiteral) -> str:
        return f'"{node.value}"'
    
    def visit_identifier(self, node: Identifier) -> str:
        return node.name
    
    def visit_binary_operation(self, node: BinaryOperation) -> str:
        left = accept_visitor(node.left, self)
        right = accept_visitor(node.right, self)
        temp = self.new_temp()
        self.emit(node.operator, left, right, temp)
        return temp
    
    def visit_unary_operation(self, node: UnaryOperation) -> str:
        operand = accept_visitor(node.operand, self)
        temp = self.new_temp()
        self.emit(node.operator, operand, None, temp)
        return temp
    
    def visit_assignment_statement(self, node: AssignmentStatement) -> Any:
        value = accept_visitor(node.value, self)
        
        if node.operator == '=':
            self.emit('ASSIGN', value, None, node.target)
        elif node.operator == '+=':
            temp = self.new_temp()
            self.emit('+', node.target, value, temp)
            self.emit('ASSIGN', temp, None, node.target)
        elif node.operator == '-=':
            temp = self.new_temp()
            self.emit('-', node.target, value, temp)
            self.emit('ASSIGN', temp, None, node.target)
        
        return None
    
    def visit_print_statement(self, node: PrintStatement) -> Any:
        expr = accept_visitor(node.expression, self)
        self.emit('PRINT', expr)
        return None
    
    def visit_scan_statement(self, node: ScanStatement) -> Any:
        self.emit('SCAN', None, None, node.target)
        return None
    
    def visit_while_statement(self, node: WhileStatement) -> Any:
        start_label = self.new_label()
        end_label = self.new_label()
        
        self.emit('LABEL', start_label)
        condition = accept_visitor(node.condition, self)
        self.emit('IF_FALSE', condition, end_label)
        
        for stmt in node.body:
            accept_visitor(stmt, self)
        
        self.emit('GOTO', start_label)
        self.emit('LABEL', end_label)
        return None
    
    def visit_if_statement(self, node: IfStatement) -> Any:
        else_label = self.new_label()
        end_label = self.new_label()
        
        condition = accept_visitor(node.condition, self)
        self.emit('IF_FALSE', condition, else_label)
        
        for stmt in node.then_body:
            accept_visitor(stmt, self)
        
        if node.else_body:
            self.emit('GOTO', end_label)
            self.emit('LABEL', else_label)
            for stmt in node.else_body:
                accept_visitor(stmt, self)
            self.emit('LABEL', end_label)
        else:
            self.emit('LABEL', else_label)
        
        return None
    
    def visit_program(self, node: Program) -> Any:
        for stmt in node.statements:
            accept_visitor(stmt, self)
        return None
