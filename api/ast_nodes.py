from abc import ABC, abstractmethod
from typing import List, Any, Optional
from dataclasses import dataclass

class ASTNode(ABC):
    pass

class Expression(ASTNode):
    pass

class Statement(ASTNode):
    pass

@dataclass
class NumberLiteral(Expression):
    value: float

@dataclass
class StringLiteral(Expression):
    value: str

@dataclass
class Identifier(Expression):
    name: str

@dataclass
class BinaryOperation(Expression):
    left: Expression
    operator: str
    right: Expression

@dataclass
class UnaryOperation(Expression):
    operator: str
    operand: Expression

@dataclass
class AssignmentStatement(Statement):
    target: str
    operator: str
    value: Expression

@dataclass
class PrintStatement(Statement):
    expression: Expression

@dataclass
class ScanStatement(Statement):
    target: str

@dataclass
class WhileStatement(Statement):
    condition: Expression
    body: List[Statement]

@dataclass
class IfStatement(Statement):
    condition: Expression
    then_body: List[Statement]
    else_body: Optional[List[Statement]] = None

@dataclass
class Program(ASTNode):
    statements: List[Statement]

class ASTVisitor(ABC):
    @abstractmethod
    def visit_number_literal(self, node: NumberLiteral) -> Any:
        pass
    
    @abstractmethod
    def visit_string_literal(self, node: StringLiteral) -> Any:
        pass
    
    @abstractmethod
    def visit_identifier(self, node: Identifier) -> Any:
        pass
    
    @abstractmethod
    def visit_binary_operation(self, node: BinaryOperation) -> Any:
        pass
    
    @abstractmethod
    def visit_unary_operation(self, node: UnaryOperation) -> Any:
        pass
    
    @abstractmethod
    def visit_assignment_statement(self, node: AssignmentStatement) -> Any:
        pass
    
    @abstractmethod
    def visit_print_statement(self, node: PrintStatement) -> Any:
        pass
    
    @abstractmethod
    def visit_scan_statement(self, node: ScanStatement) -> Any:
        pass
    
    @abstractmethod
    def visit_while_statement(self, node: WhileStatement) -> Any:
        pass
    
    @abstractmethod
    def visit_if_statement(self, node: IfStatement) -> Any:
        pass
    
    @abstractmethod
    def visit_program(self, node: Program) -> Any:
        pass

def accept_visitor(node: ASTNode, visitor: ASTVisitor) -> Any:
    if isinstance(node, NumberLiteral):
        return visitor.visit_number_literal(node)
    elif isinstance(node, StringLiteral):
        return visitor.visit_string_literal(node)
    elif isinstance(node, Identifier):
        return visitor.visit_identifier(node)
    elif isinstance(node, BinaryOperation):
        return visitor.visit_binary_operation(node)
    elif isinstance(node, UnaryOperation):
        return visitor.visit_unary_operation(node)
    elif isinstance(node, AssignmentStatement):
        return visitor.visit_assignment_statement(node)
    elif isinstance(node, PrintStatement):
        return visitor.visit_print_statement(node)
    elif isinstance(node, ScanStatement):
        return visitor.visit_scan_statement(node)
    elif isinstance(node, WhileStatement):
        return visitor.visit_while_statement(node)
    elif isinstance(node, IfStatement):
        return visitor.visit_if_statement(node)
    elif isinstance(node, Program):
        return visitor.visit_program(node)
    else:
        raise ValueError(f"Unknown AST node type: {type(node)}")
