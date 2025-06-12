from typing import List, Dict, Any
from .intermediate_code import ThreeAddressCode
from .ast_nodes import *

class PythonCodeGenerator:
    def __init__(self):
        self.code_lines: List[str] = []
        self.indent_level = 0
        self.variables: Dict[str, Any] = {}
    
    def indent(self) -> str:
        return "    " * self.indent_level
    
    def emit(self, line: str):
        self.code_lines.append(self.indent() + line)
    
    def generate(self, intermediate_code: List[ThreeAddressCode]) -> str:
        self.code_lines = []
        self.indent_level = 0
        
        # Add necessary setup without imports (modules are provided by executor)
        self.emit("# Generated Python code")
        self.emit("# output_buffer and other modules are provided by the execution environment")
        self.emit("")
        self.emit("def custom_input():")
        self.indent_level += 1
        self.emit("global input_index")
        self.emit("if input_index < len(input_values):")
        self.indent_level += 1
        self.emit("value = input_values[input_index]")
        self.emit("input_index += 1")
        self.emit("return value")
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
        self.emit("# Main execution")
        
        # Process intermediate code
        i = 0
        while i < len(intermediate_code):
            instr = intermediate_code[i]
            i = self.generate_instruction(instr, intermediate_code, i)
        
        self.emit("")
        self.emit("# Return output")
        self.emit("result = output_buffer.getvalue()")
        
        return "\n".join(self.code_lines)
    
    def generate_instruction(self, instr: ThreeAddressCode, all_code: List[ThreeAddressCode], index: int) -> int:
        if instr.operation == 'ASSIGN':
            self.emit(f"{instr.result} = {self.format_operand(instr.arg1)}")
        
        elif instr.operation == 'PRINT':
            self.emit(f"custom_print({self.format_operand(instr.arg1)})")
        
        elif instr.operation == 'SCAN':
            self.emit(f"{instr.result} = float(custom_input())")
        
        elif instr.operation == 'LABEL':
            # Python doesn't have goto, so we'll use while loops for control flow
            pass
        
        elif instr.operation == 'GOTO':
            # Handle with while loops
            pass
        
        elif instr.operation == 'IF_FALSE':
            # Find the corresponding label
            label_pos = self.find_label_position(all_code, instr.arg2)
            if label_pos != -1:
                self.emit(f"if not ({self.format_operand(instr.arg1)}):")
                self.indent_level += 1
                # Generate code until the label
                for i in range(index + 1, label_pos):
                    if i < len(all_code):
                        self.generate_instruction(all_code[i], all_code, i)
                self.indent_level -= 1
                return label_pos
        
        elif instr.operation in ['+', '-', '*', '/']:
            left = self.format_operand(instr.arg1)
            right = self.format_operand(instr.arg2)
            
            if instr.operation == '/':
                # Handle division by zero
                self.emit(f"{instr.result} = {left} / {right} if {right} != 0 else 0")
            else:
                self.emit(f"{instr.result} = {left} {instr.operation} {right}")
        
        elif instr.operation in ['>', '<', '>=', '<=', '==', '!=']:
            left = self.format_operand(instr.arg1)
            right = self.format_operand(instr.arg2)
            self.emit(f"{instr.result} = {left} {instr.operation} {right}")
        
        elif instr.operation == '&&':
            left = self.format_operand(instr.arg1)
            right = self.format_operand(instr.arg2)
            self.emit(f"{instr.result} = {left} and {right}")
        
        elif instr.operation == '||':
            left = self.format_operand(instr.arg1)
            right = self.format_operand(instr.arg2)
            self.emit(f"{instr.result} = {left} or {right}")
        
        elif instr.operation == '!':
            operand = self.format_operand(instr.arg1)
            self.emit(f"{instr.result} = not {operand}")
        
        elif instr.operation == '-' and instr.arg2 is None:  # Unary minus
            operand = self.format_operand(instr.arg1)
            self.emit(f"{instr.result} = -{operand}")
        
        return index + 1
    
    def format_operand(self, operand: str) -> str:
        if operand is None:
            return "None"
        
        # Check if it's a string literal
        if operand.startswith('"') and operand.endswith('"'):
            return operand
        
        # Check if it's a number
        try:
            float(operand)
            return operand
        except ValueError:
            # It's a variable
            return operand
    
    def find_label_position(self, code: List[ThreeAddressCode], label: str) -> int:
        for i, instr in enumerate(code):
            if instr.operation == 'LABEL' and instr.arg1 == label:
                return i
        return -1

class DirectASTToPythonGenerator(ASTVisitor):
    """Alternative generator that works directly from AST for better control flow handling"""
    
    def __init__(self):
        self.code_lines: List[str] = []
        self.indent_level = 0
    
    def indent(self) -> str:
        return "    " * self.indent_level
    
    def emit(self, line: str):
        self.code_lines.append(self.indent() + line)
    
    def generate(self, ast: Program) -> str:
        self.code_lines = []
        self.indent_level = 0
        
        # Add setup code without imports (modules provided by executor)
        self.emit("# Generated Python code")
        self.emit("# Modules and variables are provided by the execution environment")
        self.emit("")
        self.emit("def custom_input():")
        self.indent_level += 1
        self.emit("global input_index")
        self.emit("if input_index < len(input_values):")
        self.indent_level += 1
        self.emit("value = input_values[input_index]")
        self.emit("input_index += 1")
        self.emit("return value")
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
        self.emit("# Main execution")
        
        # Generate from AST
        accept_visitor(ast, self)
        
        self.emit("")
        self.emit("result = output_buffer.getvalue()")
        
        return "\n".join(self.code_lines)
    
    def visit_number_literal(self, node: NumberLiteral) -> str:
        return str(node.value)
    
    def visit_string_literal(self, node: StringLiteral) -> str:
        return f'"{node.value}"'
    
    def visit_identifier(self, node: Identifier) -> str:
        return node.name
    
    def visit_binary_operation(self, node: BinaryOperation) -> str:
        left = accept_visitor(node.left, self)
        right = accept_visitor(node.right, self)
        
        if node.operator == '&&':
            return f"({left} and {right})"
        elif node.operator == '||':
            return f"({left} or {right})"
        else:
            return f"({left} {node.operator} {right})"
    
    def visit_unary_operation(self, node: UnaryOperation) -> str:
        operand = accept_visitor(node.operand, self)
        if node.operator == '!':
            return f"(not {operand})"
        else:
            return f"({node.operator}{operand})"
    
    def visit_assignment_statement(self, node: AssignmentStatement) -> Any:
        value = accept_visitor(node.value, self)
        
        if node.operator == '=':
            self.emit(f"{node.target} = {value}")
        elif node.operator == '+=':
            self.emit(f"{node.target} += {value}")
        elif node.operator == '-=':
            self.emit(f"{node.target} -= {value}")
        
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
    
    def visit_program(self, node: Program) -> Any:
        for stmt in node.statements:
            accept_visitor(stmt, self)
        return None
