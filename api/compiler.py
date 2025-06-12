from typing import Dict, List, Any, Tuple
from .lexer import Lexer, Token
from .parser import Parser
from .semantic_analyzer import SemanticAnalyzer
from .intermediate_code import IntermediateCodeGenerator
from .rl_optimizer import RLOptimizer
from .code_generator import PythonCodeGenerator, DirectASTToPythonGenerator
from .executor import SecureExecutor
import json
import time

class CompilerError(Exception):
    pass

class SourceToSourceCompiler:
    def __init__(self):
        self.lexer = None
        self.parser = None
        self.semantic_analyzer = SemanticAnalyzer()
        self.intermediate_generator = IntermediateCodeGenerator()
        self.optimizer = RLOptimizer()
        self.code_generator = DirectASTToPythonGenerator()  # Use direct AST generator for better control flow
        self.executor = SecureExecutor()
        
        # Compilation stages data
        self.compilation_data = {
            'tokens': [],
            'ast': None,
            'semantic_errors': [],
            'intermediate_code': [],
            'optimized_code': [],
            'optimization_log': [],
            'python_code': '',
            'output': '',
            'errors': '',
            'success': False
        }
    
    def compile_and_execute(self, source_code: str, input_data: str = "") -> Dict[str, Any]:
        """Main compilation and execution pipeline"""
        try:
            start_time = time.time()
            
            # Reset compilation data
            self.compilation_data = {
                'tokens': [],
                'ast': None,
                'semantic_errors': [],
                'intermediate_code': [],
                'optimized_code': [],
                'optimization_log': [],
                'python_code': '',
                'output': '',
                'errors': '',
                'success': False,
                'execution_time': 0
            }
            
            # Stage 1: Lexical Analysis
            self.lexer = Lexer(source_code)
            tokens = self.lexer.tokenize()
            self.compilation_data['tokens'] = [
                {
                    'type': token.type.value,
                    'value': token.value,
                    'line': token.line,
                    'column': token.column
                }
                for token in tokens
            ]
            
            # Stage 2: Parsing (AST Generation)
            self.parser = Parser(tokens)
            ast = self.parser.parse()
            self.compilation_data['ast'] = self.ast_to_dict(ast)
            
            # Stage 3: Semantic Analysis
            semantic_errors = self.semantic_analyzer.analyze(ast)
            self.compilation_data['semantic_errors'] = semantic_errors
            
            if semantic_errors:
                self.compilation_data['errors'] = '\n'.join(semantic_errors)
                return self.compilation_data
            
            # Stage 4: Intermediate Code Generation
            intermediate_code = self.intermediate_generator.generate(ast)
            self.compilation_data['intermediate_code'] = [str(instr) for instr in intermediate_code]
            
            # Stage 5: RL-based Optimization
            optimized_code, optimization_log = self.optimizer.optimize(intermediate_code)
            self.compilation_data['optimized_code'] = [str(instr) for instr in optimized_code]
            self.compilation_data['optimization_log'] = optimization_log
            
            # Stage 6: Target Code Generation (Python)
            python_code = self.code_generator.generate(ast)  # Use AST directly for better results
            self.compilation_data['python_code'] = python_code
            
            # Stage 7: Secure Execution
            output, errors, success = self.executor.execute(python_code, input_data)
            self.compilation_data['output'] = output
            self.compilation_data['errors'] = errors
            self.compilation_data['success'] = success
            
            # Calculate execution time
            self.compilation_data['execution_time'] = time.time() - start_time
            
            return self.compilation_data
            
        except Exception as e:
            self.compilation_data['errors'] = f"Compilation Error: {str(e)}"
            self.compilation_data['success'] = False
            return self.compilation_data
    
    def ast_to_dict(self, node) -> Dict[str, Any]:
        """Convert AST node to dictionary for JSON serialization"""
        if node is None:
            return None
        
        result = {
            'type': type(node).__name__
        }
        
        # Handle different node types
        if hasattr(node, 'value'):
            result['value'] = node.value
        if hasattr(node, 'name'):
            result['name'] = node.name
        if hasattr(node, 'operator'):
            result['operator'] = node.operator
        if hasattr(node, 'target'):
            result['target'] = node.target
        
        # Handle child nodes
        if hasattr(node, 'left'):
            result['left'] = self.ast_to_dict(node.left)
        if hasattr(node, 'right'):
            result['right'] = self.ast_to_dict(node.right)
        if hasattr(node, 'operand'):
            result['operand'] = self.ast_to_dict(node.operand)
        if hasattr(node, 'expression'):
            result['expression'] = self.ast_to_dict(node.expression)
        if hasattr(node, 'condition'):
            result['condition'] = self.ast_to_dict(node.condition)
        
        # Handle lists of nodes
        if hasattr(node, 'statements'):
            result['statements'] = [self.ast_to_dict(stmt) for stmt in node.statements]
        if hasattr(node, 'body'):
            result['body'] = [self.ast_to_dict(stmt) for stmt in node.body]
        if hasattr(node, 'then_body'):
            result['then_body'] = [self.ast_to_dict(stmt) for stmt in node.then_body]
        if hasattr(node, 'else_body') and node.else_body:
            result['else_body'] = [self.ast_to_dict(stmt) for stmt in node.else_body]
        
        return result
