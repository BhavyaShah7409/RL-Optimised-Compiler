from typing import Dict, List, Any, Tuple
from .lexer import Lexer, Token
from .parser import Parser
from .semantic_analyzer import SemanticAnalyzer
from .intermediate_code import IntermediateCodeGenerator
from .rl_optimizer import RLOptimizer
from .code_generator import DirectASTToPythonGenerator
from .executor import SecureExecutor
import time

# BUG FIX #15: do NOT set recursion limit at module level (it affects the whole
# Gunicorn process).  We only raise it locally inside compile_and_execute and
# restore it afterwards.
_DEFAULT_RECURSION_LIMIT = 1000


class CompilerError(Exception):
    pass


class SourceToSourceCompiler:
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.intermediate_generator = IntermediateCodeGenerator()
        self.optimizer = RLOptimizer()
        self.code_generator = DirectASTToPythonGenerator()
        self.executor = SecureExecutor()

    def _empty_result(self) -> Dict[str, Any]:
        return {
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
            'execution_time': 0,
        }

    def compile_and_execute(self, source_code: str, input_data: str = "") -> Dict[str, Any]:
        """Main compilation and execution pipeline."""
        data = self._empty_result()
        # BUG FIX #15: raise recursion limit locally, restore afterwards
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        try:
            start_time = time.time()

            # Stage 1: Lexical Analysis
            lexer = Lexer(source_code)
            tokens = lexer.tokenize()
            data['tokens'] = [
                {'type': t.type.value, 'value': t.value, 'line': t.line, 'column': t.column}
                for t in tokens
            ]

            # Stage 2: Parsing
            parser = Parser(tokens)
            ast = parser.parse()
            data['ast'] = self.ast_to_dict(ast)

            # Stage 3: Semantic Analysis
            semantic_errors = self.semantic_analyzer.analyze(ast)
            data['semantic_errors'] = semantic_errors
            if semantic_errors:
                data['errors'] = '\n'.join(semantic_errors)
                return data

            # Stage 4: Intermediate Code Generation
            intermediate_code = self.intermediate_generator.generate(ast)
            data['intermediate_code'] = [str(i) for i in intermediate_code]

            # Stage 5: RL-Based Optimization
            optimized_code, opt_log = self.optimizer.optimize(intermediate_code)
            data['optimized_code'] = [str(i) for i in optimized_code]
            data['optimization_log'] = opt_log

            # Stage 6: Target Code Generation (Python) — from AST for correctness
            python_code = self.code_generator.generate(ast)
            data['python_code'] = python_code

            # Stage 7: Secure Execution
            output, errors, success = self.executor.execute(python_code, input_data)
            data['output'] = output
            data['errors'] = errors
            data['success'] = success
            data['execution_time'] = time.time() - start_time

        except Exception as exc:
            data['errors'] = f"Compilation Error: {exc}"
            data['success'] = False
        finally:
            sys.setrecursionlimit(old_limit)

        return data

    # ------------------------------------------------------------------
    # AST → dict for JSON serialisation
    # ------------------------------------------------------------------
    def ast_to_dict(self, node) -> Dict[str, Any]:
        if node is None:
            return None
        result = {'type': type(node).__name__}
        for attr in ('value', 'name', 'operator', 'target'):
            if hasattr(node, attr):
                result[attr] = getattr(node, attr)
        for child_attr in ('left', 'right', 'operand', 'expression', 'condition'):
            if hasattr(node, child_attr):
                result[child_attr] = self.ast_to_dict(getattr(node, child_attr))
        for list_attr in ('statements', 'body', 'then_body'):
            if hasattr(node, list_attr):
                result[list_attr] = [self.ast_to_dict(s) for s in getattr(node, list_attr)]
        if hasattr(node, 'else_body') and node.else_body is not None:
            result['else_body'] = [self.ast_to_dict(s) for s in node.else_body]
        return result
