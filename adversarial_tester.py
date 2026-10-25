import sys
import os
import random
import traceback
import time
import json

# Add current dir to path to import api
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api.compiler import SourceToSourceCompiler

class Fuzzer:
    def __init__(self):
        self.compiler = SourceToSourceCompiler()
        self.vars = ['x', 'y', 'z', 'a', 'b', 'c', 'i', 'j', 'k', 'temp', 'result']
        
    def random_number(self):
        return str(random.choice([0, 1, -1, 42, 3.14, 0.001, 1000000]))

    def random_var(self):
        return random.choice(self.vars)
        
    def generate_expr(self, depth=0):
        if depth > 3 or random.random() < 0.4:
            if random.random() < 0.5:
                return self.random_number()
            else:
                return self.random_var()
                
        op = random.choice(['+', '-', '*', '/'])
        left = self.generate_expr(depth + 1)
        right = self.generate_expr(depth + 1)
        
        # Avoid division by zero strictly in literal generation or variables which might be zero, 
        # but for true adversarial, we might want to test how the compiler handles it.
        # Let's keep it mostly valid syntax.
        return f"({left} {op} {right})"

    def generate_statement(self, depth=0, in_loop=False, active_loop_vars=None):
        if active_loop_vars is None:
            active_loop_vars = set()

        if depth >= 4:
            types = ['assign', 'print']
        else:
            types = ['assign', 'print', 'if']
            if depth < 2:
                types.append('while')
            
        choice = random.choice(types)
        
        if choice == 'assign':
            var = self.random_var()
            # Try not to overwrite the loop variable of an active loop to avoid infinite loops
            if in_loop and var in ['i', 'j', 'k']:
                var = random.choice(['x', 'y', 'z', 'a', 'b', 'c'])
                
            expr = self.generate_expr(depth)
            op = random.choice(['=', '+=', '-='])
            return f"{var} {op} {expr};"
            
        elif choice == 'print':
            expr = self.generate_expr(depth)
            return f"print({expr});"
            
        elif choice == 'if':
            cond = f"{self.generate_expr(depth)} {random.choice(['==', '!=', '>', '<', '>=', '<='])} {self.generate_expr(depth)}"
            body = "\n".join([self.generate_statement(depth + 1, in_loop, active_loop_vars) for _ in range(random.randint(1, 3))])
            else_body = ""
            if random.random() < 0.5:
                else_body = " else {\n" + "\n".join([self.generate_statement(depth + 1, in_loop, active_loop_vars) for _ in range(random.randint(1, 3))]) + "\n}"
            return f"if ({cond}) {{\n{body}\n}}{else_body}"
            
        elif choice == 'while':
            # Simple loop to avoid infinite execution timeouts
            available_vars = [v for v in ['i', 'j', 'k'] if v not in active_loop_vars]
            if not available_vars:
                var = random.choice(['x', 'y', 'z', 'a', 'b', 'c'])
                return f"{var} = {self.generate_expr(depth)};"

            loop_var = random.choice(available_vars)
            limit = random.randint(1, 10)
            cond = f"{loop_var} < {limit}"
            
            # Ensure loop terminates
            init = f"{loop_var} = 0;"
            inc = f"{loop_var} += 1;"
            
            new_active = active_loop_vars | {loop_var}
            body = "\n".join([self.generate_statement(depth + 1, True, new_active) for _ in range(random.randint(1, 2))])
            return f"{init}\nwhile ({cond}) {{\n{body}\n{inc}\n}}"
            
        return ""

    def generate_program(self):
        # Always initialize some variables to avoid reference before assignment semantic errors
        inits = [f"{v} = {self.random_number()};" for v in self.vars]
        stmts = [self.generate_statement() for _ in range(random.randint(5, 15))]
        return "\n".join(inits + stmts)

    def run_fuzzing(self, iterations=100):
        print(f"Starting fuzzing for {iterations} iterations...")
        stats = {
            'success': 0,
            'compile_error': 0,
            'semantic_error': 0,
            'execution_mismatch': 0, # Cannot easily test execution mismatch without a reference interpreter, but we can check if it crashed
            'crashes': 0,
            'total_time': 0,
            'optimization_rewards': []
        }
        
        for i in range(iterations):
            code = self.generate_program()
            
            start_time = time.time()
            try:
                res = self.compiler.compile_and_execute(code)
                stats['total_time'] += time.time() - start_time
                
                if res['success']:
                    stats['success'] += 1
                    # Collect optimization metrics
                    if len(res.get('optimization_log', [])) > 0:
                        stats['optimization_rewards'].append(len(res['optimization_log']))
                else:
                    if 'semantic_error' in res['errors'].lower() or res.get('semantic_errors'):
                        stats['semantic_error'] += 1
                    elif 'compilation error' in res['errors'].lower():
                        stats['compile_error'] += 1
                        print(f"\n--- FAILED ITERATION {i} (Compile Error) ---")
                        print("CODE:\n", code)
                        print("ERROR:\n", res['errors'])
                    else:
                        stats['execution_mismatch'] += 1
                        print(f"\n--- FAILED ITERATION {i} (Execution Error) ---")
                        print("CODE:\n", code)
                        print("ERROR:\n", res['errors'])
                        
            except Exception as e:
                stats['crashes'] += 1
                stats['total_time'] += time.time() - start_time
                print(f"\n--- CRASH AT ITERATION {i} ---")
                print("CODE:\n", code)
                traceback.print_exc()
                
            if i % 10 == 0:
                print(f"Iteration {i}/{iterations} complete.")

        print("\n--- FUZZING COMPLETE ---")
        print(json.dumps(stats, indent=2))
        
        # Determine recommendations based on stats
        if stats['crashes'] > 0:
            print("CRITICAL: Compiler crashed. Fix exceptions in compiler toolchain.")
        if stats['compile_error'] > 0:
            print("WARNING: Compilation errors found on valid adversarial syntax. Fix parser/optimizer.")

if __name__ == '__main__':
    iterations = 100
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])
    fuzzer = Fuzzer()
    fuzzer.run_fuzzing(iterations)
