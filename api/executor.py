import sys
import io
import traceback
from typing import Dict, Any, Tuple
import contextlib
import threading
import time

class ExecutionTimeoutError(Exception):
    pass

def _mp_worker(q, code, inp):
    import sys
    import io
    import traceback
    try:
        output_buffer = io.StringIO()
        exec_globals = {
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float,
                'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'min': min, 'max': max, 'sum': sum, 'abs': abs,
                'round': round, 'pow': pow, 'divmod': divmod,
                'isinstance': isinstance, 'type': type,
            },
            'sys': sys,
            'io': io,
            'output_buffer': output_buffer,
            'input_values': inp.strip().split('\n') if inp else [],
            'input_index': 0,
        }
        exec_locals = {}
        old_stderr = sys.stderr
        stderr_capture = io.StringIO()
        try:
            sys.stderr = stderr_capture
            exec(code, exec_globals, exec_locals)
            output = exec_locals.get('result', output_buffer.getvalue())
            errors = stderr_capture.getvalue()
            q.put({'output': output, 'errors': errors, 'success': True, 'exception': None})
        finally:
            sys.stderr = old_stderr
            output_buffer.close()
    except Exception as e:
        q.put({'output': '', 'errors': f"Runtime Error: {str(e)}\n{traceback.format_exc()}", 'success': False, 'exception': str(e)})

class SecureExecutor:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.restricted_imports = {
            'os', 'subprocess', 'importlib', '__import__',
            'eval', 'exec', 'compile', 'open', 'file', 'input',
            'raw_input', 'reload', 'vars', 'dir', 'globals', 'locals'
        }
    
    def execute_with_timeout(self, python_code: str, input_data: str = "") -> Tuple[str, str, bool]:
        """Execute code with timeout using multiprocessing (cross-platform, killable)"""
        import multiprocessing
        
        # We need a top-level or picklable function for multiprocessing, or use a queue.
        # However, since this is Windows, spawn is used, so we must be careful with closures.
        # But wait, we can just use multiprocessing.Process with a worker function.
        
        queue = multiprocessing.Queue()

        p = multiprocessing.Process(target=_mp_worker, args=(queue, python_code, input_data))
        p.daemon = True
        p.start()
        
        p.join(timeout=self.timeout)
        
        if p.is_alive():
            p.terminate()
            p.join()
            return "", "Execution timed out", False
            
        if not queue.empty():
            res = queue.get()
            return res.get('output', ''), res.get('errors', ''), res.get('success', False)
            
        return "", "Execution crashed unexpectedly", False
    
    def execute(self, python_code: str, input_data: str = "") -> Tuple[str, str, bool]:
        """
        Execute Python code securely and return output, errors, and success status
        """
        try:
            return self.execute_with_timeout(python_code, input_data)
        except Exception as e:
            error_msg = f"Execution Error: {str(e)}\n{traceback.format_exc()}"
            return "", error_msg, False
