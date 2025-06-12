import sys
import io
import traceback
from typing import Dict, Any, Tuple
import contextlib
import threading
import time

class ExecutionTimeoutError(Exception):
    pass

class SecureExecutor:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.restricted_imports = {
            'os', 'subprocess', 'importlib', '__import__',
            'eval', 'exec', 'compile', 'open', 'file', 'input',
            'raw_input', 'reload', 'vars', 'dir', 'globals', 'locals'
        }
    
    def execute_with_timeout(self, python_code: str, input_data: str = "") -> Tuple[str, str, bool]:
        """Execute code with timeout using threading (cross-platform)"""
        result = {'output': '', 'errors': '', 'success': False, 'exception': None}
        
        def target():
            try:
                # Create output buffer for capturing prints
                output_buffer = io.StringIO()
                
                # Prepare execution environment with necessary modules and functions
                exec_globals = {
                    '__builtins__': {
                        'len': len, 'str': str, 'int': int, 'float': float,
                        'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
                        'range': range, 'enumerate': enumerate, 'zip': zip,
                        'min': min, 'max': max, 'sum': sum, 'abs': abs,
                        'round': round, 'pow': pow, 'divmod': divmod,
                        'isinstance': isinstance, 'type': type,
                    },
                    # Provide necessary modules directly
                    'sys': sys,
                    'io': io,
                    # Pre-create the output buffer
                    'output_buffer': output_buffer,
                    # Pre-define input handling
                    'input_values': [],
                    'input_index': 0,
                }
                
                exec_locals = {}
                
                # Capture stderr for error reporting
                old_stderr = sys.stderr
                stderr_capture = io.StringIO()
                
                try:
                    sys.stderr = stderr_capture
                    
                    # Prepare input data
                    if input_data:
                        input_lines = input_data.strip().split('\n')
                        exec_globals['input_values'] = input_lines
                    
                    # Execute the code
                    exec(python_code, exec_globals, exec_locals)
                    
                    # Get the result
                    output = ""
                    if 'result' in exec_locals:
                        output = exec_locals['result']
                    elif output_buffer.getvalue():
                        output = output_buffer.getvalue()
                    
                    errors = stderr_capture.getvalue()
                    
                    result['output'] = output
                    result['errors'] = errors
                    result['success'] = True
                    
                finally:
                    sys.stderr = old_stderr
                    output_buffer.close()
                    
            except Exception as e:
                result['exception'] = e
                result['success'] = False
        
        # Create and start thread
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # Wait for completion or timeout
        thread.join(timeout=self.timeout)
        
        if thread.is_alive():
            # Thread is still running, execution timed out
            return "", "Execution timed out", False
        
        if result['exception']:
            error_msg = f"Runtime Error: {str(result['exception'])}\n{traceback.format_exc()}"
            return "", error_msg, False
        
        return result['output'], result['errors'], result['success']
    
    def execute(self, python_code: str, input_data: str = "") -> Tuple[str, str, bool]:
        """
        Execute Python code securely and return output, errors, and success status
        """
        try:
            return self.execute_with_timeout(python_code, input_data)
        except Exception as e:
            error_msg = f"Execution Error: {str(e)}\n{traceback.format_exc()}"
            return "", error_msg, False
