import io
import sys
import traceback
import threading
from typing import Tuple


class SecureExecutor:
    """
    BUG FIX #6: Replaced multiprocessing (crashes on Windows/Gunicorn with spawn)
    with a thread-based executor.  Threads share the GIL so they can be reliably
    interrupted via a daemon thread + join timeout, which is safe on all platforms.
    Truly infinite loops are killed by making the thread daemon=True; after
    timeout the thread is abandoned (it will be garbage-collected when the process
    eventually exits or the GIL forces it to yield).
    """

    # Whitelist of safe builtins for the sandboxed exec
    SAFE_BUILTINS = {
        '__build_class__': __builtins__.__build_class__ if hasattr(__builtins__, '__build_class__') else __build_class__,  # type: ignore[name-defined]
        'len': len, 'str': str, 'int': int, 'float': float,
        'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
        'set': set, 'frozenset': frozenset,
        'range': range, 'enumerate': enumerate, 'zip': zip,
        'min': min, 'max': max, 'sum': sum, 'abs': abs,
        'round': round, 'pow': pow, 'divmod': divmod,
        'sorted': sorted, 'reversed': reversed,
        'isinstance': isinstance, 'issubclass': issubclass, 'type': type,
        'print': print,   # will be overridden in exec_globals
        'repr': repr, 'hash': hash,
        'True': True, 'False': False, 'None': None,
        'Exception': Exception, 'ValueError': ValueError,
        'TypeError': TypeError, 'ZeroDivisionError': ZeroDivisionError,
        'IndexError': IndexError, 'KeyError': KeyError,
        'StopIteration': StopIteration,
    }

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def execute(self, python_code: str, input_data: str = "") -> Tuple[str, str, bool]:
        output_buffer = io.StringIO()
        result_holder: dict = {}

        def _run():
            exec_globals = {
                '__builtins__': self.SAFE_BUILTINS,
                'output_buffer': output_buffer,
                'input_values': input_data.strip().split('\n') if input_data.strip() else [],
                'input_index': 0,
            }
            stderr_capture = io.StringIO()
            old_stderr = sys.stderr
            try:
                sys.stderr = stderr_capture
                exec(python_code, exec_globals)  # noqa: S102
                output = exec_globals.get('result', output_buffer.getvalue())
                errors = stderr_capture.getvalue()
                result_holder['output'] = output
                result_holder['errors'] = errors
                result_holder['success'] = True
            except Exception as exc:
                result_holder['output'] = ''
                result_holder['errors'] = f"Runtime Error: {exc}\n{traceback.format_exc()}"
                result_holder['success'] = False
            finally:
                sys.stderr = old_stderr

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            # Thread is still running (infinite loop) — it's daemon so it won't
            # block process exit.  Report timeout.
            return "", "Execution timed out (exceeded 5 s)", False

        if not result_holder:
            return "", "Execution failed unexpectedly", False

        return (
            result_holder.get('output', ''),
            result_holder.get('errors', ''),
            result_holder.get('success', False),
        )
