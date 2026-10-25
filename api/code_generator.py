from typing import List, Dict, Optional, Set
from .intermediate_code import ThreeAddressCode


# ---------------------------------------------------------------------------
# Helper: format a TAC operand as a clean Python literal / identifier
# ---------------------------------------------------------------------------
def _fmt(val: Optional[str]) -> str:
    """Format a TAC operand as a Python token (strip .0 from whole floats)."""
    if val is None:
        return ""
    if not isinstance(val, str):
        return str(val)
    try:
        f = float(val)
        if f == int(f) and "." not in val:
            return str(int(f))
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return val


# ---------------------------------------------------------------------------
# TAC → Python decompiler
# ---------------------------------------------------------------------------
class TACToPythonGenerator:
    """
    Translates a list of ThreeAddressCode instructions (the RL-optimized TAC)
    into executable Python code.

    Control-flow mapping
    --------------------
    While-loop  : LABEL Lstart  … IF_FALSE cond Lend … GOTO Lstart  LABEL Lend
    If-else     : IF_FALSE cond Lelse  … [GOTO Lend  LABEL Lelse … LABEL Lend]
    If-only     : IF_FALSE cond Lend   … LABEL Lend
    """

    def __init__(self):
        self._lines: List[str] = []
        self._indent: int = 0

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------
    def _ind(self) -> str:
        return "    " * self._indent

    def _emit(self, line: str):
        self._lines.append(self._ind() + line)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def generate(self, tac: List[ThreeAddressCode]) -> str:
        self._lines = []
        self._indent = 0

        # ---- standard preamble ----
        self._emit("# Generated Python code (from RL-Optimized TAC)")
        self._emit("# Execution environment provides: output_buffer, input_values, input_index")
        self._emit("")
        self._emit("def custom_input():")
        self._indent += 1
        self._emit("global input_index")
        self._emit("if input_index < len(input_values):")
        self._indent += 1
        self._emit("val = input_values[input_index]")
        self._emit("input_index += 1")
        self._emit("return val")
        self._indent -= 1
        self._emit("else:")
        self._indent += 1
        self._emit("return '0'")
        self._indent -= 2
        self._emit("")
        self._emit("def custom_print(value):")
        self._indent += 1
        self._emit("output_buffer.write(str(value) + '\\n')")
        self._indent -= 1
        self._emit("")
        self._emit("# --- Main execution (from RL-Optimized TAC) ---")

        # Pre-declare compiler temporaries (t1, t2, …) so NameErrors never
        # occur when a temp is defined in only one branch of an if-else.
        temps: Set[str] = set()
        for instr in tac:
            if instr.result and instr.result.startswith('t'):
                rest = instr.result[1:]
                if rest.isdigit():
                    temps.add(instr.result)
        if temps:
            for t in sorted(temps, key=lambda x: int(x[1:])):
                self._emit(f"{t} = 0")
            self._emit("")

        # Build label-position index for look-ahead
        label_pos: Dict[str, int] = {}
        for i, instr in enumerate(tac):
            if instr.operation == 'LABEL':
                label_pos[instr.arg1] = i

        # Identify which labels are back-jump targets (loop starts)
        back_jump_targets: Set[str] = set()
        for i, instr in enumerate(tac):
            if instr.operation == 'GOTO' and instr.arg1 in label_pos:
                if label_pos[instr.arg1] <= i:          # backward jump
                    back_jump_targets.add(instr.arg1)

        # Decompile
        self._decompile(tac, 0, len(tac), label_pos, back_jump_targets, set())

        self._emit("")
        self._emit("result = output_buffer.getvalue()")
        return "\n".join(self._lines)

    # ------------------------------------------------------------------
    # Core recursive decompiler
    # ------------------------------------------------------------------
    def _decompile(
        self,
        tac: List[ThreeAddressCode],
        start: int,
        end: int,
        label_pos: Dict[str, int],
        back_jump_targets: Set[str],
        loop_exit_labels: Set[str],
    ):
        idx = start
        while idx < end:
            instr = tac[idx]

            # --------------------------------------------------------
            # 1. LABEL — could be while-loop header or just a target
            # --------------------------------------------------------
            if instr.operation == 'LABEL':
                lbl = instr.arg1

                if lbl in back_jump_targets:
                    # This label is the start of a while-loop.
                    # Find the matching backward GOTO within [idx+1, end)
                    back_goto_idx = None
                    for j in range(idx + 1, end):
                        if tac[j].operation == 'GOTO' and tac[j].arg1 == lbl:
                            back_goto_idx = j
                            break

                    if back_goto_idx is not None:
                        # Find exit label: the LABEL immediately after the GOTO
                        exit_lbl = None
                        if back_goto_idx + 1 < len(tac):
                            nxt = tac[back_goto_idx + 1]
                            if nxt.operation == 'LABEL':
                                exit_lbl = nxt.arg1

                        # Try to lift the loop condition into the while header.
                        # Pattern: LABEL → t = (cond) → IF_FALSE t Lexit → body
                        body_start = idx + 1
                        cond_str = None

                        if exit_lbl and idx + 2 < back_goto_idx:
                            maybe_cond = tac[idx + 1]
                            maybe_branch = tac[idx + 2]
                            if (maybe_branch.operation == 'IF_FALSE'
                                    and maybe_branch.arg2 == exit_lbl
                                    and maybe_branch.arg1 == maybe_cond.result):
                                # Check that the temp is not used inside the body
                                t_used = any(
                                    c.arg1 == maybe_cond.result or c.arg2 == maybe_cond.result
                                    for c in tac[idx + 3: back_goto_idx]
                                )
                                if not t_used:
                                    cond_str = self._binary_expr(maybe_cond)
                                    body_start = idx + 3
                        elif exit_lbl and idx + 1 < back_goto_idx:
                            maybe_branch = tac[idx + 1]
                            if (maybe_branch.operation == 'IF_FALSE'
                                    and maybe_branch.arg2 == exit_lbl):
                                cond_str = _fmt(maybe_branch.arg1)
                                body_start = idx + 2

                        if cond_str:
                            self._emit(f"while {cond_str}:")
                        else:
                            self._emit("while True:")

                        self._indent += 1
                        new_exits = set(loop_exit_labels)
                        if exit_lbl:
                            new_exits.add(exit_lbl)

                        body_start_lines = len(self._lines)
                        self._decompile(tac, body_start, back_goto_idx,
                                        label_pos, back_jump_targets, new_exits)
                        if len(self._lines) == body_start_lines:
                            self._emit("pass")
                        self._indent -= 1

                        # Advance past the GOTO and the exit LABEL
                        idx = back_goto_idx + 1
                        if idx < end and tac[idx].operation == 'LABEL' and tac[idx].arg1 == exit_lbl:
                            idx += 1
                        continue
                    # else: label with no matching back-goto in range → skip
                else:
                    # Not a loop header; skip the label itself.
                    # If it's an exit label we've reached sequentially, just skip it.
                    pass
                idx += 1
                continue

            # --------------------------------------------------------
            # 2. IF_FALSE — if or if-else statement
            # --------------------------------------------------------
            if instr.operation == 'IF_FALSE':
                cond_var = _fmt(instr.arg1)
                target_lbl = instr.arg2

                # 2a. Loop exit condition (break)
                if target_lbl in loop_exit_labels:
                    self._emit(f"if not ({cond_var}):")
                    self._indent += 1
                    self._emit("break")
                    self._indent -= 1
                    idx += 1
                    continue

                # 2b. Standard if / if-else
                # Find the LABEL that matches target_lbl
                target_idx = label_pos.get(target_lbl)
                if target_idx is not None and target_idx < end:
                    # Check for if-else: does instruction before target_idx jump over else?
                    has_else = False
                    else_end_idx = None
                    else_body_start = target_idx + 1

                    if target_idx > 0 and tac[target_idx - 1].operation == 'GOTO':
                        else_end_lbl = tac[target_idx - 1].arg1
                        else_end_pos = label_pos.get(else_end_lbl)
                        if else_end_pos is not None and else_end_pos < end:
                            has_else = True
                            else_end_idx = else_end_pos

                    # then-body is between idx+1 and target_idx (exclusive of any trailing GOTO)
                    then_end = target_idx - 1 if has_else else target_idx

                    self._emit(f"if ({cond_var}):")
                    self._indent += 1
                    start_l = len(self._lines)
                    self._decompile(tac, idx + 1, then_end,
                                    label_pos, back_jump_targets, loop_exit_labels)
                    if len(self._lines) == start_l:
                        self._emit("pass")
                    self._indent -= 1

                    if has_else and else_end_idx is not None:
                        self._emit("else:")
                        self._indent += 1
                        start_l = len(self._lines)
                        self._decompile(tac, else_body_start, else_end_idx,
                                        label_pos, back_jump_targets, loop_exit_labels)
                        if len(self._lines) == start_l:
                            self._emit("pass")
                        self._indent -= 1
                        idx = else_end_idx + 1  # skip past end LABEL
                    else:
                        idx = target_idx + 1    # skip past target LABEL
                    continue
                # Unresolved IF_FALSE: skip
                idx += 1
                continue

            # --------------------------------------------------------
            # 3. GOTO — only back-jumps are handled above via loop detection;
            #    forward GOTOs that remain here are already "consumed" by the
            #    if-else block above. Skip silently.
            # --------------------------------------------------------
            if instr.operation == 'GOTO':
                idx += 1
                continue

            # --------------------------------------------------------
            # 4. Regular instructions
            # --------------------------------------------------------
            self._emit_instr(instr)
            idx += 1

    # ------------------------------------------------------------------
    # Emit a single non-control-flow TAC instruction as Python
    # ------------------------------------------------------------------
    def _emit_instr(self, instr: ThreeAddressCode):
        op = instr.operation
        res = instr.result
        a1 = _fmt(instr.arg1)
        a2 = _fmt(instr.arg2)

        if op == 'ASSIGN':
            self._emit(f"{res} = {a1}")

        elif op == 'PRINT':
            self._emit(f"custom_print({a1})")

        elif op == 'SCAN':
            self._emit(f"{res} = float(custom_input())")

        elif instr.arg2 is None and op in ('-', '+'):
            self._emit(f"{res} = ({op}{a1})")

        elif instr.arg2 is None and op == '!':
            self._emit(f"{res} = (not {a1})")

        elif op in ('+', '-', '*', '/', '%',
                    '==', '!=', '<', '<=', '>', '>='):
            if op == '/':
                self._emit(f"{res} = ({a1} / {a2} if ({a2}) != 0 else 0)")
            else:
                self._emit(f"{res} = ({a1} {op} {a2})")

        elif op == '&&':
            self._emit(f"{res} = ({a1} and {a2})")

        elif op == '||':
            self._emit(f"{res} = ({a1} or {a2})")

        elif op in ('LABEL', 'GOTO'):
            pass  # already consumed above

        else:
            self._emit(f"# unhandled TAC: {instr}")

    def _binary_expr(self, instr: ThreeAddressCode) -> str:
        """Return a Python expression string for a binary/comparison TAC instruction."""
        op = instr.operation
        a1 = _fmt(instr.arg1)
        a2 = _fmt(instr.arg2)
        if instr.arg2 is None and op in ('-', '+'):
            return f"({op}{a1})"
        if instr.arg2 is None and op == '!':
            return f"(not {a1})"
        if op in ('+', '-', '*', '%', '==', '!=', '<', '<=', '>', '>='):
            return f"({a1} {op} {a2})"
        if op == '/':
            return f"({a1} / {a2} if ({a2}) != 0 else 0)"
        if op == '&&':
            return f"({a1} and {a2})"
        if op == '||':
            return f"({a1} or {a2})"
        return f"({a1} {op} {a2})"


# Backward compatibility alias
PythonCodeGenerator = TACToPythonGenerator

