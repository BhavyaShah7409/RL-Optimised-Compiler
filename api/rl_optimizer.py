import copy
import random
from typing import List, Dict, Tuple, Any, Set, Optional
from dataclasses import dataclass, field
from .intermediate_code import ThreeAddressCode


@dataclass
class OptimizationAction:
    type: str   # 'constant_folding' | 'dead_code_elimination' | 'common_subexpression' | 'strength_reduction' | 'loop_invariant'
    position: int
    confidence: float = 0.0
    loop_info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.type} at position {self.position} (confidence: {self.confidence:.2f})"


def _deep_copy_code(code: List[ThreeAddressCode]) -> List[ThreeAddressCode]:
    """BUG FIX #8: deep-copy a TAC list so mutations don't bleed across references."""
    return [copy.copy(instr) for instr in code]


class RLOptimizer:
    def __init__(self):
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.optimization_history: List[Tuple[str, float]] = []
        self.variable_usage: Dict[str, Set[int]] = {}
        self.variable_defs: Dict[str, Set[int]] = {}

    # ------------------------------------------------------------------
    # State key — BUG FIX #11: encode operation *type* only, not position,
    # so Q-values remain valid after insertions / deletions.
    # ------------------------------------------------------------------
    def get_state_key(self, code: List[ThreeAddressCode], position: int) -> str:
        if position >= len(code):
            return "END"
        current = code[position]
        context = [current.operation]
        if position > 0:
            context.append(f"prev_{code[position - 1].operation}")
        if position < len(code) - 1:
            context.append(f"next_{code[position + 1].operation}")
        return "_".join(context)

    # ------------------------------------------------------------------
    # Usage / definition tracking
    # ------------------------------------------------------------------
    def analyze_code(self, code: List[ThreeAddressCode]) -> None:
        self.variable_usage = {}
        self.variable_defs = {}

        for i, instr in enumerate(code):
            if instr.result and instr.operation not in ('LABEL', 'GOTO', 'IF_FALSE', 'PRINT'):
                self.variable_defs.setdefault(instr.result, set()).add(i)
                self.variable_usage.setdefault(instr.result, set())

            for arg in (instr.arg1, instr.arg2):
                if arg and isinstance(arg, str) and not self.is_constant(arg):
                    self.variable_usage.setdefault(arg, set()).add(i)

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------
    def find_loops(self, code: List[ThreeAddressCode]) -> List[Dict[str, Any]]:
        loops = []
        labels: Dict[str, int] = {}

        for i, instr in enumerate(code):
            if instr.operation == 'LABEL':
                labels[instr.arg1] = i

        for i, instr in enumerate(code):
            if instr.operation == 'GOTO' and instr.arg1 in labels:
                target = labels[instr.arg1]
                if target < i:  # backward jump → loop
                    condition_pos = -1
                    for j in range(target, i):
                        if code[j].operation == 'IF_FALSE':
                            condition_pos = j
                            break
                    if condition_pos != -1:
                        loops.append({
                            'start': target,
                            'end': i,
                            'condition': condition_pos,
                            'label': instr.arg1,
                            'modified_vars': set(),
                            'invariant_candidates': [],
                            'parent': None,
                            'children': [],
                        })

        loops.sort(key=lambda x: x['start'])

        # Identify nesting relationships
        for i, loop in enumerate(loops):
            for j, other in enumerate(loops):
                if i != j and loop['start'] < other['start'] and loop['end'] > other['end']:
                    other['parent'] = i
                    loop['children'].append(j)

        # Collect variables modified inside each loop
        for loop in loops:
            for i in range(loop['start'], loop['end'] + 1):
                instr = code[i]
                if instr.result and instr.operation not in ('LABEL', 'GOTO', 'IF_FALSE'):
                    loop['modified_vars'].add(instr.result)

        # Find loop-invariant candidates
        for loop in loops:
            for i in range(loop['start'] + 1, loop['end']):
                instr = code[i]
                if instr.operation not in ('+', '-', '*', '/', 'ASSIGN'):
                    continue
                if not instr.result:
                    continue
                # Do not hoist if the variable is read anywhere in the loop before `i`
                # (e.g. in the loop condition or earlier statements, which is a loop-carried dependency)
                read_before = False
                for j in range(loop['start'], i):
                    cj = code[j]
                    if cj.arg1 == instr.result or cj.arg2 == instr.result:
                        read_before = True
                        break
                if read_before:
                    continue

                invariant = True
                for arg in (instr.arg1, instr.arg2):
                    if arg and isinstance(arg, str) and not self.is_constant(arg):
                        if arg in loop['modified_vars']:
                            invariant = False
                            break
                if invariant and instr.result:
                    # Only hoist if the result is defined exactly once inside
                    # the loop — hoisting it when it's assigned multiple times
                    # (e.g. j = 0 inside a nested loop) would break correctness.
                    defs_in_loop = sum(
                        1 for ii in range(loop['start'], loop['end'] + 1)
                        if code[ii].result == instr.result
                        and code[ii].operation not in ('LABEL', 'GOTO', 'IF_FALSE')
                    )
                    if defs_in_loop == 1:
                        loop['invariant_candidates'].append(i)

        return loops

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def is_constant(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return value.startswith('"') and value.endswith('"')

    def try_parse_float(self, value: str) -> Optional[float]:
        if not isinstance(value, str):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def is_variable_used_later(self, code: List[ThreeAddressCode], position: int, variable: str) -> bool:
        if variable not in self.variable_usage:
            return False
        usages = self.variable_usage[variable]
        # Simple forward check
        if any(pos > position for pos in usages):
            return True
        # Loop-awareness: if the definition at `position` is inside a loop body,
        # the variable may be *read* on a future iteration even though all its
        # usage-positions are <= position in the flat list.
        # If there is a backward GOTO after `position` that jumps to a LABEL at or before
        # `position`, and `variable` is used anywhere within that loop's range [target, j],
        # then the variable will be read again on the next iteration.
        label_pos = {instr.arg1: idx for idx, instr in enumerate(code) if instr.operation == 'LABEL'}
        for j in range(position + 1, len(code)):
            instr = code[j]
            if instr.operation == 'GOTO' and instr.arg1 in label_pos:
                target_pos = label_pos[instr.arg1]
                if target_pos <= position:
                    if any(target_pos <= u <= j for u in usages):
                        return True
        return False


    # ------------------------------------------------------------------
    # Optimisation transformations
    # ------------------------------------------------------------------
    def apply_constant_folding(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        if position >= len(code):
            return code
        instr = code[position]
        if instr.operation in ('+', '-', '*', '/') and instr.arg1 and instr.arg2 and self.is_constant(instr.arg1) and self.is_constant(instr.arg2):
            try:
                v1, v2 = float(instr.arg1), float(instr.arg2)
                if instr.operation == '+':
                    res = v1 + v2
                elif instr.operation == '-':
                    res = v1 - v2
                elif instr.operation == '*':
                    res = v1 * v2
                elif instr.operation == '/':
                    if v2 == 0:
                        return code
                    res = v1 / v2
                else:
                    return code

                res_str = str(int(res)) if res == int(res) else str(res)
                new_code = _deep_copy_code(code)
                new_code[position] = ThreeAddressCode('ASSIGN', res_str, None, instr.result)
                return new_code
            except Exception:
                return code
        elif instr.operation in ('+', '-') and instr.arg1 and instr.arg2 is None and self.is_constant(instr.arg1):
            try:
                v1 = float(instr.arg1)
                res = -v1 if instr.operation == '-' else v1
                res_str = str(int(res)) if res == int(res) else str(res)
                new_code = _deep_copy_code(code)
                new_code[position] = ThreeAddressCode('ASSIGN', res_str, None, instr.result)
                return new_code
            except Exception:
                return code
        return code

    def apply_dead_code_elimination(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        if position >= len(code):
            return code
        instr = code[position]
        if instr.operation in ('PRINT', 'SCAN', 'LABEL', 'GOTO', 'IF_FALSE'):
            return code
        if instr.result and self.is_variable_used_later(code, position, instr.result):
            return code
        new_code = _deep_copy_code(code)
        del new_code[position]
        return new_code

    def _is_valid_cse(self, code: List[ThreeAddressCode], prev_idx: int, cur_idx: int) -> bool:
        prev = code[prev_idx]
        cur = code[cur_idx]
        if prev.operation != cur.operation or prev.arg1 != cur.arg1 or prev.arg2 != cur.arg2 or not prev.result:
            return False
        # If prev.result overwrote one of its own operands (e.g. i = i + 1), it cannot be reused
        if not self.is_constant(cur.arg1) and prev.result == cur.arg1:
            return False
        if not self.is_constant(cur.arg2) and prev.result == cur.arg2:
            return False
        # Must be within the same basic block and neither operands nor prev.result can be overwritten
        for j in range(prev_idx + 1, cur_idx):
            instr = code[j]
            if instr.operation in ('LABEL', 'GOTO', 'IF_FALSE'):
                return False
            if instr.result:
                if instr.result == prev.result:
                    return False
                if not self.is_constant(cur.arg1) and instr.result == cur.arg1:
                    return False
                if not self.is_constant(cur.arg2) and instr.result == cur.arg2:
                    return False
        return True

    def apply_common_subexpression_elimination(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        if position >= len(code):
            return code
        current = code[position]
        if current.operation not in ('+', '-', '*', '/') or not current.arg1 or not current.arg2:
            return code
        for i in range(position - 1, -1, -1):
            if code[i].operation in ('LABEL', 'GOTO', 'IF_FALSE'):
                break
            if self._is_valid_cse(code, i, position):
                new_code = _deep_copy_code(code)
                new_code[position] = ThreeAddressCode('ASSIGN', code[i].result, None, current.result)
                return new_code
        return code

    def apply_strength_reduction(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        if position >= len(code):
            return code
        instr = code[position]
        new_code = _deep_copy_code(code)
        if instr.operation == '*':
            if self.is_constant(instr.arg2) and self.try_parse_float(instr.arg2) == 2.0:
                new_code[position] = ThreeAddressCode('+', instr.arg1, instr.arg1, instr.result)
                return new_code
            if self.is_constant(instr.arg1) and self.try_parse_float(instr.arg1) == 2.0:
                new_code[position] = ThreeAddressCode('+', instr.arg2, instr.arg2, instr.result)
                return new_code
        return code

    def apply_loop_invariant_motion(self, code: List[ThreeAddressCode], position: int,
                                     loop_idx: int, loops: List[Dict[str, Any]]) -> List[ThreeAddressCode]:
        if position >= len(code) or loop_idx >= len(loops):
            return code

        # Find the outermost enclosing loop
        current_idx = loop_idx
        while loops[current_idx]['parent'] is not None:
            current_idx = loops[current_idx]['parent']
        outermost = loops[current_idx]

        new_code = _deep_copy_code(code)
        instr = new_code[position]
        del new_code[position]

        # Insert before the outermost loop's LABEL instruction
        insert_at = outermost['start']
        new_code.insert(insert_at, instr)
        return new_code

    # ------------------------------------------------------------------
    # Reward — BUG FIX #10: removed phantom 'COMMENT' check, rewarded correctly
    # ------------------------------------------------------------------
    def calculate_reward(self, original_code: List[ThreeAddressCode],
                         optimized_code: List[ThreeAddressCode]) -> float:
        size_reduction = len(original_code) - len(optimized_code)
        reward = size_reduction * 10.0

        for instr in optimized_code:
            if instr.operation == 'ASSIGN' and self.is_constant(instr.arg1):
                reward += 5.0  # constant folding bonus

        if len(optimized_code) > len(original_code):
            reward -= 20.0

        return reward

    # ------------------------------------------------------------------
    # Q-learning
    # ------------------------------------------------------------------
    def update_q_value(self, state: str, action: OptimizationAction,
                       reward: float, next_state: str) -> None:
        self.q_table.setdefault(state, {})

        # BUG FIX #11: action key uses only operation type, NOT position
        action_key = action.type
        current_q = self.q_table[state].get(action_key, 0.0)
        max_next_q = max(self.q_table[next_state].values()) if next_state in self.q_table and self.q_table[next_state] else 0.0

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_key] = new_q
        action.confidence = new_q

    def get_possible_actions(self, code: List[ThreeAddressCode], position: int,
                              loops: List[Dict[str, Any]]) -> List[OptimizationAction]:
        actions = []
        if position >= len(code):
            return actions
        current = code[position]

        # Constant folding
        if (current.operation in ('+', '-', '*', '/') and current.arg1 and current.arg2
                and self.is_constant(current.arg1) and self.is_constant(current.arg2)):
            actions.append(OptimizationAction('constant_folding', position, 0.8))
        elif (current.operation in ('+', '-') and current.arg1 and current.arg2 is None
                and self.is_constant(current.arg1)):
            actions.append(OptimizationAction('constant_folding', position, 0.8))

        # Dead code elimination
        if (current.result and not self.is_variable_used_later(code, position, current.result)
                and current.operation not in ('PRINT', 'SCAN', 'LABEL', 'GOTO', 'IF_FALSE')):
            actions.append(OptimizationAction('dead_code_elimination', position, 0.7))

        # Common subexpression elimination (within basic block)
        if current.operation in ('+', '-', '*', '/') and current.arg1 and current.arg2:
            for i in range(position - 1, -1, -1):
                if code[i].operation in ('LABEL', 'GOTO', 'IF_FALSE'):
                    break
                if self._is_valid_cse(code, i, position):
                    actions.append(OptimizationAction('common_subexpression', position, 0.6))
                    break

        # Strength reduction
        if current.operation == '*' and current.arg1 and current.arg2:
            if ((self.is_constant(current.arg1) and self.try_parse_float(current.arg1) == 2.0) or
                    (self.is_constant(current.arg2) and self.try_parse_float(current.arg2) == 2.0)):
                actions.append(OptimizationAction('strength_reduction', position, 0.5))

        # Loop invariant code motion
        for loop_idx, loop in enumerate(loops):
            if position in loop['invariant_candidates']:
                outermost_idx = loop_idx
                for j, outer in enumerate(loops):
                    if j < loop_idx and outer['start'] <= loop['start'] and outer['end'] >= loop['end']:
                        is_inv = True
                        for arg in (current.arg1, current.arg2):
                            if arg and isinstance(arg, str) and not self.is_constant(arg):
                                if arg in outer['modified_vars']:
                                    is_inv = False
                                    break
                        if is_inv:
                            outermost_idx = j
                actions.append(OptimizationAction('loop_invariant', position, 0.9,
                                                   {'loop_idx': outermost_idx}))
                break

        return actions

    def choose_action(self, state: str, actions: List[OptimizationAction]) -> Optional[OptimizationAction]:
        if not actions:
            return None
        self.q_table.setdefault(state, {})

        if random.random() < self.epsilon:
            return random.choice(actions)

        best_action, best_q = None, float('-inf')
        for action in actions:
            q = self.q_table[state].get(action.type, 0.0)   # BUG FIX #11: type-only key
            if q > best_q:
                best_q = q
                best_action = action
        return best_action or random.choice(actions)

    def apply_optimization(self, code: List[ThreeAddressCode], action: OptimizationAction,
                            loops: List[Dict[str, Any]]) -> List[ThreeAddressCode]:
        if action.type == 'constant_folding':
            return self.apply_constant_folding(code, action.position)
        elif action.type == 'dead_code_elimination':
            return self.apply_dead_code_elimination(code, action.position)
        elif action.type == 'common_subexpression':
            return self.apply_common_subexpression_elimination(code, action.position)
        elif action.type == 'strength_reduction':
            return self.apply_strength_reduction(code, action.position)
        elif action.type == 'loop_invariant':
            return self.apply_loop_invariant_motion(
                code, action.position, action.loop_info.get('loop_idx', 0), loops)
        return code

    # ------------------------------------------------------------------
    # Main optimise loop
    # BUG FIX #3 / #4: removed the broken fix_loop_control_flow entirely.
    # BUG FIX #9: re-analyse after every change so variable_usage is fresh.
    # ------------------------------------------------------------------
    def optimize(self, code: List[ThreeAddressCode],
                 max_steps: int = 50) -> Tuple[List[ThreeAddressCode], List[str]]:
        optimized_code = _deep_copy_code(code)
        optimization_log: List[str] = []

        self.analyze_code(optimized_code)
        loops = self.find_loops(optimized_code)

        step = 0
        changed = True

        while changed and step < max_steps:
            changed = False
            position = 0

            while position < len(optimized_code):
                state = self.get_state_key(optimized_code, position)
                actions = self.get_possible_actions(optimized_code, position, loops)

                if actions:
                    action = self.choose_action(state, actions)
                    if action:
                        old_code = optimized_code
                        new_code = self.apply_optimization(optimized_code, action, loops)

                        code_changed = (len(new_code) != len(old_code) or
                                        any(new_code[i] != old_code[i] for i in range(len(old_code))))

                        if code_changed:
                            reward = self.calculate_reward(old_code, new_code)
                            next_state = self.get_state_key(new_code, position)
                            self.update_q_value(state, action, reward, next_state)

                            action_key = action.type
                            q_val = self.q_table[state].get(action_key, 0.0)
                            optimized_code = new_code
                            optimization_log.append(
                                f"Applied {action.type} at position {position} "
                                f"(reward: {reward:.2f}, Q: {q_val:.2f})"
                            )

                            # BUG FIX #9: re-analyse immediately so is_variable_used_later
                            # uses fresh data for the rest of this pass
                            self.analyze_code(optimized_code)
                            loops = self.find_loops(optimized_code)
                            changed = True
                            self.epsilon = max(0.01, self.epsilon * 0.99)

                            if len(new_code) < len(old_code):
                                continue  # don't advance; same position may have a new instruction

                position += 1
            step += 1

        return optimized_code, optimization_log
