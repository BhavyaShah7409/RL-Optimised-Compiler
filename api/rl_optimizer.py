import random
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from .intermediate_code import ThreeAddressCode

@dataclass
class OptimizationAction:
    type: str  # 'constant_folding', 'dead_code_elimination', 'common_subexpression'
    position: int
    confidence: float = 0.0

class RLOptimizer:
    def __init__(self):
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # exploration rate
        self.optimization_history: List[Tuple[str, float]] = []

    def get_state_key(self, code: List[ThreeAddressCode], position: int) -> str:
        if position >= len(code):
            return "END"
        current = code[position]
        context = [current.operation]
        if position > 0:
            context.append(f"prev_{code[position-1].operation}")
        if position < len(code) - 1:
            context.append(f"next_{code[position+1].operation}")
        return "_".join(context)

    def is_constant(self, value: str) -> bool:
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return value.isnumeric() or (value.startswith('"') and value.endswith('"'))

    def is_variable_used_later(self, code: List[ThreeAddressCode], position: int, variable: str) -> bool:
        for i in range(position + 1, len(code)):
            instr = code[i]
            if instr.arg1 == variable or instr.arg2 == variable:
                return True
        return False

    def get_possible_actions(self, code: List[ThreeAddressCode], position: int) -> List[OptimizationAction]:
        print(f"[DEBUG] Checking instruction at position {position}: {code[position]}")
        actions = []
        if position >= len(code):
            return actions
        current = code[position]

        if current.operation in ['+', '-', '*', '/'] and current.arg1 and current.arg2:
            if self.is_constant(current.arg1) and self.is_constant(current.arg2):
                actions.append(OptimizationAction('constant_folding', position))

        if current.result and not self.is_variable_used_later(code, position, current.result):
            actions.append(OptimizationAction('dead_code_elimination', position))

        if current.operation in ['+', '-', '*', '/']:
            for i in range(position):
                prev = code[i]
                if (prev.operation == current.operation and 
                    prev.arg1 == current.arg1 and 
                    prev.arg2 == current.arg2):
                    actions.append(OptimizationAction('common_subexpression', position))
                    break

        return actions

    def choose_action(self, state: str, actions: List[OptimizationAction]) -> OptimizationAction:
        if not actions:
            return None
        if state not in self.q_table:
            self.q_table[state] = {}
        if random.random() < self.epsilon:
            return random.choice(actions)

        best_action = None
        best_value = float('-inf')
        for action in actions:
            action_key = f"{action.type}_{action.position}"
            q_value = self.q_table[state].get(action_key, 0.0)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action or random.choice(actions)

    def apply_optimization(self, code: List[ThreeAddressCode], action: OptimizationAction) -> List[ThreeAddressCode]:
        if action.type == 'constant_folding':
            return self.apply_constant_folding(code, action.position)
        elif action.type == 'dead_code_elimination':
            return self.apply_dead_code_elimination(code, action.position)
        elif action.type == 'common_subexpression':
            return self.apply_common_subexpression_elimination(code, action.position)
        return code

    def apply_constant_folding(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
    """Apply constant folding optimization to 3AC if both arguments are constants."""
    if position >= len(code):
        return code

    instr = code[position]
    op = instr.operation
    arg1 = instr.arg1
    arg2 = instr.arg2

    if op not in ['+', '-', '*', '/']:
        return code

    if not (self.is_constant(arg1) and self.is_constant(arg2)):
        return code

    try:
        val1 = float(arg1)
        val2 = float(arg2)

        if op == '+':
            result = val1 + val2
        elif op == '-':
            result = val1 - val2
        elif op == '*':
            result = val1 * val2
        elif op == '/':
            if val2 == 0:
                print(f"[DEBUG] Skipped division by zero at position {position}")
                return code
            result = val1 / val2
        else:
            return code

        # Replace the original instruction with an ASSIGN of the folded value
        new_instr = ThreeAddressCode('ASSIGN', str(result), None, instr.result)
        new_code = code.copy()
        new_code[position] = new_instr

        print(f"[DEBUG] Constant Folding at position {position}: {arg1} {op} {arg2} → {result}")
        return new_code

    except Exception as e:
        print(f"[DEBUG] Folding error at position {position}: {e}")
        return code


    def apply_dead_code_elimination(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        if position >= len(code):
            return code
        print(f"[DEBUG] Dead Code Elimination at position {position}")
        new_code = code.copy()
        del new_code[position]
        return new_code

   def apply_common_subexpression_elimination(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
    """Eliminate common subexpressions more effectively using hash mapping"""
    if position >= len(code):
        return code

    current = code[position]
    key = (current.operation, current.arg1, current.arg2)

    for i in range(position):
        prev = code[i]
        prev_key = (prev.operation, prev.arg1, prev.arg2)
        if key == prev_key and prev.result:
            print(f"[DEBUG] Common Subexpression Elimination: Found duplicate of instruction at position {i} for position {position}")
            new_code = code.copy()
            new_code[position] = ThreeAddressCode('ASSIGN', prev.result, None, current.result)
            return new_code

    return code

    def calculate_reward(self, original_code: List[ThreeAddressCode], optimized_code: List[ThreeAddressCode], execution_time: float = None) -> float:
        size_reduction = len(original_code) - len(optimized_code)
        reward = size_reduction * 10
        if execution_time is not None:
            time_improvement = max(0, 1.0 - execution_time)
            reward += time_improvement * 50
        if len(optimized_code) > len(original_code):
            reward -= 20
        return reward

    def update_q_value(self, state: str, action: OptimizationAction, reward: float, next_state: str):
        if state not in self.q_table:
            self.q_table[state] = {}
        action_key = f"{action.type}_{action.position}"
        current_q = self.q_table[state].get(action_key, 0.0)
        max_next_q = max(self.q_table.get(next_state, {}).values(), default=0.0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_key] = new_q

    def optimize(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        optimized_code = code.copy()
        optimization_log = []
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            improved = False
            for position in range(len(optimized_code)):
                state = self.get_state_key(optimized_code, position)
                actions = self.get_possible_actions(optimized_code, position)
                if not actions:
                    continue
                action = self.choose_action(state, actions)
                if not action:
                    continue
                new_code = self.apply_optimization(optimized_code, action)
                reward = self.calculate_reward(optimized_code, new_code)
                next_state = self.get_state_key(new_code, position)
                self.update_q_value(state, action, reward, next_state)

                if reward > 0:
                    print(f"[DEBUG] Reward: {reward:.2f}, Applied: {action.type} at {action.position}")
                    optimized_code = new_code
                    optimization_log.append(f"Applied {action.type} at position {action.position} (reward: {reward:.2f})")
                    improved = True
                    break
            if not improved:
                break
            iteration += 1
        self.epsilon = max(0.01, self.epsilon * 0.95)
        return optimized_code, optimization_log
