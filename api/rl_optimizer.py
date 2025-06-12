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
        """Generate a state key based on the current code context"""
        if position >= len(code):
            return "END"
        
        current = code[position]
        context = []
        
        # Add current instruction type
        context.append(current.operation)
        
        # Add context from previous and next instructions
        if position > 0:
            context.append(f"prev_{code[position-1].operation}")
        if position < len(code) - 1:
            context.append(f"next_{code[position+1].operation}")
        
        return "_".join(context)
    
    def get_possible_actions(self, code: List[ThreeAddressCode], position: int) -> List[OptimizationAction]:
        """Get possible optimization actions for the current position"""
        actions = []
        
        if position >= len(code):
            return actions
        
        current = code[position]
        
        # Constant folding opportunities
        if current.operation in ['+', '-', '*', '/'] and current.arg1 and current.arg2:
            if self.is_constant(current.arg1) and self.is_constant(current.arg2):
                actions.append(OptimizationAction('constant_folding', position))
        
        # Dead code elimination
        if current.result and not self.is_variable_used_later(code, position, current.result):
            actions.append(OptimizationAction('dead_code_elimination', position))
        
        # Common subexpression elimination
        if current.operation in ['+', '-', '*', '/']:
            for i in range(position):
                prev = code[i]
                if (prev.operation == current.operation and 
                    prev.arg1 == current.arg1 and 
                    prev.arg2 == current.arg2):
                    actions.append(OptimizationAction('common_subexpression', position))
                    break
        
        return actions
    
    def is_constant(self, value: str) -> bool:
        """Check if a value is a constant"""
        try:
            float(value)
            return True
        except ValueError:
            return value.startswith('"') and value.endswith('"')
    
    def is_variable_used_later(self, code: List[ThreeAddressCode], position: int, variable: str) -> bool:
        """Check if a variable is used after the current position"""
        for i in range(position + 1, len(code)):
            instr = code[i]
            if instr.arg1 == variable or instr.arg2 == variable:
                return True
        return False
    
    def choose_action(self, state: str, actions: List[OptimizationAction]) -> OptimizationAction:
        """Choose an action using epsilon-greedy strategy"""
        if not actions:
            return None
        
        if state not in self.q_table:
            self.q_table[state] = {}
        
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Choose best action based on Q-values
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
        """Apply the chosen optimization action"""
        if action.type == 'constant_folding':
            return self.apply_constant_folding(code, action.position)
        elif action.type == 'dead_code_elimination':
            return self.apply_dead_code_elimination(code, action.position)
        elif action.type == 'common_subexpression':
            return self.apply_common_subexpression_elimination(code, action.position)
        
        return code
    
    def apply_constant_folding(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        """Apply constant folding optimization"""
        if position >= len(code):
            return code
        
        instr = code[position]
        if instr.operation in ['+', '-', '*', '/'] and self.is_constant(instr.arg1) and self.is_constant(instr.arg2):
            try:
                val1 = float(instr.arg1)
                val2 = float(instr.arg2)
                
                if instr.operation == '+':
                    result = val1 + val2
                elif instr.operation == '-':
                    result = val1 - val2
                elif instr.operation == '*':
                    result = val1 * val2
                elif instr.operation == '/':
                    if val2 != 0:
                        result = val1 / val2
                    else:
                        return code
                
                new_code = code.copy()
                new_code[position] = ThreeAddressCode('ASSIGN', str(result), None, instr.result)
                return new_code
            except:
                pass
        
        return code
    
    def apply_dead_code_elimination(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        """Remove dead code"""
        if position >= len(code):
            return code
        
        new_code = code.copy()
        del new_code[position]
        return new_code
    
    def apply_common_subexpression_elimination(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        """Eliminate common subexpressions"""
        if position >= len(code):
            return code
        
        current = code[position]
        for i in range(position):
            prev = code[i]
            if (prev.operation == current.operation and 
                prev.arg1 == current.arg1 and 
                prev.arg2 == current.arg2):
                new_code = code.copy()
                new_code[position] = ThreeAddressCode('ASSIGN', prev.result, None, current.result)
                return new_code
        
        return code
    
    def calculate_reward(self, original_code: List[ThreeAddressCode], optimized_code: List[ThreeAddressCode], 
                        execution_time: float = None) -> float:
        """Calculate reward for the optimization"""
        # Base reward for code size reduction
        size_reduction = len(original_code) - len(optimized_code)
        reward = size_reduction * 10
        
        # Bonus for execution time improvement (if available)
        if execution_time is not None:
            # Simulate execution time improvement
            time_improvement = max(0, 1.0 - execution_time)
            reward += time_improvement * 50
        
        # Penalty for making code worse
        if len(optimized_code) > len(original_code):
            reward -= 20
        
        return reward
    
    def update_q_value(self, state: str, action: OptimizationAction, reward: float, next_state: str):
        """Update Q-value using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        action_key = f"{action.type}_{action.position}"
        current_q = self.q_table[state].get(action_key, 0.0)
        
        # Get max Q-value for next state
        max_next_q = 0.0
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_key] = new_q
    
    def optimize(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Main optimization function using reinforcement learning"""
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
                
                # Apply optimization
                new_code = self.apply_optimization(optimized_code, action)
                
                # Calculate reward
                reward = self.calculate_reward(optimized_code, new_code)
                
                # Update Q-value
                next_state = self.get_state_key(new_code, position)
                self.update_q_value(state, action, reward, next_state)
                
                # Accept improvement
                if reward > 0:
                    optimized_code = new_code
                    optimization_log.append(f"Applied {action.type} at position {action.position} (reward: {reward:.2f})")
                    improved = True
                    break
            
            if not improved:
                break
            
            iteration += 1
        
        # Decay epsilon for less exploration over time
        self.epsilon = max(0.01, self.epsilon * 0.95)
        
        return optimized_code, optimization_log
