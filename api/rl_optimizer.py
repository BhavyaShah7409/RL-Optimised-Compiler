import random
import numpy as np
from typing import List, Dict, Tuple, Any, Set, Optional
from dataclasses import dataclass
from .intermediate_code import ThreeAddressCode

@dataclass
class OptimizationAction:
    type: str  # 'constant_folding', 'dead_code_elimination', 'common_subexpression', 'strength_reduction'
    position: int
    confidence: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.type} at position {self.position} (confidence: {self.confidence:.2f})"

class RLOptimizer:
    def __init__(self):
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # exploration rate
        self.optimization_history: List[Tuple[str, float]] = []
        self.variable_usage: Dict[str, Set[int]] = {}  # Track variable usage
    
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
    
    def analyze_variable_usage(self, code: List[ThreeAddressCode]) -> None:
        """Analyze where variables are used in the code"""
        self.variable_usage = {}
        
        # First pass: collect all variable definitions
        for i, instr in enumerate(code):
            if instr.result and instr.result not in ('LABEL', 'GOTO', 'IF_FALSE', 'PRINT'):
                if instr.result not in self.variable_usage:
                    self.variable_usage[instr.result] = set()
        
        # Second pass: collect all variable usages
        for i, instr in enumerate(code):
            # Check arg1
            if instr.arg1 and isinstance(instr.arg1, str) and not self.is_constant(instr.arg1):
                if instr.arg1 in self.variable_usage:
                    self.variable_usage[instr.arg1].add(i)
            
            # Check arg2
            if instr.arg2 and isinstance(instr.arg2, str) and not self.is_constant(instr.arg2):
                if instr.arg2 in self.variable_usage:
                    self.variable_usage[instr.arg2].add(i)
    
    def get_possible_actions(self, code: List[ThreeAddressCode], position: int) -> List[OptimizationAction]:
        """Get possible optimization actions for the current position"""
        actions = []
        
        if position >= len(code):
            return actions
        
        current = code[position]
        
        # Constant folding opportunities
        if current.operation in ['+', '-', '*', '/'] and current.arg1 and current.arg2:
            if self.is_constant(current.arg1) and self.is_constant(current.arg2):
                actions.append(OptimizationAction('constant_folding', position, 0.8))
        
        # Dead code elimination
        if current.result and not self.is_variable_used_later(code, position, current.result):
            # Don't eliminate code with side effects like PRINT
            if current.operation not in ['PRINT', 'SCAN']:
                actions.append(OptimizationAction('dead_code_elimination', position, 0.7))
        
        # Common subexpression elimination
        if current.operation in ['+', '-', '*', '/'] and current.arg1 and current.arg2:
            for i in range(position):
                prev = code[i]
                if (prev.operation == current.operation and 
                    prev.arg1 == current.arg1 and 
                    prev.arg2 == current.arg2 and
                    prev.result):  # Make sure there's a result to reuse
                    actions.append(OptimizationAction('common_subexpression', position, 0.6))
                    break
        
        # Strength reduction
        if current.operation == '*' and current.arg1 and current.arg2:
            # Replace multiplication by 2 with addition
            if (self.is_constant(current.arg1) and self.try_parse_float(current.arg1) == 2) or \
               (self.is_constant(current.arg2) and self.try_parse_float(current.arg2) == 2):
                actions.append(OptimizationAction('strength_reduction', position, 0.5))
            
            # Replace multiplication by power of 2 with shift
            if self.is_constant(current.arg2) and self.is_power_of_two(self.try_parse_float(current.arg2)):
                actions.append(OptimizationAction('strength_reduction', position, 0.5))
        
        return actions
    
    def is_constant(self, value: str) -> bool:
        """Check if a value is a constant"""
        if not isinstance(value, str):
            return False
            
        # Check if it's a numeric constant
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            # Check if it's a string constant
            return value.startswith('"') and value.endswith('"')
    
    def try_parse_float(self, value: str) -> Optional[float]:
        """Safely try to parse a string as float"""
        if not isinstance(value, str):
            return None
            
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def is_power_of_two(self, num: Optional[float]) -> bool:
        """Check if a number is a power of 2"""
        if num is None or num <= 0:
            return False
            
        # Check if it's an integer
        if num != int(num):
            return False
            
        num = int(num)
        return (num & (num - 1)) == 0
    
    def is_variable_used_later(self, code: List[ThreeAddressCode], position: int, variable: str) -> bool:
        """Check if a variable is used after the current position"""
        if variable not in self.variable_usage:
            return False
            
        # Check if there are any usages after the current position
        for usage_pos in self.variable_usage[variable]:
            if usage_pos > position:
                return True
                
        return False
    
    def choose_action(self, state: str, actions: List[OptimizationAction]) -> Optional[OptimizationAction]:
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
        elif action.type == 'strength_reduction':
            return self.apply_strength_reduction(code, action.position)
        
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
                result = None
                
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
                        return code  # Avoid division by zero
                
                if result is not None:
                    # Format result to avoid unnecessary decimal places
                    if result == int(result):
                        result_str = str(int(result))
                    else:
                        result_str = str(result)
                        
                    new_code = code.copy()
                    new_code[position] = ThreeAddressCode('ASSIGN', result_str, None, instr.result)
                    return new_code
            except Exception as e:
                # If any error occurs during constant folding, return original code
                return code
        
        return code
    
    def apply_dead_code_elimination(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        """Remove dead code"""
        if position >= len(code):
            return code
        
        # Don't remove code with side effects
        instr = code[position]
        if instr.operation in ['PRINT', 'SCAN', 'LABEL', 'GOTO', 'IF_FALSE']:
            return code
            
        # Don't remove code that defines variables used later
        if instr.result and self.is_variable_used_later(code, position, instr.result):
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
                prev.arg2 == current.arg2 and
                prev.result):  # Make sure there's a result to reuse
                
                # Check that the previous result hasn't been modified
                modified = False
                for j in range(i + 1, position):
                    if code[j].result == prev.result:
                        modified = True
                        break
                
                if not modified:
                    new_code = code.copy()
                    new_code[position] = ThreeAddressCode('ASSIGN', prev.result, None, current.result)
                    return new_code
        
        return code
    
    def apply_strength_reduction(self, code: List[ThreeAddressCode], position: int) -> List[ThreeAddressCode]:
        """Apply strength reduction optimizations"""
        if position >= len(code):
            return code
        
        instr = code[position]
        new_code = code.copy()
        
        # Replace multiplication by 2 with addition
        if instr.operation == '*':
            if self.is_constant(instr.arg2) and self.try_parse_float(instr.arg2) == 2:
                # x * 2 -> x + x
                new_code[position] = ThreeAddressCode('+', instr.arg1, instr.arg1, instr.result)
                return new_code
            elif self.is_constant(instr.arg1) and self.try_parse_float(instr.arg1) == 2:
                # 2 * x -> x + x
                new_code[position] = ThreeAddressCode('+', instr.arg2, instr.arg2, instr.result)
                return new_code
            
            # Replace multiplication by power of 2 with shift
            if self.is_constant(instr.arg2):
                val = self.try_parse_float(instr.arg2)
                if val and self.is_power_of_two(val):
                    # Calculate log2(val) to get shift amount
                    shift_amount = int(np.log2(val))
                    # We can't directly represent shifts in our IR, so we'll add a comment
                    # In a real compiler, we'd use a SHIFT operation
                    new_code[position] = ThreeAddressCode('*', instr.arg1, str(val), instr.result)
                    new_code[position].operation = f"* (shift left by {shift_amount})"
                    return new_code
        
        return code
    
    def calculate_reward(self, original_code: List[ThreeAddressCode], optimized_code: List[ThreeAddressCode]) -> float:
        """Calculate reward for the optimization"""
        # Base reward for code size reduction
        size_reduction = len(original_code) - len(optimized_code)
        reward = size_reduction * 10
        
        # Bonus for specific optimizations
        for i, instr in enumerate(optimized_code):
            # Reward for constant folding
            if instr.operation == 'ASSIGN' and self.is_constant(instr.arg1):
                reward += 5
                
            # Reward for strength reduction
            if isinstance(instr.operation, str) and "shift" in instr.operation:
                reward += 8
        
        # Penalty for making code worse
        if len(optimized_code) > len(original_code):
            reward -= 20
        
        return reward
    
    def update_q_value(self, state: str, action: OptimizationAction, reward: float, next_state: str) -> None:
        """Update Q-value using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        action_key = f"{action.type}_{action.position}"
        current_q = self.q_table[state].get(action_key, 0.0)
        
        # Get max Q-value for next state
        max_next_q = 0.0
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_key] = new_q
        
        # Update action confidence based on Q-value
        action.confidence = new_q
    
    def optimize(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Main optimization function using reinforcement learning"""
        optimized_code = code.copy()
        optimization_log = []
        
        # Analyze variable usage for dead code elimination
        self.analyze_variable_usage(optimized_code)
        
        max_iterations = 20  # Increased from 10 to allow more optimizations
        iteration = 0
        
        while iteration < max_iterations:
            improved = False
            
            # Try to apply optimizations at each position
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
                
                # Check if code actually changed
                if len(new_code) != len(optimized_code) or any(new_code[i] != optimized_code[i] for i in range(min(len(new_code), len(optimized_code)))):
                    # Calculate reward
                    reward = self.calculate_reward(optimized_code, new_code)
                    
                    # Update Q-value
                    next_state = self.get_state_key(new_code, min(position, len(new_code) - 1) if new_code else 0)
                    self.update_q_value(state, action, reward, next_state)
                    
                    # Accept improvement
                    if reward > 0:
                        optimized_code = new_code
                        optimization_log.append(f"Applied {action.type} at position {action.position} (reward: {reward:.2f})")
                        
                        # Re-analyze variable usage after code changes
                        self.analyze_variable_usage(optimized_code)
                        
                        improved = True
                        break
            
            if not improved:
                break
            
            iteration += 1
        
        # Decay epsilon for less exploration over time
        self.epsilon = max(0.01, self.epsilon * 0.95)
        
        return optimized_code, optimization_log
