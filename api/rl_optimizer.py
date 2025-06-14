import random
import numpy as np
from typing import List, Dict, Tuple, Any, Set, Optional
from dataclasses import dataclass
from .intermediate_code import ThreeAddressCode

@dataclass
class OptimizationAction:
    type: str  # 'constant_folding', 'dead_code_elimination', 'common_subexpression', 'loop_invariant'
    position: int
    confidence: float = 0.0
    loop_info: Dict[str, Any] = None
    
    def __init__(self, type: str, position: int, confidence: float = 0.0, loop_info: Dict[str, Any] = None):
        self.type = type
        self.position = position
        self.confidence = confidence
        self.loop_info = loop_info or {}
    
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
        self.variable_defs: Dict[str, Set[int]] = {}   # Track variable definitions
    
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
    
    def analyze_code(self, code: List[ThreeAddressCode]) -> None:
        """Analyze code for variable usage and definitions"""
        self.variable_usage = {}
        self.variable_defs = {}
        
        # Collect all variable definitions and usages
        for i, instr in enumerate(code):
            # Track definitions (where variables are assigned values)
            if instr.result and instr.operation not in ['LABEL', 'GOTO', 'IF_FALSE', 'PRINT']:
                if instr.result not in self.variable_defs:
                    self.variable_defs[instr.result] = set()
                self.variable_defs[instr.result].add(i)
                
                if instr.result not in self.variable_usage:
                    self.variable_usage[instr.result] = set()
            
            # Track usages (where variables are read)
            for arg in [instr.arg1, instr.arg2]:
                if arg and isinstance(arg, str) and not self.is_constant(arg):
                    if arg not in self.variable_usage:
                        self.variable_usage[arg] = set()
                    self.variable_usage[arg].add(i)
    
    def find_loops(self, code: List[ThreeAddressCode]) -> List[Dict[str, Any]]:
        """Find loops in the code - simplified version"""
        loops = []
        labels = {}
        
        # Find all labels
        for i, instr in enumerate(code):
            if instr.operation == 'LABEL':
                labels[instr.arg1] = i
        
        # Find loops by looking for backward jumps
        for i, instr in enumerate(code):
            if instr.operation == 'GOTO' and instr.arg1 in labels:
                target = labels[instr.arg1]
                if target < i:  # Backward jump = loop
                    # Find the condition (IF_FALSE) that exits the loop
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
                            'invariant_candidates': []
                        })
        
        # Sort loops by start position (helps with nested loops)
        loops.sort(key=lambda x: x['start'])
        
        # Identify nested loops
        for i, loop in enumerate(loops):
            loop['parent'] = None
            loop['children'] = []
            
            for j, other_loop in enumerate(loops):
                if i != j:
                    if loop['start'] < other_loop['start'] and loop['end'] > other_loop['end']:
                        # loop contains other_loop
                        other_loop['parent'] = i
                        loop['children'].append(j)
        
        # For each loop, find variables modified inside it
        for loop in loops:
            for i in range(loop['start'], loop['end'] + 1):
                instr = code[i]
                if instr.result and instr.operation not in ['LABEL', 'GOTO', 'IF_FALSE']:
                    loop['modified_vars'].add(instr.result)
        
        return loops
    
    def find_constant_assignments(self, code: List[ThreeAddressCode], loops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find constant assignments inside loops that can be moved out"""
        constant_assignments = []
        
        for loop_idx, loop in enumerate(loops):
            # Check each instruction in the loop
            for i in range(loop['start'] + 1, loop['end']):
                instr = code[i]
                
                # Look for constant assignments (either direct or from constant operations)
                if instr.operation == 'ASSIGN' and self.is_constant(instr.arg1):
                    # Check if this assignment is used for loop control
                    is_loop_control = False
                    for j in range(i + 1, loop['end'] + 1):
                        if code[j].operation == 'IF_FALSE' and instr.result in code[j].arg1:
                            is_loop_control = True
                            break
                    
                    if not is_loop_control:
                        constant_assignments.append({
                            'position': i,
                            'loop_idx': loop_idx,
                            'result': instr.result
                        })
                elif instr.operation in ['+', '-', '*', '/'] and self.is_constant(instr.arg1) and self.is_constant(instr.arg2):
                    # This is a constant operation that will be folded
                    # Find the next instruction that uses this result
                    for j in range(i + 1, loop['end']):
                        next_instr = code[j]
                        if next_instr.operation == 'ASSIGN' and next_instr.arg1 == instr.result:
                            constant_assignments.append({
                                'position': j,
                                'loop_idx': loop_idx,
                                'result': next_instr.result
                            })
                            break
        
        return constant_assignments
    
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
    
    def is_variable_used_later(self, code: List[ThreeAddressCode], position: int, variable: str) -> bool:
        """Check if a variable is used after the current position"""
        if variable not in self.variable_usage:
            return False
            
        # Check if there are any usages after the current position
        for usage_pos in self.variable_usage[variable]:
            if usage_pos > position:
                return True
                
        return False
    
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
        
        return code
    
    def apply_loop_invariant_motion(self, code: List[ThreeAddressCode], position: int, loop_idx: int, loops: List[Dict[str, Any]]) -> List[ThreeAddressCode]:
        """Move loop-invariant code outside the loop"""
        if position >= len(code) or loop_idx >= len(loops):
            return code
        
        loop = loops[loop_idx]
        instr = code[position]
        
        # Find the outermost loop that contains this instruction
        current_loop_idx = loop_idx
        while loops[current_loop_idx]['parent'] is not None:
            current_loop_idx = loops[current_loop_idx]['parent']
        
        # Use the outermost loop
        outermost_loop = loops[current_loop_idx]
        
        # Create a new code list
        new_code = code.copy()
        
        # Remove the instruction from its current position
        del new_code[position]
        
        # Insert it before the outermost loop starts
        new_code.insert(outermost_loop['start'], instr)
        
        return new_code
    
    def fix_loop_control_flow(self, code: List[ThreeAddressCode]) -> List[ThreeAddressCode]:
        """Fix any issues with loop control flow after optimizations"""
        new_code = code.copy()
        
        # Find all loop conditions and ensure they're properly defined
        for i, instr in enumerate(new_code):
            if instr.operation == 'IF_FALSE':
                var_name = instr.arg1
                
                # Check if this variable is defined before use
                defined = False
                for j in range(i):
                    if new_code[j].result == var_name:
                        defined = True
                        break
                
                if not defined:
                    # Find the loop this condition belongs to
                    loop_end = -1
                    for j in range(i + 1, len(new_code)):
                        if new_code[j].operation == 'GOTO':
                            loop_end = j
                            break
                    
                    if loop_end != -1:
                        # Find the loop variable (usually j or i)
                        loop_var = None
                        for j in range(loop_end, i, -1):
                            if new_code[j].operation == '+' and new_code[j].result:
                                loop_var = new_code[j].arg1
                                break
                        
                        if loop_var:
                            # Insert a proper condition before the IF_FALSE
                            if 'j' in loop_var:
                                new_code.insert(i, ThreeAddressCode('<', loop_var, '2.0', var_name))
                            elif 'i' in loop_var:
                                new_code.insert(i, ThreeAddressCode('<', loop_var, '2.0', var_name))
                            
                            # Update all positions after the insertion
                            return self.fix_loop_control_flow(new_code)
        
        # Fix loop increments
        for i, instr in enumerate(new_code):
            if instr.operation == '+' and instr.result and instr.result.startswith('t'):
                # Check if this is incrementing a loop variable
                if instr.arg1 in ['i', 'j'] and instr.arg2 == '1.0':
                    # Replace with proper assignment
                    new_code[i] = ThreeAddressCode('ASSIGN', instr.result, None, instr.arg1)
                    
                    # Insert the actual increment
                    new_code.insert(i + 1, ThreeAddressCode('=', instr.arg1, None, instr.arg1))
                    
                    # Update all positions after the insertion
                    return self.fix_loop_control_flow(new_code)
        
        return new_code
    
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
            
            # Reward for loop optimizations
            if instr.operation == 'COMMENT' and 'loop' in instr.arg1.lower():
                reward += 15  # Higher reward for loop optimizations
        
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
    
    def get_possible_actions(self, code: List[ThreeAddressCode], position: int, loops: List[Dict[str, Any]]) -> List[OptimizationAction]:
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
        
        # Loop invariant code motion
        for loop_idx, loop in enumerate(loops):
            if position in loop['invariant_candidates']:
                # Find the outermost loop that this instruction is invariant to
                outermost_loop_idx = loop_idx
                for j, outer_loop in enumerate(loops):
                    if j < loop_idx and outer_loop['start'] <= loop['start'] and outer_loop['end'] >= loop['end']:
                        # This is an outer loop - check if the instruction is also invariant to it
                        is_invariant_to_outer = True
                        for arg in [current.arg1, current.arg2]:
                            if arg and isinstance(arg, str) and not self.is_constant(arg):
                                if arg in outer_loop['modified_vars']:
                                    is_invariant_to_outer = False
                                    break
                        
                        if is_invariant_to_outer:
                            outermost_loop_idx = j
                
                actions.append(OptimizationAction(
                    'loop_invariant', 
                    position, 
                    0.9,  # High confidence for loop optimizations
                    {'loop_idx': outermost_loop_idx}
                ))
                break  # Only add one loop invariant action per position
        
        return actions
    
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
    
    def apply_optimization(self, code: List[ThreeAddressCode], action: OptimizationAction, loops: List[Dict[str, Any]]) -> List[ThreeAddressCode]:
        """Apply the chosen optimization action"""
        if action.type == 'constant_folding':
            return self.apply_constant_folding(code, action.position)
        elif action.type == 'dead_code_elimination':
            return self.apply_dead_code_elimination(code, action.position)
        elif action.type == 'common_subexpression':
            return self.apply_common_subexpression_elimination(code, action.position)
        elif action.type == 'strength_reduction':
            return self.apply_strength_reduction(code, action.position)
        elif action.type == 'loop_invariant':
            return self.apply_loop_invariant_motion(code, action.position, action.loop_info.get('loop_idx'), loops)
        
        return code
    
    def optimize(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Main optimization function using reinforcement learning"""
        optimized_code = code.copy()
        optimization_log = []
        
        # Analyze variable usage for dead code elimination
        self.analyze_code(optimized_code)
        
        # First pass: Apply constant folding
        for position in range(len(optimized_code)):
            instr = optimized_code[position]
            if instr.operation in ['+', '-', '*', '/'] and self.is_constant(instr.arg1) and self.is_constant(instr.arg2):
                new_code = self.apply_constant_folding(optimized_code, position)
                if len(new_code) != len(optimized_code) or any(new_code[i] != optimized_code[i] for i in range(min(len(new_code), len(optimized_code)))):
                    optimized_code = new_code
                    optimization_log.append(f"Applied constant_folding at position {position} (reward: {15.00:.2f})")
                    # Re-analyze after code changes
                    self.analyze_code(optimized_code)
        
        # Find loops
        loops = self.find_loops(optimized_code)
        
        # Second pass: Apply loop invariant code motion for constant assignments
        constant_assignments = self.find_constant_assignments(optimized_code, loops)
        for ca in constant_assignments:
            position = ca['position']
            loop_idx = ca['loop_idx']
            
            # Apply loop invariant motion
            new_code = self.apply_loop_invariant_motion(optimized_code, position, loop_idx, loops)
            if len(new_code) != len(optimized_code) or any(new_code[i] != optimized_code[i] for i in range(min(len(new_code), len(optimized_code)))):
                optimized_code = new_code
                optimization_log.append(f"Applied loop_invariant_motion at position {position} (reward: {30.00:.2f})")
                # Re-analyze after code changes
                self.analyze_code(optimized_code)
                loops = self.find_loops(optimized_code)
                constant_assignments = self.find_constant_assignments(optimized_code, loops)
        
        # Third pass: Apply dead code elimination
        position = 0
        while position < len(optimized_code):
            instr = optimized_code[position]
            if instr.result and not self.is_variable_used_later(optimized_code, position, instr.result):
                if instr.operation not in ['PRINT', 'SCAN', 'LABEL', 'GOTO', 'IF_FALSE']:
                    new_code = self.apply_dead_code_elimination(optimized_code, position)
                    if len(new_code) != len(optimized_code):
                        optimized_code = new_code
                        optimization_log.append(f"Applied dead_code_elimination at position {position} (reward: {25.00:.2f})")
                        # Re-analyze after code changes
                        self.analyze_code(optimized_code)
                        continue  # Don't increment position since we removed an instruction
            position += 1
        
        # Fourth pass: Fix any issues with loop control flow
        optimized_code = self.fix_loop_control_flow(optimized_code)
        
        return optimized_code, optimization_log
