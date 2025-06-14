import json
import numpy as np
from typing import List, Dict, Tuple, Any, Set, Optional
from dataclasses import dataclass
from .intermediate_code import ThreeAddressCode
import pickle
import os

@dataclass
class GraphNode:
    """Represents a node in the code graph"""
    id: int
    node_type: str  # 'operation', 'variable', 'constant', 'control'
    value: str
    features: List[float]
    neighbors: List[int]
    metadata: Dict[str, Any]

@dataclass
class CodeGraph:
    """Represents the code as a graph structure"""
    nodes: List[GraphNode]
    edges: List[Tuple[int, int, str]]  # (source, target, edge_type)
    node_features: np.ndarray
    edge_features: np.ndarray
    adjacency_matrix: np.ndarray

@dataclass
class OptimizationPrediction:
    """Represents a predicted optimization"""
    optimization_type: str
    confidence: float
    target_nodes: List[int]
    parameters: Dict[str, Any]
    expected_benefit: float

class GNNOptimizer:
    """Graph Neural Network-based code optimizer"""
    
    def __init__(self, model_path: str = "gnn_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_dim = 16
        self.optimization_types = [
            'constant_folding',
            'dead_code_elimination', 
            'common_subexpression_elimination',
            'strength_reduction',
            'loop_unrolling',
            'instruction_scheduling',
            'function_inlining',
            'loop_invariant_motion'
        ]
        self.load_model()
        
    def load_model(self):
        """Load pre-trained GNN model from local file"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                # Create a mock model for demonstration
                self.model = self._create_mock_model()
                self._save_mock_model()
        except Exception as e:
            print(f"Warning: Could not load GNN model: {e}")
            self.model = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a mock GNN model for demonstration purposes"""
        return {
            'weights': {
                'node_embedding': np.random.randn(self.feature_dim, 32),
                'gnn_layer1': np.random.randn(32, 64),
                'gnn_layer2': np.random.randn(64, 32),
                'output_layer': np.random.randn(32, len(self.optimization_types))
            },
            'biases': {
                'gnn_layer1': np.random.randn(64),
                'gnn_layer2': np.random.randn(32),
                'output_layer': np.random.randn(len(self.optimization_types))
            },
            'version': '1.0'
        }
    
    def _save_mock_model(self):
        """Save the mock model to file"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as e:
            print(f"Warning: Could not save mock model: {e}")
    
    def build_code_graph(self, intermediate_code: List[ThreeAddressCode]) -> CodeGraph:
        """Convert intermediate code to graph representation"""
        nodes = []
        edges = []
        node_id = 0
        variable_nodes = {}
        
        # Create nodes for each instruction
        for i, instr in enumerate(intermediate_code):
            # Create operation node
            op_features = self._extract_operation_features(instr)
            op_node = GraphNode(
                id=node_id,
                node_type='operation',
                value=instr.operation,
                features=op_features,
                neighbors=[],
                metadata={'instruction_index': i, 'instruction': instr}
            )
            nodes.append(op_node)
            op_node_id = node_id
            node_id += 1
            
            # Create variable/constant nodes and edges
            for arg in [instr.arg1, instr.arg2, instr.result]:
                if arg is not None:
                    if self._is_constant(arg):
                        # Create constant node
                        const_features = self._extract_constant_features(arg)
                        const_node = GraphNode(
                            id=node_id,
                            node_type='constant',
                            value=arg,
                            features=const_features,
                            neighbors=[],
                            metadata={'value': arg}
                        )
                        nodes.append(const_node)
                        edges.append((const_node.id, op_node_id, 'uses'))
                        node_id += 1
                    else:
                        # Handle variable
                        if arg not in variable_nodes:
                            var_features = self._extract_variable_features(arg)
                            var_node = GraphNode(
                                id=node_id,
                                node_type='variable',
                                value=arg,
                                features=var_features,
                                neighbors=[],
                                metadata={'name': arg}
                            )
                            nodes.append(var_node)
                            variable_nodes[arg] = var_node.id
                            node_id += 1
                        
                        var_node_id = variable_nodes[arg]
                        if arg == instr.result:
                            edges.append((op_node_id, var_node_id, 'defines'))
                        else:
                            edges.append((var_node_id, op_node_id, 'uses'))
        
        # Add control flow edges
        self._add_control_flow_edges(nodes, edges, intermediate_code)
        
        # Build adjacency matrix and feature matrices
        num_nodes = len(nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        node_features = np.zeros((num_nodes, self.feature_dim))
        
        for i, node in enumerate(nodes):
            node_features[i] = node.features[:self.feature_dim]
        
        for source, target, edge_type in edges:
            adjacency_matrix[source][target] = 1
            # Update neighbor lists
            nodes[source].neighbors.append(target)
        
        edge_features = self._extract_edge_features(edges)
        
        return CodeGraph(
            nodes=nodes,
            edges=edges,
            node_features=node_features,
            edge_features=edge_features,
            adjacency_matrix=adjacency_matrix
        )
    
    def _extract_operation_features(self, instr: ThreeAddressCode) -> List[float]:
        """Extract features for operation nodes"""
        features = [0.0] * self.feature_dim
        
        # Operation type encoding
        op_types = {
            'ASSIGN': 0, 'PRINT': 1, 'SCAN': 2, 'LABEL': 3, 'GOTO': 4,
            'IF_FALSE': 5, '+': 6, '-': 7, '*': 8, '/': 9,
            '>': 10, '<': 11, '>=': 12, '<=': 13, '==': 14, '!=': 15
        }
        
        if instr.operation in op_types:
            features[0] = op_types[instr.operation] / len(op_types)
        
        # Arity (number of operands)
        arity = sum(1 for arg in [instr.arg1, instr.arg2] if arg is not None)
        features[1] = arity / 2.0
        
        # Has result
        features[2] = 1.0 if instr.result else 0.0
        
        # Computational complexity estimate
        complexity_map = {'+': 1, '-': 1, '*': 2, '/': 3, '==': 1, '!=': 1}
        features[3] = complexity_map.get(instr.operation, 0) / 3.0
        
        return features
    
    def _extract_variable_features(self, var_name: str) -> List[float]:
        """Extract features for variable nodes"""
        features = [0.0] * self.feature_dim
        features[0] = 0.5  # Variable type indicator
        features[1] = len(var_name) / 10.0  # Name length normalized
        features[2] = 1.0 if var_name.startswith('t') else 0.0  # Temporary variable
        return features
    
    def _extract_constant_features(self, const_value: str) -> List[float]:
        """Extract features for constant nodes"""
        features = [0.0] * self.feature_dim
        features[0] = 1.0  # Constant type indicator
        
        try:
            val = float(const_value.strip('"'))
            features[1] = min(abs(val) / 100.0, 1.0)  # Magnitude
            features[2] = 1.0 if val == int(val) else 0.0  # Is integer
        except:
            features[1] = len(const_value) / 20.0  # String length
            features[2] = 0.0
        
        return features
    
    def _add_control_flow_edges(self, nodes: List[GraphNode], edges: List[Tuple[int, int, str]], 
                               intermediate_code: List[ThreeAddressCode]):
        """Add control flow edges to the graph"""
        # Find labels and their positions
        labels = {}
        for i, instr in enumerate(intermediate_code):
            if instr.operation == 'LABEL':
                labels[instr.arg1] = i
        
        # Add control flow edges
        for i, instr in enumerate(intermediate_code):
            if instr.operation == 'GOTO':
                target_pos = labels.get(instr.arg1)
                if target_pos is not None:
                    edges.append((i, target_pos, 'control_flow'))
            elif instr.operation == 'IF_FALSE':
                target_pos = labels.get(instr.arg2)
                if target_pos is not None:
                    edges.append((i, target_pos, 'conditional_flow'))
                # Also add fall-through edge
                if i + 1 < len(intermediate_code):
                    edges.append((i, i + 1, 'fall_through'))
    
    def _extract_edge_features(self, edges: List[Tuple[int, int, str]]) -> np.ndarray:
        """Extract features for edges"""
        edge_types = {'uses': 0, 'defines': 1, 'control_flow': 2, 'conditional_flow': 3, 'fall_through': 4}
        edge_features = np.zeros((len(edges), 4))
        
        for i, (source, target, edge_type) in enumerate(edges):
            if edge_type in edge_types:
                edge_features[i][edge_types[edge_type]] = 1.0
        
        return edge_features
    
    def _is_constant(self, value: str) -> bool:
        """Check if a value is a constant"""
        if not isinstance(value, str):
            return False
        try:
            float(value)
            return True
        except ValueError:
            return value.startswith('"') and value.endswith('"')
    
    def gnn_inference(self, graph: CodeGraph) -> List[OptimizationPrediction]:
        """Perform GNN inference to predict optimizations"""
        if self.model is None:
            return []
        
        try:
            # Simulate GNN forward pass
            node_embeddings = self._forward_pass(graph)
            
            # Predict optimizations
            predictions = self._predict_optimizations(graph, node_embeddings)
            
            return predictions
        except Exception as e:
            print(f"Warning: GNN inference failed: {e}")
            return []
    
    def _forward_pass(self, graph: CodeGraph) -> np.ndarray:
        """Simulate GNN forward pass"""
        # Initial node embeddings
        h = np.dot(graph.node_features, self.model['weights']['node_embedding'])
        
        # GNN layers with message passing
        for layer in ['gnn_layer1', 'gnn_layer2']:
            # Message passing: aggregate neighbor features
            messages = np.zeros_like(h)
            for i, node in enumerate(graph.nodes):
                neighbor_features = []
                for neighbor_id in node.neighbors:
                    if neighbor_id < len(h):
                        neighbor_features.append(h[neighbor_id])
                
                if neighbor_features:
                    messages[i] = np.mean(neighbor_features, axis=0)
            
            # Update node representations
            combined = h + messages
            h = np.tanh(np.dot(combined, self.model['weights'][layer]) + self.model['biases'][layer])
        
        return h
    
    def _predict_optimizations(self, graph: CodeGraph, embeddings: np.ndarray) -> List[OptimizationPrediction]:
        """Predict optimizations from node embeddings"""
        predictions = []
        
        # Global graph representation (mean pooling)
        graph_embedding = np.mean(embeddings, axis=0)
        
        # Predict optimization scores
        scores = np.dot(graph_embedding, self.model['weights']['output_layer']) + self.model['biases']['output_layer']
        scores = 1 / (1 + np.exp(-scores))  # Sigmoid activation
        
        # Generate predictions for high-scoring optimizations
        for i, score in enumerate(scores):
            if score > 0.5:  # Threshold for applying optimization
                opt_type = self.optimization_types[i]
                target_nodes = self._find_optimization_targets(graph, opt_type, embeddings)
                
                prediction = OptimizationPrediction(
                    optimization_type=opt_type,
                    confidence=float(score),
                    target_nodes=target_nodes,
                    parameters=self._get_optimization_parameters(opt_type),
                    expected_benefit=float(score * 10)  # Estimated benefit
                )
                predictions.append(prediction)
        
        return sorted(predictions, key=lambda x: x.confidence, reverse=True)
    
    def _find_optimization_targets(self, graph: CodeGraph, opt_type: str, embeddings: np.ndarray) -> List[int]:
        """Find target nodes for specific optimization"""
        targets = []
        
        if opt_type == 'constant_folding':
            # Find arithmetic operations with constant operands
            for i, node in enumerate(graph.nodes):
                if node.node_type == 'operation' and node.value in ['+', '-', '*', '/']:
                    targets.append(i)
        
        elif opt_type == 'dead_code_elimination':
            # Find operations that don't affect output
            for i, node in enumerate(graph.nodes):
                if node.node_type == 'operation' and node.value not in ['PRINT', 'SCAN']:
                    # Simple heuristic: operations with low connectivity
                    if len(node.neighbors) < 2:
                        targets.append(i)
        
        elif opt_type == 'common_subexpression_elimination':
            # Find repeated operations
            operation_patterns = {}
            for i, node in enumerate(graph.nodes):
                if node.node_type == 'operation':
                    pattern = node.value
                    if pattern in operation_patterns:
                        targets.extend([operation_patterns[pattern], i])
                    else:
                        operation_patterns[pattern] = i
        
        return targets[:5]  # Limit to top 5 targets
    
    def _get_optimization_parameters(self, opt_type: str) -> Dict[str, Any]:
        """Get parameters for specific optimization"""
        params = {
            'constant_folding': {'aggressive': True},
            'dead_code_elimination': {'preserve_side_effects': True},
            'common_subexpression_elimination': {'scope': 'local'},
            'strength_reduction': {'target_operations': ['*', '/']},
            'loop_unrolling': {'max_iterations': 4},
            'instruction_scheduling': {'window_size': 8},
            'function_inlining': {'size_threshold': 50},
            'loop_invariant_motion': {'safety_check': True}
        }
        return params.get(opt_type, {})
    
    def apply_optimizations(self, intermediate_code: List[ThreeAddressCode], 
                          predictions: List[OptimizationPrediction]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Apply predicted optimizations to intermediate code"""
        optimized_code = intermediate_code.copy()
        optimization_log = []
        
        for prediction in predictions:
            if prediction.confidence > 0.6:  # Apply high-confidence optimizations
                try:
                    if prediction.optimization_type == 'constant_folding':
                        optimized_code, log = self._apply_constant_folding(optimized_code)
                        optimization_log.extend(log)
                    
                    elif prediction.optimization_type == 'dead_code_elimination':
                        optimized_code, log = self._apply_dead_code_elimination(optimized_code)
                        optimization_log.extend(log)
                    
                    elif prediction.optimization_type == 'common_subexpression_elimination':
                        optimized_code, log = self._apply_cse(optimized_code)
                        optimization_log.extend(log)
                    
                    elif prediction.optimization_type == 'strength_reduction':
                        optimized_code, log = self._apply_strength_reduction(optimized_code)
                        optimization_log.extend(log)
                    
                    # Add more optimization implementations as needed
                    
                except Exception as e:
                    optimization_log.append(f"Failed to apply {prediction.optimization_type}: {str(e)}")
        
        return optimized_code, optimization_log
    
    def _apply_constant_folding(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Apply constant folding optimization"""
        optimized_code = []
        log = []
        
        for instr in code:
            if instr.operation in ['+', '-', '*', '/'] and self._is_constant(instr.arg1) and self._is_constant(instr.arg2):
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
                            optimized_code.append(instr)
                            continue
                    
                    # Create new assignment instruction
                    result_str = str(int(result)) if result == int(result) else str(result)
                    new_instr = ThreeAddressCode('ASSIGN', result_str, None, instr.result)
                    optimized_code.append(new_instr)
                    log.append(f"GNN-guided constant folding: {instr.arg1} {instr.operation} {instr.arg2} = {result_str}")
                    
                except Exception:
                    optimized_code.append(instr)
            else:
                optimized_code.append(instr)
        
        return optimized_code, log
    
    def _apply_dead_code_elimination(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Apply dead code elimination"""
        # Build use-def chains
        used_vars = set()
        definitions = {}
        
        # Find all used variables
        for instr in code:
            if instr.operation in ['PRINT', 'SCAN', 'IF_FALSE']:
                if instr.arg1:
                    used_vars.add(instr.arg1)
                if instr.arg2:
                    used_vars.add(instr.arg2)
            
            for arg in [instr.arg1, instr.arg2]:
                if arg and not self._is_constant(arg):
                    used_vars.add(arg)
        
        # Mark definitions of used variables
        live_vars = used_vars.copy()
        changed = True
        while changed:
            changed = False
            for instr in code:
                if instr.result and instr.result in live_vars:
                    for arg in [instr.arg1, instr.arg2]:
                        if arg and not self._is_constant(arg) and arg not in live_vars:
                            live_vars.add(arg)
                            changed = True
        
        # Remove dead code
        optimized_code = []
        log = []
        
        for instr in code:
            if (instr.operation in ['PRINT', 'SCAN', 'LABEL', 'GOTO', 'IF_FALSE'] or
                (instr.result and instr.result in live_vars)):
                optimized_code.append(instr)
            else:
                log.append(f"GNN-guided dead code elimination: removed {instr}")
        
        return optimized_code, log
    
    def _apply_cse(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Apply common subexpression elimination"""
        optimized_code = []
        log = []
        expression_map = {}
        
        for instr in code:
            if instr.operation in ['+', '-', '*', '/']:
                expr_key = f"{instr.arg1}_{instr.operation}_{instr.arg2}"
                
                if expr_key in expression_map:
                    # Replace with assignment from previous result
                    prev_result = expression_map[expr_key]
                    new_instr = ThreeAddressCode('ASSIGN', prev_result, None, instr.result)
                    optimized_code.append(new_instr)
                    log.append(f"GNN-guided CSE: reused {prev_result} for {expr_key}")
                else:
                    expression_map[expr_key] = instr.result
                    optimized_code.append(instr)
            else:
                optimized_code.append(instr)
        
        return optimized_code, log
    
    def _apply_strength_reduction(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Apply strength reduction optimization"""
        optimized_code = []
        log = []
        
        for instr in code:
            if instr.operation == '*':
                # Replace multiplication by 2 with addition
                if (self._is_constant(instr.arg2) and float(instr.arg2) == 2) or \
                   (self._is_constant(instr.arg1) and float(instr.arg1) == 2):
                    
                    operand = instr.arg1 if self._is_constant(instr.arg2) else instr.arg2
                    new_instr = ThreeAddressCode('+', operand, operand, instr.result)
                    optimized_code.append(new_instr)
                    log.append(f"GNN-guided strength reduction: {operand} * 2 → {operand} + {operand}")
                else:
                    optimized_code.append(instr)
            else:
                optimized_code.append(instr)
        
        return optimized_code, log
    
    def optimize(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Main optimization function using GNN"""
        optimization_log = []
        
        try:
            # Build code graph
            graph = self.build_code_graph(code)
            optimization_log.append(f"GNN: Built code graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            
            # Perform GNN inference
            predictions = self.gnn_inference(graph)
            optimization_log.append(f"GNN: Generated {len(predictions)} optimization predictions")
            
            # Apply optimizations
            optimized_code, opt_log = self.apply_optimizations(code, predictions)
            optimization_log.extend(opt_log)
            
            # Log prediction details
            for pred in predictions:
                optimization_log.append(
                    f"GNN prediction: {pred.optimization_type} "
                    f"(confidence: {pred.confidence:.3f}, benefit: {pred.expected_benefit:.1f})"
                )
            
            return optimized_code, optimization_log
            
        except Exception as e:
            optimization_log.append(f"GNN optimization failed: {str(e)}")
            return code, optimization_log

# Maintain compatibility with existing interface
class RLOptimizer(GNNOptimizer):
    """Compatibility wrapper for the GNN optimizer"""
    
    def __init__(self):
        super().__init__()
        self.optimization_history = []
    
    def optimize(self, code: List[ThreeAddressCode]) -> Tuple[List[ThreeAddressCode], List[str]]:
        """Main optimization function - now uses GNN instead of RL"""
        optimized_code, log = super().optimize(code)
        
        # Update history for compatibility
        self.optimization_history.extend([(opt, 1.0) for opt in log])
        
        return optimized_code, log
