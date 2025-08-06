"""
Step 16: Logic Flow Analysis
Advanced analysis of control flow patterns and summary logic creation for PLC systems.

This module provides comprehensive logic flow analysis including:
- Control flow pattern detection and classification
- Logic path analysis and execution flow mapping
- Conditional logic analysis and branch detection  
- Loop detection and iteration pattern analysis
- Logic summarization and simplification
- Critical path identification and bottleneck analysis
- Logic optimization recommendations
- Flow control validation and safety analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from enum import Enum
import re
import logging
from collections import defaultdict, deque
import networkx as nx
from pathlib import Path

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Types of control flow patterns."""
    SEQUENTIAL = "SEQUENTIAL"           # Linear execution
    CONDITIONAL = "CONDITIONAL"         # IF-THEN-ELSE patterns
    LOOP = "LOOP"                      # Iterative patterns
    PARALLEL = "PARALLEL"              # Concurrent execution
    INTERLOCK = "INTERLOCK"            # Safety interlocks
    STATE_MACHINE = "STATE_MACHINE"     # State-based logic
    ALARM = "ALARM"                    # Alarm/fault handling
    TIMER_CHAIN = "TIMER_CHAIN"        # Sequential timing
    COUNTER_CHAIN = "COUNTER_CHAIN"    # Sequential counting


class LogicPattern(Enum):
    """Common logic patterns in PLC systems."""
    START_STOP = "START_STOP"          # Motor start/stop logic
    SEAL_IN = "SEAL_IN"                # Self-holding logic
    EDGE_DETECT = "EDGE_DETECT"        # Rising/falling edge
    DEBOUNCE = "DEBOUNCE"              # Input debouncing
    SEQUENCER = "SEQUENCER"            # Sequential operations
    SAFETY_CHAIN = "SAFETY_CHAIN"     # Safety system logic
    ANALOG_SCALING = "ANALOG_SCALING"   # Analog input processing
    ALARM_LOGIC = "ALARM_LOGIC"        # Alarm detection
    PERMISSIVE = "PERMISSIVE"          # Permissive conditions


class ExecutionPath(Enum):
    """Types of execution paths."""
    MAIN_PATH = "MAIN_PATH"            # Primary execution flow
    ALTERNATE_PATH = "ALTERNATE_PATH"   # Alternative execution
    ERROR_PATH = "ERROR_PATH"          # Error handling flow
    SAFETY_PATH = "SAFETY_PATH"        # Safety shutdown flow
    STARTUP_PATH = "STARTUP_PATH"      # System startup flow
    SHUTDOWN_PATH = "SHUTDOWN_PATH"    # System shutdown flow


@dataclass
class LogicCondition:
    """Represents a logical condition in the flow."""
    condition_type: str  # XIC, XIO, EQU, etc.
    operands: List[str]
    expression: str
    routine_name: str
    rung_number: int
    is_negated: bool = False
    complexity_score: float = 1.0
    
    @property
    def readable_expression(self) -> str:
        """Get human-readable expression."""
        if self.condition_type in ['XIC', 'XIO']:
            tag = self.operands[0] if self.operands else 'UNKNOWN'
            return f"NOT {tag}" if self.condition_type == 'XIO' else tag
        elif self.condition_type in ['EQU', 'NEQ', 'GRT', 'LES', 'GEQ', 'LEQ']:
            if len(self.operands) >= 2:
                op_map = {'EQU': '==', 'NEQ': '!=', 'GRT': '>', 'LES': '<', 'GEQ': '>=', 'LEQ': '<='}
                return f"{self.operands[0]} {op_map.get(self.condition_type, '?')} {self.operands[1]}"
        return self.expression


@dataclass
class LogicAction:
    """Represents an action in the logic flow."""
    action_type: str  # OTE, OTL, OTU, MOV, etc.
    targets: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    routine_name: str = ""
    rung_number: int = 0
    side_effects: List[str] = field(default_factory=list)
    
    @property
    def readable_action(self) -> str:
        """Get human-readable action description."""
        if self.action_type == 'OTE':
            return f"SET {self.targets[0]}" if self.targets else "SET UNKNOWN"
        elif self.action_type == 'OTL':
            return f"LATCH {self.targets[0]}" if self.targets else "LATCH UNKNOWN"
        elif self.action_type == 'OTU':
            return f"UNLATCH {self.targets[0]}" if self.targets else "UNLATCH UNKNOWN"
        elif self.action_type == 'MOV':
            if len(self.targets) >= 1 and 'source' in self.parameters:
                return f"MOVE {self.parameters['source']} TO {self.targets[0]}"
        elif self.action_type in ['TON', 'TOF', 'RTO']:
            timer_name = self.targets[0] if self.targets else 'UNKNOWN'
            preset = self.parameters.get('preset', '?')
            return f"{self.action_type} {timer_name} PRESET:{preset}"
        return f"{self.action_type} {', '.join(self.targets)}"


@dataclass
class LogicBlock:
    """Represents a block of related logic."""
    block_id: str
    block_type: FlowType
    conditions: List[LogicCondition] = field(default_factory=list)
    actions: List[LogicAction] = field(default_factory=list)
    routine_name: str = ""
    rung_range: Tuple[int, int] = (0, 0)
    description: str = ""
    complexity_score: float = 0.0
    execution_probability: float = 1.0
    
    @property
    def condition_count(self) -> int:
        """Get number of conditions in this block."""
        return len(self.conditions)
        
    @property
    def action_count(self) -> int:
        """Get number of actions in this block."""
        return len(self.actions)
        
    def get_referenced_tags(self) -> Set[str]:
        """Get all tags referenced in this block."""
        tags = set()
        for condition in self.conditions:
            tags.update(condition.operands)
        for action in self.actions:
            tags.update(action.targets)
            tags.update(str(v) for v in action.parameters.values() if isinstance(v, str))
        return tags


@dataclass
class LogicFlow:
    """Represents a complete logic flow path."""
    flow_id: str
    flow_type: ExecutionPath
    blocks: List[LogicBlock] = field(default_factory=list)
    entry_conditions: List[LogicCondition] = field(default_factory=list)
    exit_conditions: List[LogicCondition] = field(default_factory=list)
    pattern: Optional[LogicPattern] = None
    priority: int = 0
    is_critical: bool = False
    
    @property
    def total_complexity(self) -> float:
        """Calculate total complexity of this flow."""
        return sum(block.complexity_score for block in self.blocks)
        
    @property
    def execution_probability(self) -> float:
        """Estimate execution probability of this flow."""
        if not self.blocks:
            return 0.0
        return sum(block.execution_probability for block in self.blocks) / len(self.blocks)


@dataclass
class FlowAnalysisResult:
    """Results of flow analysis."""
    flows: List[LogicFlow] = field(default_factory=list)
    patterns: Dict[LogicPattern, int] = field(default_factory=dict)
    critical_paths: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    safety_concerns: List[str] = field(default_factory=list)
    execution_graph: Optional[nx.DiGraph] = None


class LogicFlowAnalyzer:
    """Comprehensive analyzer for logic flow patterns and control flow."""
    
    def __init__(self):
        self.logic_blocks: List[LogicBlock] = []
        self.logic_flows: List[LogicFlow] = []
        self.execution_graph = nx.DiGraph()
        self.pattern_library = self._initialize_pattern_library()
        self.analysis_results = FlowAnalysisResult()
        
    def _initialize_pattern_library(self) -> Dict[LogicPattern, Dict[str, Any]]:
        """Initialize library of common logic patterns."""
        return {
            LogicPattern.START_STOP: {
                'signature': ['XIC.*START', 'XIO.*STOP', 'OTE.*RUN'],
                'description': 'Start/Stop motor control logic',
                'safety_critical': True
            },
            LogicPattern.SEAL_IN: {
                'signature': ['XIC.*START', 'XIC.*RUN', 'OTE.*RUN'],
                'description': 'Self-holding seal-in logic',
                'safety_critical': False
            },
            LogicPattern.EDGE_DETECT: {
                'signature': ['XIC.*', 'XIO.*_OLD', 'OTE.*_EDGE'],
                'description': 'Rising/falling edge detection',
                'safety_critical': False
            },
            LogicPattern.SAFETY_CHAIN: {
                'signature': ['XIC.*SAFE', 'XIC.*OK', 'XIO.*FAULT'],
                'description': 'Safety interlock chain',
                'safety_critical': True
            },
            LogicPattern.TIMER_CHAIN: {
                'signature': ['TON.*', 'XIC.*\\.DN', 'TON.*'],
                'description': 'Sequential timer chain',
                'safety_critical': False
            },
            LogicPattern.ALARM_LOGIC: {
                'signature': ['GRT.*', 'LES.*', 'OTE.*ALARM'],
                'description': 'Alarm detection logic',
                'safety_critical': True
            }
        }
        
    def analyze_routine_flow(self, routine_analysis) -> None:
        """Analyze logic flow for a single routine."""
        if not routine_analysis or not hasattr(routine_analysis, 'analyzed_routines'):
            logger.warning("No routine analysis provided for flow analysis")
            return
            
        logger.info("Analyzing logic flow patterns...")
        
        try:
            for routine_name, routine_data in routine_analysis.analyzed_routines.items():
                logger.debug(f"Analyzing flow for routine: {routine_name}")
                
                # Extract logic blocks from routine
                routine_blocks = self._extract_logic_blocks(routine_name, routine_data)
                self.logic_blocks.extend(routine_blocks)
                
                # Build execution graph for routine
                self._build_execution_graph(routine_name, routine_blocks)
                
                # Detect flow patterns
                routine_flows = self._detect_flow_patterns(routine_name, routine_blocks)
                self.logic_flows.extend(routine_flows)
                
            logger.info(f"Found {len(self.logic_blocks)} logic blocks and {len(self.logic_flows)} flows")
            
        except Exception as e:
            logger.error(f"Error analyzing routine flow: {e}")
            
    def _extract_logic_blocks(self, routine_name: str, routine_data) -> List[LogicBlock]:
        """Extract logic blocks from routine rungs."""
        blocks = []
        current_block = None
        block_counter = 0
        
        try:
            for rung_idx, rung in enumerate(routine_data.rungs):
                rung_text = rung.get('Text', '')
                if not rung_text:
                    continue
                    
                # Parse conditions and actions from rung
                conditions = self._parse_rung_conditions(rung_text, routine_name, rung_idx)
                actions = self._parse_rung_actions(rung_text, routine_name, rung_idx)
                
                # Determine if this starts a new block or continues current one
                if self._should_start_new_block(conditions, actions, current_block):
                    if current_block:
                        blocks.append(current_block)
                        
                    block_counter += 1
                    current_block = LogicBlock(
                        block_id=f"{routine_name}_Block_{block_counter}",
                        block_type=self._classify_block_type(conditions, actions),
                        routine_name=routine_name,
                        rung_range=(rung_idx, rung_idx)
                    )
                    
                # Add conditions and actions to current block
                if current_block:
                    current_block.conditions.extend(conditions)
                    current_block.actions.extend(actions)
                    current_block.rung_range = (current_block.rung_range[0], rung_idx)
                    current_block.complexity_score += len(conditions) + len(actions) * 0.5
                    
            # Add final block
            if current_block:
                blocks.append(current_block)
                
        except Exception as e:
            logger.error(f"Error extracting logic blocks from {routine_name}: {e}")
            
        return blocks
        
    def _parse_rung_conditions(self, rung_text: str, routine_name: str, rung_idx: int) -> List[LogicCondition]:
        """Parse conditions from rung text."""
        conditions = []
        
        # Pattern to match common condition instructions
        condition_patterns = [
            (r'XIC\(([^)]+)\)', 'XIC', False),
            (r'XIO\(([^)]+)\)', 'XIO', True),
            (r'EQU\(([^,]+),([^)]+)\)', 'EQU', False),
            (r'NEQ\(([^,]+),([^)]+)\)', 'NEQ', False),
            (r'GRT\(([^,]+),([^)]+)\)', 'GRT', False),
            (r'LES\(([^,]+),([^)]+)\)', 'LES', False),
            (r'GEQ\(([^,]+),([^)]+)\)', 'GEQ', False),
            (r'LEQ\(([^,]+),([^)]+)\)', 'LEQ', False),
        ]
        
        for pattern, instr_type, is_negated in condition_patterns:
            matches = re.finditer(pattern, rung_text)
            for match in matches:
                operands = [group.strip() for group in match.groups()]
                
                condition = LogicCondition(
                    condition_type=instr_type,
                    operands=operands,
                    expression=match.group(0),
                    routine_name=routine_name,
                    rung_number=rung_idx,
                    is_negated=is_negated,
                    complexity_score=self._calculate_condition_complexity(instr_type, operands)
                )
                conditions.append(condition)
                
        return conditions
        
    def _parse_rung_actions(self, rung_text: str, routine_name: str, rung_idx: int) -> List[LogicAction]:
        """Parse actions from rung text."""
        actions = []
        
        # Pattern to match common action instructions
        action_patterns = [
            (r'OTE\(([^)]+)\)', 'OTE'),
            (r'OTL\(([^)]+)\)', 'OTL'),
            (r'OTU\(([^)]+)\)', 'OTU'),
            (r'MOV\(([^,]+),([^)]+)\)', 'MOV'),
            (r'TON\(([^,]+),([^,]+),([^)]+)\)', 'TON'),
            (r'TOF\(([^,]+),([^,]+),([^)]+)\)', 'TOF'),
            (r'RTO\(([^,]+),([^,]+),([^)]+)\)', 'RTO'),
            (r'CTU\(([^,]+),([^,]+),([^)]+)\)', 'CTU'),
            (r'CTD\(([^,]+),([^,]+),([^)]+)\)', 'CTD'),
        ]
        
        for pattern, instr_type in action_patterns:
            matches = re.finditer(pattern, rung_text)
            for match in matches:
                operands = [group.strip() for group in match.groups()]
                
                # Determine targets and parameters based on instruction type
                if instr_type in ['OTE', 'OTL', 'OTU']:
                    targets = operands
                    parameters = {}
                elif instr_type == 'MOV':
                    targets = [operands[1]] if len(operands) > 1 else []
                    parameters = {'source': operands[0]} if operands else {}
                elif instr_type in ['TON', 'TOF', 'RTO']:
                    targets = [operands[0]] if operands else []
                    parameters = {
                        'preset': operands[1] if len(operands) > 1 else '0',
                        'accumulator': operands[2] if len(operands) > 2 else '0'
                    }
                elif instr_type in ['CTU', 'CTD']:
                    targets = [operands[0]] if operands else []
                    parameters = {
                        'preset': operands[1] if len(operands) > 1 else '0',
                        'accumulator': operands[2] if len(operands) > 2 else '0'
                    }
                else:
                    targets = operands
                    parameters = {}
                    
                action = LogicAction(
                    action_type=instr_type,
                    targets=targets,
                    parameters=parameters,
                    routine_name=routine_name,
                    rung_number=rung_idx
                )
                actions.append(action)
                
        return actions
        
    def _calculate_condition_complexity(self, instr_type: str, operands: List[str]) -> float:
        """Calculate complexity score for a condition."""
        base_score = {
            'XIC': 1.0, 'XIO': 1.0,
            'EQU': 1.5, 'NEQ': 1.5,
            'GRT': 2.0, 'LES': 2.0, 'GEQ': 2.0, 'LEQ': 2.0
        }.get(instr_type, 1.0)
        
        # Add complexity for complex operands
        complexity_bonus = 0.0
        for operand in operands:
            if '.' in operand:  # UDT member access
                complexity_bonus += 0.5
            if '[' in operand:  # Array access
                complexity_bonus += 0.3
            if any(op in operand for op in ['+', '-', '*', '/']):  # Expression
                complexity_bonus += 1.0
                
        return base_score + complexity_bonus
        
    def _should_start_new_block(self, conditions: List[LogicCondition], actions: List[LogicAction], 
                               current_block: Optional[LogicBlock]) -> bool:
        """Determine if conditions/actions should start a new logic block."""
        if not current_block:
            return True
            
        # Start new block if we have a major control flow change
        if any(action.action_type in ['JSR', 'SBR', 'RET'] for action in actions):
            return True
            
        # Start new block if complexity gets too high
        if current_block.complexity_score > 10.0:
            return True
            
        # Start new block if we detect a different pattern
        current_pattern = self._classify_block_type(current_block.conditions, current_block.actions)
        new_pattern = self._classify_block_type(conditions, actions)
        
        if current_pattern != new_pattern and new_pattern != FlowType.SEQUENTIAL:
            return True
            
        return False
        
    def _classify_block_type(self, conditions: List[LogicCondition], actions: List[LogicAction]) -> FlowType:
        """Classify the type of logic block based on conditions and actions."""
        # Check for timer patterns
        if any(action.action_type in ['TON', 'TOF', 'RTO'] for action in actions):
            return FlowType.TIMER_CHAIN
            
        # Check for counter patterns  
        if any(action.action_type in ['CTU', 'CTD'] for action in actions):
            return FlowType.COUNTER_CHAIN
            
        # Check for conditional patterns
        comparison_conditions = [c for c in conditions if c.condition_type in ['EQU', 'NEQ', 'GRT', 'LES', 'GEQ', 'LEQ']]
        if len(comparison_conditions) > 1:
            return FlowType.CONDITIONAL
            
        # Check for interlock patterns (multiple safety conditions)
        safety_keywords = ['safe', 'ok', 'fault', 'alarm', 'emergency', 'stop']
        safety_conditions = [c for c in conditions 
                           if any(keyword in op.lower() for op in c.operands for keyword in safety_keywords)]
        if len(safety_conditions) > 2:
            return FlowType.INTERLOCK
            
        # Check for parallel execution (multiple outputs)
        if len(actions) > 2 and len(set(action.action_type for action in actions)) > 1:
            return FlowType.PARALLEL
            
        return FlowType.SEQUENTIAL
        
    def _build_execution_graph(self, routine_name: str, blocks: List[LogicBlock]) -> None:
        """Build execution graph from logic blocks."""
        # Add nodes for each block
        for block in blocks:
            self.execution_graph.add_node(
                block.block_id,
                routine=routine_name,
                block_type=block.block_type.value,
                complexity=block.complexity_score,
                rung_range=block.rung_range
            )
            
        # Add edges between sequential blocks
        for i in range(len(blocks) - 1):
            current_block = blocks[i]
            next_block = blocks[i + 1]
            
            self.execution_graph.add_edge(
                current_block.block_id,
                next_block.block_id,
                edge_type='sequential',
                weight=1.0
            )
            
        # Add conditional edges based on logic analysis
        self._add_conditional_edges(blocks)
        
    def _add_conditional_edges(self, blocks: List[LogicBlock]) -> None:
        """Add conditional edges to execution graph."""
        for block in blocks:
            if block.block_type == FlowType.CONDITIONAL:
                # Find blocks that might be conditionally executed
                condition_tags = set()
                for condition in block.conditions:
                    condition_tags.update(condition.operands)
                    
                for action in block.actions:
                    action_tags = set(action.targets)
                    
                    # Look for other blocks that reference the same tags
                    for other_block in blocks:
                        if other_block.block_id == block.block_id:
                            continue
                            
                        other_tags = other_block.get_referenced_tags()
                        if condition_tags.intersection(other_tags) or action_tags.intersection(other_tags):
                            self.execution_graph.add_edge(
                                block.block_id,
                                other_block.block_id,
                                edge_type='conditional',
                                weight=0.5
                            )
                            
    def _detect_flow_patterns(self, routine_name: str, blocks: List[LogicBlock]) -> List[LogicFlow]:
        """Detect common flow patterns in logic blocks."""
        flows = []
        
        # Detect each pattern type
        for pattern, pattern_info in self.pattern_library.items():
            pattern_flows = self._detect_specific_pattern(routine_name, blocks, pattern, pattern_info)
            flows.extend(pattern_flows)
            
        # Detect custom flows based on block sequences
        sequence_flows = self._detect_sequence_flows(routine_name, blocks)
        flows.extend(sequence_flows)
        
        return flows
        
    def _detect_specific_pattern(self, routine_name: str, blocks: List[LogicBlock], 
                                pattern: LogicPattern, pattern_info: Dict[str, Any]) -> List[LogicFlow]:
        """Detect a specific pattern in the logic blocks."""
        flows = []
        signatures = pattern_info.get('signature', [])
        
        if not signatures:
            return flows
            
        # Look for pattern signatures in block sequences
        for i in range(len(blocks) - len(signatures) + 1):
            block_sequence = blocks[i:i + len(signatures)]
            
            if self._matches_pattern_signature(block_sequence, signatures):
                flow = LogicFlow(
                    flow_id=f"{routine_name}_{pattern.value}_{i}",
                    flow_type=ExecutionPath.MAIN_PATH,
                    blocks=block_sequence,
                    pattern=pattern,
                    priority=10 if pattern_info.get('safety_critical', False) else 5,
                    is_critical=pattern_info.get('safety_critical', False)
                )
                flows.append(flow)
                
        return flows
        
    def _matches_pattern_signature(self, blocks: List[LogicBlock], signatures: List[str]) -> bool:
        """Check if block sequence matches pattern signature."""
        if len(blocks) != len(signatures):
            return False
            
        for block, signature in zip(blocks, signatures):
            # Check if block matches signature pattern
            block_text = self._get_block_text_representation(block)
            if not re.search(signature, block_text, re.IGNORECASE):
                return False
                
        return True
        
    def _get_block_text_representation(self, block: LogicBlock) -> str:
        """Get text representation of block for pattern matching."""
        text_parts = []
        
        for condition in block.conditions:
            text_parts.append(f"{condition.condition_type}({','.join(condition.operands)})")
            
        for action in block.actions:
            text_parts.append(f"{action.action_type}({','.join(action.targets)})")
            
        return ' '.join(text_parts)
        
    def _detect_sequence_flows(self, routine_name: str, blocks: List[LogicBlock]) -> List[LogicFlow]:
        """Detect sequence-based flows."""
        flows = []
        
        # Group consecutive blocks by type
        current_sequence = []
        current_type = None
        
        for block in blocks:
            if block.block_type != current_type:
                if current_sequence and len(current_sequence) > 1:
                    flow = LogicFlow(
                        flow_id=f"{routine_name}_Sequence_{len(flows)}",
                        flow_type=ExecutionPath.MAIN_PATH,
                        blocks=current_sequence.copy(),
                        priority=1
                    )
                    flows.append(flow)
                    
                current_sequence = [block]
                current_type = block.block_type
            else:
                current_sequence.append(block)
                
        # Add final sequence
        if current_sequence and len(current_sequence) > 1:
            flow = LogicFlow(
                flow_id=f"{routine_name}_Sequence_{len(flows)}",
                flow_type=ExecutionPath.MAIN_PATH,
                blocks=current_sequence,
                priority=1
            )
            flows.append(flow)
            
        return flows
        
    def analyze_flow_performance(self) -> None:
        """Analyze performance characteristics of flows."""
        logger.info("Analyzing flow performance...")
        
        try:
            # Identify critical paths
            self._identify_critical_paths()
            
            # Find bottlenecks
            self._find_bottlenecks()
            
            # Generate optimization opportunities
            self._generate_optimization_opportunities()
            
            # Identify safety concerns
            self._identify_safety_concerns()
            
        except Exception as e:
            logger.error(f"Error analyzing flow performance: {e}")
            
    def _identify_critical_paths(self) -> None:
        """Identify critical execution paths."""
        critical_paths = []
        
        # Find paths with high complexity or safety criticality
        for flow in self.logic_flows:
            if flow.is_critical or flow.total_complexity > 20.0:
                critical_paths.append(flow.flow_id)
                
        # Find longest paths in execution graph
        if self.execution_graph.number_of_nodes() > 0:
            try:
                # Use topological sorting to find longest paths
                if nx.is_directed_acyclic_graph(self.execution_graph):
                    topo_order = list(nx.topological_sort(self.execution_graph))
                    
                    # Find paths with high cumulative complexity
                    for node in topo_order:
                        node_data = self.execution_graph.nodes[node]
                        if node_data.get('complexity', 0) > 5.0:
                            critical_paths.append(node)
                            
            except Exception as e:
                logger.warning(f"Error finding critical paths: {e}")
                
        self.analysis_results.critical_paths = critical_paths
        
    def _find_bottlenecks(self) -> None:
        """Find potential bottlenecks in execution flow."""
        bottlenecks = []
        
        # Find nodes with high in-degree (many dependencies)
        for node in self.execution_graph.nodes():
            in_degree = self.execution_graph.in_degree(node)
            if in_degree > 3:  # Arbitrary threshold
                bottlenecks.append(f"Node {node} has {in_degree} dependencies")
                
        # Find flows with high complexity
        for flow in self.logic_flows:
            if flow.total_complexity > 15.0:
                bottlenecks.append(f"Flow {flow.flow_id} has high complexity ({flow.total_complexity:.1f})")
                
        self.analysis_results.bottlenecks = bottlenecks
        
    def _generate_optimization_opportunities(self) -> None:
        """Generate optimization recommendations."""
        opportunities = []
        
        # Look for redundant conditions
        condition_map = defaultdict(list)
        for block in self.logic_blocks:
            for condition in block.conditions:
                key = (condition.condition_type, tuple(condition.operands))
                condition_map[key].append(block.block_id)
                
        for key, block_ids in condition_map.items():
            if len(block_ids) > 2:
                opportunities.append(f"Condition {key[0]}({','.join(key[1])}) appears in {len(block_ids)} blocks - consider consolidation")
                
        # Look for complex blocks that could be simplified
        for block in self.logic_blocks:
            if block.complexity_score > 12.0:
                opportunities.append(f"Block {block.block_id} has high complexity ({block.complexity_score:.1f}) - consider splitting")
                
        # Look for unused patterns
        pattern_counts = defaultdict(int)
        for flow in self.logic_flows:
            if flow.pattern:
                pattern_counts[flow.pattern] += 1
                
        for pattern, count in pattern_counts.items():
            if count == 1:
                opportunities.append(f"Pattern {pattern.value} appears only once - verify if it's necessary")
                
        self.analysis_results.optimization_opportunities = opportunities
        
    def _identify_safety_concerns(self) -> None:
        """Identify potential safety concerns."""
        concerns = []
        
        # Look for safety-critical flows without redundancy
        safety_flows = [flow for flow in self.logic_flows if flow.is_critical]
        if safety_flows:
            for flow in safety_flows:
                if len(flow.blocks) == 1:
                    concerns.append(f"Safety-critical flow {flow.flow_id} has no redundancy")
                    
        # Look for missing safety patterns
        has_safety_chain = any(flow.pattern == LogicPattern.SAFETY_CHAIN for flow in self.logic_flows)
        if not has_safety_chain and len(self.logic_blocks) > 10:
            concerns.append("No safety chain pattern detected in complex logic")
            
        # Look for alarm logic without proper handling
        has_alarm_logic = any(flow.pattern == LogicPattern.ALARM_LOGIC for flow in self.logic_flows)
        alarm_actions = [block for block in self.logic_blocks 
                        if any('alarm' in target.lower() for action in block.actions for target in action.targets)]
        
        if alarm_actions and not has_alarm_logic:
            concerns.append("Alarm outputs detected without proper alarm logic pattern")
            
        self.analysis_results.safety_concerns = concerns
        
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get comprehensive flow analysis summary."""
        flow_types = defaultdict(int)
        pattern_counts = defaultdict(int)
        
        for flow in self.logic_flows:
            flow_types[flow.flow_type.value] += 1
            if flow.pattern:
                pattern_counts[flow.pattern.value] += 1
                
        block_types = defaultdict(int)
        for block in self.logic_blocks:
            block_types[block.block_type.value] += 1
            
        return {
            'total_blocks': len(self.logic_blocks),
            'total_flows': len(self.logic_flows),
            'block_types': dict(block_types),
            'flow_types': dict(flow_types),
            'detected_patterns': dict(pattern_counts),
            'critical_paths': len(self.analysis_results.critical_paths),
            'bottlenecks': len(self.analysis_results.bottlenecks),
            'optimization_opportunities': len(self.analysis_results.optimization_opportunities),
            'safety_concerns': len(self.analysis_results.safety_concerns),
            'average_block_complexity': sum(b.complexity_score for b in self.logic_blocks) / len(self.logic_blocks) if self.logic_blocks else 0,
            'total_complexity': sum(b.complexity_score for b in self.logic_blocks)
        }
