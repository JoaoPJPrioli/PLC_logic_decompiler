#!/usr/bin/env python3
"""
Routine and Program Analysis Module for Step 12

This module provides comprehensive analysis of PLC program structure,
subroutine calls, parameter passing, and execution flow patterns.

Key Features:
- Subroutine call detection and analysis (JSR, SBR, RET)
- Program hierarchy mapping and structure analysis
- Cross-routine parameter passing analysis
- Call stack modeling and recursion detection
- Program execution flow analysis
- Performance pattern analysis

Author: GitHub Copilot
Date: July 31, 2025
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

logger = logging.getLogger(__name__)


class RoutineType(Enum):
    """Types of routines in PLC programs"""
    MAIN = "main"                    # Main routine
    SUBROUTINE = "subroutine"       # Subroutine called by JSR
    FAULT = "fault"                 # Fault handling routine
    EVENT = "event"                 # Event-driven routine
    INTERRUPT = "interrupt"         # Interrupt service routine
    TASK = "task"                   # Task routine
    UNKNOWN = "unknown"             # Unknown routine type


class CallType(Enum):
    """Types of subroutine calls"""
    JSR = "JSR"                     # Jump to Subroutine
    SBR = "SBR"                     # Subroutine entry point
    RET = "RET"                     # Return from subroutine
    FOR = "FOR"                     # For loop
    NEXT = "NEXT"                   # Next in for loop
    GOTO = "GOTO"                   # Goto statement


@dataclass
class SubroutineCall:
    """Represents a subroutine call in the program"""
    call_type: CallType
    caller_routine: str
    caller_rung: int
    target_routine: Optional[str]
    parameters: List[str] = field(default_factory=list)
    call_id: Optional[str] = None
    line_number: Optional[int] = None
    raw_instruction: Optional[str] = None


@dataclass
class RoutineInfo:
    """Information about a routine"""
    name: str
    routine_type: RoutineType
    description: Optional[str] = None
    rung_count: int = 0
    instruction_count: int = 0
    calls_made: List[SubroutineCall] = field(default_factory=list)
    called_by: List[SubroutineCall] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    local_tags: Set[str] = field(default_factory=set)
    complexity_score: int = 0
    execution_time_estimate: float = 0.0


@dataclass
class ProgramStructure:
    """Overall program structure analysis"""
    routines: Dict[str, RoutineInfo] = field(default_factory=dict)
    call_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    max_call_depth: int = 0
    total_routines: int = 0
    total_subroutines: int = 0
    recursive_calls: List[Tuple[str, str]] = field(default_factory=list)
    unreachable_routines: Set[str] = field(default_factory=set)


class RoutineAnalyzer:
    """Analyzer for routine and program structure"""
    
    def __init__(self):
        """Initialize the routine analyzer"""
        self.call_graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.program_structure = ProgramStructure()
        self.analysis_cache: Dict[str, Any] = {}
        
        # Pattern recognition for different call types
        self.call_patterns = {
            CallType.JSR: re.compile(r'JSR\s+(\w+)', re.IGNORECASE),
            CallType.SBR: re.compile(r'SBR\s*\(([^)]*)\)', re.IGNORECASE),
            CallType.RET: re.compile(r'RET\s*\(\)', re.IGNORECASE),
            CallType.FOR: re.compile(r'FOR\s+(\w+)', re.IGNORECASE),
            CallType.GOTO: re.compile(r'GOTO\s+(\w+)', re.IGNORECASE)
        }
    
    def analyze_program_structure(self, ladder_routines: List[Any]) -> Dict[str, Any]:
        """
        Analyze the overall program structure
        
        Args:
            ladder_routines: List of ladder logic routines to analyze
            
        Returns:
            Dictionary containing comprehensive program structure analysis
        """
        logger.info("Starting comprehensive program structure analysis...")
        
        # Step 1: Initialize program structure
        self.program_structure = ProgramStructure()
        
        # Step 2: Analyze individual routines
        for routine in ladder_routines:
            routine_info = self._analyze_routine(routine)
            self.program_structure.routines[routine.name] = routine_info
        
        # Step 3: Build call hierarchy
        self._build_call_hierarchy()
        
        # Step 4: Detect execution patterns
        self._analyze_execution_patterns()
        
        # Step 5: Identify performance characteristics
        self._analyze_performance_patterns()
        
        # Step 6: Generate analysis results
        analysis_results = self._generate_analysis_results()
        
        logger.info(f"Program structure analysis completed for {len(ladder_routines)} routines")
        return analysis_results
    
    def _analyze_routine(self, routine: Any) -> RoutineInfo:
        """Analyze a single routine"""
        routine_info = RoutineInfo(
            name=routine.name,
            routine_type=self._determine_routine_type(routine),
            rung_count=len(getattr(routine, 'rungs', [])),
            instruction_count=0
        )
        
        # Analyze rungs for calls and complexity
        if hasattr(routine, 'rungs'):
            for rung_idx, rung in enumerate(routine.rungs):
                if hasattr(rung, 'instructions'):
                    routine_info.instruction_count += len(rung.instructions)
                    
                    # Look for subroutine calls in instructions
                    for instruction in rung.instructions:
                        call = self._detect_subroutine_call(instruction, routine.name, rung_idx)
                        if call:
                            routine_info.calls_made.append(call)
        
        # Calculate complexity score
        routine_info.complexity_score = self._calculate_routine_complexity(routine_info)
        
        return routine_info
    
    def _determine_routine_type(self, routine: Any) -> RoutineType:
        """Determine the type of routine"""
        routine_name = routine.name.lower()
        
        if 'main' in routine_name:
            return RoutineType.MAIN
        elif 'fault' in routine_name or 'error' in routine_name:
            return RoutineType.FAULT
        elif 'event' in routine_name:
            return RoutineType.EVENT
        elif 'interrupt' in routine_name or 'isr' in routine_name:
            return RoutineType.INTERRUPT
        elif 'task' in routine_name:
            return RoutineType.TASK
        else:
            return RoutineType.SUBROUTINE
    
    def _detect_subroutine_call(self, instruction: Any, caller_routine: str, rung_idx: int) -> Optional[SubroutineCall]:
        """Detect if an instruction is a subroutine call"""
        if not hasattr(instruction, 'raw_text') or not instruction.raw_text:
            return None
        
        raw_text = instruction.raw_text.strip()
        
        # Check each call pattern
        for call_type, pattern in self.call_patterns.items():
            match = pattern.search(raw_text)
            if match:
                target_routine = match.group(1) if match.groups() else None
                
                call = SubroutineCall(
                    call_type=call_type,
                    caller_routine=caller_routine,
                    caller_rung=rung_idx,
                    target_routine=target_routine,
                    raw_instruction=raw_text
                )
                
                # Extract parameters if available
                if hasattr(instruction, 'parameters') and instruction.parameters:
                    call.parameters = [param.value for param in instruction.parameters if hasattr(param, 'value')]
                
                return call
        
        return None
    
    def _build_call_hierarchy(self):
        """Build the call hierarchy graph"""
        logger.info("Building call hierarchy...")
        
        # Build call relationships
        for routine_name, routine_info in self.program_structure.routines.items():
            self.program_structure.call_hierarchy[routine_name] = []
            
            for call in routine_info.calls_made:
                if call.target_routine:
                    self.program_structure.call_hierarchy[routine_name].append(call.target_routine)
                    
                    # Add to call graph if NetworkX is available
                    if self.call_graph is not None:
                        self.call_graph.add_edge(routine_name, call.target_routine, call_type=call.call_type.value)
                    
                    # Update called_by relationships
                    if call.target_routine in self.program_structure.routines:
                        self.program_structure.routines[call.target_routine].called_by.append(call)
        
        # Calculate statistics
        self.program_structure.total_routines = len(self.program_structure.routines)
        self.program_structure.total_subroutines = sum(
            1 for routine_info in self.program_structure.routines.values()
            if routine_info.routine_type == RoutineType.SUBROUTINE
        )
    
    def _analyze_execution_patterns(self):
        """Analyze execution patterns and call depth"""
        logger.info("Analyzing execution patterns...")
        
        # Find main routines (entry points)
        main_routines = [
            name for name, routine_info in self.program_structure.routines.items()
            if routine_info.routine_type == RoutineType.MAIN or not routine_info.called_by
        ]
        
        # Calculate call depth from each main routine
        max_depth = 0
        for main_routine in main_routines:
            depth = self._calculate_call_depth(main_routine, set())
            max_depth = max(max_depth, depth)
        
        self.program_structure.max_call_depth = max_depth
        
        # Detect recursive calls
        self._detect_recursive_calls()
        
        # Find unreachable routines
        self._find_unreachable_routines(main_routines)
    
    def _calculate_call_depth(self, routine_name: str, visited: Set[str]) -> int:
        """Calculate maximum call depth from a routine"""
        if routine_name in visited:
            return 0  # Avoid infinite recursion
        
        visited.add(routine_name)
        
        max_depth = 0
        called_routines = self.program_structure.call_hierarchy.get(routine_name, [])
        
        for called_routine in called_routines:
            depth = 1 + self._calculate_call_depth(called_routine, visited.copy())
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _detect_recursive_calls(self):
        """Detect recursive call patterns"""
        if not NETWORKX_AVAILABLE or self.call_graph is None:
            return
        
        try:
            # Find cycles in the call graph
            cycles = list(nx.simple_cycles(self.call_graph))
            
            for cycle in cycles:
                if len(cycle) >= 2:
                    # Add recursive call pairs
                    for i in range(len(cycle)):
                        next_i = (i + 1) % len(cycle)
                        self.program_structure.recursive_calls.append((cycle[i], cycle[next_i]))
        except Exception as e:
            logger.warning(f"Could not detect recursive calls: {e}")
    
    def _find_unreachable_routines(self, main_routines: List[str]):
        """Find routines that are never called"""
        reachable = set()
        
        def mark_reachable(routine_name: str):
            if routine_name in reachable or routine_name not in self.program_structure.routines:
                return
            
            reachable.add(routine_name)
            called_routines = self.program_structure.call_hierarchy.get(routine_name, [])
            
            for called_routine in called_routines:
                mark_reachable(called_routine)
        
        # Mark all reachable routines from main routines
        for main_routine in main_routines:
            mark_reachable(main_routine)
        
        # Find unreachable routines
        all_routines = set(self.program_structure.routines.keys())
        self.program_structure.unreachable_routines = all_routines - reachable
    
    def _analyze_performance_patterns(self):
        """Analyze performance-related patterns"""
        logger.info("Analyzing performance patterns...")
        
        for routine_name, routine_info in self.program_structure.routines.items():
            # Estimate execution time based on complexity
            routine_info.execution_time_estimate = self._estimate_execution_time(routine_info)
    
    def _calculate_routine_complexity(self, routine_info: RoutineInfo) -> int:
        """Calculate complexity score for a routine"""
        complexity = 0
        
        # Base complexity from instruction count
        complexity += routine_info.instruction_count
        
        # Additional complexity for calls
        complexity += len(routine_info.calls_made) * 5
        
        # Additional complexity for being called frequently
        complexity += len(routine_info.called_by) * 2
        
        return complexity
    
    def _estimate_execution_time(self, routine_info: RoutineInfo) -> float:
        """Estimate execution time for a routine (in milliseconds)"""
        # Simple estimation based on complexity
        base_time = routine_info.instruction_count * 0.001  # 1µs per instruction
        call_overhead = len(routine_info.calls_made) * 0.01  # 10µs per call
        
        return base_time + call_overhead
    
    def _generate_analysis_results(self) -> Dict[str, Any]:
        """Generate comprehensive analysis results"""
        return {
            'success': True,
            'program_structure': {
                'total_routines': self.program_structure.total_routines,
                'total_subroutines': self.program_structure.total_subroutines,
                'max_call_depth': self.program_structure.max_call_depth,
                'recursive_calls_count': len(self.program_structure.recursive_calls),
                'unreachable_routines_count': len(self.program_structure.unreachable_routines)
            },
            'routines': {
                name: {
                    'type': info.routine_type.value,
                    'rung_count': info.rung_count,
                    'instruction_count': info.instruction_count,
                    'calls_made': len(info.calls_made),
                    'called_by_count': len(info.called_by),
                    'complexity_score': info.complexity_score,
                    'execution_time_estimate': info.execution_time_estimate
                }
                for name, info in self.program_structure.routines.items()
            },
            'call_hierarchy': self.program_structure.call_hierarchy,
            'recursive_calls': self.program_structure.recursive_calls,
            'unreachable_routines': list(self.program_structure.unreachable_routines),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check for unreachable routines
        if self.program_structure.unreachable_routines:
            recommendations.append({
                'type': 'optimization',
                'priority': 'medium',
                'title': 'Unreachable Routines Found',
                'description': f'Found {len(self.program_structure.unreachable_routines)} routines that are never called. Consider removing unused code.',
                'affected_routines': list(self.program_structure.unreachable_routines)
            })
        
        # Check for deep call hierarchies
        if self.program_structure.max_call_depth > 5:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'title': 'Deep Call Hierarchy',
                'description': f'Maximum call depth is {self.program_structure.max_call_depth}. Consider flattening the call structure for better performance.',
                'max_depth': self.program_structure.max_call_depth
            })
        
        # Check for recursive calls
        if self.program_structure.recursive_calls:
            recommendations.append({
                'type': 'warning',
                'priority': 'high',
                'title': 'Recursive Calls Detected',
                'description': f'Found {len(self.program_structure.recursive_calls)} recursive call patterns. Verify these are intentional.',
                'recursive_pairs': self.program_structure.recursive_calls
            })
        
        # Check for high complexity routines
        high_complexity_routines = [
            (name, info.complexity_score)
            for name, info in self.program_structure.routines.items()
            if info.complexity_score > 100
        ]
        
        if high_complexity_routines:
            recommendations.append({
                'type': 'maintenance',
                'priority': 'low',
                'title': 'High Complexity Routines',
                'description': f'Found {len(high_complexity_routines)} routines with high complexity. Consider breaking them down.',
                'complex_routines': high_complexity_routines[:5]
            })
        
        return recommendations
    
    def get_routine_call_graph_data(self) -> Dict[str, Any]:
        """Get call graph data for visualization"""
        if not NETWORKX_AVAILABLE or self.call_graph is None:
            return {'error': 'NetworkX not available for call graph generation'}
        
        # Convert NetworkX graph to visualization format
        nodes = []
        for node in self.call_graph.nodes():
            routine_info = self.program_structure.routines.get(node)
            nodes.append({
                'id': node,
                'label': node,
                'type': routine_info.routine_type.value if routine_info else 'unknown',
                'complexity': routine_info.complexity_score if routine_info else 0,
                'instruction_count': routine_info.instruction_count if routine_info else 0
            })
        
        edges = []
        for source, target, data in self.call_graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'call_type': data.get('call_type', 'unknown')
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metrics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'max_call_depth': self.program_structure.max_call_depth
            }
        }
