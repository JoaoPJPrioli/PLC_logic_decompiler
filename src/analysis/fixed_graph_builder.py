#!/usr/bin/env python3
"""
Fixed Graph Builder for PLC Logic Analysis
This module fixes the graph generation to create meaningful nodes and edges from L5X data
"""

import networkx as nx
import json
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class GraphType(Enum):
    CONTROL_FLOW = "control_flow"
    DATA_DEPENDENCY = "data_dependency" 
    INSTRUCTION_NETWORK = "instruction_network"
    EXECUTION_FLOW = "execution_flow"

@dataclass
class GraphNode:
    id: str
    node_type: str
    label: str
    properties: Dict[str, Any]

@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str
    properties: Dict[str, Any]

class FixedAdvancedGraphBuilder:
    """Fixed graph builder that actually creates meaningful graphs from L5X data"""
    
    def __init__(self):
        self.graphs = {}
        self.instruction_patterns = {
            'XIC': r'XIC\(([^)]+)\)',  # Examine if Closed
            'XIO': r'XIO\(([^)]+)\)',  # Examine if Open
            'OTE': r'OTE\(([^)]+)\)',  # Output Energize
            'OTL': r'OTL\(([^)]+)\)',  # Output Latch
            'OTU': r'OTU\(([^)]+)\)',  # Output Unlatch
            'TON': r'TON\(([^,)]+)[^)]*\)',  # Timer On Delay
            'TOF': r'TOF\(([^,)]+)[^)]*\)',  # Timer Off Delay
            'CTU': r'CTU\(([^,)]+)[^)]*\)',  # Count Up
            'CTD': r'CTD\(([^,)]+)[^)]*\)',  # Count Down
            'JSR': r'JSR\(([^,)]+)[^)]*\)',  # Jump to Subroutine
            'MOV': r'MOV\(([^,)]+),([^)]+)\)',  # Move
            'EQU': r'EQU\(([^,)]+),([^)]+)\)',  # Equal
            'GEQ': r'GEQ\(([^,)]+),([^)]+)\)',  # Greater or Equal
            'LEQ': r'LEQ\(([^,)]+),([^)]+)\)',  # Less or Equal
        }
    
    def build_comprehensive_graph(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive graphs from L5X analysis data"""
        try:
            # Extract ladder routines from analysis data
            ladder_routines = self._extract_ladder_routines(analysis_data)
            
            if not ladder_routines:
                print("No ladder routines found in analysis data")
                return self._empty_result()
            
            # Build different graph types
            control_flow_graph = self._build_control_flow_graph(ladder_routines)
            data_dependency_graph = self._build_data_dependency_graph(ladder_routines)
            instruction_network_graph = self._build_instruction_network_graph(ladder_routines)
            execution_flow_graph = self._build_execution_flow_graph(ladder_routines)
            
            # Store graphs
            self.graphs = {
                GraphType.CONTROL_FLOW: control_flow_graph,
                GraphType.DATA_DEPENDENCY: data_dependency_graph,
                GraphType.INSTRUCTION_NETWORK: instruction_network_graph,
                GraphType.EXECUTION_FLOW: execution_flow_graph
            }
            
            # Generate summary and recommendations
            summary = self._generate_summary()
            recommendations = self._generate_recommendations()
            
            return {
                'build_successful': True,
                'graphs': {gt.value: self._graph_to_dict(g) for gt, g in self.graphs.items()},
                'summary': summary,
                'recommendations': recommendations,
                'statistics': self._generate_statistics()
            }
            
        except Exception as e:
            print(f"Error building graphs: {e}")
            return {
                'build_successful': False,
                'error': str(e),
                'graphs': {},
                'recommendations': []
            }
    
    def _extract_ladder_routines(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ladder routines from analysis data"""
        routines = []
        
        # Try different paths in the analysis data
        possible_paths = [
            ['l5x_parsing', 'programs'],
            ['analysis_results', 'l5x_parsing', 'programs'],
            ['final_data', 'extracted_data', 'detailed_data', 'programs'],
            ['programs']
        ]
        
        programs_data = None
        for path in possible_paths:
            current = analysis_data
            try:
                for key in path:
                    current = current[key]
                programs_data = current
                break
            except (KeyError, TypeError):
                continue
        
        if not programs_data:
            print("No programs found in analysis data")
            return []
        
        # Extract routines from programs
        for program in programs_data:
            if isinstance(program, dict) and 'routines' in program:
                for routine in program['routines']:
                    if isinstance(routine, dict):
                        routines.append({
                            'program_name': program.get('name', 'Unknown'),
                            'routine_name': routine.get('name', 'Unknown'),
                            'routine_type': routine.get('type', 'RLL'),
                            'rungs': routine.get('rungs', []),
                            'ladder_text': routine.get('ladder_text', ''),
                            'instructions': routine.get('instructions', [])
                        })
        
        return routines
    
    def _build_control_flow_graph(self, ladder_routines: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build control flow graph showing program execution flow"""
        G = nx.DiGraph()
        
        for routine in ladder_routines:
            program_name = routine['program_name']
            routine_name = routine['routine_name']
            
            # Add routine node
            routine_node = f"{program_name}:{routine_name}"
            G.add_node(routine_node, type='routine', program=program_name, routine=routine_name)
            
            # Process rungs in the routine
            rungs = routine.get('rungs', [])
            prev_rung = None
            
            for i, rung in enumerate(rungs):
                rung_node = f"{routine_node}:Rung_{i}"
                G.add_node(rung_node, type='rung', rung_number=i, 
                          text=rung.get('text', ''), comment=rung.get('comment', ''))
                
                # Connect routine to first rung
                if i == 0:
                    G.add_edge(routine_node, rung_node, type='start')
                
                # Connect sequential rungs
                if prev_rung:
                    G.add_edge(prev_rung, rung_node, type='sequence')
                
                # Look for JSR calls
                rung_text = rung.get('text', '')
                jsr_matches = re.findall(self.instruction_patterns['JSR'], rung_text)
                for jsr_target in jsr_matches:
                    target_routine = f"{program_name}:{jsr_target}"
                    if target_routine != routine_node:  # Avoid self-loops
                        G.add_edge(rung_node, target_routine, type='call')
                
                prev_rung = rung_node
        
        return G
    
    def _build_data_dependency_graph(self, ladder_routines: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build data dependency graph showing variable relationships"""
        G = nx.DiGraph()
        
        # Track variable usage
        variable_writes = {}  # variable -> list of nodes that write to it
        variable_reads = {}   # variable -> list of nodes that read from it
        
        for routine in ladder_routines:
            routine_name = f"{routine['program_name']}:{routine['routine_name']}"
            
            for i, rung in enumerate(routine.get('rungs', [])):
                rung_node = f"{routine_name}:Rung_{i}"
                rung_text = rung.get('text', '')
                
                # Add rung node
                G.add_node(rung_node, type='rung', routine=routine_name)
                
                # Find variables being read (XIC, XIO, etc.)
                read_vars = set()
                for pattern in ['XIC', 'XIO', 'EQU', 'GEQ', 'LEQ']:
                    if pattern in self.instruction_patterns:
                        matches = re.findall(self.instruction_patterns[pattern], rung_text)
                        for match in matches:
                            if isinstance(match, tuple):
                                read_vars.update(match)
                            else:
                                read_vars.add(match)
                
                # Find variables being written (OTE, OTL, MOV, etc.)
                write_vars = set()
                for pattern in ['OTE', 'OTL', 'OTU', 'MOV']:
                    if pattern in self.instruction_patterns:
                        matches = re.findall(self.instruction_patterns[pattern], rung_text)
                        for match in matches:
                            if isinstance(match, tuple):
                                if pattern == 'MOV':
                                    write_vars.add(match[1])  # MOV writes to second operand
                                else:
                                    write_vars.update(match)
                            else:
                                write_vars.add(match)
                
                # Track timers as both read and write
                for pattern in ['TON', 'TOF']:
                    if pattern in self.instruction_patterns:
                        matches = re.findall(self.instruction_patterns[pattern], rung_text)
                        for match in matches:
                            write_vars.add(match)
                            read_vars.add(f"{match}.DN")  # Timer done bit
                            read_vars.add(f"{match}.EN")  # Timer enable bit
                
                # Record variable usage
                for var in read_vars:
                    if var:
                        var = var.strip()
                        if var not in variable_reads:
                            variable_reads[var] = []
                        variable_reads[var].append(rung_node)
                
                for var in write_vars:
                    if var:
                        var = var.strip()
                        if var not in variable_writes:
                            variable_writes[var] = []
                        variable_writes[var].append(rung_node)
        
        # Add variable nodes and dependencies
        all_variables = set(variable_reads.keys()) | set(variable_writes.keys())
        
        for var in all_variables:
            G.add_node(var, type='variable')
            
            # Connect writers to variable
            for writer in variable_writes.get(var, []):
                G.add_edge(writer, var, type='write')
            
            # Connect variable to readers
            for reader in variable_reads.get(var, []):
                G.add_edge(var, reader, type='read')
        
        return G
    
    def _build_instruction_network_graph(self, ladder_routines: List[Dict[str, Any]]) -> nx.Graph:
        """Build instruction network graph showing instruction relationships"""
        G = nx.Graph()
        
        instruction_count = {}
        instruction_cooccurrence = {}
        
        for routine in ladder_routines:
            for rung in routine.get('rungs', []):
                rung_text = rung.get('text', '')
                
                # Find all instructions in this rung
                rung_instructions = []
                for instr_type, pattern in self.instruction_patterns.items():
                    if re.search(pattern, rung_text):
                        rung_instructions.append(instr_type)
                        instruction_count[instr_type] = instruction_count.get(instr_type, 0) + 1
                
                # Track co-occurrence of instructions
                for i, instr1 in enumerate(rung_instructions):
                    for instr2 in rung_instructions[i+1:]:
                        pair = tuple(sorted([instr1, instr2]))
                        instruction_cooccurrence[pair] = instruction_cooccurrence.get(pair, 0) + 1
        
        # Add instruction nodes
        for instr, count in instruction_count.items():
            G.add_node(instr, type='instruction', count=count)
        
        # Add edges for co-occurrence
        for (instr1, instr2), weight in instruction_cooccurrence.items():
            G.add_edge(instr1, instr2, weight=weight, type='cooccurrence')
        
        return G
    
    def _build_execution_flow_graph(self, ladder_routines: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build execution flow graph showing logical execution paths"""
        G = nx.DiGraph()
        
        for routine in ladder_routines:
            routine_name = f"{routine['program_name']}:{routine['routine_name']}"
            
            # Add start node for routine
            start_node = f"{routine_name}:START"
            G.add_node(start_node, type='start', routine=routine_name)
            
            prev_node = start_node
            
            for i, rung in enumerate(routine.get('rungs', [])):
                rung_node = f"{routine_name}:Rung_{i}"
                rung_text = rung.get('text', '')
                
                G.add_node(rung_node, type='rung', text=rung_text)
                G.add_edge(prev_node, rung_node, type='flow')
                
                # Look for conditional branches (timers create conditional paths)
                if re.search(r'TON|TOF|CTU|CTD', rung_text):
                    # Create conditional execution paths
                    true_path = f"{rung_node}:TRUE"
                    false_path = f"{rung_node}:FALSE"
                    
                    G.add_node(true_path, type='condition', state='true')
                    G.add_node(false_path, type='condition', state='false')
                    
                    G.add_edge(rung_node, true_path, type='condition_true')
                    G.add_edge(rung_node, false_path, type='condition_false')
                    
                    prev_node = rung_node
                else:
                    prev_node = rung_node
            
            # Add end node
            end_node = f"{routine_name}:END"
            G.add_node(end_node, type='end', routine=routine_name)
            G.add_edge(prev_node, end_node, type='flow')
        
        return G
    
    def _graph_to_dict(self, graph: nx.Graph) -> Dict[str, Any]:
        """Convert NetworkX graph to dictionary format"""
        return {
            'nodes': [
                {
                    'id': node,
                    'properties': data
                }
                for node, data in graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'properties': data
                }
                for source, target, data in graph.edges(data=True)
            ],
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of built graphs"""
        summary = {}
        
        for graph_type, graph in self.graphs.items():
            summary[graph_type.value] = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'is_directed': isinstance(graph, nx.DiGraph),
                'density': nx.density(graph) if graph.number_of_nodes() > 1 else 0
            }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on graph analysis"""
        recommendations = []
        
        # Control flow recommendations
        if GraphType.CONTROL_FLOW in self.graphs:
            cf_graph = self.graphs[GraphType.CONTROL_FLOW]
            
            # Find routines with no incoming calls
            isolated_routines = []
            for node in cf_graph.nodes():
                if cf_graph.nodes[node].get('type') == 'routine':
                    incoming_calls = [edge for edge in cf_graph.in_edges(node) 
                                    if cf_graph.edges[edge].get('type') == 'call']
                    if not incoming_calls and not node.endswith(':MainRoutine'):
                        isolated_routines.append(node)
            
            if isolated_routines:
                recommendations.append(f"Found {len(isolated_routines)} potentially unused routines")
        
        # Data dependency recommendations
        if GraphType.DATA_DEPENDENCY in self.graphs:
            dd_graph = self.graphs[GraphType.DATA_DEPENDENCY]
            
            # Find variables that are written but never read
            unused_vars = []
            for node in dd_graph.nodes():
                if dd_graph.nodes[node].get('type') == 'variable':
                    outgoing_reads = [edge for edge in dd_graph.out_edges(node)
                                    if dd_graph.edges[edge].get('type') == 'read']
                    if not outgoing_reads:
                        unused_vars.append(node)
            
            if unused_vars:
                recommendations.append(f"Found {len(unused_vars)} variables that are written but never read")
        
        # Instruction network recommendations
        if GraphType.INSTRUCTION_NETWORK in self.graphs:
            in_graph = self.graphs[GraphType.INSTRUCTION_NETWORK]
            
            if in_graph.number_of_nodes() > 0:
                # Find most commonly used instructions
                most_common = max(in_graph.nodes(data=True), 
                                key=lambda x: x[1].get('count', 0))
                recommendations.append(f"Most frequently used instruction: {most_common[0]} ({most_common[1].get('count', 0)} times)")
        
        if not recommendations:
            recommendations.append("Code structure appears well-organized")
        
        return recommendations
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate detailed statistics"""
        total_nodes = sum(g.number_of_nodes() for g in self.graphs.values())
        total_edges = sum(g.number_of_edges() for g in self.graphs.values())
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'graph_types_built': len(self.graphs),
            'complexity_score': min(100, total_nodes + total_edges * 0.5)
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'build_successful': False,
            'graphs': {gt.value: {'nodes': [], 'edges': [], 'node_count': 0, 'edge_count': 0} 
                      for gt in GraphType},
            'summary': {gt.value: {'nodes': 0, 'edges': 0, 'is_directed': True, 'density': 0} 
                       for gt in GraphType},
            'recommendations': ['No ladder logic found in the L5X file'],
            'statistics': {'total_nodes': 0, 'total_edges': 0, 'graph_types_built': 0, 'complexity_score': 0}
        }
