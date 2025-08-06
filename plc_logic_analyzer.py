#!/usr/bin/env python3
"""
PLC Logic Search and Analysis Tool
This script analyzes the analysis_report.json file to search variables and build logic representations
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class Variable:
    name: str
    data_type: str
    scope: str
    description: str
    value: Any
    program_name: str = ""
    external_access: str = ""
    array_dimensions: List = None

@dataclass
class Instruction:
    type: str
    operands: List[str]
    rung_number: int
    routine_name: str
    program_name: str
    text: str

@dataclass
class LogicPath:
    start_condition: str
    end_action: str
    intermediate_steps: List[str]
    variables_involved: Set[str]
    instructions_used: List[str]

class PLCLogicAnalyzer:
    """Analyzes PLC logic from analysis report JSON"""
    
    def __init__(self, report_file: str):
        self.report_file = Path(report_file)
        self.data = self._load_report()
        self.variables = {}
        self.instructions = []
        self.logic_paths = []
        self.variable_dependencies = nx.DiGraph()
        
        # Instruction parsing patterns
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
            'NEQ': r'NEQ\(([^,)]+),([^)]+)\)',  # Not Equal
            'LES': r'LES\(([^,)]+),([^)]+)\)',  # Less Than
            'GRT': r'GRT\(([^,)]+),([^)]+)\)',  # Greater Than
        }
        
        self._parse_variables()
        self._parse_instructions()
        self._build_dependencies()
    
    def _load_report(self) -> Dict[str, Any]:
        """Load the analysis report JSON file"""
        try:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading report file: {e}")
            sys.exit(1)
    
    def _parse_variables(self):
        """Parse all variables from the report"""
        print("Parsing variables...")
        
        # Parse controller tags
        l5x_data = self.data.get('analysis_results', {}).get('l5x_parsing', {})
        controller_tags = l5x_data.get('controller_tags', [])
        
        for tag in controller_tags:
            var = Variable(
                name=tag.get('name', ''),
                data_type=tag.get('data_type', ''),
                scope=tag.get('scope', 'controller'),
                description=tag.get('description', ''),
                value=tag.get('value', ''),
                external_access=tag.get('external_access', ''),
                array_dimensions=tag.get('array_dimensions', [])
            )
            self.variables[var.name] = var
        
        # Parse program tags
        programs = l5x_data.get('programs', [])
        for program in programs:
            program_name = program.get('name', '')
            program_tags = program.get('tags', [])
            
            for tag in program_tags:
                var = Variable(
                    name=tag.get('name', ''),
                    data_type=tag.get('data_type', ''),
                    scope='program',
                    description=tag.get('description', ''),
                    value=tag.get('value', ''),
                    program_name=program_name,
                    external_access=tag.get('external_access', ''),
                    array_dimensions=tag.get('array_dimensions', [])
                )
                full_name = f"{program_name}.{var.name}" if program_name else var.name
                self.variables[full_name] = var
        
        print(f"Found {len(self.variables)} variables")
    
    def _parse_instructions(self):
        """Parse all instructions from ladder logic"""
        print("Parsing instructions...")
        
        programs = self.data.get('analysis_results', {}).get('l5x_parsing', {}).get('programs', [])
        
        for program in programs:
            program_name = program.get('name', '')
            routines = program.get('routines', [])
            
            for routine in routines:
                routine_name = routine.get('name', '')
                rungs = routine.get('rungs', [])
                
                for rung in rungs:
                    rung_number = rung.get('number', 0)
                    rung_text = rung.get('text', '')
                    
                    # Parse instructions from rung text
                    for instr_type, pattern in self.instruction_patterns.items():
                        matches = re.finditer(pattern, rung_text)
                        for match in matches:
                            operands = [match.group(i) for i in range(1, match.lastindex + 1)] if match.lastindex else []
                            
                            instruction = Instruction(
                                type=instr_type,
                                operands=operands,
                                rung_number=rung_number,
                                routine_name=routine_name,
                                program_name=program_name,
                                text=rung_text
                            )
                            self.instructions.append(instruction)
        
        print(f"Found {len(self.instructions)} instructions")
    
    def _build_dependencies(self):
        """Build variable dependency graph"""
        print("Building dependency graph...")
        
        # Add all variables as nodes
        for var_name in self.variables:
            self.variable_dependencies.add_node(var_name, **asdict(self.variables[var_name]))
        
        # Add dependencies based on instructions
        for instruction in self.instructions:
            if instruction.type in ['XIC', 'XIO', 'EQU', 'GEQ', 'LEQ', 'NEQ', 'LES', 'GRT']:
                # These read from variables
                for operand in instruction.operands:
                    operand = operand.strip()
                    if operand in self.variables:
                        # This variable is read by this instruction location
                        self.variable_dependencies.add_edge(
                            operand, 
                            f"{instruction.program_name}.{instruction.routine_name}.Rung_{instruction.rung_number}",
                            relationship='read'
                        )
            
            elif instruction.type in ['OTE', 'OTL', 'OTU']:
                # These write to variables
                for operand in instruction.operands:
                    operand = operand.strip()
                    if operand in self.variables:
                        self.variable_dependencies.add_edge(
                            f"{instruction.program_name}.{instruction.routine_name}.Rung_{instruction.rung_number}",
                            operand,
                            relationship='write'
                        )
            
            elif instruction.type == 'MOV':
                # MOV reads from first operand, writes to second
                if len(instruction.operands) >= 2:
                    source = instruction.operands[0].strip()
                    dest = instruction.operands[1].strip()
                    
                    rung_id = f"{instruction.program_name}.{instruction.routine_name}.Rung_{instruction.rung_number}"
                    
                    if source in self.variables:
                        self.variable_dependencies.add_edge(source, rung_id, relationship='read')
                    if dest in self.variables:
                        self.variable_dependencies.add_edge(rung_id, dest, relationship='write')
        
        print(f"Built dependency graph with {self.variable_dependencies.number_of_nodes()} nodes and {self.variable_dependencies.number_of_edges()} edges")
    
    def search_variable(self, variable_name: str, fuzzy: bool = False) -> List[Variable]:
        """Search for variables by name"""
        results = []
        
        if fuzzy:
            # Fuzzy search - case insensitive partial match
            for name, var in self.variables.items():
                if variable_name.lower() in name.lower():
                    results.append(var)
        else:
            # Exact match
            if variable_name in self.variables:
                results.append(self.variables[variable_name])
        
        return results
    
    def find_variable_usage(self, variable_name: str) -> Dict[str, List[Instruction]]:
        """Find all instructions that use a specific variable"""
        usage = {'reads': [], 'writes': []}
        
        for instruction in self.instructions:
            for operand in instruction.operands:
                if operand.strip() == variable_name:
                    if instruction.type in ['XIC', 'XIO', 'EQU', 'GEQ', 'LEQ', 'NEQ', 'LES', 'GRT']:
                        usage['reads'].append(instruction)
                    elif instruction.type in ['OTE', 'OTL', 'OTU']:
                        usage['writes'].append(instruction)
                    elif instruction.type == 'MOV':
                        if instruction.operands[0].strip() == variable_name:
                            usage['reads'].append(instruction)
                        elif instruction.operands[1].strip() == variable_name:
                            usage['writes'].append(instruction)
        
        return usage
    
    def trace_logic_path(self, start_var: str, max_depth: int = 5) -> List[LogicPath]:
        """Trace logic paths starting from a variable"""
        paths = []
        visited = set()
        
        def _trace_recursive(current_var: str, path: List[str], depth: int):
            if depth > max_depth or current_var in visited:
                return
            
            visited.add(current_var)
            
            # Find instructions that read this variable
            for instruction in self.instructions:
                if any(op.strip() == current_var for op in instruction.operands):
                    if instruction.type in ['XIC', 'XIO']:
                        # This is a condition - trace what it enables
                        rung_instructions = [i for i in self.instructions 
                                           if i.rung_number == instruction.rung_number 
                                           and i.routine_name == instruction.routine_name
                                           and i.program_name == instruction.program_name]
                        
                        for rung_instr in rung_instructions:
                            if rung_instr.type in ['OTE', 'OTL', 'OTU']:
                                for operand in rung_instr.operands:
                                    if operand.strip() != current_var:
                                        logic_path = LogicPath(
                                            start_condition=current_var,
                                            end_action=operand.strip(),
                                            intermediate_steps=path + [f"{instruction.type}({current_var})"],
                                            variables_involved={current_var, operand.strip()},
                                            instructions_used=[instruction.type, rung_instr.type]
                                        )
                                        paths.append(logic_path)
                                        
                                        # Continue tracing from the output
                                        _trace_recursive(operand.strip(), 
                                                       path + [f"{instruction.type}({current_var})", f"{rung_instr.type}({operand.strip()})"],
                                                       depth + 1)
        
        _trace_recursive(start_var, [], 0)
        return paths
    
    def find_safety_circuits(self) -> List[Dict[str, Any]]:
        """Find potential safety circuits"""
        safety_keywords = ['estop', 'emergency', 'safety', 'stop', 'fault', 'alarm']
        safety_circuits = []
        
        for var_name, var in self.variables.items():
            var_lower = var_name.lower()
            desc_lower = var.description.lower()
            
            if any(keyword in var_lower or keyword in desc_lower for keyword in safety_keywords):
                usage = self.find_variable_usage(var_name)
                paths = self.trace_logic_path(var_name, max_depth=3)
                
                safety_circuits.append({
                    'variable': var,
                    'usage': usage,
                    'logic_paths': paths,
                    'safety_score': len(usage['reads']) + len(usage['writes']) + len(paths)
                })
        
        return sorted(safety_circuits, key=lambda x: x['safety_score'], reverse=True)
    
    def find_timer_circuits(self) -> List[Dict[str, Any]]:
        """Find timer-based circuits"""
        timer_circuits = []
        
        for var_name, var in self.variables.items():
            if var.data_type == 'TIMER':
                usage = self.find_variable_usage(var_name)
                paths = self.trace_logic_path(var_name, max_depth=3)
                
                # Find timer instruction
                timer_instruction = None
                for instruction in self.instructions:
                    if instruction.type in ['TON', 'TOF'] and any(op.strip() == var_name for op in instruction.operands):
                        timer_instruction = instruction
                        break
                
                timer_circuits.append({
                    'timer_variable': var,
                    'timer_instruction': timer_instruction,
                    'usage': usage,
                    'logic_paths': paths
                })
        
        return timer_circuits
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'total_variables': len(self.variables),
                'total_instructions': len(self.instructions),
                'controller_tags': len([v for v in self.variables.values() if v.scope == 'controller']),
                'program_tags': len([v for v in self.variables.values() if v.scope == 'program']),
                'instruction_types': {}
            },
            'variables_by_type': defaultdict(list),
            'safety_circuits': self.find_safety_circuits(),
            'timer_circuits': self.find_timer_circuits(),
            'variable_dependencies': {
                'nodes': self.variable_dependencies.number_of_nodes(),
                'edges': self.variable_dependencies.number_of_edges()
            }
        }
        
        # Count instruction types
        for instruction in self.instructions:
            instr_type = instruction.type
            if instr_type not in report['summary']['instruction_types']:
                report['summary']['instruction_types'][instr_type] = 0
            report['summary']['instruction_types'][instr_type] += 1
        
        # Group variables by type
        for var in self.variables.values():
            report['variables_by_type'][var.data_type].append(asdict(var))
        
        return report
    
    def export_dependency_graph(self, output_file: str = "variable_dependencies.png"):
        """Export dependency graph as image"""
        try:
            plt.figure(figsize=(16, 12))
            
            # Filter to show only variable-to-variable dependencies
            var_graph = nx.DiGraph()
            for node in self.variable_dependencies.nodes():
                if node in self.variables:
                    var_graph.add_node(node)
            
            # Add edges between variables (through intermediate nodes)
            for var1 in var_graph.nodes():
                for var2 in var_graph.nodes():
                    if var1 != var2:
                        try:
                            path = nx.shortest_path(self.variable_dependencies, var1, var2)
                            if len(path) <= 3:  # Direct or one-hop connection
                                var_graph.add_edge(var1, var2)
                        except nx.NetworkXNoPath:
                            continue
            
            if var_graph.number_of_nodes() > 0:
                pos = nx.spring_layout(var_graph, k=2, iterations=50)
                
                # Color nodes by variable type
                node_colors = []
                for node in var_graph.nodes():
                    var = self.variables[node]
                    if var.data_type == 'BOOL':
                        node_colors.append('lightblue')
                    elif var.data_type == 'TIMER':
                        node_colors.append('orange')
                    elif var.data_type in ['DINT', 'INT']:
                        node_colors.append('lightgreen')
                    else:
                        node_colors.append('lightgray')
                
                nx.draw(var_graph, pos, 
                       node_color=node_colors,
                       node_size=1000,
                       font_size=8,
                       font_weight='bold',
                       arrows=True,
                       arrowsize=20,
                       edge_color='gray',
                       alpha=0.7)
                
                # Add labels
                labels = {node: node.split('.')[-1] if '.' in node else node for node in var_graph.nodes()}
                nx.draw_networkx_labels(var_graph, pos, labels, font_size=6)
                
                plt.title("PLC Variable Dependencies", fontsize=16)
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Dependency graph saved to {output_file}")
            else:
                print("No variable dependencies found to visualize")
                
        except Exception as e:
            print(f"Error creating dependency graph: {e}")

def main():
    parser = argparse.ArgumentParser(description="PLC Logic Search and Analysis Tool")
    parser.add_argument("report_file", help="Path to analysis_report.json file")
    parser.add_argument("--search", "-s", help="Search for a variable")
    parser.add_argument("--fuzzy", "-f", action="store_true", help="Use fuzzy search")
    parser.add_argument("--trace", "-t", help="Trace logic paths from variable")
    parser.add_argument("--usage", "-u", help="Find usage of variable")
    parser.add_argument("--safety", action="store_true", help="Find safety circuits")
    parser.add_argument("--timers", action="store_true", help="Find timer circuits")
    parser.add_argument("--report", "-r", help="Generate full report to file")
    parser.add_argument("--graph", "-g", help="Export dependency graph to image file")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PLCLogicAnalyzer(args.report_file)
    
    if args.search:
        results = analyzer.search_variable(args.search, args.fuzzy)
        print(f"\\nFound {len(results)} variable(s) matching '{args.search}':")
        for var in results:
            print(f"  {var.name} ({var.data_type}) - {var.description}")
    
    if args.usage:
        usage = analyzer.find_variable_usage(args.usage)
        print(f"\\nUsage of variable '{args.usage}':")
        print(f"  Reads: {len(usage['reads'])} instructions")
        print(f"  Writes: {len(usage['writes'])} instructions")
        
        for instruction in usage['reads'][:5]:  # Show first 5
            print(f"    READ: {instruction.type} in {instruction.program_name}.{instruction.routine_name} Rung {instruction.rung_number}")
        
        for instruction in usage['writes'][:5]:  # Show first 5
            print(f"    WRITE: {instruction.type} in {instruction.program_name}.{instruction.routine_name} Rung {instruction.rung_number}")
    
    if args.trace:
        paths = analyzer.trace_logic_path(args.trace)
        print(f"\\nLogic paths from '{args.trace}':")
        for i, path in enumerate(paths[:10]):  # Show first 10
            print(f"  Path {i+1}: {path.start_condition} -> {path.end_action}")
            print(f"    Steps: {' -> '.join(path.intermediate_steps)}")
    
    if args.safety:
        safety_circuits = analyzer.find_safety_circuits()
        print(f"\\nFound {len(safety_circuits)} potential safety circuits:")
        for circuit in safety_circuits[:10]:  # Show top 10
            var = circuit['variable']
            print(f"  {var.name} ({var.data_type}) - Score: {circuit['safety_score']}")
            print(f"    Description: {var.description}")
    
    if args.timers:
        timer_circuits = analyzer.find_timer_circuits()
        print(f"\\nFound {len(timer_circuits)} timer circuits:")
        for circuit in timer_circuits:
            var = circuit['timer_variable']
            print(f"  {var.name} - {var.description}")
    
    if args.report:
        report = analyzer.generate_report()
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Full analysis report saved to {args.report}")
    
    if args.graph:
        analyzer.export_dependency_graph(args.graph)
    
    # Default: show summary
    if not any([args.search, args.usage, args.trace, args.safety, args.timers, args.report, args.graph]):
        print("\\n=== PLC Logic Analysis Summary ===")
        print(f"Variables: {len(analyzer.variables)}")
        print(f"Instructions: {len(analyzer.instructions)}")
        print(f"Controller Tags: {len([v for v in analyzer.variables.values() if v.scope == 'controller'])}")
        print(f"Program Tags: {len([v for v in analyzer.variables.values() if v.scope == 'program'])}")
        
        # Show variable types
        type_counts = defaultdict(int)
        for var in analyzer.variables.values():
            type_counts[var.data_type] += 1
        
        print("\\nVariable Types:")
        for var_type, count in sorted(type_counts.items()):
            print(f"  {var_type}: {count}")
        
        # Show instruction types
        instr_counts = defaultdict(int)
        for instruction in analyzer.instructions:
            instr_counts[instruction.type] += 1
        
        print("\\nInstruction Types:")
        for instr_type, count in sorted(instr_counts.items()):
            print(f"  {instr_type}: {count}")

if __name__ == "__main__":
    main()
