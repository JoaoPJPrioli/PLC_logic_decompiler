#!/usr/bin/env python3
"""
Enhanced L5X Parser Integration for Step 12

This module extends the existing enhanced L5X parser to work seamlessly
with the routine analyzer, providing comprehensive program structure
analysis and subroutine call detection.

Author: GitHub Copilot
Date: July 31, 2025
"""

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .enhanced_l5x_parser import EnhancedL5XParser
from ..analysis.routine_analyzer import RoutineAnalyzer, ProgramStructure
from ..models.ladder_logic import LadderRoutine


@dataclass
class RoutineAnalysisResult:
    """Results from comprehensive routine analysis"""
    programs: Dict[str, ProgramStructure]
    analysis_summary: Dict[str, Any]
    call_graph_files: Dict[str, Dict[str, str]]
    performance_metrics: Dict[str, float]


class RoutineAnalysisIntegrator:
    """
    Integration layer for routine analysis with enhanced L5X parsing.
    
    This class combines the enhanced L5X parser with the routine analyzer
    to provide comprehensive program structure analysis.
    """
    
    def __init__(self, enhanced_parser: Optional[EnhancedL5XParser] = None):
        """
        Initialize the routine analysis integrator.
        
        Args:
            enhanced_parser: Optional enhanced L5X parser instance
        """
        self.logger = logging.getLogger(__name__)
        self.enhanced_parser = enhanced_parser or EnhancedL5XParser()
        self.routine_analyzer = RoutineAnalyzer()
        self.analysis_results: Optional[RoutineAnalysisResult] = None
    
    def analyze_l5x_file(self, file_path: str, output_dir: str = "routine_analysis") -> RoutineAnalysisResult:
        """
        Perform comprehensive routine analysis on an L5X file.
        
        Args:
            file_path: Path to the L5X file
            output_dir: Directory for output files
            
        Returns:
            RoutineAnalysisResult with comprehensive analysis data
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting comprehensive routine analysis of {file_path}")
        
        try:
            # Step 1: Parse L5X file with enhanced parser
            self.logger.info("Step 1: Enhanced L5X parsing")
            parsing_start = time.time()
            
            if not self.enhanced_parser.load_file(file_path):
                raise RuntimeError("Failed to parse L5X file")
            
            parsing_time = time.time() - parsing_start
            self.logger.info(f"L5X parsing completed in {parsing_time:.2f}s")
            
            # Step 2: Extract program and routine information
            self.logger.info("Step 2: Extracting program information")
            programs_data = self._extract_programs_data()
            ladder_routines = self._extract_ladder_routines()
            
            # Step 3: Perform routine analysis
            self.logger.info("Step 3: Performing routine analysis")
            analysis_start = time.time()
            
            programs = self.routine_analyzer.analyze_programs(programs_data, ladder_routines)
            
            analysis_time = time.time() - analysis_start
            self.logger.info(f"Routine analysis completed in {analysis_time:.2f}s")
            
            # Step 4: Generate analysis summary
            self.logger.info("Step 4: Generating analysis summary")
            summary = self.routine_analyzer.get_analysis_summary()
            
            # Step 5: Export call graphs
            self.logger.info("Step 5: Exporting call graphs")
            export_start = time.time()
            
            call_graph_files = self.routine_analyzer.export_call_graphs(output_dir)
            
            export_time = time.time() - export_start
            self.logger.info(f"Call graph export completed in {export_time:.2f}s")
            
            # Step 6: Calculate performance metrics
            total_time = time.time() - start_time
            performance_metrics = {
                'total_analysis_time': round(total_time, 3),
                'parsing_time': round(parsing_time, 3),
                'routine_analysis_time': round(analysis_time, 3),
                'export_time': round(export_time, 3),
                'routines_per_second': round(summary['statistics']['routines_analyzed'] / total_time, 2),
                'calls_per_second': round(summary['statistics']['calls_detected'] / total_time, 2)
            }
            
            # Create comprehensive result
            self.analysis_results = RoutineAnalysisResult(
                programs=programs,
                analysis_summary=summary,
                call_graph_files=call_graph_files,
                performance_metrics=performance_metrics
            )
            
            self.logger.info(f"Comprehensive routine analysis completed successfully in {total_time:.2f}s")
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Error during routine analysis: {e}")
            raise
    
    def _extract_programs_data(self) -> Dict[str, Dict[str, Any]]:
        """Extract program data from enhanced parser"""
        programs_data = {}
        
        # Get basic program information from enhanced parser
        basic_programs = self.enhanced_parser.get_programs()
        
        if not basic_programs:
            self.logger.warning("No programs found in L5X file")
            return programs_data
        
        # Enhanced program data extraction
        for program_name, program_info in basic_programs.items():
            enhanced_program_info = {
                'name': program_name,
                'type': program_info.get('type', 'PROGRAM'),
                'description': program_info.get('description', ''),
                'disabled': program_info.get('disabled', False),
                'routines': {}
            }
            
            # Extract routine information
            routines = program_info.get('routines', [])
            
            # Handle both list and dictionary formats for routines
            if isinstance(routines, list):
                # Routines are stored as a list of names
                for routine_name in routines:
                    enhanced_routine_info = {
                        'name': routine_name,
                        'type': 'RLL',  # Default type
                        'description': '',
                        'routine_path': f"{program_name}.{routine_name}"
                    }
                    enhanced_program_info['routines'][routine_name] = enhanced_routine_info
            elif isinstance(routines, dict):
                # Routines are stored as a dictionary with detailed info
                for routine_name, routine_info in routines.items():
                    enhanced_routine_info = {
                        'name': routine_name,
                        'type': routine_info.get('type', 'RLL'),
                        'description': routine_info.get('description', ''),
                        'routine_path': f"{program_name}.{routine_name}"
                    }
                    enhanced_program_info['routines'][routine_name] = enhanced_routine_info
            
            programs_data[program_name] = enhanced_program_info
        
        self.logger.info(f"Extracted data for {len(programs_data)} programs")
        return programs_data
    
    def _extract_ladder_routines(self) -> Dict[str, LadderRoutine]:
        """Extract ladder logic routines from enhanced parser"""
        ladder_routines = {}
        
        # Get ladder logic from enhanced parser
        if hasattr(self.enhanced_parser, 'ladder_extractor') and self.enhanced_parser.ladder_extractor:
            basic_routines = self.enhanced_parser.ladder_extractor.routines
            
            for routine_key, basic_routine in basic_routines.items():
                # Convert basic routine to comprehensive ladder routine
                ladder_routine = self._convert_to_ladder_routine(routine_key, basic_routine)
                if ladder_routine:
                    ladder_routines[routine_key] = ladder_routine
        
        self.logger.info(f"Extracted {len(ladder_routines)} ladder logic routines")
        return ladder_routines
    
    def _convert_to_ladder_routine(self, routine_key: str, basic_routine: Any) -> Optional[LadderRoutine]:
        """Convert basic routine to comprehensive LadderRoutine"""
        try:
            # Extract program and routine names
            if '.' in routine_key:
                program_name, routine_name = routine_key.split('.', 1)
            else:
                program_name = 'Unknown'
                routine_name = routine_key
            
            # Create comprehensive ladder routine
            ladder_routine = LadderRoutine(
                name=routine_name,
                program_name=program_name,
                routine_type=getattr(basic_routine, 'routine_type', 'RLL'),
                rungs=getattr(basic_routine, 'rungs', [])
            )
            
            return ladder_routine
            
        except Exception as e:
            self.logger.error(f"Error converting routine {routine_key}: {e}")
            return None
    
    def generate_routine_report(self, output_file: str = "routine_analysis_report.html") -> str:
        """
        Generate comprehensive HTML report of routine analysis.
        
        Args:
            output_file: Output file path for the report
            
        Returns:
            Path to the generated report file
        """
        if not self.analysis_results:
            raise RuntimeError("No analysis results available. Run analyze_l5x_file first.")
        
        self.logger.info(f"Generating routine analysis report: {output_file}")
        
        html_content = self._generate_html_report()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Routine analysis report generated: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content"""
        results = self.analysis_results
        summary = results.analysis_summary
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PLC Routine Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .program-section {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }}
        .program-header {{
            background: #007bff;
            color: white;
            padding: 15px;
            font-weight: bold;
        }}
        .program-content {{
            padding: 20px;
        }}
        .routine-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .routine-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
        .routine-name {{
            font-weight: bold;
            color: #495057;
        }}
        .routine-stats {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #6c757d;
        }}
        .call-graph-link {{
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 8px 16px;
            background: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .call-graph-link:hover {{
            background: #218838;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        .error {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .performance-section {{
            background: #e9ecef;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PLC Routine Analysis Report</h1>
        <p><strong>Generated:</strong> {results.performance_metrics.get('timestamp', 'Unknown')}</p>
        
        <h2>Analysis Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{summary['global_metrics']['total_programs']}</div>
                <div class="metric-label">Programs Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['global_metrics']['total_routines']}</div>
                <div class="metric-label">Total Routines</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['global_metrics']['total_calls']}</div>
                <div class="metric-label">Subroutine Calls</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['global_metrics']['max_call_depth_global']}</div>
                <div class="metric-label">Max Call Depth</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['global_metrics']['programs_with_recursion']}</div>
                <div class="metric-label">Programs with Recursion</div>
            </div>
        </div>
        
        <div class="performance-section">
            <h3>Performance Metrics</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{results.performance_metrics['total_analysis_time']}s</div>
                    <div class="metric-label">Total Analysis Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results.performance_metrics['routines_per_second']}</div>
                    <div class="metric-label">Routines/Second</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results.performance_metrics['calls_per_second']}</div>
                    <div class="metric-label">Calls/Second</div>
                </div>
            </div>
        </div>
        
        <h2>Program Details</h2>
"""
        
        # Add program sections
        for program_name, program_summary in summary['programs'].items():
            program_structure = results.programs[program_name]
            
            # Recursion warning
            recursion_warning = ""
            if program_structure.has_recursion:
                cycles_str = ", ".join([" → ".join(cycle + [cycle[0]]) for cycle in program_structure.recursive_cycles])
                recursion_warning = f"""
                <div class="warning">
                    <strong>⚠️ Recursion Detected:</strong> {cycles_str}
                </div>
                """
            
            # Call graph links
            call_graph_links = ""
            if program_name in results.call_graph_files:
                for format_type, file_path in results.call_graph_files[program_name].items():
                    call_graph_links += f'<a href="{file_path}" class="call-graph-link">{format_type.upper()} Export</a>'
            
            html += f"""
        <div class="program-section">
            <div class="program-header">{program_name}</div>
            <div class="program-content">
                {recursion_warning}
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{program_summary['total_routines']}</div>
                        <div class="metric-label">Routines</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{program_summary['total_calls']}</div>
                        <div class="metric-label">Subroutine Calls</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{program_summary['max_call_depth']}</div>
                        <div class="metric-label">Max Call Depth</div>
                    </div>
                </div>
                
                <p><strong>Main Routine:</strong> {program_summary.get('main_routine', 'Not identified')}</p>
                <p><strong>Fault Routine:</strong> {program_summary.get('fault_routine', 'None')}</p>
                
                <h4>Call Graph Exports</h4>
                <p>{call_graph_links}</p>
                
                <h4>Most Complex Routines</h4>
                <table>
                    <tr>
                        <th>Routine Name</th>
                        <th>Complexity Score</th>
                        <th>Instructions</th>
                        <th>Calls Made</th>
                        <th>Est. Execution Time (ms)</th>
                    </tr>
"""
            
            for routine in program_summary['top_complex_routines']:
                html += f"""
                    <tr>
                        <td>{routine['name']}</td>
                        <td>{routine['complexity_score']}</td>
                        <td>{routine['instruction_count']}</td>
                        <td>{routine['calls_made']}</td>
                        <td>{routine['execution_time_estimate']}</td>
                    </tr>
"""
            
            html += """
                </table>
                
                <h4>All Routines</h4>
                <div class="routine-list">
"""
            
            for routine_name, routine_info in program_structure.routines.items():
                html += f"""
                    <div class="routine-card">
                        <div class="routine-name">{routine_name}</div>
                        <div>Type: {routine_info.routine_type.value}</div>
                        <div class="routine-stats">
                            Rungs: {routine_info.rung_count} | 
                            Instructions: {routine_info.instruction_count} | 
                            Calls: {len(routine_info.calls_made)} | 
                            Called by: {len(routine_info.called_by)}
                        </div>
                    </div>
"""
            
            html += """
                </div>
            </div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        return html
    
    def get_routine_call_chain(self, program_name: str, start_routine: str, 
                              end_routine: str) -> List[List[str]]:
        """
        Get call chains between two routines.
        
        Args:
            program_name: Name of the program
            start_routine: Starting routine name
            end_routine: Target routine name
            
        Returns:
            List of possible call chains
        """
        if not self.analysis_results:
            return []
        
        return self.routine_analyzer.get_call_chain(program_name, start_routine, end_routine)
    
    def simulate_program_execution(self, program_name: str, start_routine: str = None) -> Dict[str, Any]:
        """
        Simulate program execution and return call stack analysis.
        
        Args:
            program_name: Name of the program to simulate
            start_routine: Starting routine (uses main if not specified)
            
        Returns:
            Dictionary with simulation results
        """
        if not self.analysis_results:
            return {}
        
        call_stack = self.routine_analyzer.simulate_call_stack(program_name, start_routine)
        
        return {
            'max_depth_reached': call_stack.max_depth_reached,
            'final_stack_size': len(call_stack.stack),
            'execution_path': [f"{routine}:{rung}" for routine, rung in call_stack.stack]
        }
    
    def get_routine_dependencies(self, program_name: str, routine_name: str) -> Dict[str, List[str]]:
        """
        Get dependencies for a specific routine.
        
        Args:
            program_name: Name of the program
            routine_name: Name of the routine
            
        Returns:
            Dictionary with 'calls_made' and 'called_by' lists
        """
        if (not self.analysis_results or 
            program_name not in self.analysis_results.programs or
            routine_name not in self.analysis_results.programs[program_name].routines):
            return {'calls_made': [], 'called_by': []}
        
        routine_info = self.analysis_results.programs[program_name].routines[routine_name]
        
        return {
            'calls_made': [call.called_routine for call in routine_info.calls_made],
            'called_by': [call.caller_routine for call in routine_info.called_by]
        }
