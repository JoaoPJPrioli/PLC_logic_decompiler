"""
Timer and Counter Analysis Integration Module

This module provides integration between the timer/counter analyzer and the enhanced L5X parser,
building on the Step 12 routine analysis foundation to add specialized timer and counter capabilities.
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the enhanced L5X parser from Step 10
from src.parsers.enhanced_l5x_parser import EnhancedL5XParser
from src.parsers.l5x_parser import L5XParser

# Import the timer/counter analyzer
from src.analysis.timer_counter_analyzer import (
    TimerCounterAnalyzer, 
    TimerCounterAnalysisResult,
    TimerInfo,
    CounterInfo,
    TimingChain,
    CountingChain
)

# Import the routine analysis integration from Step 12
from src.parsers.routine_integration import RoutineAnalysisIntegrator


class TimerCounterAnalysisIntegrator:
    """
    Integration layer for timer and counter analysis with enhanced L5X parsing.
    
    This class combines the capabilities of:
    - Enhanced L5X Parser (Step 10)
    - Routine Analysis (Step 12) 
    - Timer/Counter Analysis (Step 13)
    
    To provide comprehensive timer and counter analysis of PLC systems.
    """
    
    def __init__(self, l5x_parser: Optional[L5XParser] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize parsers and analyzers
        self.enhanced_parser = EnhancedL5XParser()
        self.routine_integrator = RoutineAnalysisIntegrator(self.enhanced_parser)
        self.timer_counter_analyzer = TimerCounterAnalyzer()
        
        # Analysis results storage
        self.current_analysis_result: Optional[TimerCounterAnalysisResult] = None
        self.current_routine_analysis: Optional[Dict[str, Any]] = None
    
    def analyze_file(self, file_path: str, output_dir: str = "timer_counter_output") -> Dict[str, Any]:
        """
        Perform comprehensive timer and counter analysis on an L5X file.
        
        Args:
            file_path: Path to the L5X file
            output_dir: Directory for output files
            
        Returns:
            Dict containing complete analysis results
        """
        self.logger.info(f"Starting comprehensive timer and counter analysis of {file_path}")
        analysis_start = time.time()
        
        try:
            # Step 1: Enhanced L5X parsing
            self.logger.info("Step 1: Enhanced L5X parsing")
            parsing_start = time.time()
            
            if not self.enhanced_parser.load_file(file_path):
                raise RuntimeError("Failed to parse L5X file")
            
            parsing_time = time.time() - parsing_start
            self.logger.info(f"L5X parsing completed in {parsing_time:.2f}s")
            
            # Step 2: Extract ladder routines
            self.logger.info("Step 2: Extracting ladder routines")
            ladder_routines = self._extract_ladder_routines()
            self.logger.info(f"Extracted {len(ladder_routines)} ladder routines")
            
            # Step 3: Perform routine analysis (Step 12 integration)
            self.logger.info("Step 3: Performing routine analysis")
            routine_start = time.time()
            self.current_routine_analysis = self.routine_integrator.analyze_l5x_file(file_path, output_dir)
            routine_time = time.time() - routine_start
            self.logger.info(f"Routine analysis completed in {routine_time:.2f}s")
            
            # Step 4: Perform timer and counter analysis
            self.logger.info("Step 4: Performing timer and counter analysis")  
            timer_counter_start = time.time()
            self.current_analysis_result = self.timer_counter_analyzer.analyze_timers_and_counters(ladder_routines)
            timer_counter_time = time.time() - timer_counter_start
            self.logger.info(f"Timer and counter analysis completed in {timer_counter_time:.2f}s")
            
            # Step 5: Generate comprehensive analysis summary
            self.logger.info("Step 5: Generating analysis summary")
            analysis_summary = self._generate_analysis_summary()
            
            # Step 6: Export results and visualizations
            self.logger.info("Step 6: Exporting results")
            export_start = time.time()
            output_files = self._export_results(output_dir, file_path)
            export_time = time.time() - export_start
            self.logger.info(f"Results export completed in {export_time:.2f}s")
            
            total_time = time.time() - analysis_start
            self.logger.info(f"Comprehensive analysis completed successfully in {total_time:.2f}s")
            
            return {
                'timer_counter_analysis': self.current_analysis_result,
                'routine_analysis': self.current_routine_analysis,
                'analysis_summary': analysis_summary,
                'output_files': output_files,
                'performance_metrics': {
                    'total_time': total_time,
                    'parsing_time': parsing_time,
                    'routine_analysis_time': routine_time,
                    'timer_counter_analysis_time': timer_counter_time,
                    'export_time': export_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _extract_ladder_routines(self) -> Dict[str, Any]:
        """Extract ladder routine data from the enhanced parser."""
        try:
            # Get programs from the enhanced parser
            programs = self.enhanced_parser.get_programs()
            ladder_routines = {}
            
            for program_name, program_info in programs.items():
                routines = program_info.get('routines', [])
                
                for routine_name in routines:
                    try:
                        # Use the enhanced parser to get detailed routine information
                        routine_data = self._get_routine_details(program_name, routine_name)
                        if routine_data:
                            full_routine_name = f"{program_name}.{routine_name}"
                            ladder_routines[full_routine_name] = routine_data
                    except Exception as e:
                        self.logger.warning(f"Failed to extract routine {routine_name}: {e}")
            
            return ladder_routines
            
        except Exception as e:
            self.logger.error(f"Failed to extract ladder routines: {e}")
            return {}
    
    def _get_routine_details(self, program_name: str, routine_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific routine."""
        try:
            # Get the raw routine data from the L5X parser
            root = self.enhanced_parser.root
            if root is None:
                return None
            
            # Find the routine in the XML
            programs_element = root.find('.//Programs')
            if programs_element is None:
                return None
            
            for program_element in programs_element.findall('Program'):
                if program_element.get('Name') == program_name:
                    routines_element = program_element.find('Routines')
                    if routines_element is not None:
                        for routine_element in routines_element.findall('Routine'):
                            if routine_element.get('Name') == routine_name:
                                return self._parse_routine_xml(routine_element, program_name, routine_name)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get routine details for {program_name}.{routine_name}: {e}")
            return None
    
    def _parse_routine_xml(self, routine_element, program_name: str, routine_name: str) -> Dict[str, Any]:
        """Parse routine XML element to extract rung and instruction data."""
        routine_data = {
            'name': routine_name,
            'program': program_name,
            'type': routine_element.get('Type', 'RLL'),
            'rungs': []
        }
        
        # Find RLLContent for ladder logic
        rll_content = routine_element.find('.//RLLContent')
        if rll_content is not None:
            for rung_idx, rung_element in enumerate(rll_content.findall('Rung')):
                rung_data = self._parse_rung_xml(rung_element, rung_idx)
                if rung_data:
                    routine_data['rungs'].append(rung_data)
        
        return routine_data
    
    def _parse_rung_xml(self, rung_element, rung_idx: int) -> Optional[Dict[str, Any]]:
        """Parse a rung XML element to extract instruction data."""
        try:
            rung_data = {
                'number': rung_idx,
                'instructions': []
            }
            
            # Get rung text content which contains the instruction text
            text_element = rung_element.find('Text')
            if text_element is not None:
                rung_text = text_element.text or ""
                instructions = self._parse_rung_text(rung_text)
                rung_data['instructions'] = instructions
            
            return rung_data
            
        except Exception as e:
            self.logger.warning(f"Failed to parse rung {rung_idx}: {e}")
            return None
    
    def _parse_rung_text(self, rung_text: str) -> List[Dict[str, Any]]:
        """Parse rung text to extract individual instructions."""
        instructions = []
        
        # Simple instruction extraction - look for common timer/counter patterns
        timer_counter_patterns = [
            (r'TON\s*\([^)]+\)', 'TON'),
            (r'TOF\s*\([^)]+\)', 'TOF'), 
            (r'RTO\s*\([^)]+\)', 'RTO'),
            (r'CTU\s*\([^)]+\)', 'CTU'),
            (r'CTD\s*\([^)]+\)', 'CTD')
        ]
        
        import re
        for pattern, inst_name in timer_counter_patterns:
            matches = re.finditer(pattern, rung_text, re.IGNORECASE)
            for match in matches:
                instructions.append({
                    'name': inst_name,
                    'text': match.group(0),
                    'position': match.start()
                })
        
        # Sort instructions by position in the rung
        instructions.sort(key=lambda x: x.get('position', 0))
        
        return instructions
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis summary."""
        if not self.current_analysis_result:
            return {}
        
        result = self.current_analysis_result
        
        # Timer analysis summary
        timer_summary = {
            'total_timers': len(result.timers),
            'timer_types': {},
            'average_preset_time': 0.0,
            'total_timing_operations': len(result.timing_chains)
        }
        
        if result.timers:
            # Calculate timer type distribution
            for timer in result.timers.values():
                timer_type = timer.timer_type.value
                timer_summary['timer_types'][timer_type] = timer_summary['timer_types'].get(timer_type, 0) + 1
            
            # Calculate average preset time
            preset_times = [t.preset_value for t in result.timers.values() if t.preset_value]
            if preset_times:
                timer_summary['average_preset_time'] = sum(preset_times) / len(preset_times)
        
        # Counter analysis summary
        counter_summary = {
            'total_counters': len(result.counters),
            'counter_types': {},
            'average_preset_count': 0.0,
            'total_counting_operations': len(result.counting_chains)
        }
        
        if result.counters:
            # Calculate counter type distribution
            for counter in result.counters.values():
                counter_type = counter.counter_type.value
                counter_summary['counter_types'][counter_type] = counter_summary['counter_types'].get(counter_type, 0) + 1
            
            # Calculate average preset count
            preset_counts = [c.preset_value for c in result.counters.values() if c.preset_value]
            if preset_counts:
                counter_summary['average_preset_count'] = sum(preset_counts) / len(preset_counts)
        
        # Chain analysis summary
        chain_summary = {
            'timing_chains': len(result.timing_chains),
            'counting_chains': len(result.counting_chains),
            'critical_timing_paths': len([c for c in result.timing_chains if c.critical_path]),
            'critical_counting_paths': len([c for c in result.counting_chains if c.critical_path]),
            'longest_timing_chain': max([len(c.timers) for c in result.timing_chains], default=0),
            'longest_counting_chain': max([len(c.counters) for c in result.counting_chains], default=0)
        }
        
        return {
            'timer_summary': timer_summary,
            'counter_summary': counter_summary,
            'chain_summary': chain_summary,
            'analysis_metrics': result.analysis_metrics,
            'performance_data': result.performance_data
        }
    
    def _export_results(self, output_dir: str, source_file: str) -> Dict[str, str]:
        """Export analysis results to various formats."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        output_files = {}
        
        try:
            # Export timer analysis report
            timer_report_file = output_path / "timer_analysis_report.html"
            self._generate_timer_report(timer_report_file, source_file)
            output_files['timer_report'] = str(timer_report_file)
            
            # Export counter analysis report
            counter_report_file = output_path / "counter_analysis_report.html"
            self._generate_counter_report(counter_report_file, source_file)
            output_files['counter_report'] = str(counter_report_file)
            
            # Export timing chains
            timing_chains_file = output_path / "timing_chains.json"
            self._export_timing_chains(timing_chains_file)
            output_files['timing_chains'] = str(timing_chains_file)
            
            # Export counting chains
            counting_chains_file = output_path / "counting_chains.json"
            self._export_counting_chains(counting_chains_file)
            output_files['counting_chains'] = str(counting_chains_file)
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
        
        return output_files
    
    def _generate_timer_report(self, output_file: Path, source_file: str) -> None:
        """Generate HTML report for timer analysis."""
        if not self.current_analysis_result:
            return
        
        html_content = self._create_timer_html_report(source_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_counter_report(self, output_file: Path, source_file: str) -> None:
        """Generate HTML report for counter analysis."""
        if not self.current_analysis_result:
            return
        
        html_content = self._create_counter_html_report(source_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _create_timer_html_report(self, source_file: str) -> str:
        """Create HTML report content for timer analysis."""
        result = self.current_analysis_result
        if not result:
            return "<html><body><h1>No timer analysis results available</h1></body></html>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Timer Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .timer {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007acc; }}
                .chain {{ background-color: #fff5e6; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9900; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Timer Analysis Report</h1>
                <p><strong>Source File:</strong> {source_file}</p>
                <p><strong>Analysis Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total Timers:</strong> {len(result.timers)}</p>
                <p><strong>Timing Chains:</strong> {len(result.timing_chains)}</p>
            </div>
            
            <div class="section">
                <h2>Timer Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Timers</td><td>{len(result.timers)}</td></tr>
                    <tr><td>TON Timers</td><td>{len([t for t in result.timers.values() if t.timer_type.value == 'TON'])}</td></tr>
                    <tr><td>TOF Timers</td><td>{len([t for t in result.timers.values() if t.timer_type.value == 'TOF'])}</td></tr>
                    <tr><td>RTO Timers</td><td>{len([t for t in result.timers.values() if t.timer_type.value == 'RTO'])}</td></tr>
                    <tr><td>Timing Chains</td><td>{len(result.timing_chains)}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Timer Details</h2>
        """
        
        for timer_name, timer in result.timers.items():
            html += f"""
                <div class="timer">
                    <h3>{timer.name}</h3>
                    <p><strong>Type:</strong> {timer.timer_type.value}</p>
                    <p><strong>Tag:</strong> {timer.tag_name}</p>
                    <p><strong>Routine:</strong> {timer.routine_name}</p>
                    <p><strong>Rung:</strong> {timer.rung_number}</p>
                    <p><strong>Preset:</strong> {timer.preset_value or timer.preset_tag or 'N/A'}</p>
                    <p><strong>Est. Cycle Time:</strong> {timer.estimated_cycle_time:.3f}s</p>
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Timing Chains</h2>
        """
        
        for chain in result.timing_chains:
            html += f"""
                <div class="chain">
                    <h3>{chain.chain_id}</h3>
                    <p><strong>Type:</strong> {chain.chain_type}</p>
                    <p><strong>Total Time:</strong> {chain.total_time:.3f}s</p>
                    <p><strong>Critical Path:</strong> {'Yes' if chain.critical_path else 'No'}</p>
                    <p><strong>Timers:</strong> {', '.join(chain.timers)}</p>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_counter_html_report(self, source_file: str) -> str:
        """Create HTML report content for counter analysis."""
        result = self.current_analysis_result
        if not result:
            return "<html><body><h1>No counter analysis results available</h1></body></html>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Counter Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .counter {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #28a745; }}
                .chain {{ background-color: #e6f7ff; padding: 10px; margin: 10px 0; border-left: 4px solid #1890ff; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Counter Analysis Report</h1>
                <p><strong>Source File:</strong> {source_file}</p>
                <p><strong>Analysis Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total Counters:</strong> {len(result.counters)}</p>
                <p><strong>Counting Chains:</strong> {len(result.counting_chains)}</p>
            </div>
            
            <div class="section">
                <h2>Counter Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Counters</td><td>{len(result.counters)}</td></tr>
                    <tr><td>CTU Counters</td><td>{len([c for c in result.counters.values() if c.counter_type.value == 'CTU'])}</td></tr>
                    <tr><td>CTD Counters</td><td>{len([c for c in result.counters.values() if c.counter_type.value == 'CTD'])}</td></tr>
                    <tr><td>Counting Chains</td><td>{len(result.counting_chains)}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Counter Details</h2>
        """
        
        for counter_name, counter in result.counters.items():
            html += f"""
                <div class="counter">
                    <h3>{counter.name}</h3>
                    <p><strong>Type:</strong> {counter.counter_type.value}</p>
                    <p><strong>Tag:</strong> {counter.tag_name}</p>
                    <p><strong>Routine:</strong> {counter.routine_name}</p>
                    <p><strong>Rung:</strong> {counter.rung_number}</p>
                    <p><strong>Preset:</strong> {counter.preset_value or counter.preset_tag or 'N/A'}</p>
                    <p><strong>Est. Count Rate:</strong> {counter.estimated_count_rate:.2f}/s</p>
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Counting Chains</h2>
        """
        
        for chain in result.counting_chains:
            html += f"""
                <div class="chain">
                    <h3>{chain.chain_id}</h3>
                    <p><strong>Type:</strong> {chain.chain_type}</p>
                    <p><strong>Total Count:</strong> {chain.total_count}</p>
                    <p><strong>Critical Path:</strong> {'Yes' if chain.critical_path else 'No'}</p>
                    <p><strong>Counters:</strong> {', '.join(chain.counters)}</p>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _export_timing_chains(self, output_file: Path) -> None:
        """Export timing chains to JSON format."""
        if not self.current_analysis_result:
            return
        
        import json
        
        chains_data = []
        for chain in self.current_analysis_result.timing_chains:
            chains_data.append({
                'chain_id': chain.chain_id,
                'timers': chain.timers,
                'total_time': chain.total_time,
                'critical_path': chain.critical_path,
                'chain_type': chain.chain_type
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chains_data, f, indent=2)
    
    def _export_counting_chains(self, output_file: Path) -> None:
        """Export counting chains to JSON format."""
        if not self.current_analysis_result:
            return
        
        import json
        
        chains_data = []
        for chain in self.current_analysis_result.counting_chains:
            chains_data.append({
                'chain_id': chain.chain_id,
                'counters': chain.counters,
                'total_count': chain.total_count,
                'critical_path': chain.critical_path,
                'chain_type': chain.chain_type
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chains_data, f, indent=2)


def main():
    """Main function for testing the timer and counter analysis integration."""
    integrator = TimerCounterAnalysisIntegrator()
    
    # Test file path (update with actual file path)
    test_file = "sample.L5X"
    
    if os.path.exists(test_file):
        result = integrator.analyze_file(test_file, "test_output")
        print(f"Analysis completed successfully!")
        print(f"Output files: {result.get('output_files', {})}")
    else:
        print(f"Test file {test_file} not found")


if __name__ == "__main__":
    main()
