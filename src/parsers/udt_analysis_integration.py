"""
Step 14: UDT Analysis Integration
Integration layer connecting UDT analysis with enhanced L5X parsing and routine analysis.

This module provides comprehensive integration for UDT analysis including:
- Integration with enhanced L5X parser for UDT definitions and instances
- Connection with routine analysis for member access detection
- HTML report generation for UDT analysis results
- JSON export for UDT structures and relationships
- Performance metrics and analysis timing
"""

import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import asdict

# Import from the analysis module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from analysis.udt_analyzer import UDTAnalyzer, UDTType, MemberType

logger = logging.getLogger(__name__)


class SimpleL5XParser:
    """Simple L5X parser for UDT analysis."""
    
    def __init__(self, l5x_file_path: str):
        """Initialize simple L5X parser."""
        self.l5x_file_path = l5x_file_path
        self.root = None
        self.controller_name = ""
        self.controller_type = ""
        self.software_revision = ""
        
    def parse_controller_info(self):
        """Parse controller information."""
        try:
            tree = ET.parse(self.l5x_file_path)
            self.root = tree.getroot()
            
            # Get controller info from root attributes or Controller element
            controller = self.root.find('.//Controller')
            if controller is not None:
                self.controller_name = controller.get('Name', 'Unknown')
                self.controller_type = controller.get('Type', 'Unknown')
            else:
                # Try from root element
                self.controller_name = self.root.get('TargetName', 'Unknown')
                self.controller_type = self.root.get('TargetType', 'Unknown')
                
            self.software_revision = self.root.get('SoftwareRevision', 'Unknown')
            
        except Exception as e:
            logger.error(f"Error parsing controller info: {e}")
            
    def parse_tags(self):
        """Parse tags - placeholder for compatibility."""
        pass


class SimpleRoutineAnalyzer:
    """Simple routine analyzer for UDT analysis."""
    
    def __init__(self):
        """Initialize simple routine analyzer."""
        self.analyzed_routines = {}
        
    def add_analyzed_routine(self, name: str, rungs: List):
        """Add analyzed routine."""
        # Create a simple mock object for rungs
        class MockRoutine:
            def __init__(self, rungs):
                self.rungs = rungs
                
        self.analyzed_routines[name] = MockRoutine(rungs)


class UDTAnalysisIntegrator:
    """Integration layer for comprehensive UDT analysis."""
    
    def __init__(self, l5x_file_path: str):
        """Initialize UDT analysis integrator."""
        self.l5x_file_path = l5x_file_path
        self.enhanced_parser = SimpleL5XParser(l5x_file_path)
        self.routine_integrator = SimpleRoutineAnalyzer()
        self.udt_analyzer = UDTAnalyzer()
        self.analysis_start_time = None
        self.analysis_end_time = None
        
    def perform_comprehensive_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive UDT analysis."""
        logger.info("Starting comprehensive UDT analysis...")
        self.analysis_start_time = time.time()
        
        try:
            # Step 1: Parse L5X file
            logger.info("Step 1: Parsing L5X file...")
            self.enhanced_parser.parse_controller_info()
            self.enhanced_parser.parse_tags()
            
            # Step 2: Analyze UDT definitions (from DataTypes section)
            logger.info("Step 2: Analyzing UDT definitions...")
            data_types = self.enhanced_parser.root.find('.//DataTypes') if self.enhanced_parser.root is not None else None
            self.udt_analyzer.analyze_udt_definitions(data_types)
            
            # Step 3: Analyze UDT instances (from Tags section)
            logger.info("Step 3: Analyzing UDT instances...")
            tags = self.enhanced_parser.root.find('.//Tags') if self.enhanced_parser.root is not None else None
            self.udt_analyzer.analyze_tag_instances(tags)
            
            # Step 4: Perform routine analysis for member access detection
            logger.info("Step 4: Analyzing routines for member access patterns...")
            
            # Parse routines from L5X and add to routine analyzer
            if self.enhanced_parser.root is not None:
                routines = self.enhanced_parser.root.findall('.//Routine')
                for routine in routines:
                    routine_name = routine.get('Name', 'Unknown')
                    rungs = routine.findall('.//Rung')
                    self.routine_integrator.add_analyzed_routine(routine_name, rungs)
            
            self.udt_analyzer.analyze_member_accesses(self.routine_integrator)
            
            # Step 5: Analyze UDT relationships
            logger.info("Step 5: Analyzing UDT relationships...")
            self.udt_analyzer.analyze_udt_relationships()
            
            self.analysis_end_time = time.time()
            
            # Prepare comprehensive results
            results = self._prepare_analysis_results()
            
            logger.info(f"UDT analysis completed in {self.get_analysis_duration():.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive UDT analysis: {e}")
            self.analysis_end_time = time.time()
            raise
            
    def _prepare_analysis_results(self) -> Dict[str, Any]:
        """Prepare comprehensive analysis results."""
        return {
            'controller_info': {
                'name': self.enhanced_parser.controller_name,
                'type': self.enhanced_parser.controller_type,
                'revision': self.enhanced_parser.software_revision
            },
            'udt_definitions': {
                name: {
                    'name': udt_def.name,
                    'type': udt_def.udt_type.value,
                    'description': udt_def.description,
                    'member_count': len(udt_def.members),
                    'size_bytes': udt_def.size_bytes,
                    'is_built_in': udt_def.is_built_in,
                    'usage_count': udt_def.usage_count,
                    'members': {
                        member_name: {
                            'name': member.name,
                            'data_type': member.data_type.value,
                            'description': member.description,
                            'full_type': member.get_full_type(),
                            'is_array': bool(member.array_dimensions),
                            'array_dimensions': member.array_dimensions,
                            'nested_udt': member.nested_udt
                        }
                        for member_name, member in udt_def.members.items()
                    }
                }
                for name, udt_def in self.udt_analyzer.udt_definitions.items()
            },
            'udt_instances': {
                name: {
                    'name': instance.name,
                    'udt_definition': instance.udt_definition,
                    'tag_type': instance.tag_type,
                    'scope': instance.scope,
                    'member_access_count': len(instance.member_accesses),
                    'accessed_members': list(instance.get_accessed_members()),
                    'initial_values': instance.initial_values
                }
                for name, instance in self.udt_analyzer.udt_instances.items()
            },
            'member_accesses': [
                {
                    'tag_name': access.tag_name,
                    'member_path': access.member_path,
                    'full_path': access.full_path,
                    'access_type': access.access_type,
                    'instruction': access.instruction,
                    'routine_name': access.routine_name,
                    'rung_number': access.rung_number,
                    'context': access.context
                }
                for access in self.udt_analyzer.member_accesses
            ],
            'udt_relationships': [
                {
                    'source': rel.source,
                    'target': rel.target,
                    'relationship_type': rel.relationship_type,
                    'strength': rel.strength,
                    'context': rel.context
                }
                for rel in self.udt_analyzer.udt_relationships
            ],
            'usage_summary': self.udt_analyzer.get_udt_usage_summary(),
            'access_patterns': {
                tag_name: [
                    {
                        'member_path': access.member_path,
                        'access_type': access.access_type,
                        'instruction': access.instruction,
                        'routine_name': access.routine_name
                    }
                    for access in accesses
                ]
                for tag_name, accesses in self.udt_analyzer.get_member_access_patterns().items()
            },
            'unused_members': self.udt_analyzer.find_unused_udt_members(),
            'analysis_metrics': self.udt_analyzer.get_analysis_metrics(),
            'performance_metrics': self.get_performance_metrics()
        }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the analysis."""
        if not self.analysis_start_time or not self.analysis_end_time:
            return {}
            
        duration = self.analysis_end_time - self.analysis_start_time
        total_operations = (
            len(self.udt_analyzer.udt_definitions) +
            len(self.udt_analyzer.udt_instances) +
            len(self.udt_analyzer.member_accesses) +
            len(self.udt_analyzer.udt_relationships)
        )
        
        return {
            'analysis_duration': duration,
            'total_operations': total_operations,
            'operations_per_second': total_operations / duration if duration > 0 else 0,
            'udts_per_second': len(self.udt_analyzer.udt_definitions) / duration if duration > 0 else 0,
            'instances_per_second': len(self.udt_analyzer.udt_instances) / duration if duration > 0 else 0,
            'accesses_per_second': len(self.udt_analyzer.member_accesses) / duration if duration > 0 else 0,
            'memory_efficiency_score': self._calculate_memory_efficiency(),
            'complexity_score': self._calculate_complexity_score()
        }
        
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score based on UDT usage."""
        if not self.udt_analyzer.udt_definitions:
            return 1.0
            
        total_defined_members = sum(len(udt.members) for udt in self.udt_analyzer.udt_definitions.values())
        total_used_members = len(set(access.member_path for access in self.udt_analyzer.member_accesses))
        
        if total_defined_members == 0:
            return 1.0
            
        return min(1.0, total_used_members / total_defined_members)
        
    def _calculate_complexity_score(self) -> float:
        """Calculate complexity score based on UDT relationships and nesting."""
        if not self.udt_analyzer.udt_definitions:
            return 0.0
            
        # Base complexity from number of UDTs
        base_complexity = len(self.udt_analyzer.udt_definitions) * 0.1
        
        # Relationship complexity
        relationship_complexity = len(self.udt_analyzer.udt_relationships) * 0.05
        
        # Nesting complexity
        nesting_complexity = 0
        for udt in self.udt_analyzer.udt_definitions.values():
            for member in udt.members.values():
                if member.nested_udt:
                    nesting_complexity += 0.2
                if member.array_dimensions:
                    nesting_complexity += 0.1 * len(member.array_dimensions)
                    
        return min(10.0, base_complexity + relationship_complexity + nesting_complexity)
        
    def get_analysis_duration(self) -> float:
        """Get the duration of the analysis in seconds."""
        if self.analysis_start_time and self.analysis_end_time:
            return self.analysis_end_time - self.analysis_start_time
        return 0.0
        
    def generate_html_report(self, output_dir: str = "step14_output") -> str:
        """Generate comprehensive HTML report for UDT analysis."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            report_file = output_path / "udt_analysis_report.html"
            
            # Get analysis results
            results = self._prepare_analysis_results()
            
            html_content = self._generate_html_content(results)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"UDT analysis HTML report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise
            
    def _generate_html_content(self, results: Dict[str, Any]) -> str:
        """Generate HTML content for the UDT analysis report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UDT Analysis Report - {results['controller_info']['name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric .label {{ color: #7f8c8d; margin-top: 5px; }}
        .udt-definition {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
        .member {{ background-color: #ffffff; padding: 10px; margin: 5px 0; border: 1px solid #dee2e6; border-radius: 3px; }}
        .access {{ background-color: #fff3cd; padding: 8px; margin: 3px 0; border-left: 3px solid #ffc107; font-family: monospace; font-size: 12px; }}
        .relationship {{ background-color: #d1ecf1; padding: 10px; margin: 5px 0; border-left: 3px solid #17a2b8; }}
        .built-in {{ border-left-color: #28a745 !important; }}
        .custom {{ border-left-color: #007bff !important; }}
        .unused {{ background-color: #f8d7da; border-left-color: #dc3545 !important; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>UDT Analysis Report</h1>
            <p>Controller: {results['controller_info']['name']} ({results['controller_info']['type']})</p>
            <p>Software Revision: {results['controller_info']['revision']}</p>
            <p>Analysis Duration: {results['performance_metrics']['analysis_duration']:.2f} seconds</p>
        </div>
        
        <div class="section">
            <h2>Analysis Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{results['analysis_metrics']['total_udt_definitions']}</div>
                    <div class="label">Total UDT Definitions</div>
                </div>
                <div class="metric">
                    <div class="value">{results['analysis_metrics']['custom_udts']}</div>
                    <div class="label">Custom UDTs</div>
                </div>
                <div class="metric">
                    <div class="value">{results['analysis_metrics']['total_instances']}</div>
                    <div class="label">UDT Instances</div>
                </div>
                <div class="metric">
                    <div class="value">{results['analysis_metrics']['total_member_accesses']}</div>
                    <div class="label">Member Accesses</div>
                </div>
                <div class="metric">
                    <div class="value">{results['analysis_metrics']['total_relationships']}</div>
                    <div class="label">UDT Relationships</div>
                </div>
                <div class="metric">
                    <div class="value">{results['performance_metrics']['operations_per_second']:.1f}</div>
                    <div class="label">Operations/Second</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>UDT Definitions</h2>
            {self._generate_udt_definitions_html(results['udt_definitions'])}
        </div>
        
        <div class="section">
            <h2>UDT Instances</h2>
            {self._generate_udt_instances_html(results['udt_instances'])}
        </div>
        
        <div class="section">
            <h2>Member Access Patterns</h2>
            {self._generate_access_patterns_html(results['access_patterns'])}
        </div>
        
        <div class="section">
            <h2>UDT Relationships</h2>
            {self._generate_relationships_html(results['udt_relationships'])}
        </div>
        
        <div class="section">
            <h2>Usage Summary</h2>
            {self._generate_usage_summary_html(results['usage_summary'])}
        </div>
        
        <div class="section">
            <h2>Unused Members Analysis</h2>
            {self._generate_unused_members_html(results['unused_members'])}
        </div>
        
        <div class="section">
            <h2>Performance Analysis</h2>
            {self._generate_performance_html(results['performance_metrics'])}
        </div>
    </div>
</body>
</html>
        """
        
        return html
        
    def _generate_udt_definitions_html(self, udt_definitions: Dict) -> str:
        """Generate HTML for UDT definitions section."""
        html = ""
        
        for name, udt_def in udt_definitions.items():
            class_name = "built-in" if udt_def['is_built_in'] else "custom"
            
            html += f"""
            <div class="udt-definition {class_name}">
                <h3>{udt_def['name']} ({udt_def['type']})</h3>
                <p><strong>Description:</strong> {udt_def['description'] or 'No description'}</p>
                <p><strong>Members:</strong> {udt_def['member_count']} | 
                   <strong>Size:</strong> {udt_def['size_bytes']} bytes | 
                   <strong>Usage Count:</strong> {udt_def['usage_count']}</p>
                
                <h4>Members:</h4>
            """
            
            for member_name, member in udt_def['members'].items():
                html += f"""
                <div class="member">
                    <strong>{member['name']}</strong> ({member['full_type']})
                    {f"<br><em>{member['description']}</em>" if member['description'] else ""}
                    {f"<br><span style='color: #007bff;'>Nested UDT: {member['nested_udt']}</span>" if member['nested_udt'] else ""}
                </div>
                """
                
            html += "</div>"
            
        return html
        
    def _generate_udt_instances_html(self, udt_instances: Dict) -> str:
        """Generate HTML for UDT instances section."""
        html = "<table><tr><th>Instance Name</th><th>UDT Type</th><th>Scope</th><th>Member Accesses</th><th>Accessed Members</th></tr>"
        
        for name, instance in udt_instances.items():
            accessed_members = ", ".join(instance['accessed_members']) if instance['accessed_members'] else "None"
            
            html += f"""
            <tr>
                <td><strong>{instance['name']}</strong></td>
                <td>{instance['udt_definition']}</td>
                <td>{instance['scope']}</td>
                <td>{instance['member_access_count']}</td>
                <td style="font-family: monospace; font-size: 12px;">{accessed_members}</td>
            </tr>
            """
            
        html += "</table>"
        return html
        
    def _generate_access_patterns_html(self, access_patterns: Dict) -> str:
        """Generate HTML for access patterns section."""
        html = ""
        
        for tag_name, accesses in access_patterns.items():
            html += f"""
            <div class="udt-definition">
                <h3>{tag_name}</h3>
                <p><strong>Total Accesses:</strong> {len(accesses)}</p>
            """
            
            for access in accesses[:10]:  # Show first 10 accesses
                html += f"""
                <div class="access">
                    {access['member_path']} | {access['access_type']} | 
                    {access['instruction']} | {access['routine_name']}
                </div>
                """
                
            if len(accesses) > 10:
                html += f"<p><em>... and {len(accesses) - 10} more accesses</em></p>"
                
            html += "</div>"
            
        return html
        
    def _generate_relationships_html(self, relationships: List) -> str:
        """Generate HTML for relationships section."""
        html = ""
        
        relationship_types = {}
        for rel in relationships:
            rel_type = rel['relationship_type']
            if rel_type not in relationship_types:
                relationship_types[rel_type] = []
            relationship_types[rel_type].append(rel)
            
        for rel_type, rels in relationship_types.items():
            html += f"<h3>{rel_type} Relationships</h3>"
            for rel in rels:
                html += f"""
                <div class="relationship">
                    <strong>{rel['source']}</strong> → <strong>{rel['target']}</strong>
                    (Strength: {rel['strength']:.2f})
                    <br><em>{rel['context']}</em>
                </div>
                """
                
        return html
        
    def _generate_usage_summary_html(self, usage_summary: Dict) -> str:
        """Generate HTML for usage summary section."""
        html = "<table><tr><th>UDT</th><th>Instances</th><th>Total Accesses</th><th>Most Used Member</th><th>Instance Names</th></tr>"
        
        for udt_name, summary in usage_summary.items():
            instances_str = ", ".join(summary['instances'][:5])
            if len(summary['instances']) > 5:
                instances_str += f" ... and {len(summary['instances']) - 5} more"
                
            html += f"""
            <tr>
                <td><strong>{udt_name}</strong></td>
                <td>{summary['instance_count']}</td>
                <td>{summary['total_accesses']}</td>
                <td>{summary['most_used_member'] or 'None'}</td>
                <td style="font-family: monospace; font-size: 12px;">{instances_str}</td>
            </tr>
            """
            
        html += "</table>"
        return html
        
    def _generate_unused_members_html(self, unused_members: Dict) -> str:
        """Generate HTML for unused members section."""
        if not unused_members:
            return "<p class='status-good'>All UDT members are being used - excellent memory efficiency!</p>"
            
        html = ""
        for udt_name, members in unused_members.items():
            html += f"""
            <div class="udt-definition unused">
                <h3>{udt_name}</h3>
                <p><strong>Unused Members:</strong> {', '.join(members)}</p>
                <p><em>Consider removing unused members to improve memory efficiency.</em></p>
            </div>
            """
            
        return html
        
    def _generate_performance_html(self, performance: Dict) -> str:
        """Generate HTML for performance section."""
        efficiency_class = "status-good" if performance['memory_efficiency_score'] > 0.8 else "status-warning" if performance['memory_efficiency_score'] > 0.5 else "status-error"
        
        html = f"""
        <div class="metrics">
            <div class="metric">
                <div class="value">{performance['analysis_duration']:.2f}s</div>
                <div class="label">Analysis Duration</div>
            </div>
            <div class="metric">
                <div class="value">{performance['operations_per_second']:.1f}</div>
                <div class="label">Operations/Second</div>
            </div>
            <div class="metric">
                <div class="value {efficiency_class}">{performance['memory_efficiency_score']:.2f}</div>
                <div class="label">Memory Efficiency</div>
            </div>
            <div class="metric">
                <div class="value">{performance['complexity_score']:.1f}</div>
                <div class="label">Complexity Score</div>
            </div>
        </div>
        """
        
        return html
        
    def export_udt_structures(self, output_dir: str = "step14_output") -> str:
        """Export UDT structures to JSON file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            structures_file = output_path / "udt_structures.json"
            
            results = self._prepare_analysis_results()
            
            # Prepare structured data for export
            export_data = {
                'controller_info': results['controller_info'],
                'udt_definitions': results['udt_definitions'],
                'udt_instances': results['udt_instances'],
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_metrics': results['analysis_metrics']
            }
            
            with open(structures_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"UDT structures exported to: {structures_file}")
            return str(structures_file)
            
        except Exception as e:
            logger.error(f"Error exporting UDT structures: {e}")
            raise
            
    def export_member_accesses(self, output_dir: str = "step14_output") -> str:
        """Export member access patterns to JSON file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            accesses_file = output_path / "member_accesses.json"
            
            results = self._prepare_analysis_results()
            
            # Prepare access data for export
            export_data = {
                'member_accesses': results['member_accesses'],
                'access_patterns': results['access_patterns'],
                'udt_relationships': results['udt_relationships'],
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'access_summary': {
                    'total_accesses': len(results['member_accesses']),
                    'unique_tags': len(results['access_patterns']),
                    'relationships': len(results['udt_relationships'])
                }
            }
            
            with open(accesses_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Member accesses exported to: {accesses_file}")
            return str(accesses_file)
            
        except Exception as e:
            logger.error(f"Error exporting member accesses: {e}")
            raise
