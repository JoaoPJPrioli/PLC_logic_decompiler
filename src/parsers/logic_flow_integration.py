"""
Logic Flow Analysis Integration
Integration layer for logic flow analysis with routine analysis and reporting.

This module provides:
- Integration with existing routine analysis for flow extraction
- Logic flow analysis pipeline with performance optimization
- Comprehensive reporting (HTML, JSON) for flow analysis results
- Flow visualization and graph export capabilities
- Logic summary generation and optimization recommendations
"""

import json
import time
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import logging
from collections import defaultdict
import networkx as nx

from src.analysis.logic_flow_analyzer import (
    LogicFlowAnalyzer, FlowType, LogicPattern, ExecutionPath,
    LogicBlock, LogicFlow, LogicCondition, LogicAction
)

logger = logging.getLogger(__name__)


class LogicFlowIntegrator:
    """Integration layer for logic flow analysis with routine analysis and reporting."""
    
    def __init__(self):
        self.analyzer = LogicFlowAnalyzer()
        self.routine_analysis = None
        
        # Performance metrics
        self.analysis_start_time = None
        self.analysis_end_time = None
        self.performance_metrics = {}
        
    def set_routine_analysis(self, routine_analysis) -> None:
        """Set routine analysis for flow analysis."""
        self.routine_analysis = routine_analysis
        logger.info("Routine analysis set for logic flow analysis integration")
        
    def run_comprehensive_flow_analysis(self) -> Dict[str, Any]:
        """Run comprehensive logic flow analysis pipeline."""
        logger.info("Starting comprehensive logic flow analysis...")
        self.analysis_start_time = time.time()
        
        try:
            results = {
                'success': False,
                'error': None,
                'flow_summary': {},
                'logic_blocks': [],
                'logic_flows': [],
                'execution_graph': {},
                'patterns_detected': {},
                'critical_analysis': {},
                'performance_metrics': {},
                'recommendations': []
            }
            
            # Step 1: Analyze routine flows
            logger.info("Step 1: Analyzing routine flow patterns...")
            self._analyze_routine_flows()
            
            # Step 2: Analyze flow performance
            logger.info("Step 2: Analyzing flow performance...")
            self._analyze_flow_performance()
            
            # Step 3: Generate comprehensive results
            logger.info("Step 3: Generating comprehensive results...")
            results.update(self._generate_comprehensive_results())
            
            # Step 4: Generate recommendations
            logger.info("Step 4: Generating flow optimization recommendations...")
            results['recommendations'] = self._generate_flow_recommendations()
            
            results['success'] = True
            self.analysis_end_time = time.time()
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            results['performance_metrics'] = self.performance_metrics
            
            logger.info(f"Logic flow analysis completed successfully in {self.performance_metrics.get('total_time_seconds', 0):.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error during logic flow analysis: {e}")
            results['error'] = str(e)
            return results
            
    def _analyze_routine_flows(self) -> None:
        """Analyze flows from routine analysis."""
        if not self.routine_analysis:
            logger.warning("No routine analysis available for flow analysis")
            return
            
        self.analyzer.analyze_routine_flow(self.routine_analysis)
        
    def _analyze_flow_performance(self) -> None:
        """Analyze flow performance characteristics."""
        self.analyzer.analyze_flow_performance()
        
    def _generate_comprehensive_results(self) -> Dict[str, Any]:
        """Generate comprehensive flow analysis results."""
        return {
            'flow_summary': self.analyzer.get_flow_summary(),
            'logic_blocks': self._serialize_logic_blocks(),
            'logic_flows': self._serialize_logic_flows(),
            'execution_graph': self._serialize_execution_graph(),
            'patterns_detected': self._serialize_patterns(),
            'critical_analysis': self._serialize_critical_analysis()
        }
        
    def _serialize_logic_blocks(self) -> List[Dict[str, Any]]:
        """Serialize logic blocks for output."""
        serialized = []
        
        for block in self.analyzer.logic_blocks:
            conditions_data = []
            for condition in block.conditions:
                conditions_data.append({
                    'type': condition.condition_type,
                    'operands': condition.operands,
                    'expression': condition.expression,
                    'readable': condition.readable_expression,
                    'routine': condition.routine_name,
                    'rung': condition.rung_number,
                    'negated': condition.is_negated,
                    'complexity': condition.complexity_score
                })
                
            actions_data = []
            for action in block.actions:
                actions_data.append({
                    'type': action.action_type,
                    'targets': action.targets,
                    'parameters': action.parameters,
                    'readable': action.readable_action,
                    'routine': action.routine_name,
                    'rung': action.rung_number,
                    'side_effects': action.side_effects
                })
                
            serialized.append({
                'block_id': block.block_id,
                'block_type': block.block_type.value,
                'routine_name': block.routine_name,
                'rung_range': block.rung_range,
                'description': block.description,
                'complexity_score': block.complexity_score,
                'execution_probability': block.execution_probability,
                'condition_count': block.condition_count,
                'action_count': block.action_count,
                'referenced_tags': list(block.get_referenced_tags()),
                'conditions': conditions_data,
                'actions': actions_data
            })
            
        return serialized
        
    def _serialize_logic_flows(self) -> List[Dict[str, Any]]:
        """Serialize logic flows for output."""
        serialized = []
        
        for flow in self.analyzer.logic_flows:
            entry_conditions = []
            for condition in flow.entry_conditions:
                entry_conditions.append({
                    'type': condition.condition_type,
                    'operands': condition.operands,
                    'readable': condition.readable_expression
                })
                
            exit_conditions = []
            for condition in flow.exit_conditions:
                exit_conditions.append({
                    'type': condition.condition_type,
                    'operands': condition.operands,
                    'readable': condition.readable_expression
                })
                
            serialized.append({
                'flow_id': flow.flow_id,
                'flow_type': flow.flow_type.value,
                'pattern': flow.pattern.value if flow.pattern else None,
                'priority': flow.priority,
                'is_critical': flow.is_critical,
                'total_complexity': flow.total_complexity,
                'execution_probability': flow.execution_probability,
                'block_count': len(flow.blocks),
                'block_ids': [block.block_id for block in flow.blocks],
                'entry_conditions': entry_conditions,
                'exit_conditions': exit_conditions
            })
            
        return serialized
        
    def _serialize_execution_graph(self) -> Dict[str, Any]:
        """Serialize execution graph for output."""
        if not self.analyzer.execution_graph.nodes():
            return {'nodes': [], 'edges': [], 'metrics': {}}
            
        nodes = []
        for node_id, node_data in self.analyzer.execution_graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'routine': node_data.get('routine', ''),
                'block_type': node_data.get('block_type', ''),
                'complexity': node_data.get('complexity', 0.0),
                'rung_range': node_data.get('rung_range', (0, 0))
            })
            
        edges = []
        for source, target, edge_data in self.analyzer.execution_graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'edge_type': edge_data.get('edge_type', 'sequential'),
                'weight': edge_data.get('weight', 1.0)
            })
            
        # Calculate graph metrics
        graph = self.analyzer.execution_graph
        metrics = {
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges(),
            'density': nx.density(graph) if graph.number_of_nodes() > 0 else 0.0,
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'weakly_connected_components': nx.number_weakly_connected_components(graph),
            'average_clustering': nx.average_clustering(graph.to_undirected()) if graph.number_of_nodes() > 0 else 0.0
        }
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metrics': metrics
        }
        
    def _serialize_patterns(self) -> Dict[str, Any]:
        """Serialize detected patterns for output."""
        pattern_counts = defaultdict(int)
        pattern_details = defaultdict(list)
        
        for flow in self.analyzer.logic_flows:
            if flow.pattern:
                pattern_counts[flow.pattern.value] += 1
                pattern_details[flow.pattern.value].append({
                    'flow_id': flow.flow_id,
                    'routine': flow.blocks[0].routine_name if flow.blocks else 'Unknown',
                    'complexity': flow.total_complexity,
                    'is_critical': flow.is_critical
                })
                
        return {
            'pattern_counts': dict(pattern_counts),
            'pattern_details': dict(pattern_details),
            'total_patterns': sum(pattern_counts.values()),
            'unique_patterns': len(pattern_counts)
        }
        
    def _serialize_critical_analysis(self) -> Dict[str, Any]:
        """Serialize critical analysis results."""
        results = self.analyzer.analysis_results
        
        return {
            'critical_paths': results.critical_paths,
            'bottlenecks': results.bottlenecks,
            'optimization_opportunities': results.optimization_opportunities,
            'safety_concerns': results.safety_concerns,
            'critical_path_count': len(results.critical_paths),
            'bottleneck_count': len(results.bottlenecks),
            'optimization_count': len(results.optimization_opportunities),
            'safety_concern_count': len(results.safety_concerns)
        }
        
    def _generate_flow_recommendations(self) -> List[Dict[str, Any]]:
        """Generate flow optimization recommendations."""
        recommendations = []
        results = self.analyzer.analysis_results
        summary = self.analyzer.get_flow_summary()
        
        # High complexity recommendations
        avg_complexity = summary.get('average_block_complexity', 0)
        if avg_complexity > 8.0:
            recommendations.append({
                'type': 'OPTIMIZATION',
                'priority': 'HIGH',
                'title': 'High Average Logic Complexity',
                'description': f'Average block complexity is {avg_complexity:.1f}, which is high. Consider simplifying complex logic blocks.',
                'metric_value': avg_complexity,
                'threshold': 8.0,
                'suggestion': 'Break down complex blocks into smaller, simpler components'
            })
            
        # Safety recommendations
        if len(results.safety_concerns) > 0:
            recommendations.append({
                'type': 'SAFETY',
                'priority': 'HIGH',
                'title': 'Safety Concerns Detected',
                'description': f'Found {len(results.safety_concerns)} safety concerns that need attention.',
                'concerns': results.safety_concerns,
                'suggestion': 'Review and address all safety concerns before deployment'
            })
            
        # Pattern recommendations
        pattern_counts = summary.get('detected_patterns', {})
        if 'START_STOP' in pattern_counts and pattern_counts['START_STOP'] > 5:
            recommendations.append({
                'type': 'ARCHITECTURE',
                'priority': 'MEDIUM',
                'title': 'Multiple Start/Stop Patterns',
                'description': f'Found {pattern_counts["START_STOP"]} start/stop patterns. Consider consolidating similar patterns.',
                'pattern_count': pattern_counts['START_STOP'],
                'suggestion': 'Create reusable function blocks for common start/stop logic'
            })
            
        # Performance recommendations
        if len(results.bottlenecks) > 3:
            recommendations.append({
                'type': 'PERFORMANCE',
                'priority': 'MEDIUM',
                'title': 'Multiple Bottlenecks Detected',
                'description': f'Found {len(results.bottlenecks)} potential bottlenecks in execution flow.',
                'bottlenecks': results.bottlenecks[:5],  # Show first 5
                'suggestion': 'Optimize high-dependency nodes and complex flows'
            })
            
        # Logic flow recommendations
        flow_types = summary.get('flow_types', {})
        if flow_types.get('MAIN_PATH', 0) < flow_types.get('ALTERNATE_PATH', 0):
            recommendations.append({
                'type': 'LOGIC',
                'priority': 'LOW',
                'title': 'More Alternate Paths Than Main Paths',
                'description': 'The logic has more alternate execution paths than main paths, which may indicate complex branching.',
                'main_paths': flow_types.get('MAIN_PATH', 0),
                'alternate_paths': flow_types.get('ALTERNATE_PATH', 0),
                'suggestion': 'Review logic structure for potential simplification'
            })
            
        # Optimization opportunities
        if len(results.optimization_opportunities) > 0:
            recommendations.append({
                'type': 'OPTIMIZATION',
                'priority': 'LOW',
                'title': 'Optimization Opportunities Available',
                'description': f'Found {len(results.optimization_opportunities)} optimization opportunities.',
                'opportunities': results.optimization_opportunities[:3],  # Show first 3
                'suggestion': 'Review and implement suggested optimizations'
            })
            
        return recommendations
        
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics for the analysis."""
        if self.analysis_start_time and self.analysis_end_time:
            total_time = self.analysis_end_time - self.analysis_start_time
            
            self.performance_metrics = {
                'total_time_seconds': total_time,
                'blocks_per_second': len(self.analyzer.logic_blocks) / total_time if total_time > 0 else 0,
                'flows_per_second': len(self.analyzer.logic_flows) / total_time if total_time > 0 else 0,
                'total_blocks_analyzed': len(self.analyzer.logic_blocks),
                'total_flows_analyzed': len(self.analyzer.logic_flows),
                'graph_nodes': self.analyzer.execution_graph.number_of_nodes(),
                'graph_edges': self.analyzer.execution_graph.number_of_edges(),
                'complexity_analysis_speed': sum(b.complexity_score for b in self.analyzer.logic_blocks) / total_time if total_time > 0 else 0
            }
            
    def generate_html_report(self, output_path: str) -> bool:
        """Generate HTML report for logic flow analysis."""
        try:
            logger.info(f"Generating HTML report: {output_path}")
            
            # Run analysis if not already done
            if not self.analyzer.logic_blocks and not self.analyzer.logic_flows:
                logger.info("Running analysis for HTML report generation...")
                self.run_comprehensive_flow_analysis()
                
            html_content = self._generate_html_content()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"HTML report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return False
            
    def _generate_html_content(self) -> str:
        """Generate HTML content for the flow analysis report."""
        summary = self.analyzer.get_flow_summary()
        results = self.analyzer.analysis_results
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logic Flow Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #007acc; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .table th {{ background-color: #f2f2f2; }}
        .flow-type-SEQUENTIAL {{ color: #28a745; }}
        .flow-type-CONDITIONAL {{ color: #007bff; }}
        .flow-type-LOOP {{ color: #6f42c1; }}
        .flow-type-PARALLEL {{ color: #fd7e14; }}
        .flow-type-INTERLOCK {{ color: #dc3545; }}
        .flow-type-TIMER_CHAIN {{ color: #ffc107; }}
        .pattern-START_STOP {{ color: #dc3545; font-weight: bold; }}
        .pattern-SAFETY_CHAIN {{ color: #dc3545; font-weight: bold; }}
        .pattern-TIMER_CHAIN {{ color: #ffc107; }}
        .pattern-ALARM_LOGIC {{ color: #fd7e14; }}
        .recommendation-HIGH {{ border-left-color: #dc3545; }}
        .recommendation-MEDIUM {{ border-left-color: #ffc107; }}
        .recommendation-LOW {{ border-left-color: #28a745; }}
        .critical {{ color: #dc3545; font-weight: bold; }}
        .warning {{ color: #ffc107; font-weight: bold; }}
        .info {{ color: #007bff; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Logic Flow Analysis Report</h1>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        {f'<p><strong>Analysis Time:</strong> {self.performance_metrics.get("total_time_seconds", 0):.2f} seconds</p>' if self.performance_metrics else ''}
    </div>

    <div class="section">
        <h2>📊 Flow Analysis Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_blocks', 0)}</div>
                <div class="metric-label">Logic Blocks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_flows', 0)}</div>
                <div class="metric-label">Logic Flows</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(summary.get('detected_patterns', {}))}</div>
                <div class="metric-label">Patterns Detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('average_block_complexity', 0):.1f}</div>
                <div class="metric-label">Avg Block Complexity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('critical_paths', 0)}</div>
                <div class="metric-label">Critical Paths</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('safety_concerns', 0)}</div>
                <div class="metric-label">Safety Concerns</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🏗️ Block Type Distribution</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Block Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_block_types_table(summary.get('block_types', {}))}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🔍 Detected Patterns</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Pattern</th>
                    <th>Count</th>
                    <th>Safety Critical</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_patterns_table(summary.get('detected_patterns', {}))}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>⚠️ Critical Analysis</h2>
        <div class="metric-grid">
            <div class="metric-card recommendation-HIGH">
                <div class="metric-value critical">{len(results.critical_paths)}</div>
                <div class="metric-label">Critical Paths</div>
            </div>
            <div class="metric-card recommendation-MEDIUM">
                <div class="metric-value warning">{len(results.bottlenecks)}</div>
                <div class="metric-label">Bottlenecks</div>
            </div>
            <div class="metric-card recommendation-LOW">
                <div class="metric-value info">{len(results.optimization_opportunities)}</div>
                <div class="metric-label">Optimizations</div>
            </div>
            <div class="metric-card recommendation-HIGH">
                <div class="metric-value critical">{len(results.safety_concerns)}</div>
                <div class="metric-label">Safety Concerns</div>
            </div>
        </div>
        
        {self._generate_critical_analysis_details(results)}
    </div>

    <div class="section">
        <h2>💡 Flow Optimization Recommendations</h2>
        {self._generate_recommendations_section()}
    </div>

    <div class="section">
        <h2>⚡ Performance Metrics</h2>
        {self._generate_performance_section()}
    </div>

    <div class="section">
        <h2>📈 Execution Graph Metrics</h2>
        {self._generate_graph_metrics_section()}
    </div>
</body>
</html>
        """
        
        return html
        
    def _generate_block_types_table(self, block_types: Dict[str, int]) -> str:
        """Generate HTML table rows for block types."""
        if not block_types:
            return "<tr><td colspan='4'>No block types found</td></tr>"
            
        total_blocks = sum(block_types.values())
        type_descriptions = {
            'SEQUENTIAL': 'Linear execution blocks',
            'CONDITIONAL': 'IF-THEN-ELSE logic blocks',
            'LOOP': 'Iterative logic blocks',
            'PARALLEL': 'Concurrent execution blocks',
            'INTERLOCK': 'Safety interlock blocks',
            'TIMER_CHAIN': 'Timer sequence blocks',
            'COUNTER_CHAIN': 'Counter sequence blocks'
        }
        
        rows = []
        for block_type, count in sorted(block_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
            description = type_descriptions.get(block_type, 'Unknown block type')
            
            rows.append(f"""
                <tr>
                    <td><span class="flow-type-{block_type}">{block_type}</span></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{description}</td>
                </tr>
            """)
            
        return "".join(rows)
        
    def _generate_patterns_table(self, detected_patterns: Dict[str, int]) -> str:
        """Generate HTML table rows for detected patterns."""
        if not detected_patterns:
            return "<tr><td colspan='4'>No patterns detected</td></tr>"
            
        pattern_info = {
            'START_STOP': ('Yes', 'Motor start/stop control logic'),
            'SEAL_IN': ('No', 'Self-holding logic patterns'),
            'EDGE_DETECT': ('No', 'Rising/falling edge detection'),
            'SAFETY_CHAIN': ('Yes', 'Safety interlock chains'),
            'TIMER_CHAIN': ('No', 'Sequential timer operations'),
            'ALARM_LOGIC': ('Yes', 'Alarm detection and handling'),
            'DEBOUNCE': ('No', 'Input signal debouncing'),
            'SEQUENCER': ('No', 'Sequential operation control')
        }
        
        rows = []
        for pattern, count in sorted(detected_patterns.items(), key=lambda x: x[1], reverse=True):
            safety_critical, description = pattern_info.get(pattern, ('Unknown', 'Pattern description not available'))
            
            rows.append(f"""
                <tr>
                    <td><span class="pattern-{pattern}">{pattern}</span></td>
                    <td>{count}</td>
                    <td>{safety_critical}</td>
                    <td>{description}</td>
                </tr>
            """)
            
        return "".join(rows)
        
    def _generate_critical_analysis_details(self, results) -> str:
        """Generate detailed critical analysis section."""
        html_parts = []
        
        if results.safety_concerns:
            html_parts.append("""
                <h3 class="critical">🚨 Safety Concerns</h3>
                <ul>
            """)
            for concern in results.safety_concerns:
                html_parts.append(f"<li class='critical'>{concern}</li>")
            html_parts.append("</ul>")
            
        if results.bottlenecks:
            html_parts.append("""
                <h3 class="warning">⚠️ Bottlenecks</h3>
                <ul>
            """)
            for bottleneck in results.bottlenecks[:5]:  # Show first 5
                html_parts.append(f"<li class='warning'>{bottleneck}</li>")
            html_parts.append("</ul>")
            
        if results.critical_paths:
            html_parts.append("""
                <h3 class="info">🎯 Critical Paths</h3>
                <ul>
            """)
            for path in results.critical_paths[:5]:  # Show first 5
                html_parts.append(f"<li class='info'>{path}</li>")
            html_parts.append("</ul>")
            
        return "".join(html_parts) if html_parts else "<p>No critical issues detected.</p>"
        
    def _generate_recommendations_section(self) -> str:
        """Generate HTML content for recommendations."""
        if not hasattr(self, '_flow_recommendations'):
            self._flow_recommendations = self._generate_flow_recommendations()
            
        if not self._flow_recommendations:
            return "<p>No specific recommendations at this time.</p>"
            
        html = []
        # Group by priority
        by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for rec in self._flow_recommendations:
            priority = rec.get('priority', 'LOW')
            by_priority[priority].append(rec)
            
        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            recs = by_priority[priority]
            if recs:
                html.append(f"<h3>🔥 {priority} Priority ({len(recs)} recommendations)</h3>")
                for i, rec in enumerate(recs, 1):
                    priority_class = f"recommendation-{priority}"
                    html.append(f"""
                        <div class="metric-card {priority_class}">
                            <h4>{i}. {rec.get('title', 'Untitled')}</h4>
                            <p><strong>Type:</strong> {rec.get('type', 'N/A')}</p>
                            <p>{rec.get('description', 'No description available')}</p>
                            {f"<p><strong>Suggestion:</strong> {rec['suggestion']}</p>" if 'suggestion' in rec else ''}
                        </div>
                    """)
                    
        return "".join(html)
        
    def _generate_performance_section(self) -> str:
        """Generate HTML content for performance metrics."""
        if not self.performance_metrics:
            return "<p>No performance metrics available.</p>"
            
        return f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{self.performance_metrics.get('blocks_per_second', 0):.1f}</div>
                    <div class="metric-label">Blocks/Second</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.performance_metrics.get('flows_per_second', 0):.1f}</div>
                    <div class="metric-label">Flows/Second</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.performance_metrics.get('complexity_analysis_speed', 0):.1f}</div>
                    <div class="metric-label">Complexity Units/Second</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.performance_metrics.get('total_time_seconds', 0):.2f}s</div>
                    <div class="metric-label">Analysis Time</div>
                </div>
            </div>
        """
        
    def _generate_graph_metrics_section(self) -> str:
        """Generate HTML content for execution graph metrics."""
        graph_data = self._serialize_execution_graph()
        metrics = graph_data.get('metrics', {})
        
        if not metrics:
            return "<p>No graph metrics available.</p>"
            
        return f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('node_count', 0)}</div>
                    <div class="metric-label">Graph Nodes</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('edge_count', 0)}</div>
                    <div class="metric-label">Graph Edges</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('density', 0):.3f}</div>
                    <div class="metric-label">Graph Density</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{'Yes' if metrics.get('is_dag', False) else 'No'}</div>
                    <div class="metric-label">Is DAG</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('weakly_connected_components', 0)}</div>
                    <div class="metric-label">Connected Components</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('average_clustering', 0):.3f}</div>
                    <div class="metric-label">Average Clustering</div>
                </div>
            </div>
        """
        
    def export_json_report(self, output_path: str) -> bool:
        """Export analysis results as JSON."""
        try:
            logger.info(f"Exporting JSON report: {output_path}")
            
            # Run analysis if not already done
            if not self.analyzer.logic_blocks and not self.analyzer.logic_flows:
                logger.info("Running analysis for JSON export...")
                results = self.run_comprehensive_flow_analysis()
            else:
                results = self._generate_comprehensive_results()
                results['performance_metrics'] = self.performance_metrics
                results['recommendations'] = self._generate_flow_recommendations()
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            logger.info(f"JSON report exported successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export JSON report: {e}")
            return False
            
    def export_execution_graph(self, output_path: str, format: str = 'graphml') -> bool:
        """Export execution graph to file."""
        try:
            logger.info(f"Exporting execution graph: {output_path}")
            
            if not self.analyzer.execution_graph.nodes():
                logger.warning("No execution graph available for export")
                return False
                
            if format.lower() == 'graphml':
                nx.write_graphml(self.analyzer.execution_graph, output_path)
            elif format.lower() == 'dot':
                nx.drawing.nx_pydot.write_dot(self.analyzer.execution_graph, output_path)
            elif format.lower() == 'gexf':
                nx.write_gexf(self.analyzer.execution_graph, output_path)
            else:
                logger.error(f"Unsupported graph format: {format}")
                return False
                
            logger.info(f"Execution graph exported successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export execution graph: {e}")
            return False
