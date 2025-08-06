"""
Array Analysis Integration
Integration layer for array analysis with L5X parsing and reporting.

This module provides:
- Integration with XML parsing for array definition extraction
- Array analysis pipeline with routine analysis integration
- Comprehensive reporting (HTML, JSON) for array analysis results
- Performance metrics and optimization recommendations
"""

import json
import time
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import logging
from collections import defaultdict

from src.analysis.array_analyzer import (
    ArrayAnalyzer, ArrayType, IndexType, ArrayDefinition, 
    ArrayAccess, ArrayUsagePattern, ArrayRelationship
)

logger = logging.getLogger(__name__)


class SimpleL5XParser:
    """Simplified L5X parser for testing and integration purposes."""
    
    def __init__(self, l5x_content: str = None):
        self.l5x_content = l5x_content or ""
        
    def get_tags(self):
        """Mock implementation of tags extraction."""
        # Return empty structure for testing
        return None
        
    def get_routines(self):
        """Mock implementation of routine extraction."""
        return {}


class ArrayAnalysisIntegrator:
    """Integration layer for array analysis with L5X parsing and reporting."""
    
    def __init__(self, l5x_file_path: Optional[str] = None):
        self.l5x_file_path = l5x_file_path
        self.analyzer = ArrayAnalyzer()
        self.l5x_parser = None
        self.routine_analysis = None
        
        # Performance metrics
        self.analysis_start_time = None
        self.analysis_end_time = None
        self.performance_metrics = {}
        
    def load_l5x_file(self, file_path: str) -> bool:
        """Load L5X file for analysis."""
        try:
            self.l5x_file_path = file_path
            logger.info(f"Loading L5X file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.l5x_parser = SimpleL5XParser(content)
            logger.info("L5X file loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load L5X file: {e}")
            return False
            
    def set_routine_analysis(self, routine_analysis) -> None:
        """Set routine analysis for array access analysis."""
        self.routine_analysis = routine_analysis
        logger.info("Routine analysis set for array analysis integration")
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive array analysis pipeline."""
        logger.info("Starting comprehensive array analysis...")
        self.analysis_start_time = time.time()
        
        try:
            results = {
                'success': False,
                'error': None,
                'analysis_summary': {},
                'array_definitions': {},
                'array_accesses': [],
                'usage_patterns': {},
                'relationships': [],
                'performance_metrics': {},
                'recommendations': []
            }
            
            # Step 1: Analyze array definitions
            logger.info("Step 1: Analyzing array definitions...")
            self._analyze_array_definitions()
            
            # Step 2: Analyze array accesses
            logger.info("Step 2: Analyzing array access patterns...")
            self._analyze_array_accesses()
            
            # Step 3: Analyze array relationships
            logger.info("Step 3: Analyzing array relationships...")
            self._analyze_array_relationships()
            
            # Step 4: Generate comprehensive results
            logger.info("Step 4: Generating comprehensive results...")
            results.update(self._generate_comprehensive_results())
            
            # Step 5: Generate recommendations
            logger.info("Step 5: Generating optimization recommendations...")
            results['recommendations'] = self._generate_recommendations()
            
            results['success'] = True
            self.analysis_end_time = time.time()
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            results['performance_metrics'] = self.performance_metrics
            
            logger.info(f"Array analysis completed successfully in {self.performance_metrics.get('total_time_seconds', 0):.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error during array analysis: {e}")
            results['error'] = str(e)
            return results
            
    def _analyze_array_definitions(self) -> None:
        """Analyze array definitions from L5X file."""
        if not self.l5x_parser:
            logger.warning("No L5X parser available for array definition analysis")
            return
            
        tags = self.l5x_parser.get_tags()
        self.analyzer.analyze_array_definitions(tags)
        
    def _analyze_array_accesses(self) -> None:
        """Analyze array access patterns."""
        if not self.routine_analysis:
            logger.warning("No routine analysis available for array access analysis")
            return
            
        self.analyzer.analyze_array_accesses(self.routine_analysis)
        
    def _analyze_array_relationships(self) -> None:
        """Analyze relationships between arrays."""
        self.analyzer.analyze_array_relationships()
        
    def _generate_comprehensive_results(self) -> Dict[str, Any]:
        """Generate comprehensive analysis results."""
        return {
            'analysis_summary': self.analyzer.get_array_analysis_summary(),
            'array_definitions': self._serialize_array_definitions(),
            'array_accesses': self._serialize_array_accesses(),
            'usage_patterns': self._serialize_usage_patterns(),
            'relationships': self._serialize_relationships()
        }
        
    def _serialize_array_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Serialize array definitions for output."""
        serialized = {}
        
        for name, array_def in self.analyzer.array_definitions.items():
            serialized[name] = {
                'name': array_def.name,
                'array_type': array_def.array_type.value,
                'element_data_type': array_def.element_data_type,
                'dimension_count': array_def.dimension_count,
                'total_elements': array_def.total_elements,
                'element_counts': array_def.get_element_count_by_dimension(),
                'bounds': array_def.get_bounds_by_dimension(),
                'scope': array_def.scope,
                'description': array_def.description,
                'initial_values_count': len(array_def.initial_values)
            }
            
        return serialized
        
    def _serialize_array_accesses(self) -> List[Dict[str, Any]]:
        """Serialize array accesses for output."""
        serialized = []
        
        for access in self.analyzer.array_accesses:
            indices_data = []
            for idx in access.indices:
                indices_data.append({
                    'dimension': idx.dimension,
                    'index_value': str(idx.index_value),
                    'index_type': idx.index_type.value,
                    'is_static': idx.is_static(),
                    'source_instruction': idx.source_instruction
                })
                
            serialized.append({
                'array_name': access.array_name,
                'full_path': access.full_path,
                'indices': indices_data,
                'access_type': access.access_type,
                'instruction': access.instruction,
                'routine_name': access.routine_name,
                'rung_number': access.rung_number,
                'dimension_count': access.dimension_count,
                'is_fully_static': access.is_fully_static(),
                'context': access.context[:100] + '...' if len(access.context) > 100 else access.context
            })
            
        return serialized
        
    def _serialize_usage_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Serialize usage patterns for output."""
        serialized = {}
        
        for name, pattern in self.analyzer.array_usage_patterns.items():
            serialized[name] = {
                'array_name': pattern.array_name,
                'total_accesses': pattern.total_accesses,
                'static_accesses': pattern.static_accesses,
                'dynamic_accesses': pattern.dynamic_accesses,
                'read_accesses': pattern.read_accesses,
                'write_accesses': pattern.write_accesses,
                'unique_indices_count': len(pattern.unique_indices),
                'accessed_elements_count': len(pattern.accessed_elements),
                'unused_elements_count': len(pattern.unused_elements),
                'usage_ratio': pattern.usage_ratio,
                'static_ratio': pattern.static_ratio,
                'bounds_violations_count': len(pattern.bounds_violations)
            }
            
        return serialized
        
    def _serialize_relationships(self) -> List[Dict[str, Any]]:
        """Serialize array relationships for output."""
        serialized = []
        
        for rel in self.analyzer.array_relationships:
            serialized.append({
                'source_array': rel.source_array,
                'target_array': rel.target_array,
                'relationship_type': rel.relationship_type,
                'strength': rel.strength,
                'context': rel.context,
                'routine_count': len(rel.routine_names),
                'routines': list(rel.routine_names)
            })
            
        return serialized
        
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Analyze usage patterns for recommendations
        for name, pattern in self.analyzer.array_usage_patterns.items():
            if name in self.analyzer.array_definitions:
                array_def = self.analyzer.array_definitions[name]
                
                # Low usage recommendation
                if pattern.usage_ratio < 0.3 and pattern.total_accesses > 5:
                    recommendations.append({
                        'type': 'OPTIMIZATION',
                        'priority': 'MEDIUM',
                        'target': name,
                        'title': 'Low Array Usage Detected',
                        'description': f"Array '{name}' has low usage ratio ({pattern.usage_ratio:.1%}). Consider reducing size or reviewing necessity.",
                        'current_size': array_def.total_elements,
                        'suggested_size': len(pattern.accessed_elements),
                        'potential_memory_savings': array_def.total_elements - len(pattern.accessed_elements)
                    })
                    
                # High dynamic access recommendation
                if pattern.dynamic_accesses > pattern.static_accesses and pattern.total_accesses > 10:
                    recommendations.append({
                        'type': 'PERFORMANCE',
                        'priority': 'LOW',
                        'target': name,
                        'title': 'High Dynamic Access Usage',
                        'description': f"Array '{name}' has {pattern.dynamic_accesses} dynamic accesses vs {pattern.static_accesses} static. Consider bounds checking.",
                        'dynamic_ratio': pattern.dynamic_accesses / pattern.total_accesses,
                        'suggestion': 'Add bounds checking for dynamic array accesses'
                    })
                    
        # Relationship-based recommendations
        strong_relationships = [rel for rel in self.analyzer.array_relationships if rel.strength > 0.7]
        if strong_relationships:
            recommendations.append({
                'type': 'ARCHITECTURE',
                'priority': 'LOW',
                'target': 'MULTIPLE',
                'title': 'Strong Array Relationships Detected',
                'description': f"Found {len(strong_relationships)} strong array relationships. Consider data structure consolidation.",
                'relationships': len(strong_relationships),
                'suggestion': 'Review array relationships for potential UDT consolidation'
            })
            
        # Size optimization recommendations
        large_arrays = [
            (name, array_def) for name, array_def in self.analyzer.array_definitions.items() 
            if array_def.total_elements > 100
        ]
        
        for name, array_def in large_arrays:
            if name in self.analyzer.array_usage_patterns:
                pattern = self.analyzer.array_usage_patterns[name]
                if pattern.usage_ratio < 0.5:
                    recommendations.append({
                        'type': 'MEMORY',
                        'priority': 'HIGH',
                        'target': name,
                        'title': 'Large Array with Low Usage',
                        'description': f"Large array '{name}' ({array_def.total_elements} elements) has low usage ({pattern.usage_ratio:.1%}). Significant memory optimization possible.",
                        'current_memory_estimate': array_def.total_elements * 4,  # Assume 4 bytes per element
                        'optimized_memory_estimate': len(pattern.accessed_elements) * 4,
                        'memory_savings_bytes': (array_def.total_elements - len(pattern.accessed_elements)) * 4
                    })
                    
        return recommendations
        
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics for the analysis."""
        if self.analysis_start_time and self.analysis_end_time:
            total_time = self.analysis_end_time - self.analysis_start_time
            
            self.performance_metrics = {
                'total_time_seconds': total_time,
                'arrays_per_second': len(self.analyzer.array_definitions) / total_time if total_time > 0 else 0,
                'accesses_per_second': len(self.analyzer.array_accesses) / total_time if total_time > 0 else 0,
                'relationships_per_second': len(self.analyzer.array_relationships) / total_time if total_time > 0 else 0,
                'total_arrays_analyzed': len(self.analyzer.array_definitions),
                'total_accesses_analyzed': len(self.analyzer.array_accesses),
                'total_relationships_found': len(self.analyzer.array_relationships),
                'memory_usage_estimate': self._estimate_memory_usage()
            }
            
    def _estimate_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage for arrays."""
        total_elements = sum(arr.total_elements for arr in self.analyzer.array_definitions.values())
        
        # Rough estimates (bytes per element by type)
        type_sizes = {
            ArrayType.BOOL_ARRAY: 1,
            ArrayType.INT_ARRAY: 2,
            ArrayType.DINT_ARRAY: 4,
            ArrayType.REAL_ARRAY: 4,
            ArrayType.STRING_ARRAY: 82,  # Typical string size
            ArrayType.UDT_ARRAY: 16,     # Estimated UDT size
            ArrayType.IO_ARRAY: 4        # Estimated I/O size
        }
        
        total_memory = 0
        for array_def in self.analyzer.array_definitions.values():
            element_size = type_sizes.get(array_def.array_type, 4)
            total_memory += array_def.total_elements * element_size
            
        return {
            'total_elements': total_elements,
            'estimated_bytes': total_memory,
            'estimated_kb': total_memory // 1024,
            'estimated_mb': total_memory // (1024 * 1024)
        }
        
    def generate_html_report(self, output_path: str) -> bool:
        """Generate HTML report for array analysis."""
        try:
            logger.info(f"Generating HTML report: {output_path}")
            
            # Run analysis if not already done
            if not self.analyzer.array_definitions and not self.analyzer.array_accesses:
                logger.info("Running analysis for HTML report generation...")
                self.run_comprehensive_analysis()
            
            html_content = self._generate_html_content()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"HTML report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return False
            
    def _generate_html_content(self) -> str:
        """Generate HTML content for the report."""
        # Get analysis data
        summary = self.analyzer.get_array_analysis_summary()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Array Analysis Report</title>
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
        .array-type-BOOL_ARRAY {{ color: #28a745; }}
        .array-type-INT_ARRAY {{ color: #007bff; }}
        .array-type-DINT_ARRAY {{ color: #6f42c1; }}
        .array-type-REAL_ARRAY {{ color: #fd7e14; }}
        .array-type-UDT_ARRAY {{ color: #dc3545; }}
        .array-type-IO_ARRAY {{ color: #6c757d; }}
        .recommendation-HIGH {{ border-left-color: #dc3545; }}
        .recommendation-MEDIUM {{ border-left-color: #ffc107; }}
        .recommendation-LOW {{ border-left-color: #28a745; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔢 Array Analysis Report</h1>
        <p><strong>File:</strong> {self.l5x_file_path or 'Unknown'}</p>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        {f'<p><strong>Analysis Time:</strong> {self.performance_metrics.get("total_time_seconds", 0):.2f} seconds</p>' if self.performance_metrics else ''}
    </div>

    <div class="section">
        <h2>📊 Analysis Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_arrays', 0)}</div>
                <div class="metric-label">Total Arrays</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_accesses', 0)}</div>
                <div class="metric-label">Total Accesses</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_relationships', 0)}</div>
                <div class="metric-label">Array Relationships</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('access_patterns', {}).get('static_ratio', 0):.1%}</div>
                <div class="metric-label">Static Access Ratio</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🏷️ Array Definitions</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Array Name</th>
                    <th>Type</th>
                    <th>Element Type</th>
                    <th>Dimensions</th>
                    <th>Total Elements</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_array_definitions_table()}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>📈 Usage Patterns</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Array Name</th>
                    <th>Total Accesses</th>
                    <th>Static/Dynamic</th>
                    <th>Read/Write</th>
                    <th>Usage Ratio</th>
                    <th>Unused Elements</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_usage_patterns_table()}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🔗 Array Relationships</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Source Array</th>
                    <th>Target Array</th>
                    <th>Relationship Type</th>
                    <th>Strength</th>
                    <th>Context</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_relationships_table()}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>💡 Optimization Recommendations</h2>
        {self._generate_recommendations_section()}
    </div>

    <div class="section">
        <h2>⚡ Performance Metrics</h2>
        {self._generate_performance_section()}
    </div>
</body>
</html>
        """
        
        return html
        
    def _generate_array_definitions_table(self) -> str:
        """Generate HTML table rows for array definitions."""
        rows = []
        for name, array_def in self.analyzer.array_definitions.items():
            dimensions_str = " × ".join(str(dim.size) for dim in array_def.dimensions)
            rows.append(f"""
                <tr>
                    <td><strong>{array_def.name}</strong></td>
                    <td><span class="array-type-{array_def.array_type.value}">{array_def.array_type.value}</span></td>
                    <td>{array_def.element_data_type}</td>
                    <td>{dimensions_str}</td>
                    <td>{array_def.total_elements:,}</td>
                    <td>{array_def.description[:50] + '...' if len(array_def.description) > 50 else array_def.description}</td>
                </tr>
            """)
        return "".join(rows) if rows else "<tr><td colspan='6'>No array definitions found</td></tr>"
        
    def _generate_usage_patterns_table(self) -> str:
        """Generate HTML table rows for usage patterns."""
        rows = []
        for name, pattern in self.analyzer.array_usage_patterns.items():
            static_dynamic = f"{pattern.static_accesses}/{pattern.dynamic_accesses}"
            read_write = f"{pattern.read_accesses}/{pattern.write_accesses}"
            usage_ratio = f"{pattern.usage_ratio:.1%}" if hasattr(pattern, 'usage_ratio') else "N/A"
            
            rows.append(f"""
                <tr>
                    <td><strong>{pattern.array_name}</strong></td>
                    <td>{pattern.total_accesses}</td>
                    <td>{static_dynamic}</td>
                    <td>{read_write}</td>
                    <td>{usage_ratio}</td>
                    <td>{len(pattern.unused_elements)}</td>
                </tr>
            """)
        return "".join(rows) if rows else "<tr><td colspan='6'>No usage patterns found</td></tr>"
        
    def _generate_relationships_table(self) -> str:
        """Generate HTML table rows for relationships."""
        rows = []
        for rel in self.analyzer.array_relationships:
            rows.append(f"""
                <tr>
                    <td><strong>{rel.source_array}</strong></td>
                    <td><strong>{rel.target_array}</strong></td>
                    <td>{rel.relationship_type}</td>
                    <td>{rel.strength:.2f}</td>
                    <td>{rel.context[:100] + '...' if len(rel.context) > 100 else rel.context}</td>
                </tr>
            """)
        return "".join(rows) if rows else "<tr><td colspan='5'>No relationships found</td></tr>"
        
    def _generate_recommendations_section(self) -> str:
        """Generate HTML content for recommendations."""
        if not hasattr(self, '_recommendations'):
            self._recommendations = self._generate_recommendations()
            
        if not self._recommendations:
            return "<p>No optimization recommendations at this time.</p>"
            
        html = []
        for rec in self._recommendations:
            priority_class = f"recommendation-{rec.get('priority', 'LOW')}"
            html.append(f"""
                <div class="metric-card {priority_class}">
                    <h3>{rec.get('title', 'Recommendation')}</h3>
                    <p><strong>Priority:</strong> {rec.get('priority', 'LOW')} | <strong>Target:</strong> {rec.get('target', 'N/A')}</p>
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
                    <div class="metric-value">{self.performance_metrics.get('arrays_per_second', 0):.1f}</div>
                    <div class="metric-label">Arrays/Second</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.performance_metrics.get('accesses_per_second', 0):.1f}</div>
                    <div class="metric-label">Accesses/Second</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.performance_metrics.get('memory_usage_estimate', {}).get('estimated_kb', 0)}</div>
                    <div class="metric-label">Estimated Memory (KB)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.performance_metrics.get('total_time_seconds', 0):.2f}s</div>
                    <div class="metric-label">Analysis Time</div>
                </div>
            </div>
        """
        
    def export_json_report(self, output_path: str) -> bool:
        """Export analysis results as JSON."""
        try:
            logger.info(f"Exporting JSON report: {output_path}")
            
            # Run analysis if not already done
            if not self.analyzer.array_definitions and not self.analyzer.array_accesses:
                logger.info("Running analysis for JSON export...")
                results = self.run_comprehensive_analysis()
            else:
                results = self._generate_comprehensive_results()
                results['performance_metrics'] = self.performance_metrics
                results['recommendations'] = self._generate_recommendations()
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            logger.info(f"JSON report exported successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export JSON report: {e}")
            return False
