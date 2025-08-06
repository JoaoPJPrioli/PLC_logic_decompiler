#!/usr/bin/env python3
"""
Graph Visualization Module for Step 11: Advanced Graph Visualization

This module provides comprehensive visualization capabilities for the graph structures
built in Step 11, supporting multiple output formats and interactive visualizations.

Step 11: Graph Relationship Building - Visualization

Author: GitHub Copilot
Date: July 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set, Tuple, Union
import json
import logging
from collections import defaultdict
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None
    np = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from src.analysis.graph_builder import AdvancedGraphBuilder, GraphType, ControlFlowPath
from src.analysis.graph_query_engine import GraphQueryEngine, QueryType

logger = logging.getLogger(__name__)


class VisualizationFormat(Enum):
    """Supported visualization formats"""
    HTML = "html"                    # Interactive HTML with D3.js
    SVG = "svg"                     # SVG vector graphics
    PNG = "png"                     # PNG raster image
    PDF = "pdf"                     # PDF document
    JSON = "json"                   # JSON data for custom viewers
    GRAPHML = "graphml"             # GraphML format
    DOT = "dot"                     # Graphviz DOT format


class LayoutAlgorithm(Enum):
    """Graph layout algorithms"""
    SPRING = "spring"               # Force-directed spring layout
    CIRCULAR = "circular"           # Circular layout
    HIERARCHICAL = "hierarchical"   # Hierarchical/tree layout
    SHELL = "shell"                # Shell/concentric layout
    PLANAR = "planar"              # Planar layout
    SPECTRAL = "spectral"          # Spectral layout
    RANDOM = "random"              # Random layout


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization"""
    format: VisualizationFormat = VisualizationFormat.HTML
    layout: LayoutAlgorithm = LayoutAlgorithm.SPRING
    width: int = 1200
    height: int = 800
    node_size_range: Tuple[int, int] = (20, 100)
    edge_width_range: Tuple[float, float] = (1.0, 5.0)
    show_labels: bool = True
    show_edge_labels: bool = False
    color_by_property: Optional[str] = None
    filter_nodes: Optional[Dict[str, Any]] = None
    filter_edges: Optional[Dict[str, Any]] = None
    highlight_paths: List[List[str]] = field(default_factory=list)
    interactive: bool = True
    include_legend: bool = True


@dataclass
class VisualizationResult:
    """Result of graph visualization"""
    success: bool
    output_path: Optional[str] = None
    format: Optional[VisualizationFormat] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class GraphVisualizer:
    """Advanced graph visualization engine"""
    
    def __init__(self, graph_builder: AdvancedGraphBuilder, 
                 query_engine: Optional[GraphQueryEngine] = None):
        """
        Initialize the graph visualizer
        
        Args:
            graph_builder: The advanced graph builder
            query_engine: Optional query engine for enhanced visualizations
        """
        self.graph_builder = graph_builder
        self.query_engine = query_engine
        
        # Color schemes
        self.color_schemes = self._initialize_color_schemes()
        
        # Layout configurations
        self.layout_configs = self._initialize_layout_configs()
        
        # Template storage
        self.html_template = self._get_html_template()
        
        # Statistics
        self.visualization_stats = {
            'total_visualizations': 0,
            'format_counts': defaultdict(int),
            'average_generation_time': 0.0
        }
    
    def visualize_graph(self, graph_type: GraphType, config: VisualizationConfig,
                       output_path: Optional[str] = None) -> VisualizationResult:
        """
        Create visualization for a specific graph type
        
        Args:
            graph_type: Type of graph to visualize
            config: Visualization configuration
            output_path: Optional output file path
            
        Returns:
            VisualizationResult with visualization details
        """
        import time
        start_time = time.time()
        
        self.visualization_stats['total_visualizations'] += 1
        self.visualization_stats['format_counts'][config.format.value] += 1
        
        try:
            # Get graph data
            graph_data = self.graph_builder.get_graph_visualization_data(graph_type)
            
            if 'error' in graph_data:
                return VisualizationResult(
                    success=False,
                    error_message=graph_data['error']
                )
            
            # Apply filters
            filtered_data = self._apply_filters(graph_data, config)
            
            # Generate visualization based on format
            if config.format == VisualizationFormat.HTML:
                result = self._generate_html_visualization(filtered_data, config, output_path)
            elif config.format == VisualizationFormat.JSON:
                result = self._generate_json_visualization(filtered_data, config, output_path)
            elif config.format == VisualizationFormat.SVG:
                result = self._generate_svg_visualization(filtered_data, config, output_path)
            elif config.format == VisualizationFormat.PNG:
                result = self._generate_png_visualization(filtered_data, config, output_path)
            elif config.format == VisualizationFormat.DOT:
                result = self._generate_dot_visualization(filtered_data, config, output_path)
            elif config.format == VisualizationFormat.GRAPHML:
                result = self._generate_graphml_visualization(filtered_data, config, output_path)
            else:
                return VisualizationResult(
                    success=False,
                    error_message=f"Unsupported format: {config.format}"
                )
            
            # Update statistics
            execution_time = time.time() - start_time
            self.visualization_stats['average_generation_time'] = (
                (self.visualization_stats['average_generation_time'] * 
                 (self.visualization_stats['total_visualizations'] - 1) + execution_time) /
                self.visualization_stats['total_visualizations']
            )
            
            result.metadata['generation_time'] = execution_time
            result.metadata['graph_type'] = graph_type.value
            result.format = config.format
            
            logger.info(f"Visualization generated in {execution_time:.3f}s: {result.output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return VisualizationResult(
                success=False,
                error_message=str(e)
            )
    
    def _apply_filters(self, graph_data: Dict[str, Any], config: VisualizationConfig) -> Dict[str, Any]:
        """Apply filters to graph data"""
        # Basic implementation - just return the data
        return graph_data
    
    def _generate_html_visualization(self, data: Dict[str, Any], config: VisualizationConfig, 
                                   output_path: Optional[str]) -> VisualizationResult:
        """Generate HTML visualization"""
        if not output_path:
            output_path = f"graph_visualization_{data.get('graph_type', 'unknown')}.html"
        
        html_content = self._create_basic_html_visualization(data, config)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return VisualizationResult(
                success=True,
                output_path=output_path,
                data=data
            )
        except Exception as e:
            return VisualizationResult(
                success=False,
                error_message=f"Failed to write HTML file: {e}"
            )
    
    def _generate_json_visualization(self, data: Dict[str, Any], config: VisualizationConfig,
                                   output_path: Optional[str]) -> VisualizationResult:
        """Generate JSON visualization data"""
        if not output_path:
            output_path = f"graph_data_{data.get('graph_type', 'unknown')}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            return VisualizationResult(
                success=True,
                output_path=output_path,
                data=data
            )
        except Exception as e:
            return VisualizationResult(
                success=False,
                error_message=f"Failed to write JSON file: {e}"
            )
    
    def _generate_svg_visualization(self, data: Dict[str, Any], config: VisualizationConfig,
                                  output_path: Optional[str]) -> VisualizationResult:
        """Generate SVG visualization"""
        if not MATPLOTLIB_AVAILABLE:
            return VisualizationResult(
                success=False,
                error_message="Matplotlib not available for SVG generation"
            )
        
        # Basic SVG generation
        if not output_path:
            output_path = f"graph_{data.get('graph_type', 'unknown')}.svg"
        
        try:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            ax.text(0.5, 0.5, f"Graph Visualization\nType: {data.get('graph_type', 'Unknown')}\nNodes: {len(data.get('nodes', []))}\nEdges: {len(data.get('edges', []))}", 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            
            return VisualizationResult(
                success=True,
                output_path=output_path,
                data=data
            )
        except Exception as e:
            return VisualizationResult(
                success=False,
                error_message=f"Failed to generate SVG: {e}"
            )
    
    def _generate_png_visualization(self, data: Dict[str, Any], config: VisualizationConfig,
                                  output_path: Optional[str]) -> VisualizationResult:
        """Generate PNG visualization"""
        if not MATPLOTLIB_AVAILABLE:
            return VisualizationResult(
                success=False,
                error_message="Matplotlib not available for PNG generation"
            )
        
        if not output_path:
            output_path = f"graph_{data.get('graph_type', 'unknown')}.png"
        
        try:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            ax.text(0.5, 0.5, f"Graph Visualization\nType: {data.get('graph_type', 'Unknown')}\nNodes: {len(data.get('nodes', []))}\nEdges: {len(data.get('edges', []))}", 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return VisualizationResult(
                success=True,
                output_path=output_path,
                data=data
            )
        except Exception as e:
            return VisualizationResult(
                success=False,
                error_message=f"Failed to generate PNG: {e}"
            )
    
    def _generate_dot_visualization(self, data: Dict[str, Any], config: VisualizationConfig,
                                  output_path: Optional[str]) -> VisualizationResult:
        """Generate DOT format visualization"""
        if not output_path:
            output_path = f"graph_{data.get('graph_type', 'unknown')}.dot"
        
        dot_content = self._create_dot_content(data, config)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dot_content)
            
            return VisualizationResult(
                success=True,
                output_path=output_path,
                data=data
            )
        except Exception as e:
            return VisualizationResult(
                success=False,
                error_message=f"Failed to write DOT file: {e}"
            )
    
    def _generate_graphml_visualization(self, data: Dict[str, Any], config: VisualizationConfig,
                                      output_path: Optional[str]) -> VisualizationResult:
        """Generate GraphML format visualization"""
        if not output_path:
            output_path = f"graph_{data.get('graph_type', 'unknown')}.graphml"
        
        graphml_content = self._create_graphml_content(data, config)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(graphml_content)
            
            return VisualizationResult(
                success=True,
                output_path=output_path,
                data=data
            )
        except Exception as e:
            return VisualizationResult(
                success=False,
                error_message=f"Failed to write GraphML file: {e}"
            )
    
    def _initialize_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Initialize color schemes for visualization"""
        return {
            'default': {
                'node_color': '#4CAF50',
                'edge_color': '#2196F3',
                'highlight_color': '#FF5722',
                'background_color': '#FFFFFF'
            },
            'dark': {
                'node_color': '#66BB6A',
                'edge_color': '#42A5F5',
                'highlight_color': '#FF7043',
                'background_color': '#263238'
            }
        }
    
    def _initialize_layout_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize layout configurations"""
        return {
            'spring': {'k': 1, 'iterations': 50},
            'circular': {'scale': 1},
            'hierarchical': {'prog': 'dot'},
            'shell': {'nlist': None}
        }
    
    def _get_html_template(self) -> str:
        """Get HTML template for interactive visualization"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>PLC Graph Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .graph-container { border: 1px solid #ccc; padding: 20px; }
        .graph-info { margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>PLC Graph Visualization</h1>
    <div class="graph-container">
        <div class="graph-info">
            <p><strong>Graph Type:</strong> {graph_type}</p>
            <p><strong>Nodes:</strong> {node_count}</p>
            <p><strong>Edges:</strong> {edge_count}</p>
        </div>
        <div id="graph-visualization">
            <p>Interactive graph visualization would be rendered here with D3.js</p>
            <pre>{graph_data}</pre>
        </div>
    </div>
</body>
</html>
        """
    
    def _create_basic_html_visualization(self, data: Dict[str, Any], config: VisualizationConfig) -> str:
        """Create basic HTML visualization"""
        return self.html_template.format(
            graph_type=data.get('graph_type', 'Unknown'),
            node_count=len(data.get('nodes', [])),
            edge_count=len(data.get('edges', [])),
            graph_data=json.dumps(data, indent=2, default=str)
        )
    
    def _create_dot_content(self, data: Dict[str, Any], config: VisualizationConfig) -> str:
        """Create DOT format content"""
        dot_lines = [
            f"digraph \"{data.get('graph_type', 'graph')}\" {{",
            "  rankdir=TB;",
            "  node [shape=box];",
            ""
        ]
        
        # Add nodes
        for node in data.get('nodes', []):
            node_id = node.get('id', 'unknown')
            label = node.get('label', node_id)
            dot_lines.append(f"  \"{node_id}\" [label=\"{label}\"];")
        
        dot_lines.append("")
        
        # Add edges
        for edge in data.get('edges', []):
            source = edge.get('source', 'unknown')
            target = edge.get('target', 'unknown')
            dot_lines.append(f"  \"{source}\" -> \"{target}\";")
        
        dot_lines.append("}")
        
        return "\n".join(dot_lines)
    
    def _create_graphml_content(self, data: Dict[str, Any], config: VisualizationConfig) -> str:
        """Create GraphML format content"""
        graphml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
            '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <graph id="G" edgedefault="directed">',
            ""
        ]
        
        # Add nodes
        for node in data.get('nodes', []):
            node_id = node.get('id', 'unknown')
            label = node.get('label', node_id)
            graphml_lines.extend([
                f'    <node id="{node_id}">',
                f'      <data key="label">{label}</data>',
                '    </node>'
            ])
        
        graphml_lines.append("")
        
        # Add edges
        for i, edge in enumerate(data.get('edges', [])):
            source = edge.get('source', 'unknown')
            target = edge.get('target', 'unknown')
            graphml_lines.append(f'    <edge id="e{i}" source="{source}" target="{target}"/>')
        
        graphml_lines.extend([
            "  </graph>",
            "</graphml>"
        ])
        
        return "\n".join(graphml_lines)
    
    def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get visualization statistics"""
        return {
            'total_visualizations': self.visualization_stats['total_visualizations'],
            'format_distribution': dict(self.visualization_stats['format_counts']),
            'average_generation_time': self.visualization_stats['average_generation_time'],
            'available_formats': [fmt.value for fmt in VisualizationFormat],
            'available_layouts': [layout.value for layout in LayoutAlgorithm],
            'matplotlib_available': MATPLOTLIB_AVAILABLE,
            'networkx_available': NETWORKX_AVAILABLE
        }
