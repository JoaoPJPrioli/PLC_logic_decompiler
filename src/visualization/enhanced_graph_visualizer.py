#!/usr/bin/env python3
"""
Enhanced Graph Visualizer for PLC Logic
Creates meaningful visualizations of PLC logic graphs with actual data
"""

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

class VisualizationFormat(Enum):
    HTML = "html"
    PNG = "png"
    SVG = "svg"
    JSON = "json"

@dataclass
class VisualizationConfig:
    format: VisualizationFormat = VisualizationFormat.HTML
    width: int = 1200
    height: int = 800
    show_node_labels: bool = True
    show_edge_labels: bool = False
    interactive: bool = True
    color_scheme: str = "viridis"

class EnhancedGraphVisualizer:
    """Enhanced visualizer that creates meaningful graphs from PLC data"""
    
    def __init__(self):
        self.color_schemes = {
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'rainbow': px.colors.qualitative.Plotly,
            'pastel': px.colors.qualitative.Pastel
        }
    
    def visualize_control_flow(self, graph_data: Dict[str, Any], config: VisualizationConfig, output_path: str = None) -> str:
        """Visualize control flow graph"""
        if not graph_data.get('nodes') or not graph_data.get('edges'):
            return self._create_empty_visualization("No control flow data available", config, output_path)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in graph_data['nodes']:
            G.add_node(node['id'], **node['properties'])
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge['properties'])
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare data for Plotly
        edge_x, edge_y = [], []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = G.edges[edge]
            edge_type = edge_data.get('type', 'unknown')
            edge_info.append(f"{edge[0]} → {edge[1]} ({edge_type})")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_text = []
        node_colors = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'unknown')
            
            # Color by node type
            if node_type == 'routine':
                node_colors.append('#FF6B6B')  # Red
            elif node_type == 'rung':
                node_colors.append('#4ECDC4')  # Teal
            elif node_type == 'start':
                node_colors.append('#45B7D1')  # Blue
            elif node_type == 'end':
                node_colors.append('#96CEB4')  # Green
            else:
                node_colors.append('#FFEAA7')  # Yellow
            
            # Create hover text
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Type: {node_type}<br>"
            if 'program' in node_data:
                hover_text += f"Program: {node_data['program']}<br>"
            if 'routine' in node_data:
                hover_text += f"Routine: {node_data['routine']}<br>"
            if 'text' in node_data and node_data['text']:
                preview = node_data['text'][:100] + "..." if len(node_data['text']) > 100 else node_data['text']
                hover_text += f"Text: {preview}"
            
            node_text.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if config.show_node_labels else 'markers',
            hoverinfo='text',
            text=[node.split(':')[-1] for node in G.nodes()] if config.show_node_labels else None,
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='PLC Control Flow Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text=f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=config.width,
                           height=config.height
                       ))
        
        return self._save_figure(fig, config, output_path, "control_flow")
    
    def visualize_data_dependency(self, graph_data: Dict[str, Any], config: VisualizationConfig, output_path: str = None) -> str:
        """Visualize data dependency graph"""
        if not graph_data.get('nodes') or not graph_data.get('edges'):
            return self._create_empty_visualization("No data dependency information available", config, output_path)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        for node in graph_data['nodes']:
            G.add_node(node['id'], **node['properties'])
        
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge['properties'])
        
        # Use hierarchical layout for better dependency visualization
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Separate variables and rungs
        variables = [n for n in G.nodes() if G.nodes[n].get('type') == 'variable']
        rungs = [n for n in G.nodes() if G.nodes[n].get('type') == 'rung']
        
        # Create traces for different node types
        traces = []
        
        # Variable nodes
        if variables:
            var_x = [pos[node][0] for node in variables]
            var_y = [pos[node][1] for node in variables]
            var_text = [node.split('.')[-1] for node in variables]
            var_hover = [f"<b>Variable: {node}</b><br>Reads: {G.in_degree(node)}<br>Writes: {G.out_degree(node)}" 
                        for node in variables]
            
            var_trace = go.Scatter(
                x=var_x, y=var_y,
                mode='markers+text' if config.show_node_labels else 'markers',
                name='Variables',
                text=var_text if config.show_node_labels else None,
                textposition="middle center",
                hovertext=var_hover,
                hoverinfo='text',
                marker=dict(
                    size=15,
                    color='#FF6B6B',
                    symbol='circle',
                    line=dict(width=1, color='white')
                )
            )
            traces.append(var_trace)
        
        # Rung nodes
        if rungs:
            rung_x = [pos[node][0] for node in rungs]
            rung_y = [pos[node][1] for node in rungs]
            rung_text = [node.split(':')[-1] for node in rungs]
            rung_hover = [f"<b>{node}</b><br>Reads: {G.out_degree(node)}<br>Writes: {G.in_degree(node)}" 
                         for node in rungs]
            
            rung_trace = go.Scatter(
                x=rung_x, y=rung_y,
                mode='markers+text' if config.show_node_labels else 'markers',
                name='Rungs',
                text=rung_text if config.show_node_labels else None,
                textposition="middle center",
                hovertext=rung_hover,
                hoverinfo='text',
                marker=dict(
                    size=10,
                    color='#4ECDC4',
                    symbol='square',
                    line=dict(width=1, color='white')
                )
            )
            traces.append(rung_trace)
        
        # Add edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Dependencies'
        )
        traces.insert(0, edge_trace)
        
        fig = go.Figure(data=traces,
                       layout=go.Layout(
                           title='PLC Data Dependencies',
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=config.width,
                           height=config.height
                       ))
        
        return self._save_figure(fig, config, output_path, "data_dependency")
    
    def visualize_instruction_network(self, graph_data: Dict[str, Any], config: VisualizationConfig, output_path: str = None) -> str:
        """Visualize instruction network graph"""
        if not graph_data.get('nodes') or not graph_data.get('edges'):
            return self._create_empty_visualization("No instruction network data available", config, output_path)
        
        # Create NetworkX graph
        G = nx.Graph()  # Undirected for instruction co-occurrence
        
        for node in graph_data['nodes']:
            G.add_node(node['id'], **node['properties'])
        
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge['properties'])
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Get node sizes based on instruction count
        node_sizes = []
        node_colors = []
        node_text = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            count = node_data.get('count', 1)
            node_sizes.append(10 + count * 2)  # Scale size by count
            node_colors.append(count)
            node_text.append(f"{node}<br>Count: {count}")
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if config.show_node_labels else 'markers',
            text=[node for node in G.nodes()] if config.show_node_labels else None,
            textposition="middle center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Instruction Count"),
                line=dict(width=1, color='white')
            )
        )
        
        # Create edge traces with weights
        edge_traces = []
        
        for edge in G.edges():
            edge_data = G.edges[edge]
            weight = edge_data.get('weight', 1)
            
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(width=max(1, weight/2), color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        traces = edge_traces + [node_trace]
        
        fig = go.Figure(data=traces,
                       layout=go.Layout(
                           title='PLC Instruction Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=config.width,
                           height=config.height
                       ))
        
        return self._save_figure(fig, config, output_path, "instruction_network")
    
    def visualize_execution_flow(self, graph_data: Dict[str, Any], config: VisualizationConfig, output_path: str = None) -> str:
        """Visualize execution flow graph"""
        if not graph_data.get('nodes') or not graph_data.get('edges'):
            return self._create_empty_visualization("No execution flow data available", config, output_path)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        for node in graph_data['nodes']:
            G.add_node(node['id'], **node['properties'])
        
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge['properties'])
        
        # Use hierarchical layout for execution flow
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Color nodes by type
        node_colors = []
        node_text = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'unknown')
            
            if node_type == 'start':
                node_colors.append('#45B7D1')  # Blue
            elif node_type == 'end':
                node_colors.append('#96CEB4')  # Green
            elif node_type == 'rung':
                node_colors.append('#4ECDC4')  # Teal
            elif node_type == 'condition':
                state = node_data.get('state', '')
                if state == 'true':
                    node_colors.append('#55A3FF')  # Light blue
                else:
                    node_colors.append('#FFB74D')  # Orange
            else:
                node_colors.append('#FFEAA7')  # Yellow
            
            # Create hover text
            hover_text = f"<b>{node}</b><br>Type: {node_type}"
            if 'routine' in node_data:
                hover_text += f"<br>Routine: {node_data['routine']}"
            if 'state' in node_data:
                hover_text += f"<br>State: {node_data['state']}"
            
            node_text.append(hover_text)
        
        # Create traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if config.show_node_labels else 'markers',
            text=[node.split(':')[-1] for node in G.nodes()] if config.show_node_labels else None,
            textposition="middle center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=15,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='PLC Execution Flow',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=config.width,
                           height=config.height
                       ))
        
        return self._save_figure(fig, config, output_path, "execution_flow")
    
    def create_summary_dashboard(self, all_graphs: Dict[str, Dict[str, Any]], config: VisualizationConfig, output_path: str = None) -> str:
        """Create a dashboard showing all graph types"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Control Flow', 'Data Dependencies', 'Instruction Network', 'Execution Flow'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        graph_types = ['control_flow', 'data_dependency', 'instruction_network', 'execution_flow']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for graph_type, (row, col) in zip(graph_types, positions):
            graph_data = all_graphs.get(graph_type, {})
            
            if graph_data.get('nodes') and graph_data.get('edges'):
                # Create simple scatter plot for each graph type
                node_count = len(graph_data['nodes'])
                edge_count = len(graph_data['edges'])
                
                # Show basic statistics as text
                fig.add_trace(
                    go.Scatter(
                        x=[0], y=[0],
                        mode='text',
                        text=[f"Nodes: {node_count}<br>Edges: {edge_count}"],
                        textfont=dict(size=14),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            else:
                # Show "No data" message
                fig.add_trace(
                    go.Scatter(
                        x=[0], y=[0],
                        mode='text',
                        text=["No data available"],
                        textfont=dict(size=12, color='gray'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="PLC Graph Analysis Dashboard",
            height=config.height,
            width=config.width,
            showlegend=False
        )
        
        # Hide axes for all subplots
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
        
        return self._save_figure(fig, config, output_path, "dashboard")
    
    def _create_empty_visualization(self, message: str, config: VisualizationConfig, output_path: str = None) -> str:
        """Create visualization for empty data"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='text',
            text=[message],
            textfont=dict(size=16, color='gray'),
            showlegend=False
        ))
        
        fig.update_layout(
            title='PLC Graph Visualization',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=config.width,
            height=config.height
        )
        
        return self._save_figure(fig, config, output_path, "empty")
    
    def _save_figure(self, fig: go.Figure, config: VisualizationConfig, output_path: str = None, graph_type: str = "graph") -> str:
        """Save figure based on configuration"""
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"plc_{graph_type}_visualization_{timestamp}.{config.format.value}"
        
        try:
            if config.format == VisualizationFormat.HTML:
                fig.write_html(output_path)
            elif config.format == VisualizationFormat.PNG:
                fig.write_image(output_path, format='png')
            elif config.format == VisualizationFormat.SVG:
                fig.write_image(output_path, format='svg')
            elif config.format == VisualizationFormat.JSON:
                with open(output_path, 'w') as f:
                    json.dump(fig.to_dict(), f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"Error saving figure: {e}")
            # Fallback to HTML
            html_path = output_path.replace(f'.{config.format.value}', '.html')
            fig.write_html(html_path)
            return html_path
