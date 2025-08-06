"""
Graph Visualization Module

Provides visualization capabilities for the PLC knowledge graph using matplotlib
and other visualization libraries. Supports different layouts, filtering,
and export formats.

Author: GitHub Copilot
Date: December 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple
import numpy as np
from pathlib import Path
import logging

from .knowledge_graph import PLCKnowledgeGraph, NodeType, EdgeType


class GraphVisualizer:
    """Visualizer for PLC knowledge graphs"""
    
    def __init__(self, graph: PLCKnowledgeGraph):
        """
        Initialize the visualizer.
        
        Args:
            graph: The knowledge graph to visualize
        """
        self.graph = graph
        self.color_map = self._create_color_map()
        self.layout_cache = {}
    
    def _create_color_map(self) -> Dict[NodeType, str]:
        """Create color mapping for different node types"""
        return {
            NodeType.TAG: '#6495ED',          # Cornflower blue
            NodeType.PROGRAM: '#FF6347',      # Tomato red
            NodeType.ROUTINE: '#3CB371',      # Medium sea green
            NodeType.IO_MODULE: '#FFA500',    # Orange
            NodeType.IO_CONNECTION: '#DAA520', # Golden rod
            NodeType.IO_POINT: '#FFD700',     # Gold
            NodeType.CONTROLLER: '#800080',   # Purple
            NodeType.RUNG: '#A9A9A9',         # Dark gray
            NodeType.INSTRUCTION: '#00BFFF',  # Deep sky blue
            NodeType.DATA_TYPE: '#9370DB'     # Medium purple
        }
    
    def create_overview_visualization(self, 
                                    figsize: Tuple[int, int] = (15, 10),
                                    layout: str = 'spring',
                                    save_path: Optional[str] = None,
                                    show_labels: bool = True,
                                    node_size_by_degree: bool = True) -> bool:
        """
        Create an overview visualization of the entire graph.
        
        Args:
            figsize: Figure size (width, height)
            layout: Layout algorithm ('spring', 'circular', 'shell', 'spectral')
            save_path: Path to save the image
            show_labels: Whether to show node labels
            node_size_by_degree: Scale node size by degree
            
        Returns:
            True if successful
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get layout positions
            pos = self._get_layout_positions(layout)
            
            # Prepare node attributes
            node_colors = []
            node_sizes = []
            labels = {}
            
            for node_id in self.graph.graph.nodes():
                node = self.graph.get_node(node_id)
                if node:
                    # Color by type
                    color = self.color_map.get(node.node_type, '#808080')
                    node_colors.append(color)
                    
                    # Size by degree if requested
                    if node_size_by_degree:
                        degree = self.graph.graph.degree(node_id)
                        size = max(300, min(1000, 300 + degree * 50))
                    else:
                        size = 500
                    node_sizes.append(size)
                    
                    # Label
                    if show_labels:
                        labels[node_id] = node.name[:20] + '...' if len(node.name) > 20 else node.name
            
            # Draw the graph
            nx.draw_networkx_nodes(
                self.graph.graph, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                ax=ax
            )
            
            nx.draw_networkx_edges(
                self.graph.graph, pos,
                edge_color='gray',
                alpha=0.5,
                arrows=True,
                arrowsize=20,
                ax=ax
            )
            
            if show_labels:
                nx.draw_networkx_labels(
                    self.graph.graph, pos,
                    labels,
                    font_size=8,
                    font_weight='bold',
                    ax=ax
                )
            
            # Create legend
            self._create_legend(ax)
            
            # Set title and layout
            stats = self.graph.get_statistics()
            title = f"PLC Knowledge Graph Overview\n{stats['total_nodes']} nodes, {stats['total_edges']} edges"
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved overview visualization to: {save_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to create overview visualization: {e}")
            return False
    
    def create_filtered_visualization(self,
                                    node_types: List[NodeType],
                                    figsize: Tuple[int, int] = (12, 8),
                                    layout: str = 'spring',
                                    save_path: Optional[str] = None) -> bool:
        """
        Create a visualization showing only specific node types.
        
        Args:
            node_types: List of node types to include
            figsize: Figure size
            layout: Layout algorithm
            save_path: Path to save the image
            
        Returns:
            True if successful
        """
        try:
            # Create subgraph with only specified types
            filtered_nodes = []
            for node_id, node in self.graph.nodes.items():
                if node.node_type in node_types:
                    filtered_nodes.append(node_id)
            
            subgraph = self.graph.graph.subgraph(filtered_nodes)
            
            if len(subgraph.nodes()) == 0:
                logging.warning("No nodes found for specified types")
                return False
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get positions for subgraph
            if layout == 'spring':
                pos = nx.spring_layout(subgraph, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(subgraph)
            elif layout == 'shell':
                pos = nx.shell_layout(subgraph)
            else:
                pos = nx.spectral_layout(subgraph)
            
            # Prepare node attributes
            node_colors = []
            labels = {}
            
            for node_id in subgraph.nodes():
                node = self.graph.get_node(node_id)
                if node:
                    color = self.color_map.get(node.node_type, '#808080')
                    node_colors.append(color)
                    labels[node_id] = node.name
            
            # Draw
            nx.draw_networkx_nodes(
                subgraph, pos,
                node_color=node_colors,
                node_size=600,
                alpha=0.8,
                ax=ax
            )
            
            nx.draw_networkx_edges(
                subgraph, pos,
                edge_color='gray',
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                ax=ax
            )
            
            nx.draw_networkx_labels(
                subgraph, pos,
                labels,
                font_size=10,
                font_weight='bold',
                ax=ax
            )
            
            # Create filtered legend
            filtered_legend = []
            for node_type in node_types:
                color = self.color_map.get(node_type, '#808080')
                patch = mpatches.Patch(color=color, label=node_type.value)
                filtered_legend.append(patch)
            
            ax.legend(handles=filtered_legend, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            type_names = ', '.join([nt.value for nt in node_types])
            ax.set_title(f"Filtered View: {type_names}\n{len(subgraph.nodes())} nodes", 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved filtered visualization to: {save_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to create filtered visualization: {e}")
            return False
    
    def create_node_degree_histogram(self,
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: Optional[str] = None) -> bool:
        """
        Create a histogram of node degrees.
        
        Args:
            figsize: Figure size
            save_path: Path to save the image
            
        Returns:
            True if successful
        """
        try:
            degrees = [self.graph.graph.degree(node) for node in self.graph.graph.nodes()]
            
            if not degrees:
                logging.warning("No nodes found for degree histogram")
                return False
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Node Degree')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Node Degrees in PLC Knowledge Graph')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_degree = np.mean(degrees)
            max_degree = max(degrees)
            ax.axvline(mean_degree, color='red', linestyle='--', 
                      label=f'Mean: {mean_degree:.1f}')
            ax.axvline(max_degree, color='orange', linestyle='--', 
                      label=f'Max: {max_degree}')
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved degree histogram to: {save_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to create degree histogram: {e}")
            return False
    
    def create_connectivity_matrix(self,
                                 node_types: List[NodeType],
                                 figsize: Tuple[int, int] = (8, 8),
                                 save_path: Optional[str] = None) -> bool:
        """
        Create a connectivity matrix between different node types.
        
        Args:
            node_types: Node types to include in matrix
            figsize: Figure size
            save_path: Path to save the image
            
        Returns:
            True if successful
        """
        try:
            # Create connectivity matrix
            n_types = len(node_types)
            matrix = np.zeros((n_types, n_types))
            type_to_idx = {node_type: i for i, node_type in enumerate(node_types)}
            
            # Count connections between types
            for edge in self.graph.edges:
                source_node = self.graph.get_node(edge.source_id)
                target_node = self.graph.get_node(edge.target_id)
                
                if (source_node and target_node and 
                    source_node.node_type in type_to_idx and 
                    target_node.node_type in type_to_idx):
                    
                    i = type_to_idx[source_node.node_type]
                    j = type_to_idx[target_node.node_type]
                    matrix[i][j] += 1
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=figsize)
            
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            
            # Set ticks and labels
            type_names = [nt.value for nt in node_types]
            ax.set_xticks(range(n_types))
            ax.set_yticks(range(n_types))
            ax.set_xticklabels(type_names, rotation=45, ha='right')
            ax.set_yticklabels(type_names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Number of Connections')
            
            # Add text annotations
            for i in range(n_types):
                for j in range(n_types):
                    if matrix[i][j] > 0:
                        ax.text(j, i, int(matrix[i][j]), 
                               ha='center', va='center', color='black', fontweight='bold')
            
            ax.set_title('Node Type Connectivity Matrix')
            ax.set_xlabel('Target Node Type')
            ax.set_ylabel('Source Node Type')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved connectivity matrix to: {save_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to create connectivity matrix: {e}")
            return False
    
    def _get_layout_positions(self, layout: str) -> Dict[str, Tuple[float, float]]:
        """Get or compute layout positions"""
        cache_key = f"{layout}_{len(self.graph.graph.nodes())}"
        
        if cache_key in self.layout_cache:
            return self.layout_cache[cache_key]
        
        if layout == 'spring':
            pos = nx.spring_layout(self.graph.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph.graph)
        elif layout == 'shell':
            # Group nodes by type for shell layout
            shells = []
            for node_type in NodeType:
                shell = [node_id for node_id, node in self.graph.nodes.items() 
                        if node.node_type == node_type]
                if shell:
                    shells.append(shell)
            pos = nx.shell_layout(self.graph.graph, nlist=shells)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.graph.graph)
        else:
            pos = nx.spring_layout(self.graph.graph)
        
        self.layout_cache[cache_key] = pos
        return pos
    
    def _create_legend(self, ax):
        """Create legend for node types"""
        legend_elements = []
        
        # Only include types that exist in the graph
        existing_types = set()
        for node in self.graph.nodes.values():
            existing_types.add(node.node_type)
        
        for node_type in existing_types:
            color = self.color_map.get(node_type, '#808080')
            patch = mpatches.Patch(color=color, label=node_type.value)
            legend_elements.append(patch)
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def create_summary_report(self, output_dir: str) -> bool:
        """
        Create a comprehensive visual summary report.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Overview visualization
            self.create_overview_visualization(
                figsize=(15, 10),
                save_path=str(output_path / 'graph_overview.png')
            )
            
            # Degree histogram
            self.create_node_degree_histogram(
                save_path=str(output_path / 'degree_histogram.png')
            )
            
            # Node type connectivity matrix
            existing_types = list(set(node.node_type for node in self.graph.nodes.values()))
            if len(existing_types) > 1:
                self.create_connectivity_matrix(
                    existing_types,
                    save_path=str(output_path / 'connectivity_matrix.png')
                )
            
            # Filtered views for major types
            major_types = [NodeType.TAG, NodeType.PROGRAM, NodeType.IO_MODULE]
            for node_type in major_types:
                if any(node.node_type == node_type for node in self.graph.nodes.values()):
                    self.create_filtered_visualization(
                        [node_type],
                        save_path=str(output_path / f'{node_type.value.lower()}_view.png')
                    )
            
            logging.info(f"Created visual summary report in: {output_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create summary report: {e}")
            return False


def create_quick_visualization(graph: PLCKnowledgeGraph, 
                             output_path: str,
                             layout: str = 'spring') -> bool:
    """
    Quick function to create a basic visualization.
    
    Args:
        graph: Knowledge graph to visualize
        output_path: Path to save the image
        layout: Layout algorithm
        
    Returns:
        True if successful
    """
    visualizer = GraphVisualizer(graph)
    return visualizer.create_overview_visualization(
        save_path=output_path,
        layout=layout
    )
