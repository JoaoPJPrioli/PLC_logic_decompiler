"""
Knowledge Graph Module
Creates and manages knowledge graphs from PLC data for advanced analysis

This module provides:
- Graph construction from PLC components
- Relationship mapping between tags, programs, and routines
- Graph-based analysis and insights
- Visualization support
- Pattern detection through graph algorithms
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import networkx as nx
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of nodes in the PLC knowledge graph"""
    CONTROLLER = "CONTROLLER"
    PROGRAM = "PROGRAM"
    ROUTINE = "ROUTINE"
    TAG = "TAG"
    INSTRUCTION = "INSTRUCTION"
    IO_MODULE = "IO_MODULE"
    UDT = "UDT"  # User Defined Type

class EdgeType(Enum):
    """Types of edges in the PLC knowledge graph"""
    CONTAINS = "CONTAINS"           # Program contains routine
    USES = "USES"                   # Routine uses tag
    DEPENDS_ON = "DEPENDS_ON"       # Tag depends on another tag
    CALLS = "CALLS"                 # Routine calls another routine
    READS = "READS"                 # Instruction reads tag
    WRITES = "WRITES"               # Instruction writes tag
    CONNECTS_TO = "CONNECTS_TO"     # IO connections
    IMPLEMENTS = "IMPLEMENTS"       # UDT implementation

@dataclass
class GraphNode:
    """Node in the PLC knowledge graph"""
    id: str
    type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'properties': self.properties
        }

@dataclass
class GraphEdge:
    """Edge in the PLC knowledge graph"""
    source: str
    target: str
    type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.type.value,
            'properties': self.properties
        }

class PLCKnowledgeGraph:
    """
    Knowledge graph for PLC systems that captures relationships
    between all components and enables advanced analysis
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.logger = logging.getLogger(__name__)
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the knowledge graph"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.to_dict())
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the knowledge graph"""
        self.edges.append(edge)
        self.graph.add_edge(edge.source, edge.target, 
                           edge_type=edge.type.value, **edge.properties)
    
    def build_from_l5x_data(self, l5x_data: Dict[str, Any]) -> None:
        """
        Build knowledge graph from L5X analysis data
        
        Args:
            l5x_data: Parsed L5X data from processing pipeline
        """
        self.logger.info("Building knowledge graph from L5X data")
        
        # Extract data components
        final_data = l5x_data.get('final_data', {})
        extracted_data = final_data.get('extracted_data', {})
        
        # Add controller node
        self._add_controller_node(extracted_data.get('controller', {}))
        
        # Add tag nodes
        self._add_tag_nodes(extracted_data.get('detailed_data', {}).get('controller_tags', []))
        
        # Add program and routine nodes
        self._add_program_nodes(extracted_data.get('detailed_data', {}).get('programs', []))
        
        # Add I/O module nodes
        self._add_io_module_nodes(extracted_data.get('detailed_data', {}).get('io_modules', []))
        
        # Create relationships
        self._create_relationships(extracted_data)
        
        self.logger.info(f"Knowledge graph built with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def _add_controller_node(self, controller_data: Dict[str, Any]) -> None:
        """Add controller node to graph"""
        controller_id = f"controller_{controller_data.get('name', 'unknown')}"
        
        node = GraphNode(
            id=controller_id,
            type=NodeType.CONTROLLER,
            name=controller_data.get('name', 'Unknown'),
            properties={
                'type': controller_data.get('type', 'Unknown'),
                'firmware': controller_data.get('firmware', 'Unknown'),
                'tag_count': controller_data.get('tag_count', 0),
                'program_count': controller_data.get('program_count', 0)
            }
        )
        
        self.add_node(node)
    
    def _add_tag_nodes(self, tags_data: List[Dict[str, Any]]) -> None:
        """Add tag nodes to graph"""
        for tag_data in tags_data:
            tag_id = f"tag_{tag_data.get('name', 'unknown')}"
            
            node = GraphNode(
                id=tag_id,
                type=NodeType.TAG,
                name=tag_data.get('name', 'Unknown'),
                properties={
                    'data_type': tag_data.get('data_type', 'UNKNOWN'),
                    'description': tag_data.get('description', ''),
                    'scope': tag_data.get('scope', 'controller'),
                    'constant': tag_data.get('constant', False),
                    'array_dimensions': tag_data.get('array_dimensions', []),
                    'external_access': tag_data.get('external_access', 'Read/Write')
                }
            )
            
            self.add_node(node)
    
    def _add_program_nodes(self, programs_data: List[Dict[str, Any]]) -> None:
        """Add program and routine nodes to graph"""
        for program_data in programs_data:
            program_name = program_data.get('name', 'Unknown')
            program_id = f"program_{program_name}"
            
            # Add program node
            program_node = GraphNode(
                id=program_id,
                type=NodeType.PROGRAM,
                name=program_name,
                properties={
                    'type': program_data.get('type', 'Normal'),
                    'description': program_data.get('description', ''),
                    'main_routine': program_data.get('main_routine', ''),
                    'disabled': program_data.get('disabled', False),
                    'routine_count': len(program_data.get('routines', []))
                }
            )
            
            self.add_node(program_node)
            
            # Add routine nodes for this program
            for routine_name in program_data.get('routines', []):
                routine_id = f"routine_{program_name}_{routine_name}"
                
                routine_node = GraphNode(
                    id=routine_id,
                    type=NodeType.ROUTINE,
                    name=routine_name,
                    properties={
                        'program_name': program_name,
                        'type': 'RLL'  # Assume ladder logic for now
                    }
                )
                
                self.add_node(routine_node)
            
            # Add program tag nodes
            for tag_data in program_data.get('tags', []):
                tag_id = f"tag_{program_name}_{tag_data.get('name', 'unknown')}"
                
                node = GraphNode(
                    id=tag_id,
                    type=NodeType.TAG,
                    name=tag_data.get('name', 'Unknown'),
                    properties={
                        'data_type': tag_data.get('data_type', 'UNKNOWN'),
                        'description': tag_data.get('description', ''),
                        'scope': 'program',
                        'program_name': program_name,
                        'constant': tag_data.get('constant', False),
                        'array_dimensions': tag_data.get('array_dimensions', [])
                    }
                )
                
                self.add_node(node)
    
    def _add_io_module_nodes(self, io_modules_data: List[Dict[str, Any]]) -> None:
        """Add I/O module nodes to graph"""
        for module_data in io_modules_data:
            module_id = f"io_module_{module_data.get('name', 'unknown')}"
            
            node = GraphNode(
                id=module_id,
                type=NodeType.IO_MODULE,
                name=module_data.get('name', 'Unknown'),
                properties={
                    'catalog_number': module_data.get('catalog_number', 'Unknown'),
                    'vendor': module_data.get('vendor', 'Unknown'),
                    'product_type': module_data.get('product_type', 'Unknown'),
                    'major_rev': module_data.get('major_rev', 'Unknown'),
                    'minor_rev': module_data.get('minor_rev', 'Unknown')
                }
            )
            
            self.add_node(node)
    
    def _create_relationships(self, extracted_data: Dict[str, Any]) -> None:
        """Create relationships between nodes"""
        self._create_containment_relationships(extracted_data)
        self._create_tag_usage_relationships(extracted_data)
        self._create_io_relationships(extracted_data)
    
    def _create_containment_relationships(self, extracted_data: Dict[str, Any]) -> None:
        """Create containment relationships (program contains routine, etc.)"""
        controller_name = extracted_data.get('controller', {}).get('name', 'unknown')
        controller_id = f"controller_{controller_name}"
        
        # Controller contains programs
        for program_data in extracted_data.get('detailed_data', {}).get('programs', []):
            program_name = program_data.get('name', 'Unknown')
            program_id = f"program_{program_name}"
            
            edge = GraphEdge(
                source=controller_id,
                target=program_id,
                type=EdgeType.CONTAINS,
                properties={'relationship': 'controller_program'}
            )
            self.add_edge(edge)
            
            # Program contains routines
            for routine_name in program_data.get('routines', []):
                routine_id = f"routine_{program_name}_{routine_name}"
                
                edge = GraphEdge(
                    source=program_id,
                    target=routine_id,
                    type=EdgeType.CONTAINS,
                    properties={'relationship': 'program_routine'}
                )
                self.add_edge(edge)
                
                # Check if this is the main routine
                if routine_name == program_data.get('main_routine', ''):
                    edge = GraphEdge(
                        source=program_id,
                        target=routine_id,
                        type=EdgeType.CALLS,
                        properties={'relationship': 'main_routine', 'priority': 'high'}
                    )
                    self.add_edge(edge)
    
    def _create_tag_usage_relationships(self, extracted_data: Dict[str, Any]) -> None:
        """Create tag usage relationships"""
        # This is simplified - in a full implementation, you'd analyze
        # the actual ladder logic to determine which routines use which tags
        
        controller_name = extracted_data.get('controller', {}).get('name', 'unknown')
        controller_id = f"controller_{controller_name}"
        
        # Controller contains controller tags
        for tag_data in extracted_data.get('detailed_data', {}).get('controller_tags', []):
            tag_id = f"tag_{tag_data.get('name', 'unknown')}"
            
            edge = GraphEdge(
                source=controller_id,
                target=tag_id,
                type=EdgeType.CONTAINS,
                properties={'scope': 'controller'}
            )
            self.add_edge(edge)
        
        # Programs contain program tags
        for program_data in extracted_data.get('detailed_data', {}).get('programs', []):
            program_name = program_data.get('name', 'Unknown')
            program_id = f"program_{program_name}"
            
            for tag_data in program_data.get('tags', []):
                tag_id = f"tag_{program_name}_{tag_data.get('name', 'unknown')}"
                
                edge = GraphEdge(
                    source=program_id,
                    target=tag_id,
                    type=EdgeType.CONTAINS,
                    properties={'scope': 'program'}
                )
                self.add_edge(edge)
    
    def _create_io_relationships(self, extracted_data: Dict[str, Any]) -> None:
        """Create I/O module relationships"""
        controller_name = extracted_data.get('controller', {}).get('name', 'unknown')
        controller_id = f"controller_{controller_name}"
        
        # Controller contains I/O modules
        for module_data in extracted_data.get('detailed_data', {}).get('io_modules', []):
            module_id = f"io_module_{module_data.get('name', 'unknown')}"
            
            edge = GraphEdge(
                source=controller_id,
                target=module_id,
                type=EdgeType.CONTAINS,
                properties={'relationship': 'controller_io_module'}
            )
            self.add_edge(edge)
    
    def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze the structure and properties of the knowledge graph"""
        analysis = {
            'node_statistics': self._analyze_nodes(),
            'edge_statistics': self._analyze_edges(),
            'connectivity': self._analyze_connectivity(),
            'centrality': self._analyze_centrality(),
            'communities': self._detect_communities(),
            'patterns': self._detect_patterns()
        }
        
        return analysis
    
    def _analyze_nodes(self) -> Dict[str, Any]:
        """Analyze node statistics"""
        node_types = defaultdict(int)
        node_properties = defaultdict(list)
        
        for node in self.nodes.values():
            node_types[node.type.value] += 1
            for prop, value in node.properties.items():
                node_properties[prop].append(value)
        
        return {
            'total_nodes': len(self.nodes),
            'node_type_distribution': dict(node_types),
            'property_summary': {prop: len(values) for prop, values in node_properties.items()}
        }
    
    def _analyze_edges(self) -> Dict[str, Any]:
        """Analyze edge statistics"""
        edge_types = defaultdict(int)
        
        for edge in self.edges:
            edge_types[edge.type.value] += 1
        
        return {
            'total_edges': len(self.edges),
            'edge_type_distribution': dict(edge_types),
            'average_degree': len(self.edges) * 2 / len(self.nodes) if self.nodes else 0
        }
    
    def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze graph connectivity"""
        if not self.graph.nodes():
            return {'connected': False, 'components': 0}
        
        # Convert to undirected for connectivity analysis
        undirected = self.graph.to_undirected()
        
        return {
            'connected': nx.is_connected(undirected),
            'components': nx.number_connected_components(undirected),
            'largest_component_size': len(max(nx.connected_components(undirected), key=len, default=[])),
            'diameter': nx.diameter(undirected) if nx.is_connected(undirected) else None
        }
    
    def _analyze_centrality(self) -> Dict[str, Any]:
        """Analyze node centrality measures"""
        if not self.graph.nodes():
            return {}
        
        # Calculate different centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # Find most central nodes
        most_connected = max(degree_centrality.items(), key=lambda x: x[1], default=('', 0))
        most_between = max(betweenness_centrality.items(), key=lambda x: x[1], default=('', 0))
        
        return {
            'most_connected_node': {'id': most_connected[0], 'score': most_connected[1]},
            'most_between_node': {'id': most_between[0], 'score': most_between[1]},
            'average_degree_centrality': sum(degree_centrality.values()) / len(degree_centrality),
            'average_betweenness_centrality': sum(betweenness_centrality.values()) / len(betweenness_centrality)
        }
    
    def _detect_communities(self) -> Dict[str, Any]:
        """Detect communities/clusters in the graph"""
        if not self.graph.nodes():
            return {'communities': [], 'modularity': 0}
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        try:
            # Use a simple community detection algorithm
            communities = list(nx.connected_components(undirected))
            
            return {
                'num_communities': len(communities),
                'communities': [list(community) for community in communities],
                'largest_community_size': len(max(communities, key=len, default=[])),
                'average_community_size': sum(len(c) for c in communities) / len(communities) if communities else 0
            }
        except Exception as e:
            self.logger.warning(f"Community detection failed: {e}")
            return {'communities': [], 'error': str(e)}
    
    def _detect_patterns(self) -> Dict[str, Any]:
        """Detect interesting patterns in the graph"""
        patterns = {
            'hub_nodes': [],
            'isolated_nodes': [],
            'bridge_nodes': [],
            'cycles': []
        }
        
        if not self.graph.nodes():
            return patterns
        
        # Find hub nodes (high degree)
        degrees = dict(self.graph.degree())
        if degrees:
            avg_degree = sum(degrees.values()) / len(degrees)
            hub_threshold = avg_degree * 2
            
            patterns['hub_nodes'] = [
                {'node_id': node_id, 'degree': degree}
                for node_id, degree in degrees.items()
                if degree > hub_threshold
            ]
        
        # Find isolated nodes
        patterns['isolated_nodes'] = [
            node_id for node_id, degree in degrees.items() if degree == 0
        ]
        
        # Find simple cycles
        try:
            simple_cycles = list(nx.simple_cycles(self.graph))
            patterns['cycles'] = [
                {'cycle': cycle, 'length': len(cycle)}
                for cycle in simple_cycles[:10]  # Limit to first 10 cycles
            ]
        except Exception as e:
            self.logger.warning(f"Cycle detection failed: {e}")
        
        return patterns
    
    def get_node_neighbors(self, node_id: str, max_distance: int = 1) -> Dict[str, Any]:
        """Get neighbors of a node within specified distance"""
        if node_id not in self.graph:
            return {'neighbors': [], 'distances': {}}
        
        try:
            # Get nodes within max_distance
            distances = nx.single_source_shortest_path_length(
                self.graph, node_id, cutoff=max_distance
            )
            
            neighbors = []
            for neighbor_id, distance in distances.items():
                if neighbor_id != node_id and neighbor_id in self.nodes:
                    neighbors.append({
                        'node_id': neighbor_id,
                        'node_info': self.nodes[neighbor_id].to_dict(),
                        'distance': distance
                    })
            
            return {
                'neighbors': neighbors,
                'distances': distances,
                'neighbor_count': len(neighbors)
            }
        except Exception as e:
            self.logger.error(f"Error getting neighbors for {node_id}: {e}")
            return {'neighbors': [], 'distances': {}, 'error': str(e)}
    
    def find_shortest_path(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Find shortest path between two nodes"""
        if source_id not in self.graph or target_id not in self.graph:
            return {'path': [], 'length': float('inf'), 'exists': False}
        
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            length = len(path) - 1
            
            # Get detailed path information
            path_details = []
            for i, node_id in enumerate(path):
                node_info = self.nodes[node_id].to_dict() if node_id in self.nodes else {'id': node_id}
                path_details.append({
                    'step': i,
                    'node_id': node_id,
                    'node_info': node_info
                })
            
            return {
                'path': path,
                'path_details': path_details,
                'length': length,
                'exists': True
            }
        except nx.NetworkXNoPath:
            return {'path': [], 'length': float('inf'), 'exists': False}
        except Exception as e:
            self.logger.error(f"Error finding path from {source_id} to {target_id}: {e}")
            return {'path': [], 'length': float('inf'), 'exists': False, 'error': str(e)}
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for visualization or external analysis"""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges],
            'statistics': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'node_types': list(set(node.type.value for node in self.nodes.values())),
                'edge_types': list(set(edge.type.value for edge in self.edges))
            }
        }
