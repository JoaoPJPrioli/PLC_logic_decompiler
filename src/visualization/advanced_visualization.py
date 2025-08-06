"""
Advanced Visualization Dashboard - Step 37
Enhanced interactive visualizations for PLC analysis

This module provides sophisticated visualization capabilities including:
- 3D network visualizations
- Interactive dashboards  
- Process flow diagrams
- Real-time analytics
- Advanced chart types
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Types of advanced visualizations available"""
    NETWORK_3D = "network_3d"
    PROCESS_FLOW = "process_flow" 
    ANALYTICS_DASHBOARD = "analytics_dashboard"
    HEATMAP = "heatmap"
    SANKEY = "sankey"
    SUNBURST = "sunburst"
    TIMELINE = "timeline"
    FORCE_DIRECTED = "force_directed"

class ChartType(Enum):
    """Advanced chart types for visualization"""
    FORCE_LAYOUT = "force_layout"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    TIMELINE = "timeline"
    HEATMAP = "heatmap"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    PARALLEL_COORDINATES = "parallel_coordinates"
    CHORD_DIAGRAM = "chord_diagram"

class InteractionMode(Enum):
    """User interaction modes"""
    VIEW_ONLY = "view_only"
    INTERACTIVE = "interactive"
    EDITABLE = "editable"
    COLLABORATIVE = "collaborative"

@dataclass
class VisualizationConfig:
    """Configuration for advanced visualizations"""
    type: VisualizationType
    title: str
    width: int = 1200
    height: int = 800
    interactive: bool = True
    animation: bool = True
    theme: str = "dark"
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["html", "svg", "png", "json"]

@dataclass
class VisualizationData:
    """Data structure for visualization content"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    config: VisualizationConfig
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics"""
    total_tags: int = 0
    total_routines: int = 0
    total_instructions: int = 0
    complexity_score: float = 0.0
    performance_score: float = 0.0
    safety_score: float = 0.0
    optimization_opportunities: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class AdvancedVisualizationEngine:
    """Main engine for creating advanced visualizations"""
    
    def __init__(self):
        self.visualizations: Dict[str, VisualizationData] = {}
        self.dashboard_metrics: Optional[DashboardMetrics] = None
        self.templates_path = Path("templates/visualization")
        self.static_path = Path("static/visualization")
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure visualization directories exist"""
        self.templates_path.mkdir(parents=True, exist_ok=True)
        self.static_path.mkdir(parents=True, exist_ok=True)
        
    async def create_3d_network(self, graph_data: Dict[str, Any], 
                               config: VisualizationConfig) -> str:
        """Create interactive 3D network visualization"""
        logger.info("Creating 3D network visualization")
        
        # Process graph data for 3D visualization
        nodes = self._process_nodes_for_3d(graph_data.get('nodes', []))
        edges = self._process_edges_for_3d(graph_data.get('edges', []))
        
        # Create visualization data
        viz_data = VisualizationData(
            nodes=nodes,
            edges=edges,
            metadata={
                'node_count': len(nodes),
                'edge_count': len(edges),
                'layout': 'force_directed_3d',
                'physics_enabled': True
            },
            config=config
        )
        
        # Generate unique ID and store
        viz_id = f"3d_network_{int(time.time())}"
        self.visualizations[viz_id] = viz_data
        
        # Generate HTML template
        html_content = self._generate_3d_network_html(viz_data)
        html_path = self.templates_path / f"{viz_id}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"3D network visualization created: {viz_id}")
        return viz_id
        
    def _process_nodes_for_3d(self, nodes: List[Dict]) -> List[Dict[str, Any]]:
        """Process nodes for 3D visualization"""
        processed_nodes = []
        
        for i, node in enumerate(nodes):
            processed_node = {
                'id': node.get('id', f'node_{i}'),
                'label': node.get('label', node.get('name', f'Node {i}')),
                'type': node.get('type', 'default'),
                'size': node.get('size', 10),
                'color': self._get_node_color(node.get('type', 'default')),
                'x': node.get('x', 0),
                'y': node.get('y', 0), 
                'z': node.get('z', 0),
                'metadata': node.get('metadata', {})
            }
            processed_nodes.append(processed_node)
            
        return processed_nodes
        
    def _process_edges_for_3d(self, edges: List[Dict]) -> List[Dict[str, Any]]:
        """Process edges for 3D visualization"""
        processed_edges = []
        
        for i, edge in enumerate(edges):
            processed_edge = {
                'id': edge.get('id', f'edge_{i}'),
                'source': edge.get('source'),
                'target': edge.get('target'),
                'weight': edge.get('weight', 1.0),
                'type': edge.get('type', 'default'),
                'color': self._get_edge_color(edge.get('type', 'default')),
                'width': edge.get('width', 2),
                'metadata': edge.get('metadata', {})
            }
            processed_edges.append(processed_edge)
            
        return processed_edges
        
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type"""
        color_map = {
            'tag': '#4CAF50',
            'routine': '#2196F3', 
            'instruction': '#FF9800',
            'io_point': '#9C27B0',
            'timer': '#F44336',
            'counter': '#607D8B',
            'program': '#795548',
            'default': '#757575'
        }
        return color_map.get(node_type, color_map['default'])
        
    def _get_edge_color(self, edge_type: str) -> str:
        """Get color for edge type"""
        color_map = {
            'data_flow': '#4CAF50',
            'control_flow': '#2196F3',
            'dependency': '#FF9800', 
            'reference': '#9C27B0',
            'call': '#F44336',
            'default': '#757575'
        }
        return color_map.get(edge_type, color_map['default'])
        
    def _generate_3d_network_html(self, viz_data: VisualizationData) -> str:
        """Generate HTML for 3D network visualization"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{viz_data.config.title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.9/dat.gui.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #000;
            overflow: hidden;
            font-family: 'Arial', sans-serif;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 5px;
            z-index: 1000;
        }}
        #controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 5px;
            z-index: 1000;
        }}
        button {{
            background: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 2px;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background: #1976D2;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="info">
            <h3>{viz_data.config.title}</h3>
            <p>Nodes: {len(viz_data.nodes)}</p>
            <p>Edges: {len(viz_data.edges)}</p>
            <p>Mouse: Rotate | Wheel: Zoom | Right-click: Pan</p>
        </div>
        <div id="controls">
            <button onclick="resetView()">Reset View</button>
            <button onclick="toggleAnimation()">Toggle Animation</button>
            <button onclick="exportView()">Export</button>
        </div>
    </div>
    
    <script>
        // 3D Network Visualization Implementation
        let scene, camera, renderer, controls;
        let nodes = {json.dumps(viz_data.nodes)};
        let edges = {json.dumps(viz_data.edges)};
        let animationEnabled = true;
        
        function init() {{
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            
            // Camera setup
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(50, 50, 50);
            
            // Renderer setup
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(50, 50, 50);
            directionalLight.castShadow = true;
            scene.add(directionalLight);
            
            // Create nodes
            createNodes();
            
            // Create edges
            createEdges();
            
            // Controls
            setupControls();
            
            // Start animation loop
            animate();
        }}
        
        function createNodes() {{
            const nodeGeometry = new THREE.SphereGeometry(1, 16, 16);
            
            nodes.forEach(node => {{
                const nodeMaterial = new THREE.MeshLambertMaterial({{ 
                    color: node.color,
                    transparent: true,
                    opacity: 0.8
                }});
                
                const nodeMesh = new THREE.Mesh(nodeGeometry, nodeMaterial);
                nodeMesh.position.set(
                    node.x || (Math.random() - 0.5) * 100,
                    node.y || (Math.random() - 0.5) * 100, 
                    node.z || (Math.random() - 0.5) * 100
                );
                nodeMesh.scale.setScalar(node.size || 1);
                nodeMesh.userData = node;
                nodeMesh.castShadow = true;
                nodeMesh.receiveShadow = true;
                
                scene.add(nodeMesh);
            }});
        }}
        
        function createEdges() {{
            edges.forEach(edge => {{
                const sourceNode = nodes.find(n => n.id === edge.source);
                const targetNode = nodes.find(n => n.id === edge.target);
                
                if (sourceNode && targetNode) {{
                    const geometry = new THREE.BufferGeometry();
                    const positions = new Float32Array([
                        sourceNode.x || 0, sourceNode.y || 0, sourceNode.z || 0,
                        targetNode.x || 0, targetNode.y || 0, targetNode.z || 0
                    ]);
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    
                    const material = new THREE.LineBasicMaterial({{ 
                        color: edge.color || '#757575',
                        linewidth: edge.width || 1
                    }});
                    
                    const line = new THREE.Line(geometry, material);
                    line.userData = edge;
                    scene.add(line);
                }}
            }});
        }}
        
        function setupControls() {{
            // Mouse controls for camera
            let mouseDown = false;
            let mouseX = 0;
            let mouseY = 0;
            
            renderer.domElement.addEventListener('mousedown', (event) => {{
                mouseDown = true;
                mouseX = event.clientX;
                mouseY = event.clientY;
            }});
            
            renderer.domElement.addEventListener('mouseup', () => {{
                mouseDown = false;
            }});
            
            renderer.domElement.addEventListener('mousemove', (event) => {{
                if (mouseDown) {{
                    const deltaX = event.clientX - mouseX;
                    const deltaY = event.clientY - mouseY;
                    
                    camera.position.x += deltaX * 0.1;
                    camera.position.y -= deltaY * 0.1;
                    
                    mouseX = event.clientX;
                    mouseY = event.clientY;
                }}
            }});
            
            renderer.domElement.addEventListener('wheel', (event) => {{
                const scale = event.deltaY > 0 ? 1.1 : 0.9;
                camera.position.multiplyScalar(scale);
            }});
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            
            if (animationEnabled) {{
                // Rotate the entire scene slowly
                scene.rotation.y += 0.002;
            }}
            
            camera.lookAt(scene.position);
            renderer.render(scene, camera);
        }}
        
        function resetView() {{
            camera.position.set(50, 50, 50);
            scene.rotation.set(0, 0, 0);
        }}
        
        function toggleAnimation() {{
            animationEnabled = !animationEnabled;
        }}
        
        function exportView() {{
            const link = document.createElement('a');
            link.download = '{viz_data.config.title}_3d_view.png';
            link.href = renderer.domElement.toDataURL();
            link.click();
        }}
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Initialize visualization
        init();
    </script>
</body>
</html>
        """

    async def create_analytics_dashboard(self, plc_data: Dict[str, Any],
                                        config: VisualizationConfig) -> str:
        """Create comprehensive analytics dashboard"""
        logger.info("Creating analytics dashboard")
        
        # Calculate dashboard metrics
        metrics = self._calculate_dashboard_metrics(plc_data)
        self.dashboard_metrics = metrics
        
        # Create dashboard data structure
        dashboard_data = {
            'metrics': asdict(metrics),
            'charts': self._prepare_dashboard_charts(plc_data),
            'alerts': self._generate_dashboard_alerts(plc_data),
            'trends': self._calculate_trends(plc_data)
        }
        
        # Generate unique ID and store
        viz_id = f"dashboard_{int(time.time())}"
        viz_data = VisualizationData(
            nodes=[],
            edges=[],
            metadata=dashboard_data,
            config=config
        )
        self.visualizations[viz_id] = viz_data
        
        # Generate HTML dashboard
        html_content = self._generate_dashboard_html(viz_data)
        html_path = self.templates_path / f"{viz_id}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Analytics dashboard created: {viz_id}")
        return viz_id
        
    def _calculate_dashboard_metrics(self, plc_data: Dict[str, Any]) -> DashboardMetrics:
        """Calculate key metrics for dashboard"""
        tags = plc_data.get('tags', [])
        routines = plc_data.get('routines', [])
        instructions = plc_data.get('instructions', [])
        
        # Calculate complexity score (0-10)
        complexity_factors = [
            len(routines) * 0.1,  # Routine count
            len(instructions) * 0.01,  # Instruction count
            len([t for t in tags if t.get('type') == 'UDT']) * 0.5,  # UDT complexity
            len([i for i in instructions if i.get('type') in ['JSR', 'SBR']]) * 0.2  # Subroutine calls
        ]
        complexity_score = min(10.0, sum(complexity_factors))
        
        # Calculate performance score (0-10)
        performance_factors = [
            10.0,  # Start with perfect score
            -len([i for i in instructions if i.get('type') in ['TON', 'TOF']]) * 0.1,  # Timer penalty
            -len([t for t in tags if t.get('scope') == 'program']) * 0.05,  # Program scope penalty
        ]
        performance_score = max(0.0, sum(performance_factors))
        
        # Calculate safety score (0-10) 
        safety_factors = [
            8.0,  # Start with good score
            len([i for i in instructions if 'SAFE' in i.get('operand', '')]) * 0.5,  # Safety elements
            -len([i for i in instructions if i.get('type') == 'JMP']) * 0.2,  # Jump penalty
        ]
        safety_score = max(0.0, min(10.0, sum(safety_factors)))
        
        # Count optimization opportunities
        optimization_opportunities = len([
            i for i in instructions 
            if i.get('type') in ['MOV', 'CPT'] and 'TIMER' in i.get('operand', '')
        ])
        
        return DashboardMetrics(
            total_tags=len(tags),
            total_routines=len(routines),
            total_instructions=len(instructions),
            complexity_score=complexity_score,
            performance_score=performance_score,
            safety_score=safety_score,
            optimization_opportunities=optimization_opportunities
        )
        
    def _prepare_dashboard_charts(self, plc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare chart data for dashboard"""
        return {
            'tag_distribution': self._get_tag_distribution(plc_data.get('tags', [])),
            'instruction_types': self._get_instruction_distribution(plc_data.get('instructions', [])),
            'routine_complexity': self._get_routine_complexity(plc_data.get('routines', [])),
            'performance_trends': self._get_performance_trends(plc_data)
        }
        
    def _get_tag_distribution(self, tags: List[Dict]) -> Dict[str, int]:
        """Get distribution of tag types"""
        distribution = {}
        for tag in tags:
            tag_type = tag.get('data_type', 'UNKNOWN')
            distribution[tag_type] = distribution.get(tag_type, 0) + 1
        return distribution
        
    def _get_instruction_distribution(self, instructions: List[Dict]) -> Dict[str, int]:
        """Get distribution of instruction types"""
        distribution = {}
        for instruction in instructions:
            inst_type = instruction.get('type', 'UNKNOWN')
            distribution[inst_type] = distribution.get(inst_type, 0) + 1
        return distribution
        
    def _get_routine_complexity(self, routines: List[Dict]) -> Dict[str, float]:
        """Calculate complexity for each routine"""
        complexity = {}
        for routine in routines:
            name = routine.get('name', 'Unknown')
            rungs = routine.get('rungs', [])
            instructions = sum(len(rung.get('instructions', [])) for rung in rungs)
            complexity[name] = instructions * 0.1  # Simple complexity metric
        return complexity
        
    def _get_performance_trends(self, plc_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Get performance trend data"""
        # Simulated trend data - in real implementation, this would come from historical data
        return {
            'response_time': [1.2, 1.1, 1.3, 1.0, 0.9, 1.1, 1.0],
            'memory_usage': [45.2, 46.1, 44.8, 47.2, 48.1, 46.9, 45.5],
            'cpu_utilization': [12.5, 13.2, 11.8, 14.1, 12.9, 13.5, 12.1]
        }
        
    def _generate_dashboard_alerts(self, plc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dashboard alerts"""
        alerts = []
        
        instructions = plc_data.get('instructions', [])
        tags = plc_data.get('tags', [])
        
        # Check for potential issues
        timer_count = len([i for i in instructions if i.get('type') in ['TON', 'TOF', 'RTO']])
        if timer_count > 20:
            alerts.append({
                'type': 'warning',
                'message': f'High timer usage detected: {timer_count} timers',
                'severity': 'medium'
            })
            
        unused_tags = len([t for t in tags if t.get('references', 0) == 0])
        if unused_tags > 10:
            alerts.append({
                'type': 'info',
                'message': f'Unused tags found: {unused_tags} tags not referenced',
                'severity': 'low'
            })
            
        return alerts
        
    def _calculate_trends(self, plc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trend analysis"""
        return {
            'complexity_trend': 'stable',
            'performance_trend': 'improving',
            'safety_trend': 'stable',
            'recommendations': [
                'Consider consolidating similar routines',
                'Review unused tags for cleanup',
                'Add more safety interlocks'
            ]
        }
        
    def _generate_dashboard_html(self, viz_data: VisualizationData) -> str:
        """Generate HTML for analytics dashboard"""
        dashboard_data = viz_data.metadata
        metrics = dashboard_data['metrics']
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{viz_data.config.title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        .alert-item {{
            border-left: 4px solid #17a2b8;
            padding-left: 15px;
            margin: 10px 0;
        }}
        .score-bar {{
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .score-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .score-excellent {{ background: #28a745; }}
        .score-good {{ background: #17a2b8; }}
        .score-warning {{ background: #ffc107; }}
        .score-danger {{ background: #dc3545; }}
    </style>
</head>
<body class="bg-light">
    <div class="container-fluid">
        <div class="row">
            <div class="col-12">
                <h1 class="text-center my-4">{viz_data.config.title}</h1>
            </div>
        </div>
        
        <!-- Metrics Row -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <h3>{metrics['total_tags']}</h3>
                    <p>Total Tags</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <h3>{metrics['total_routines']}</h3>
                    <p>Total Routines</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <h3>{metrics['total_instructions']}</h3>
                    <p>Total Instructions</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <h3>{metrics['optimization_opportunities']}</h3>
                    <p>Optimization Opportunities</p>
                </div>
            </div>
        </div>
        
        <!-- Scores Row -->
        <div class="row my-4">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <h5>Complexity Score</h5>
                        <div class="score-bar">
                            <div class="score-fill score-{self._get_score_class(metrics['complexity_score'])}" 
                                 style="width: {metrics['complexity_score'] * 10}%"></div>
                        </div>
                        <small>{metrics['complexity_score']:.1f} / 10.0</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <h5>Performance Score</h5>
                        <div class="score-bar">
                            <div class="score-fill score-{self._get_score_class(metrics['performance_score'])}" 
                                 style="width: {metrics['performance_score'] * 10}%"></div>
                        </div>
                        <small>{metrics['performance_score']:.1f} / 10.0</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <h5>Safety Score</h5>
                        <div class="score-bar">
                            <div class="score-fill score-{self._get_score_class(metrics['safety_score'])}" 
                                 style="width: {metrics['safety_score'] * 10}%"></div>
                        </div>
                        <small>{metrics['safety_score']:.1f} / 10.0</small>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Charts Row -->
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Tag Distribution</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="tagChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Instruction Types</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="instructionChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Alerts and Trends Row -->
        <div class="row my-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>System Alerts</h5>
                    </div>
                    <div class="card-body">
                        {self._generate_alerts_html(dashboard_data['alerts'])}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Performance Trends</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="trendsChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Chart data
        const chartData = {json.dumps(dashboard_data['charts'])};
        
        // Tag Distribution Chart
        const tagCtx = document.getElementById('tagChart').getContext('2d');
        new Chart(tagCtx, {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(chartData.tag_distribution),
                datasets: [{{
                    data: Object.values(chartData.tag_distribution),
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB', 
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF',
                        '#FF9F40'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        // Instruction Types Chart
        const instCtx = document.getElementById('instructionChart').getContext('2d');
        new Chart(instCtx, {{
            type: 'bar',
            data: {{
                labels: Object.keys(chartData.instruction_types),
                datasets: [{{
                    label: 'Instructions',
                    data: Object.values(chartData.instruction_types),
                    backgroundColor: '#36A2EB'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Performance Trends Chart
        const trendsCtx = document.getElementById('trendsChart').getContext('2d');
        new Chart(trendsCtx, {{
            type: 'line',
            data: {{
                labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
                datasets: [
                    {{
                        label: 'Response Time (ms)',
                        data: chartData.performance_trends.response_time,
                        borderColor: '#FF6384',
                        tension: 0.1
                    }},
                    {{
                        label: 'Memory Usage (%)',
                        data: chartData.performance_trends.memory_usage,
                        borderColor: '#36A2EB',
                        tension: 0.1
                    }},
                    {{
                        label: 'CPU Utilization (%)',
                        data: chartData.performance_trends.cpu_utilization,
                        borderColor: '#FFCE56',
                        tension: 0.1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Auto-refresh dashboard every 30 seconds
        setInterval(() => {{
            location.reload();
        }}, 30000);
    </script>
</body>
</html>
        """
        
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score color"""
        if score >= 8.0:
            return 'excellent'
        elif score >= 6.0:
            return 'good'
        elif score >= 4.0:
            return 'warning'
        else:
            return 'danger'
            
    def _generate_alerts_html(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate HTML for alerts section"""
        if not alerts:
            return '<p class="text-muted">No alerts at this time.</p>'
            
        html = ''
        for alert in alerts:
            icon = '⚠️' if alert['type'] == 'warning' else 'ℹ️'
            html += f'''
            <div class="alert-item">
                <small class="text-muted">{alert['severity'].upper()}</small>
                <p>{icon} {alert['message']}</p>
            </div>
            '''
        return html

    async def create_process_flow(self, flow_data: Dict[str, Any],
                                 config: VisualizationConfig) -> str:
        """Create interactive process flow diagram"""
        logger.info("Creating process flow diagram")
        
        # Process flow data for visualization
        flow_nodes = self._process_flow_nodes(flow_data.get('processes', []))
        flow_edges = self._process_flow_edges(flow_data.get('connections', []))
        
        viz_data = VisualizationData(
            nodes=flow_nodes,
            edges=flow_edges,
            metadata={
                'flow_type': flow_data.get('type', 'sequential'),
                'process_count': len(flow_nodes),
                'connection_count': len(flow_edges)
            },
            config=config
        )
        
        viz_id = f"process_flow_{int(time.time())}"
        self.visualizations[viz_id] = viz_data
        
        # Generate HTML for process flow
        html_content = self._generate_process_flow_html(viz_data)
        html_path = self.templates_path / f"{viz_id}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Process flow diagram created: {viz_id}")
        return viz_id
        
    def _process_flow_nodes(self, processes: List[Dict]) -> List[Dict[str, Any]]:
        """Process flow nodes for diagram"""
        flow_nodes = []
        
        for i, process in enumerate(processes):
            node = {
                'id': process.get('id', f'process_{i}'),
                'label': process.get('name', f'Process {i}'),
                'type': process.get('type', 'process'),
                'x': process.get('x', i * 150),
                'y': process.get('y', 100),
                'width': 120,
                'height': 60,
                'color': self._get_process_color(process.get('type', 'process'))
            }
            flow_nodes.append(node)
            
        return flow_nodes
        
    def _process_flow_edges(self, connections: List[Dict]) -> List[Dict[str, Any]]:
        """Process flow edges for diagram"""
        flow_edges = []
        
        for i, connection in enumerate(connections):
            edge = {
                'id': connection.get('id', f'connection_{i}'),
                'source': connection.get('from'),
                'target': connection.get('to'),
                'type': connection.get('type', 'sequential'),
                'label': connection.get('condition', ''),
                'color': '#2196F3'
            }
            flow_edges.append(edge)
            
        return flow_edges
        
    def _get_process_color(self, process_type: str) -> str:
        """Get color for process type"""
        color_map = {
            'start': '#4CAF50',
            'process': '#2196F3',
            'decision': '#FF9800',
            'end': '#F44336',
            'io': '#9C27B0',
            'delay': '#607D8B'
        }
        return color_map.get(process_type, '#757575')
        
    def _generate_process_flow_html(self, viz_data: VisualizationData) -> str:
        """Generate HTML for process flow diagram""" 
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{viz_data.config.title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.6.1/d3.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        #flowDiagram {{
            width: 100%;
            height: 600px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .process-node {{
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .process-node:hover {{
            stroke-width: 3px;
            filter: brightness(1.1);
        }}
        .connection-line {{
            fill: none;
            stroke-width: 2;
            marker-end: url(#arrowhead);
        }}
        .node-label {{
            text-anchor: middle;
            dominant-baseline: middle;
            font-size: 12px;
            font-weight: bold;
            fill: white;
        }}
        .connection-label {{
            text-anchor: middle;
            font-size: 10px;
            fill: #666;
        }}
        #controls {{
            margin-bottom: 20px;
        }}
        button {{
            background: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 2px;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background: #1976D2;
        }}
    </style>
</head>
<body>
    <h1>{viz_data.config.title}</h1>
    
    <div id="controls">
        <button onclick="zoomIn()">Zoom In</button>
        <button onclick="zoomOut()">Zoom Out</button>
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="exportDiagram()">Export SVG</button>
    </div>
    
    <div id="flowDiagram"></div>
    
    <script>
        const nodes = {json.dumps(viz_data.nodes)};
        const edges = {json.dumps(viz_data.edges)};
        
        const svg = d3.select("#flowDiagram")
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%");
            
        const g = svg.append("g");
        
        // Define arrowhead marker
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 8)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#666");
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
            
        svg.call(zoom);
        
        // Draw connections
        const connections = g.selectAll(".connection")
            .data(edges)
            .enter()
            .append("g")
            .attr("class", "connection");
            
        connections.append("line")
            .attr("class", "connection-line")
            .attr("x1", d => {{
                const sourceNode = nodes.find(n => n.id === d.source);
                return sourceNode ? sourceNode.x + sourceNode.width/2 : 0;
            }})
            .attr("y1", d => {{
                const sourceNode = nodes.find(n => n.id === d.source);
                return sourceNode ? sourceNode.y + sourceNode.height/2 : 0;
            }})
            .attr("x2", d => {{
                const targetNode = nodes.find(n => n.id === d.target);
                return targetNode ? targetNode.x + targetNode.width/2 : 0;
            }})
            .attr("y2", d => {{
                const targetNode = nodes.find(n => n.id === d.target);
                return targetNode ? targetNode.y + targetNode.height/2 : 0;
            }})
            .attr("stroke", d => d.color);
            
        // Add connection labels
        connections.append("text")
            .attr("class", "connection-label")
            .attr("x", d => {{
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                if (sourceNode && targetNode) {{
                    return (sourceNode.x + targetNode.x) / 2 + (sourceNode.width + targetNode.width) / 4;
                }}
                return 0;
            }})
            .attr("y", d => {{
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                if (sourceNode && targetNode) {{
                    return (sourceNode.y + targetNode.y) / 2 + (sourceNode.height + targetNode.height) / 4;
                }}
                return 0;
            }})
            .text(d => d.label);
        
        // Draw process nodes
        const processNodes = g.selectAll(".process-node")
            .data(nodes)
            .enter()
            .append("g")
            .attr("class", "process-node");
        
        processNodes.append("rect")
            .attr("x", d => d.x)
            .attr("y", d => d.y)
            .attr("width", d => d.width)
            .attr("height", d => d.height)
            .attr("fill", d => d.color)
            .attr("stroke", "#333")
            .attr("stroke-width", 1)
            .attr("rx", 5);
            
        processNodes.append("text")
            .attr("class", "node-label")
            .attr("x", d => d.x + d.width/2)
            .attr("y", d => d.y + d.height/2)
            .text(d => d.label);
        
        // Zoom controls
        function zoomIn() {{
            svg.transition().call(zoom.scaleBy, 1.5);
        }}
        
        function zoomOut() {{
            svg.transition().call(zoom.scaleBy, 1 / 1.5);
        }}
        
        function resetZoom() {{
            svg.transition().call(zoom.transform, d3.zoomIdentity);
        }}
        
        function exportDiagram() {{
            const svgData = svg.node().outerHTML;
            const blob = new Blob([svgData], {{type: "image/svg+xml"}});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "{viz_data.config.title}_process_flow.svg";
            link.click();
        }}
    </script>
</body>
</html>
        """

    async def export_visualization(self, viz_id: str, 
                                  export_format: str = "html") -> str:
        """Export visualization in specified format"""
        if viz_id not in self.visualizations:
            raise ValueError(f"Visualization {viz_id} not found")
            
        viz_data = self.visualizations[viz_id]
        export_path = self.static_path / f"{viz_id}.{export_format}"
        
        if export_format == "json":
            export_data = {
                'visualization_data': asdict(viz_data),
                'export_timestamp': datetime.now().isoformat(),
                'export_format': export_format
            }
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif export_format == "html":
            # HTML already generated during creation
            template_path = self.templates_path / f"{viz_id}.html"
            if template_path.exists():
                import shutil
                shutil.copy(template_path, export_path)
            else:
                raise FileNotFoundError(f"HTML template for {viz_id} not found")
                
        logger.info(f"Visualization {viz_id} exported as {export_format}")
        return str(export_path)

    def get_visualization_list(self) -> List[Dict[str, Any]]:
        """Get list of all available visualizations"""
        viz_list = []
        
        for viz_id, viz_data in self.visualizations.items():
            viz_info = {
                'id': viz_id,
                'title': viz_data.config.title,
                'type': viz_data.config.type.value,
                'created_at': viz_data.created_at.isoformat(),
                'node_count': len(viz_data.nodes),
                'edge_count': len(viz_data.edges),
                'interactive': viz_data.config.interactive
            }
            viz_list.append(viz_info)
            
        return viz_list

# Advanced Visualization Integrator
class AdvancedVisualizationIntegrator:
    """Integration layer for advanced visualization with existing PLC analysis"""
    
    def __init__(self):
        self.engine = AdvancedVisualizationEngine()
        
    async def create_plc_visualization_suite(self, l5x_file_path: str) -> Dict[str, str]:
        """Create comprehensive visualization suite for PLC file"""
        logger.info(f"Creating visualization suite for: {l5x_file_path}")
        
        # Import necessary modules (mock for now)
        try:
            # These would import from existing PLC analysis modules
            from src.parsers.l5x_parser import L5XParser
            from src.core.processing_pipeline import PLCProcessingPipeline
        except ImportError:
            logger.warning("PLC analysis modules not available, using mock data")
            return await self._create_mock_visualization_suite()
        
        # Process PLC file
        parser = L5XParser()
        pipeline = PLCProcessingPipeline()
        
        plc_data = await pipeline.process_file(l5x_file_path)
        
        # Create visualizations
        visualizations = {}
        
        # 3D Network Visualization
        network_config = VisualizationConfig(
            type=VisualizationType.NETWORK_3D,
            title="PLC Network Architecture - 3D View"
        )
        viz_id = await self.engine.create_3d_network(plc_data.get('graph_data', {}), network_config)
        visualizations['3d_network'] = viz_id
        
        # Analytics Dashboard
        dashboard_config = VisualizationConfig(
            type=VisualizationType.ANALYTICS_DASHBOARD,
            title="PLC Analytics Dashboard"
        )
        viz_id = await self.engine.create_analytics_dashboard(plc_data, dashboard_config)
        visualizations['analytics_dashboard'] = viz_id
        
        # Process Flow Diagram
        flow_config = VisualizationConfig(
            type=VisualizationType.PROCESS_FLOW,
            title="PLC Process Flow Diagram"
        )
        flow_data = self._extract_process_flow(plc_data)
        viz_id = await self.engine.create_process_flow(flow_data, flow_config)
        visualizations['process_flow'] = viz_id
        
        logger.info(f"Visualization suite created: {len(visualizations)} visualizations")
        return visualizations
        
    async def _create_mock_visualization_suite(self) -> Dict[str, str]:
        """Create mock visualization suite for testing"""
        logger.info("Creating mock visualization suite")
        
        # Mock PLC data
        mock_plc_data = {
            'tags': [
                {'id': 'tag1', 'name': 'Motor_1_Start', 'data_type': 'BOOL', 'scope': 'controller'},
                {'id': 'tag2', 'name': 'Motor_1_Stop', 'data_type': 'BOOL', 'scope': 'controller'},
                {'id': 'tag3', 'name': 'Conveyor_Speed', 'data_type': 'REAL', 'scope': 'program'},
                {'id': 'tag4', 'name': 'Timer_1', 'data_type': 'TIMER', 'scope': 'controller'},
            ],
            'routines': [
                {'id': 'routine1', 'name': 'MainRoutine', 'type': 'ladder'},
                {'id': 'routine2', 'name': 'SafetyCheck', 'type': 'ladder'},
            ],
            'instructions': [
                {'id': 'inst1', 'type': 'XIC', 'operand': 'Motor_1_Start'},
                {'id': 'inst2', 'type': 'OTE', 'operand': 'Motor_1_Run'},
                {'id': 'inst3', 'type': 'TON', 'operand': 'Timer_1'},
                {'id': 'inst4', 'type': 'XIO', 'operand': 'Motor_1_Stop'},
            ],
            'graph_data': {
                'nodes': [
                    {'id': 'Motor_1_Start', 'type': 'tag', 'x': 0, 'y': 0, 'z': 0},
                    {'id': 'Motor_1_Run', 'type': 'tag', 'x': 50, 'y': 0, 'z': 0},
                    {'id': 'MainRoutine', 'type': 'routine', 'x': 25, 'y': 25, 'z': 0},
                ],
                'edges': [
                    {'source': 'Motor_1_Start', 'target': 'Motor_1_Run', 'type': 'data_flow'},
                    {'source': 'MainRoutine', 'target': 'Motor_1_Start', 'type': 'reference'},
                ]
            }
        }
        
        visualizations = {}
        
        # 3D Network
        network_config = VisualizationConfig(
            type=VisualizationType.NETWORK_3D,
            title="Mock PLC Network - 3D View"
        )
        viz_id = await self.engine.create_3d_network(mock_plc_data['graph_data'], network_config)
        visualizations['3d_network'] = viz_id
        
        # Analytics Dashboard
        dashboard_config = VisualizationConfig(
            type=VisualizationType.ANALYTICS_DASHBOARD,
            title="Mock PLC Analytics Dashboard"
        )
        viz_id = await self.engine.create_analytics_dashboard(mock_plc_data, dashboard_config)
        visualizations['analytics_dashboard'] = viz_id
        
        # Process Flow
        flow_config = VisualizationConfig(
            type=VisualizationType.PROCESS_FLOW,
            title="Mock Process Flow Diagram"
        )
        mock_flow_data = {
            'processes': [
                {'id': 'start', 'name': 'Start', 'type': 'start', 'x': 50, 'y': 50},
                {'id': 'check', 'name': 'Safety Check', 'type': 'decision', 'x': 200, 'y': 50},
                {'id': 'run', 'name': 'Run Motor', 'type': 'process', 'x': 350, 'y': 50},
                {'id': 'end', 'name': 'End', 'type': 'end', 'x': 500, 'y': 50},
            ],
            'connections': [
                {'from': 'start', 'to': 'check', 'condition': ''},
                {'from': 'check', 'to': 'run', 'condition': 'Safe'},
                {'from': 'run', 'to': 'end', 'condition': ''},
            ]
        }
        viz_id = await self.engine.create_process_flow(mock_flow_data, flow_config)
        visualizations['process_flow'] = viz_id
        
        return visualizations
    
    def _extract_process_flow(self, plc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract process flow data from PLC analysis"""
        # This would extract actual process flow from PLC data
        # For now, return mock data
        return {
            'processes': [
                {'id': 'init', 'name': 'Initialize', 'type': 'start'},
                {'id': 'safety', 'name': 'Safety Check', 'type': 'decision'},
                {'id': 'operate', 'name': 'Normal Operation', 'type': 'process'},
                {'id': 'shutdown', 'name': 'Shutdown', 'type': 'end'},
            ],
            'connections': [
                {'from': 'init', 'to': 'safety'},
                {'from': 'safety', 'to': 'operate', 'condition': 'Safe'},
                {'from': 'operate', 'to': 'shutdown'},
            ]
        }

# Example usage and testing functions
async def main():
    """Main function for testing advanced visualization"""
    integrator = AdvancedVisualizationIntegrator()
    
    print("🎨 Advanced Visualization System - Step 37")
    print("=" * 50)
    
    # Create visualization suite
    print("\n1. Creating visualization suite...")
    visualizations = await integrator.create_plc_visualization_suite("mock_file.l5x")
    
    print(f"Created {len(visualizations)} visualizations:")
    for viz_type, viz_id in visualizations.items():
        print(f"  - {viz_type}: {viz_id}")
    
    # Get visualization list
    print("\n2. Available visualizations:")
    viz_list = integrator.engine.get_visualization_list()
    for viz in viz_list:
        print(f"  - {viz['title']} ({viz['type']}) - {viz['node_count']} nodes")
    
    # Export visualizations
    print("\n3. Exporting visualizations...")
    for viz_type, viz_id in visualizations.items():
        try:
            json_path = await integrator.engine.export_visualization(viz_id, "json")
            print(f"  - {viz_type} exported to: {json_path}")
        except Exception as e:
            print(f"  - Error exporting {viz_type}: {e}")
    
    print("\n✅ Advanced Visualization System demonstration complete!")

if __name__ == "__main__":
    asyncio.run(main())
