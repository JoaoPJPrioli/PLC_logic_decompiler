"""
I/O Models for PLC Components

This module defines data structures for PLC I/O modules, connections, and mappings.
Supports Rockwell L5X format I/O parsing.

Author: GitHub Copilot
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ConnectionType(Enum):
    """Types of I/O connections"""
    INPUT = "Input"
    OUTPUT = "Output"
    INPUT_OUTPUT = "InputOutput"
    STANDARD_DATA_DRIVEN = "StandardDataDriven"
    LISTEN_ONLY = "ListenOnly"


class ModuleType(Enum):
    """Types of I/O modules"""
    CONTROLLER = "Controller"
    DISCRETE_IO = "DiscreteIO"
    ANALOG_IO = "AnalogIO" 
    COMMUNICATIONS = "Communications"
    GENERIC = "Generic"
    ETHERNET = "Ethernet"
    ROBOT = "Robot"


@dataclass
class IOPoint:
    """Represents a single I/O point with its mapping and comments"""
    operand: str                    # e.g., ".DATA.0", ".Fault"
    bit_number: Optional[int]       # Bit number for discrete I/O
    comment: Optional[str]          # Human-readable description
    data_type: Optional[str]        # Data type (BOOL, SINT, DINT, etc.)
    value: Optional[str]            # Default or current value


@dataclass
class IOConnection:
    """Represents an I/O connection (InputTag or OutputTag)"""
    name: str                       # Connection name (e.g., "Data", "InputData")
    connection_type: ConnectionType # Input, Output, etc.
    rpi: Optional[int]              # Request Packet Interval (ms)
    tag_suffix: Optional[str]       # Tag suffix (e.g., "I1", "O1")
    external_access: str            # External access permissions
    data_structure: Optional[str]   # Data structure type
    io_points: Dict[str, IOPoint]   # Map of operand -> IOPoint
    raw_data: Optional[Dict]        # Raw XML data for debugging
    
    def __post_init__(self):
        """Initialize collections"""
        if not self.io_points:
            self.io_points = {}


@dataclass  
class IOModule:
    """Represents a complete I/O module"""
    name: str                       # Module name
    catalog_number: str             # Hardware catalog number
    vendor_id: str                  # Vendor identifier
    product_type: str               # Product type code
    product_code: str               # Product code
    major_revision: str             # Major firmware revision
    minor_revision: str             # Minor firmware revision
    parent_module: str              # Parent module name
    parent_port_id: str             # Parent port ID
    inhibited: bool                 # Module inhibited status
    major_fault: bool               # Module fault status
    module_type: ModuleType         # Classified module type
    connections: Dict[str, IOConnection]  # Map of connection name -> IOConnection
    slot_number: Optional[int]      # Physical slot number
    ip_address: Optional[str]       # IP address for Ethernet modules
    
    def __post_init__(self):
        """Initialize collections and classify module type"""
        if not self.connections:
            self.connections = {}
        
        # Auto-classify module type based on catalog number
        if not hasattr(self, 'module_type') or self.module_type is None:
            self.module_type = self._classify_module_type()
    
    def _classify_module_type(self) -> ModuleType:
        """Classify module type based on catalog number"""
        catalog = self.catalog_number.upper()
        
        if catalog.startswith(('1769-L', '1756-L', '5069-L')):
            return ModuleType.CONTROLLER
        elif 'EMBEDDED' in catalog or 'DISCRETE' in self.name.upper():
            return ModuleType.DISCRETE_IO
        elif catalog.startswith(('1734-I', '1734-O', '1769-I', '1769-O')):
            return ModuleType.DISCRETE_IO
        elif catalog.startswith(('1734-A', '1769-A')):
            return ModuleType.ANALOG_IO
        elif 'FANUC' in catalog or 'ROBOT' in catalog:
            return ModuleType.ROBOT
        elif catalog.startswith(('1756-EN', '1769-EN')):
            return ModuleType.ETHERNET
        else:
            return ModuleType.GENERIC
    
    def get_input_connections(self) -> List[IOConnection]:
        """Get all input connections for this module"""
        return [conn for conn in self.connections.values() 
                if conn.connection_type in [ConnectionType.INPUT, ConnectionType.INPUT_OUTPUT]]
    
    def get_output_connections(self) -> List[IOConnection]:
        """Get all output connections for this module"""
        return [conn for conn in self.connections.values() 
                if conn.connection_type in [ConnectionType.OUTPUT, ConnectionType.INPUT_OUTPUT]]
    
    def get_total_io_points(self) -> int:
        """Get total number of I/O points across all connections"""
        return sum(len(conn.io_points) for conn in self.connections.values())
    
    def get_commented_io_points(self) -> List[IOPoint]:
        """Get all I/O points that have comments"""
        commented_points = []
        for conn in self.connections.values():
            for point in conn.io_points.values():
                if point.comment and point.comment.strip():
                    commented_points.append(point)
        return commented_points


@dataclass
class IOMapping:
    """Complete I/O mapping for the PLC system"""
    modules: Dict[str, IOModule] = field(default_factory=dict) # Map of module name -> IOModule
    controller_module: Optional[IOModule] = None  # Reference to controller module
    total_modules: int = 0
    total_io_points: int = 0  
    total_commented_points: int = 0
    
    def __post_init__(self):
        """Initialize and calculate statistics"""
        if not self.modules:
            self.modules = {}
        self._update_statistics()
    
    def _update_statistics(self):
        """Update mapping statistics"""
        self.total_modules = len(self.modules)
        self.total_io_points = sum(module.get_total_io_points() for module in self.modules.values())
        self.total_commented_points = sum(len(module.get_commented_io_points()) for module in self.modules.values())
        
        # Find controller module
        for module in self.modules.values():
            if module.module_type == ModuleType.CONTROLLER:
                self.controller_module = module
                break
    
    def add_module(self, module: IOModule):
        """Add a module to the mapping"""
        self.modules[module.name] = module
        self._update_statistics()
    
    def get_modules_by_type(self, module_type: ModuleType) -> List[IOModule]:
        """Get all modules of a specific type"""
        return [module for module in self.modules.values() if module.module_type == module_type]
    
    def get_all_io_points(self) -> List[IOPoint]:
        """Get all I/O points from all modules"""
        all_points = []
        for module in self.modules.values():
            for connection in module.connections.values():
                all_points.extend(connection.io_points.values())
        return all_points
    
    def get_io_point_by_operand(self, operand: str) -> Optional[IOPoint]:
        """Find an I/O point by its operand string"""
        for module in self.modules.values():
            for connection in module.connections.values():
                if operand in connection.io_points:
                    return connection.io_points[operand]
        return None
    
    def search_io_points_by_comment(self, search_term: str) -> List[IOPoint]:
        """Search I/O points by comment text"""
        search_term = search_term.lower()
        matching_points = []
        
        for module in self.modules.values():
            for connection in module.connections.values():
                for point in connection.io_points.values():
                    if point.comment and search_term in point.comment.lower():
                        matching_points.append(point)
        
        return matching_points
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive I/O mapping statistics"""
        stats = {
            'total_modules': self.total_modules,
            'total_io_points': self.total_io_points,
            'total_commented_points': self.total_commented_points,
            'comment_coverage_percent': (self.total_commented_points / self.total_io_points * 100) if self.total_io_points > 0 else 0,
            'modules_by_type': {},
            'connections_by_type': {},
            'io_points_by_module': {}
        }
        
        # Count modules by type
        for module_type in ModuleType:
            count = len(self.get_modules_by_type(module_type))
            if count > 0:
                stats['modules_by_type'][module_type.value] = count
        
        # Count connections by type
        for module in self.modules.values():
            for connection in module.connections.values():
                conn_type = connection.connection_type.value
                stats['connections_by_type'][conn_type] = stats['connections_by_type'].get(conn_type, 0) + 1
        
        # Count I/O points by module
        for module_name, module in self.modules.items():
            point_count = module.get_total_io_points()
            if point_count > 0:
                stats['io_points_by_module'][module_name] = point_count
        
        return stats


def create_io_point_from_comment_element(comment_element, data_type: Optional[str] = None) -> IOPoint:
    """Create an IOPoint from an XML Comment element"""
    operand = comment_element.get('Operand', '')
    comment_text = comment_element.text or ''
    
    # Extract bit number if it's a discrete point
    bit_number = None
    if '.DATA.' in operand:
        try:
            bit_number = int(operand.split('.DATA.')[1])
        except (IndexError, ValueError):
            pass
    
    return IOPoint(
        operand=operand,
        bit_number=bit_number,
        comment=comment_text.strip(),
        data_type=data_type,
        value=None
    )


def create_io_connection_from_xml_element(connection_element, connection_name: str, connection_type: ConnectionType) -> IOConnection:
    """Create an IOConnection from an XML Connection element"""
    # Extract connection attributes
    rpi = connection_element.get('RPI')
    rpi_int = int(rpi) if rpi and rpi.isdigit() else None
    
    tag_suffix = connection_element.get('InputTagSuffix') or connection_element.get('OutputTagSuffix')
    external_access = connection_element.get('ExternalAccess', 'Read/Write')
    
    # Create connection
    connection = IOConnection(
        name=connection_name,
        connection_type=connection_type,
        rpi=rpi_int,
        tag_suffix=tag_suffix,
        external_access=external_access,
        data_structure=None,
        io_points={},
        raw_data=connection_element.attrib
    )
    
    # Extract I/O points from comments
    comments_element = connection_element.find('Comments')
    if comments_element is not None:
        for comment_element in comments_element.findall('Comment'):
            io_point = create_io_point_from_comment_element(comment_element)
            connection.io_points[io_point.operand] = io_point
    
    # Extract data structure info
    data_element = connection_element.find('Data')
    if data_element is not None:
        structure_element = data_element.find('Structure')
        if structure_element is not None:
            connection.data_structure = structure_element.get('DataType')
    
    return connection


def create_io_module_from_xml_element(module_element) -> IOModule:
    """Create an IOModule from an XML Module element"""
    # Extract module attributes
    name = module_element.get('Name', '')
    catalog_number = module_element.get('CatalogNumber', '')
    vendor_id = module_element.get('Vendor', '')
    product_type = module_element.get('ProductType', '')
    product_code = module_element.get('ProductCode', '')
    major_revision = module_element.get('Major', '')
    minor_revision = module_element.get('Minor', '')
    parent_module = module_element.get('ParentModule', '')
    parent_port_id = module_element.get('ParentModPortId', '')
    inhibited = module_element.get('Inhibited', 'false').lower() == 'true'
    major_fault = module_element.get('MajorFault', 'false').lower() == 'true'
    
    # Create module
    module = IOModule(
        name=name,
        catalog_number=catalog_number,
        vendor_id=vendor_id,
        product_type=product_type,
        product_code=product_code,
        major_revision=major_revision,
        minor_revision=minor_revision,
        parent_module=parent_module,
        parent_port_id=parent_port_id,
        inhibited=inhibited,
        major_fault=major_fault,
        module_type=ModuleType.GENERIC,  # Will be auto-classified in __post_init__
        connections={},
        slot_number=None,
        ip_address=None
    )
    
    # Extract connections
    connections_element = module_element.find('Connections')
    if connections_element is not None:
        for connection_element in connections_element.findall('Connection'):
            connection_name = connection_element.get('Name', '')
            connection_type_str = connection_element.get('Type', 'Input')
            
            # Map connection type string to enum
            connection_type = ConnectionType.INPUT  # Default
            if connection_type_str in [ct.value for ct in ConnectionType]:
                connection_type = ConnectionType(connection_type_str)
            
            # Determine if this is InputTag or OutputTag
            input_tag = connection_element.find('InputTag')
            output_tag = connection_element.find('OutputTag')
            
            if input_tag is not None:
                io_connection = create_io_connection_from_xml_element(input_tag, connection_name, connection_type)
                module.connections[f"{connection_name}_Input"] = io_connection
            
            if output_tag is not None:
                io_connection = create_io_connection_from_xml_element(output_tag, connection_name, connection_type)
                module.connections[f"{connection_name}_Output"] = io_connection
    
    return module
