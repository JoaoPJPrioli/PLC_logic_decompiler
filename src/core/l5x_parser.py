"""
L5X Parser Module
Comprehensive XML parser for Rockwell Automation L5X files
"""

import xml.etree.ElementTree as ET
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Import the ladder logic parser
try:
    from src.analysis.ladder_logic_parser import LadderLogicParser, RoutineLadderLogic
    LADDER_LOGIC_AVAILABLE = True
except ImportError:
    LADDER_LOGIC_AVAILABLE = False
    RoutineLadderLogic = None

logger = logging.getLogger(__name__)

@dataclass
class ControllerInfo:
    """Controller information from L5X file"""
    name: str
    type: str
    firmware_revision: str
    project_creation_date: str
    project_last_modified: str
    processor_type: str = ""
    description: str = ""

@dataclass
class TagInfo:
    """Tag information from L5X parsing"""
    name: str
    data_type: str
    scope: str  # 'controller', 'program', 'io'
    description: str = ""
    value: Any = None
    program_name: str = ""
    external_access: str = "Read/Write"
    constant: bool = False
    array_dimensions: List[int] = None

    def __post_init__(self):
        if self.array_dimensions is None:
            self.array_dimensions = []

@dataclass
class ProgramInfo:
    """Program information from L5X parsing"""
    name: str
    type: str
    description: str = ""
    main_routine: str = ""
    routines: List[str] = None
    tags: List[TagInfo] = None
    disabled: bool = False

    def __post_init__(self):
        if self.routines is None:
            self.routines = []
        if self.tags is None:
            self.tags = []

@dataclass
class RoutineInfo:
    """Routine information from L5X parsing"""
    name: str
    type: str
    program_name: str
    description: str = ""
    rungs_count: int = 0
    ladder_logic: Optional[Any] = None  # Will be RoutineLadderLogic if available

@dataclass
class L5XParseResult:
    """Complete L5X parsing result"""
    success: bool
    controller_info: ControllerInfo
    controller_tags: List[TagInfo]
    programs: List[ProgramInfo]
    routines: List[RoutineInfo]
    io_modules: List[Dict[str, Any]]
    parsing_time: float
    timestamp: datetime
    error_message: str = ""

class L5XParser:
    """
    Comprehensive L5X file parser for Rockwell Automation PLC programs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if LADDER_LOGIC_AVAILABLE:
            self.ladder_parser = LadderLogicParser()
        else:
            self.ladder_parser = None
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse L5X file and extract comprehensive information
        
        Args:
            file_path: Path to the L5X file
            
        Returns:
            Dictionary containing parsed information
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting L5X file parsing: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"L5X file not found: {file_path}")
            
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract controller information
            controller_info = self._extract_controller_info(root)
            
            # Extract controller-scoped tags
            controller_tags = self._extract_controller_tags(root)
            
            # Extract programs and their information
            programs = self._extract_programs(root)
            
            # Extract routines
            routines = self._extract_routines(root, programs)
            
            # Extract I/O modules
            io_modules = self._extract_io_modules(root)
            
            parsing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'controller_info': asdict(controller_info),
                'controller_tags': [asdict(tag) for tag in controller_tags],
                'programs': [asdict(program) for program in programs],
                'routines': [asdict(routine) for routine in routines],
                'io_modules': io_modules,
                'parsing_time': parsing_time,
                'timestamp': datetime.now(),
                'statistics': {
                    'total_controller_tags': len(controller_tags),
                    'total_programs': len(programs),
                    'total_routines': len(routines),
                    'total_io_modules': len(io_modules),
                    'total_program_tags': sum(len(p.tags) for p in programs)
                }
            }
            
            self.logger.info(f"L5X parsing completed in {parsing_time:.2f} seconds")
            self.logger.info(f"Found: {len(controller_tags)} controller tags, {len(programs)} programs")
            
            return result
            
        except Exception as e:
            parsing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error parsing L5X file: {e}")
            
            return {
                'success': False,
                'controller_info': {},
                'controller_tags': [],
                'programs': [],
                'routines': [],
                'io_modules': [],
                'parsing_time': parsing_time,
                'timestamp': datetime.now(),
                'error_message': str(e),
                'statistics': {}
            }
    
    def _extract_controller_info(self, root: ET.Element) -> ControllerInfo:
        """Extract controller information from L5X"""
        controller_elem = root.find('.//Controller')
        
        if controller_elem is None:
            raise ValueError("No Controller element found in L5X file")
        
        # Get controller attributes
        name = controller_elem.get('Name', 'Unknown')
        processor_type = controller_elem.get('ProcessorType', 'Unknown')
        
        # Get firmware revision
        firmware_revision = "Unknown"
        major_rev = controller_elem.get('MajorRev')
        minor_rev = controller_elem.get('MinorRev')
        if major_rev and minor_rev:
            firmware_revision = f"{major_rev}.{minor_rev}"
        
        # Get creation and modification dates
        project_creation_date = controller_elem.get('ProjectCreationDate', 'Unknown')
        project_last_modified = controller_elem.get('LastModifiedDate', 'Unknown')
        
        return ControllerInfo(
            name=name,
            type=processor_type,
            firmware_revision=firmware_revision,
            project_creation_date=project_creation_date,
            project_last_modified=project_last_modified,
            processor_type=processor_type
        )
    
    def _extract_controller_tags(self, root: ET.Element) -> List[TagInfo]:
        """Extract controller-scoped tags"""
        tags = []
        
        # Find controller tags section
        controller_tags_elem = root.find('.//Controller/Tags')
        if controller_tags_elem is not None:
            for tag_elem in controller_tags_elem.findall('Tag'):
                tag_info = self._parse_tag_element(tag_elem, 'controller')
                if tag_info:
                    tags.append(tag_info)
        
        return tags
    
    def _extract_programs(self, root: ET.Element) -> List[ProgramInfo]:
        """Extract program information"""
        programs = []
        
        programs_elem = root.find('.//Controller/Programs')
        if programs_elem is not None:
            for program_elem in programs_elem.findall('Program'):
                program_info = self._parse_program_element(program_elem)
                if program_info:
                    programs.append(program_info)
        
        return programs
    
    def _extract_routines(self, root: ET.Element, programs: List[ProgramInfo]) -> List[RoutineInfo]:
        """Extract routine information"""
        routines = []
        
        for program in programs:
            program_elem = root.find(f'.//Program[@Name="{program.name}"]')
            if program_elem is not None:
                routines_elem = program_elem.find('Routines')
                if routines_elem is not None:
                    for routine_elem in routines_elem.findall('Routine'):
                        routine_info = self._parse_routine_element(routine_elem, program.name)
                        if routine_info:
                            routines.append(routine_info)
        
        return routines
    
    def _extract_io_modules(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract I/O module information"""
        modules = []
        
        modules_elem = root.find('.//Controller/Modules')
        if modules_elem is not None:
            for module_elem in modules_elem.findall('Module'):
                module_info = {
                    'name': module_elem.get('Name', 'Unknown'),
                    'catalog_number': module_elem.get('CatalogNumber', 'Unknown'),
                    'vendor': module_elem.get('Vendor', 'Unknown'),
                    'product_type': module_elem.get('ProductType', 'Unknown'),
                    'product_code': module_elem.get('ProductCode', 'Unknown'),
                    'major_rev': module_elem.get('Major', 'Unknown'),
                    'minor_rev': module_elem.get('Minor', 'Unknown')
                }
                modules.append(module_info)
        
        return modules
    
    def _parse_tag_element(self, tag_elem: ET.Element, scope: str, program_name: str = "") -> Optional[TagInfo]:
        """Parse individual tag element"""
        try:
            name = tag_elem.get('Name')
            data_type = tag_elem.get('DataType', 'UNKNOWN')
            external_access = tag_elem.get('ExternalAccess', 'Read/Write')
            constant = tag_elem.get('Constant', 'false').lower() == 'true'
            
            # Get description from Comments section
            description = ""
            comments_elem = tag_elem.find('Comments')
            if comments_elem is not None:
                desc_elem = comments_elem.find('Comment[@Operand="."]')
                if desc_elem is not None:
                    description = desc_elem.text or ""
            
            # Parse array dimensions if present
            array_dimensions = []
            if '[' in data_type and ']' in data_type:
                # Extract dimensions from data type like BOOL[64] or REAL[10,5]
                start = data_type.find('[')
                end = data_type.find(']')
                if start != -1 and end != -1:
                    dims_str = data_type[start+1:end]
                    try:
                        dimensions = [int(d.strip()) for d in dims_str.split(',')]
                        array_dimensions = dimensions
                        # Clean data type
                        data_type = data_type[:start]
                    except ValueError:
                        pass
            
            # Get default value if present
            value = None
            data_elem = tag_elem.find('Data')
            if data_elem is not None:
                value = data_elem.text
            
            return TagInfo(
                name=name,
                data_type=data_type,
                scope=scope,
                description=description,
                value=value,
                program_name=program_name,
                external_access=external_access,
                constant=constant,
                array_dimensions=array_dimensions
            )
            
        except Exception as e:
            self.logger.warning(f"Error parsing tag element: {e}")
            return None
    
    def _parse_program_element(self, program_elem: ET.Element) -> Optional[ProgramInfo]:
        """Parse individual program element"""
        try:
            name = program_elem.get('Name')
            program_type = program_elem.get('Type', 'Normal')
            disabled = program_elem.get('Disabled', 'false').lower() == 'true'
            
            # Get description
            description = ""
            desc_elem = program_elem.find('Description')
            if desc_elem is not None:
                description = desc_elem.text or ""
            
            # Get main routine
            main_routine = program_elem.get('MainRoutineName', '')
            
            # Get routines list
            routines = []
            routines_elem = program_elem.find('Routines')
            if routines_elem is not None:
                for routine_elem in routines_elem.findall('Routine'):
                    routine_name = routine_elem.get('Name')
                    if routine_name:
                        routines.append(routine_name)
            
            # Get program tags
            tags = []
            tags_elem = program_elem.find('Tags')
            if tags_elem is not None:
                for tag_elem in tags_elem.findall('Tag'):
                    tag_info = self._parse_tag_element(tag_elem, 'program', name)
                    if tag_info:
                        tags.append(tag_info)
            
            return ProgramInfo(
                name=name,
                type=program_type,
                description=description,
                main_routine=main_routine,
                routines=routines,
                tags=tags,
                disabled=disabled
            )
            
        except Exception as e:
            self.logger.warning(f"Error parsing program element: {e}")
            return None
    
    def _parse_routine_element(self, routine_elem: ET.Element, program_name: str) -> Optional[RoutineInfo]:
        """Parse individual routine element"""
        try:
            name = routine_elem.get('Name')
            routine_type = routine_elem.get('Type', 'RLL')  # RLL = Relay Ladder Logic
            
            # Get description
            description = ""
            desc_elem = routine_elem.find('Description')
            if desc_elem is not None:
                description = desc_elem.text or ""
            
            # Count rungs and parse ladder logic if it's a ladder logic routine
            rungs_count = 0
            ladder_logic = None
            
            if routine_type == 'RLL':
                rll_content = routine_elem.find('RLLContent')
                if rll_content is not None:
                    rungs = rll_content.findall('Rung')
                    rungs_count = len(rungs)
                    
                    # Parse ladder logic if parser is available
                    if self.ladder_parser:
                        try:
                            ladder_logic = self.ladder_parser.parse_routine_ladder_logic(
                                routine_elem, name, program_name
                            )
                        except Exception as e:
                            self.logger.warning(f"Error parsing ladder logic for routine {name}: {e}")
            
            return RoutineInfo(
                name=name,
                type=routine_type,
                program_name=program_name,
                description=description,
                rungs_count=rungs_count,
                ladder_logic=ladder_logic
            )
            
        except Exception as e:
            self.logger.warning(f"Error parsing routine element: {e}")
            return None

    def validate_l5x_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate L5X file structure and format
        
        Args:
            file_path: Path to the L5X file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, f"File does not exist: {file_path}"
            
            if not file_path.lower().endswith('.l5x'):
                return False, "File does not have .l5x extension"
            
            # Try to parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Check for required L5X structure
            if root.tag != 'RSLogix5000Content':
                return False, "Not a valid L5X file - missing RSLogix5000Content root"
            
            # Check for Controller element
            controller = root.find('.//Controller')
            if controller is None:
                return False, "Not a valid L5X file - missing Controller element"
            
            return True, "Valid L5X file"
            
        except ET.ParseError as e:
            return False, f"XML parsing error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
