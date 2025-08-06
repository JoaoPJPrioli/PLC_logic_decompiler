"""
Basic L5X XML parser for Rockwell PLC files.

This module provides fundamental XML parsing capabilities for L5X files,
including validation, controller information extraction, and error handling.

Classes:
    L5XParser: Main parser class for L5X files
    L5XParseError: Custom exception for parsing errors
"""

import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

# Import tag models
from src.models.tag_models import Tag, TagType, create_tag_from_xml_element
from src.models.io_models import (
    IOModule, IOMapping, create_io_module_from_xml_element
)
from src.models.canonicalization import (
    TagCanonicalizer, TagScope, TagReference, TagConflict
)
from src.models.knowledge_graph import (
    PLCKnowledgeGraph, PLCGraphBuilder, NodeType, EdgeType
)
from src.parsers.ladder_logic_parser import LadderLogicExtractor


# Configure logging
logger = logging.getLogger(__name__)


class L5XParseError(Exception):
    """Custom exception for L5X parsing errors."""
    pass


@dataclass
class ControllerInfo:
    """Container for controller information."""
    name: str
    processor_type: str
    major_rev: str
    minor_rev: str
    project_creation_date: Optional[str] = None
    last_modified_date: Optional[str] = None
    target_name: Optional[str] = None


class L5XParser:
    """
    Basic L5X XML parser for Rockwell PLC files.
    
    This parser provides fundamental functionality to read and validate
    L5X files, extract basic controller information, and handle parsing errors.
    """

    def __init__(self):
        """Initialize the L5X parser."""
        self.root: Optional[ET.Element] = None
        self.file_path: Optional[Path] = None
        self.controller_info: Optional[ControllerInfo] = None
        self.controller_tags: Dict[str, Tag] = {}  # Dictionary of controller-scoped tags
        self.program_tags: Dict[str, Tag] = {}     # Dictionary of program-scoped tags
        self.programs: Dict[str, Dict[str, Any]] = {}  # Dictionary of program information
        self.io_mapping: Optional[IOMapping] = None    # I/O module and mapping information
        self.canonicalizer: Optional[TagCanonicalizer] = None  # Tag canonicalization system
        self.ladder_logic_extractor: Optional[LadderLogicExtractor] = None  # Ladder logic parser
        
    def load_file(self, filepath: str) -> bool:
        """
        Load and validate an L5X file.
        
        Args:
            filepath (str): Path to the L5X file
            
        Returns:
            bool: True if file loaded successfully, False otherwise
            
        Raises:
            L5XParseError: If file cannot be parsed or is invalid L5X format
        """
        try:
            # Convert to Path object for better handling
            self.file_path = Path(filepath)
            
            # Check if file exists
            if not self.file_path.exists():
                raise L5XParseError(f"File not found: {filepath}")
            
            # Check file extension
            if self.file_path.suffix.lower() != '.l5x':
                logger.warning(f"File does not have .l5x extension: {filepath}")
            
            # Parse the XML file
            logger.info(f"Loading L5X file: {filepath}")
            tree = ET.parse(self.file_path)
            self.root = tree.getroot()
            
            # Validate L5X structure
            if not self._validate_l5x_structure():
                raise L5XParseError("Invalid L5X file structure")
            
            # Extract controller information
            self.controller_info = self._extract_controller_info()
            
            # Extract controller tags
            self.controller_tags = self._extract_controller_tags()
            
            # Extract program information and tags
            self.programs = self._extract_programs()
            self.program_tags = self._extract_program_tags()
            
            logger.info(f"Successfully loaded L5X file: {filepath}")
            logger.info(f"Found {len(self.controller_tags)} controller tags")
            logger.info(f"Found {len(self.programs)} programs")
            logger.info(f"Found {len(self.program_tags)} program-scoped tags")
            return True
            
        except ET.ParseError as e:
            error_msg = f"XML parsing error: {e}"
            logger.error(error_msg)
            raise L5XParseError(error_msg)
        
        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            logger.error(error_msg)
            raise L5XParseError(error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error loading L5X file: {e}"
            logger.error(error_msg)
            raise L5XParseError(error_msg)
    
    def _validate_l5x_structure(self) -> bool:
        """
        Validate that the XML file has proper L5X structure.
        
        Returns:
            bool: True if structure is valid, False otherwise
        """
        if self.root is None:
            return False
        
        # Check root element
        if self.root.tag != 'RSLogix5000Content':
            logger.error(f"Invalid root element: {self.root.tag}, expected 'RSLogix5000Content'")
            return False
        
        # Check for required attributes
        required_attrs = ['SchemaRevision', 'SoftwareRevision', 'TargetName', 'TargetType']
        for attr in required_attrs:
            if attr not in self.root.attrib:
                logger.error(f"Missing required attribute in root element: {attr}")
                return False
        
        # Check for Controller element
        controller = self.root.find('Controller')
        if controller is None:
            logger.error("Missing Controller element")
            return False
        
        # Check controller required attributes
        controller_required_attrs = ['Name', 'ProcessorType']
        for attr in controller_required_attrs:
            if attr not in controller.attrib:
                logger.error(f"Missing required attribute in Controller element: {attr}")
                return False
        
        logger.debug("L5X structure validation passed")
        return True
    
    def _extract_controller_info(self) -> ControllerInfo:
        """
        Extract controller information from the L5X file.
        
        Returns:
            ControllerInfo: Container with controller details
            
        Raises:
            L5XParseError: If controller information cannot be extracted
        """
        if self.root is None:
            raise L5XParseError("No L5X file loaded")
        
        controller = self.root.find('Controller')
        if controller is None:
            raise L5XParseError("Controller element not found")
        
        try:
            # Extract basic controller information
            controller_info = ControllerInfo(
                name=controller.attrib['Name'],
                processor_type=controller.attrib['ProcessorType'],
                major_rev=controller.attrib.get('MajorRev', 'Unknown'),
                minor_rev=controller.attrib.get('MinorRev', 'Unknown'),
                project_creation_date=controller.attrib.get('ProjectCreationDate'),
                last_modified_date=controller.attrib.get('LastModifiedDate'),
                target_name=self.root.attrib.get('TargetName')
            )
            
            logger.info(f"Extracted controller info - Name: {controller_info.name}, "
                       f"Type: {controller_info.processor_type}")
            
            return controller_info
            
        except KeyError as e:
            error_msg = f"Missing required controller attribute: {e}"
            logger.error(error_msg)
            raise L5XParseError(error_msg)
    
    def _extract_controller_tags(self) -> Dict[str, Tag]:
        """
        Extract controller-scoped tags from the L5X file.
        
        Returns:
            Dict[str, Tag]: Dictionary of tags with canonicalized names as keys
            
        Raises:
            L5XParseError: If tags cannot be extracted
        """
        if self.root is None:
            raise L5XParseError("No L5X file loaded")
        
        controller = self.root.find('Controller')
        if controller is None:
            raise L5XParseError("Controller element not found")
        
        tags_dict = {}
        
        # Find the Tags section under Controller
        tags_section = controller.find('Tags')
        if tags_section is None:
            logger.warning("No Tags section found in Controller")
            return tags_dict
        
        try:
            tag_count = 0
            for tag_element in tags_section.findall('Tag'):
                try:
                    # Create tag object from XML element
                    tag = create_tag_from_xml_element(
                        tag_element, 
                        tag_type=TagType.CONTROLLER
                    )
                    
                    # Use canonical name as key
                    canonical_name = tag.canonical_name
                    tags_dict[canonical_name] = tag
                    tag_count += 1
                    
                    logger.debug(f"Extracted controller tag: {canonical_name} ({tag.data_type})")
                    
                except Exception as e:
                    tag_name = tag_element.attrib.get('Name', 'Unknown')
                    logger.warning(f"Failed to parse controller tag '{tag_name}': {e}")
                    continue
            
            logger.info(f"Successfully extracted {tag_count} controller tags")
            return tags_dict
            
        except Exception as e:
            error_msg = f"Error extracting controller tags: {e}"
            logger.error(error_msg)
            raise L5XParseError(error_msg)
    
    def extract_controller_tags(self) -> Dict[str, Tag]:
        """
        Public method to extract controller tags.
        
        Returns:
            Dict[str, Tag]: Dictionary of controller tags
        """
        if not self.is_loaded():
            raise L5XParseError("No L5X file loaded")
        
        return self.controller_tags.copy()
    
    def get_tag_by_name(self, tag_name: str) -> Optional[Tag]:
        """
        Get a specific tag by name.
        
        Args:
            tag_name: Name of the tag to retrieve
            
        Returns:
            Tag: Tag object if found, None otherwise
        """
        return self.controller_tags.get(tag_name)
    
    def get_tags_by_type(self, data_type: str) -> List[Tag]:
        """
        Get all tags of a specific data type.
        
        Args:
            data_type: Data type to filter by (e.g., 'BOOL', 'INT', 'DINT')
            
        Returns:
            List[Tag]: List of tags matching the data type
        """
        return [tag for tag in self.controller_tags.values() 
                if tag.data_type.upper() == data_type.upper()]
    
    def get_array_tags(self) -> List[Tag]:
        """
        Get all array tags.
        
        Returns:
            List[Tag]: List of tags that are arrays
        """
        return [tag for tag in self.controller_tags.values() if tag.is_array]
    
    def get_tags_with_descriptions(self) -> List[Tag]:
        """
        Get all tags that have descriptions.
        
        Returns:
            List[Tag]: List of tags with descriptions
        """
        return [tag for tag in self.controller_tags.values() 
                if tag.description and tag.description.strip()]
    
    def get_tag_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the extracted tags.
        
        Returns:
            Dict[str, Any]: Dictionary containing tag statistics
        """
        if not self.controller_tags:
            return {}
        
        # Count by data type
        type_counts = {}
        array_count = 0
        with_descriptions = 0
        total_array_elements = 0
        
        for tag in self.controller_tags.values():
            # Count by type
            data_type = tag.data_type
            type_counts[data_type] = type_counts.get(data_type, 0) + 1
            
            # Count arrays
            if tag.is_array:
                array_count += 1
                total_array_elements += tag.total_elements
            
            # Count descriptions
            if tag.description and tag.description.strip():
                with_descriptions += 1
        
        return {
            'total_tags': len(self.controller_tags),
            'array_tags': array_count,
            'tags_with_descriptions': with_descriptions,
            'total_array_elements': total_array_elements,
            'types': type_counts,
            'most_common_type': max(type_counts.keys(), key=type_counts.get) if type_counts else None
        }
    
    def _extract_programs(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract program information from the L5X file.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of program information
            
        Raises:
            L5XParseError: If programs cannot be extracted
        """
        if self.root is None:
            raise L5XParseError("No L5X file loaded")
        
        controller = self.root.find('Controller')
        if controller is None:
            raise L5XParseError("Controller element not found")
        
        programs_dict = {}
        
        # Find the Programs section under Controller
        programs_section = controller.find('Programs')
        if programs_section is None:
            logger.warning("No Programs section found in Controller")
            return programs_dict
        
        try:
            for program_element in programs_section.findall('Program'):
                try:
                    program_name = program_element.attrib.get('Name', '')
                    if not program_name:
                        logger.warning("Program element missing Name attribute")
                        continue
                    
                    program_info = {
                        'name': program_name,
                        'type': program_element.attrib.get('Type', 'Sequential'),
                        'use': program_element.attrib.get('Use', 'Context'),
                        'test_edits': program_element.attrib.get('TestEdits', 'false').lower() == 'true',
                        'disabled': program_element.attrib.get('Disabled', 'false').lower() == 'true',
                        'routines': [],
                        'tags_count': 0,
                        'main_routine': program_element.attrib.get('MainRoutineName', '')
                    }
                    
                    # Count routines
                    routines_section = program_element.find('Routines')
                    if routines_section is not None:
                        routines = routines_section.findall('Routine')
                        program_info['routines'] = [r.attrib.get('Name', '') for r in routines]
                    
                    # Count local tags
                    tags_section = program_element.find('Tags')
                    if tags_section is not None:
                        tags = tags_section.findall('Tag')
                        program_info['tags_count'] = len(tags)
                    
                    programs_dict[program_name] = program_info
                    logger.debug(f"Extracted program: {program_name} with {program_info['tags_count']} tags")
                    
                except Exception as e:
                    program_name = program_element.attrib.get('Name', 'Unknown')
                    logger.warning(f"Failed to parse program '{program_name}': {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(programs_dict)} programs")
            return programs_dict
            
        except Exception as e:
            error_msg = f"Error extracting programs: {e}"
            logger.error(error_msg)
            raise L5XParseError(error_msg)
    
    def _extract_program_tags(self) -> Dict[str, Tag]:
        """
        Extract program-scoped tags from all programs in the L5X file.
        
        Returns:
            Dict[str, Tag]: Dictionary of program tags with canonicalized names as keys
            
        Raises:
            L5XParseError: If program tags cannot be extracted
        """
        if self.root is None:
            raise L5XParseError("No L5X file loaded")
        
        controller = self.root.find('Controller')
        if controller is None:
            raise L5XParseError("Controller element not found")
        
        program_tags_dict = {}
        
        # Find the Programs section under Controller
        programs_section = controller.find('Programs')
        if programs_section is None:
            logger.warning("No Programs section found in Controller")
            return program_tags_dict
        
        try:
            total_program_tags = 0
            
            for program_element in programs_section.findall('Program'):
                try:
                    program_name = program_element.attrib.get('Name', '')
                    if not program_name:
                        continue
                    
                    # Find Tags section within this program
                    tags_section = program_element.find('Tags')
                    if tags_section is None:
                        logger.debug(f"No Tags section found in program '{program_name}'")
                        continue
                    
                    program_tag_count = 0
                    for tag_element in tags_section.findall('Tag'):
                        try:
                            # Create tag object with program scope
                            tag = create_tag_from_xml_element(
                                tag_element,
                                tag_type=TagType.PROGRAM,
                                scope=program_name
                            )
                            
                            # Use canonical name as key (includes program scope)
                            canonical_name = tag.canonical_name
                            program_tags_dict[canonical_name] = tag
                            program_tag_count += 1
                            total_program_tags += 1
                            
                            logger.debug(f"Extracted program tag: {canonical_name} ({tag.data_type})")
                            
                        except Exception as e:
                            tag_name = tag_element.attrib.get('Name', 'Unknown')
                            logger.warning(f"Failed to parse program tag '{program_name}.{tag_name}': {e}")
                            continue
                    
                    logger.debug(f"Program '{program_name}': extracted {program_tag_count} tags")
                    
                except Exception as e:
                    program_name = program_element.attrib.get('Name', 'Unknown')
                    logger.warning(f"Failed to process program '{program_name}': {e}")
                    continue
            
            logger.info(f"Successfully extracted {total_program_tags} program-scoped tags from {len(self.programs)} programs")
            return program_tags_dict
            
        except Exception as e:
            error_msg = f"Error extracting program tags: {e}"
            logger.error(error_msg)
            raise L5XParseError(error_msg)
    
    def extract_program_tags(self) -> Dict[str, Tag]:
        """
        Public method to extract program-scoped tags.
        
        Returns:
            Dict[str, Tag]: Dictionary of program-scoped tags
        """
        if not self.is_loaded():
            raise L5XParseError("No L5X file loaded")
        
        return self.program_tags.copy()
    
    def get_all_tags(self) -> Dict[str, Tag]:
        """
        Get all tags (controller and program-scoped) combined.
        
        Returns:
            Dict[str, Tag]: Combined dictionary of all tags
        """
        all_tags = {}
        all_tags.update(self.controller_tags)
        all_tags.update(self.program_tags)
        return all_tags
    
    def get_programs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all programs.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of program information
        """
        return self.programs.copy()
    
    def get_program_info(self, program_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific program.
        
        Args:
            program_name: Name of the program
            
        Returns:
            Dict[str, Any]: Program information if found, None otherwise
        """
        return self.programs.get(program_name)
    
    def get_tags_for_program(self, program_name: str) -> List[Tag]:
        """
        Get all tags for a specific program.
        
        Args:
            program_name: Name of the program
            
        Returns:
            List[Tag]: List of tags in the specified program
        """
        return [tag for tag in self.program_tags.values() if tag.scope == program_name]
    
    def get_tag_by_canonical_name(self, canonical_name: str) -> Optional[Tag]:
        """
        Get a tag by its canonical name (searches both controller and program tags).
        
        Args:
            canonical_name: Canonical name of the tag
            
        Returns:
            Tag: Tag object if found, None otherwise
        """
        # Check controller tags first
        if canonical_name in self.controller_tags:
            return self.controller_tags[canonical_name]
        
        # Check program tags
        if canonical_name in self.program_tags:
            return self.program_tags[canonical_name]
        
        return None
    
    def get_combined_tag_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all tags (controller and program-scoped).
        
        Returns:
            Dict[str, Any]: Dictionary containing combined tag statistics
        """
        all_tags = self.get_all_tags()
        
        if not all_tags:
            return {}
        
        # Count by data type and scope
        type_counts = {}
        scope_counts = {'controller': 0, 'program': 0}
        array_count = 0
        with_descriptions = 0
        total_array_elements = 0
        
        for tag in all_tags.values():
            # Count by type
            data_type = tag.data_type
            type_counts[data_type] = type_counts.get(data_type, 0) + 1
            
            # Count by scope
            if tag.tag_type == TagType.CONTROLLER:
                scope_counts['controller'] += 1
            elif tag.tag_type == TagType.PROGRAM:
                scope_counts['program'] += 1
            
            # Count arrays
            if tag.is_array:
                array_count += 1
                total_array_elements += tag.total_elements
            
            # Count descriptions
            if tag.description and tag.description.strip():
                with_descriptions += 1
        
        return {
            'total_tags': len(all_tags),
            'controller_tags': scope_counts['controller'],
            'program_tags': scope_counts['program'],
            'programs_count': len(self.programs),
            'array_tags': array_count,
            'tags_with_descriptions': with_descriptions,
            'total_array_elements': total_array_elements,
            'types': type_counts,
            'most_common_type': max(type_counts.keys(), key=type_counts.get) if type_counts else None
        }
    
    def get_controller_info(self) -> Optional[ControllerInfo]:
        """
        Get controller information from the loaded L5X file.
        
        Returns:
            ControllerInfo: Controller information if file is loaded, None otherwise
        """
        return self.controller_info
    
    def is_loaded(self) -> bool:
        """
        Check if an L5X file is currently loaded.
        
        Returns:
            bool: True if file is loaded, False otherwise
        """
        return self.root is not None
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get basic file information.
        
        Returns:
            Dict[str, Any]: Dictionary containing file information
        """
        if not self.is_loaded():
            return {}
        
        info = {
            'file_path': str(self.file_path) if self.file_path else None,
            'schema_revision': self.root.attrib.get('SchemaRevision'),
            'software_revision': self.root.attrib.get('SoftwareRevision'),
            'export_date': self.root.attrib.get('ExportDate'),
            'target_name': self.root.attrib.get('TargetName'),
            'target_type': self.root.attrib.get('TargetType')
        }
        
        if self.controller_info:
            info['controller'] = {
                'name': self.controller_info.name,
                'processor_type': self.controller_info.processor_type,
                'major_rev': self.controller_info.major_rev,
                'minor_rev': self.controller_info.minor_rev,
                'project_creation_date': self.controller_info.project_creation_date,
                'last_modified_date': self.controller_info.last_modified_date
            }
        
        # Add tag statistics
        if self.controller_tags or self.program_tags:
            info['tag_statistics'] = self.get_combined_tag_statistics()
        
        # Add program information
        if self.programs:
            info['programs'] = {
                'count': len(self.programs),
                'programs': self.programs
            }
        
        return info
    
    # ===== I/O EXTRACTION METHODS (Step 5) =====
    
    def extract_io_mapping(self) -> IOMapping:
        """
        Extract I/O modules and mappings from the L5X file.
        
        Returns:
            IOMapping: Complete I/O mapping information
            
        Raises:
            L5XParseError: If L5X file is not loaded or has errors
        """
        if self.root is None:
            raise L5XParseError("No L5X file loaded")
        
        if self.io_mapping is not None:
            return self.io_mapping
        
        try:
            logging.info("Extracting I/O mapping from L5X file")
            
            # Create I/O mapping
            io_mapping = IOMapping(modules={})
            
            # Find the Controller element and its Modules section
            controller = self.root.find('.//Controller')
            if controller is None:
                raise L5XParseError("No Controller element found")
            
            modules_element = controller.find('Modules')
            if modules_element is None:
                logging.warning("No Modules section found in controller")
                self.io_mapping = io_mapping
                return io_mapping
            
            # Extract each module
            for module_element in modules_element.findall('Module'):
                try:
                    io_module = create_io_module_from_xml_element(module_element)
                    io_mapping.add_module(io_module)
                    logging.debug(f"Extracted I/O module: {io_module.name} ({io_module.catalog_number})")
                    
                except Exception as e:
                    module_name = module_element.get('Name', 'Unknown')
                    logging.warning(f"Failed to parse I/O module '{module_name}': {e}")
                    continue
            
            self.io_mapping = io_mapping
            logging.info(f"Successfully extracted {len(io_mapping.modules)} I/O modules")
            
            return io_mapping
            
        except Exception as e:
            raise L5XParseError(f"Failed to extract I/O mapping: {e}")
    
    def get_io_modules(self) -> Dict[str, IOModule]:
        """
        Get all I/O modules.
        
        Returns:
            Dict[str, IOModule]: Dictionary of module name -> IOModule
        """
        mapping = self.extract_io_mapping()
        return mapping.modules
    
    def get_io_module_by_name(self, module_name: str) -> Optional[IOModule]:
        """
        Get a specific I/O module by name.
        
        Args:
            module_name (str): Name of the module to retrieve
            
        Returns:
            Optional[IOModule]: The module if found, None otherwise
        """
        modules = self.get_io_modules()
        return modules.get(module_name)
    
    def get_io_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive I/O mapping statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing I/O statistics
        """
        mapping = self.extract_io_mapping()
        return mapping.get_statistics()
    
    def search_io_comments(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search I/O point comments for a specific term.
        
        Args:
            search_term (str): Term to search for in I/O comments
            
        Returns:
            List[Dict[str, Any]]: List of matching I/O points with context
        """
        mapping = self.extract_io_mapping()
        matching_points = mapping.search_io_points_by_comment(search_term)
        
        # Add context information
        results = []
        for point in matching_points:
            # Find the module and connection for this point
            module_name = None
            connection_name = None
            
            for mod_name, module in mapping.modules.items():
                for conn_name, connection in module.connections.items():
                    if point.operand in connection.io_points:
                        module_name = mod_name
                        connection_name = conn_name
                        break
                if module_name:
                    break
            
            results.append({
                'operand': point.operand,
                'comment': point.comment,
                'bit_number': point.bit_number,
                'data_type': point.data_type,
                'module_name': module_name,
                'connection_name': connection_name
            })
        
        return results
    
    def get_discrete_io_points(self) -> List[Dict[str, Any]]:
        """
        Get all discrete I/O points (bit-level).
        
        Returns:
            List[Dict[str, Any]]: List of discrete I/O points with details
        """
        mapping = self.extract_io_mapping()
        discrete_points = []
        
        for module in mapping.modules.values():
            for connection in module.connections.values():
                for point in connection.io_points.values():
                    if point.bit_number is not None:  # Discrete point
                        discrete_points.append({
                            'module_name': module.name,
                            'module_type': module.module_type.value,
                            'connection_name': connection.name,
                            'connection_type': connection.connection_type.value,
                            'operand': point.operand,
                            'bit_number': point.bit_number,
                            'comment': point.comment,
                            'data_type': point.data_type
                        })
        
        # Sort by module name, then bit number
        discrete_points.sort(key=lambda x: (x['module_name'], x['bit_number'] or 0))
        return discrete_points

    # ===== TAG CANONICALIZATION METHODS (Step 6) =====
    
    def create_canonicalization_system(self) -> TagCanonicalizer:
        """
        Create and populate the tag canonicalization system.
        
        Returns:
            TagCanonicalizer: Configured canonicalization system
            
        Raises:
            L5XParseError: If L5X file is not loaded
        """
        if self.root is None:
            raise L5XParseError("No L5X file loaded")
        
        if self.canonicalizer is not None:
            return self.canonicalizer
        
        try:
            logging.info("Creating tag canonicalization system")
            
            canonicalizer = TagCanonicalizer()
            
            # Add controller tags
            controller_tags = self.extract_controller_tags()
            if controller_tags:
                canonicalizer.add_controller_tags(controller_tags)
                logging.debug(f"Added {len(controller_tags)} controller tags to canonicalizer")
            
            # Add program tags
            program_tags = self.extract_program_tags()
            if program_tags:
                canonicalizer.add_program_tags(program_tags)
                logging.debug(f"Added {len(program_tags)} program tags to canonicalizer")
            
            # Add I/O module tags
            io_mapping = self.extract_io_mapping()
            if io_mapping.modules:
                canonicalizer.add_io_modules(io_mapping.modules)
                total_io_points = sum(len(module.connections) for module in io_mapping.modules.values())
                logging.debug(f"Added I/O tags from {len(io_mapping.modules)} modules ({total_io_points} connections)")
            
            # Validate all tags and find conflicts
            conflicts = canonicalizer.validate_tags()
            if conflicts:
                logging.warning(f"Found {len(conflicts)} tag conflicts during validation")
                for conflict in conflicts[:5]:  # Log first 5 conflicts
                    logging.warning(f"  {conflict.conflict_type.value}: {conflict.description}")
            else:
                logging.info("No tag conflicts found during validation")
            
            self.canonicalizer = canonicalizer
            logging.info(f"Canonicalization system created with {len(canonicalizer.tag_references)} tag references")
            
            return canonicalizer
            
        except Exception as e:
            raise L5XParseError(f"Failed to create canonicalization system: {e}")
    
    def get_canonicalization_system(self) -> TagCanonicalizer:
        """Get the tag canonicalization system"""
        return self.create_canonicalization_system()
    
    def create_knowledge_graph(self, graph_name: Optional[str] = None) -> PLCKnowledgeGraph:
        """
        Create a comprehensive knowledge graph from the parsed L5X data.
        
        Args:
            graph_name: Name for the knowledge graph
            
        Returns:
            PLCKnowledgeGraph: Populated knowledge graph
            
        Raises:
            L5XParseError: If L5X file is not loaded
        """
        if self.root is None:
            raise L5XParseError("No L5X file loaded")
        
        try:
            logging.info("Creating PLC knowledge graph")
            
            # Determine graph name
            if graph_name is None:
                controller_info = self.get_controller_info()
                graph_name = f"PLC_{controller_info.name}" if controller_info else "PLC_KnowledgeGraph"
            
            # Create graph and builder
            graph = PLCKnowledgeGraph(graph_name)
            builder = PLCGraphBuilder(graph)
            
            # Add controller node
            controller_info = self.get_controller_info()
            if controller_info:
                controller_properties = {
                    'processor_type': controller_info.processor_type,
                    'major_rev': controller_info.major_rev,
                    'minor_rev': controller_info.minor_rev,
                    'project_creation_date': controller_info.project_creation_date,
                    'last_modified_date': controller_info.last_modified_date,
                    'target_name': controller_info.target_name
                }
                builder.add_controller_node(controller_info.name, controller_properties)
                logging.debug(f"Added controller node: {controller_info.name}")
            
            # Get canonicalization system for consistent naming
            canonicalizer = self.create_canonicalization_system()
            
            # Add tag nodes from canonicalization system
            tag_node_map = builder.add_tag_nodes_from_references(list(canonicalizer.tag_references.values()))
            logging.debug(f"Added {len(tag_node_map)} tag nodes")
            
            # Add program nodes
            programs = self.get_programs_info()
            program_node_map = builder.add_program_nodes(programs)
            logging.debug(f"Added {len(program_node_map)} program nodes")
            
            # Add I/O module nodes
            io_mapping = self.extract_io_mapping()
            if io_mapping.modules:
                module_node_map = builder.add_io_module_nodes(io_mapping.modules)
                logging.debug(f"Added {len(module_node_map)} I/O module nodes")
            
            # Create relationships
            builder.create_controller_relationships()
            builder.create_program_relationships()
            logging.debug("Created basic relationships between nodes")
            
            # Add routine nodes and relationships (if we have routine data)
            routine_info = self.get_routines_info()
            if routine_info:
                self._add_routine_nodes_to_graph(builder, routine_info, program_node_map)
            
            # Get final statistics
            stats = graph.get_statistics()
            logging.info(f"Knowledge graph created: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
            logging.info(f"Node types: {', '.join(stats['nodes_by_type'].keys())}")
            
            return graph
            
        except Exception as e:
            raise L5XParseError(f"Failed to create knowledge graph: {e}")
    
    def _add_routine_nodes_to_graph(self, 
                                   builder: PLCGraphBuilder, 
                                   routine_info: Dict[str, Any],
                                   program_node_map: Dict[str, str]):
        """Add routine nodes and their relationships to the graph"""
        try:
            for program_name, program_data in routine_info.items():
                if program_name not in program_node_map:
                    continue
                
                program_node_id = program_node_map[program_name]
                routines = program_data.get('routines', [])
                
                for routine_name in routines:
                    # Add routine node
                    routine_properties = {
                        'program': program_name,
                        'type': 'Routine'
                    }
                    
                    routine_node_id = builder.graph.add_node(
                        NodeType.ROUTINE,
                        routine_name,
                        properties=routine_properties
                    )
                    
                    # Create relationship: Program contains Routine
                    builder.graph.add_edge(
                        program_node_id,
                        routine_node_id,
                        EdgeType.CONTAINS,
                        properties={'relationship': 'program_contains_routine'}
                    )
                    
                    # Store for future reference
                    builder.routine_nodes[f"{program_name}.{routine_name}"] = routine_node_id
            
            logging.debug("Added routine nodes and relationships")
            
        except Exception as e:
            logging.warning(f"Failed to add routine nodes: {e}")
    
    def get_programs_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive program information for knowledge graph.
        
        Returns:
            Dictionary with program information
        """
        if self.root is None:
            return {}
        
        programs_info = {}
        
        try:
            programs_element = self.root.find('.//Programs')
            if programs_element is None:
                return programs_info
            
            for program_element in programs_element.findall('Program'):
                program_name = program_element.get('Name', 'Unknown')
                program_type = program_element.get('Type', 'Normal')
                disabled = program_element.get('Disabled', 'false').lower() == 'true'
                
                # Get main routine
                main_routine = program_element.get('MainRoutineName', '')
                
                # Get all routines
                routines = []
                routines_element = program_element.find('Routines')
                if routines_element is not None:
                    for routine_element in routines_element.findall('Routine'):
                        routine_name = routine_element.get('Name', '')
                        if routine_name:
                            routines.append(routine_name)
                
                # Count tags
                tags_count = 0
                tags_element = program_element.find('Tags')
                if tags_element is not None:
                    tags_count = len(tags_element.findall('Tag'))
                
                programs_info[program_name] = {
                    'type': program_type,
                    'disabled': disabled,
                    'main_routine': main_routine,
                    'routines': routines,
                    'tags_count': tags_count
                }
            
            return programs_info
            
        except Exception as e:
            logging.warning(f"Failed to extract programs info: {e}")
            return programs_info
    
    def get_routines_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get routine information for knowledge graph.
        
        Returns:
            Dictionary with routine information by program
        """
        if self.root is None:
            return {}
        
        routines_info = {}
        
        try:
            programs_element = self.root.find('.//Programs')
            if programs_element is None:
                return routines_info
            
            for program_element in programs_element.findall('Program'):
                program_name = program_element.get('Name', 'Unknown')
                routines_info[program_name] = {'routines': []}
                
                routines_element = program_element.find('Routines')
                if routines_element is not None:
                    for routine_element in routines_element.findall('Routine'):
                        routine_name = routine_element.get('Name', '')
                        if routine_name:
                            routines_info[program_name]['routines'].append(routine_name)
            
            return routines_info
            
        except Exception as e:
            logging.warning(f"Failed to extract routines info: {e}")
            return routines_info
    
    def create_knowledge_graph_with_visualization(self, 
                                                 output_dir: str,
                                                 graph_name: Optional[str] = None) -> PLCKnowledgeGraph:
        """
        Create knowledge graph and generate visualization files.
        
        Args:
            output_dir: Directory to save visualization files
            graph_name: Name for the knowledge graph
            
        Returns:
            PLCKnowledgeGraph: Populated knowledge graph with visualizations
        """
        from src.models.graph_visualization import GraphVisualizer
        
        # Create the knowledge graph
        graph = self.create_knowledge_graph(graph_name)
        
        # Create visualizations
        visualizer = GraphVisualizer(graph)
        visualizer.create_summary_report(output_dir)
        
        # Export graph data
        graph.export_to_json(f"{output_dir}/knowledge_graph.json")
        graph.export_to_gexf(f"{output_dir}/knowledge_graph.gexf")
        
        logging.info(f"Knowledge graph and visualizations saved to: {output_dir}")
        return graph
    
    def get_canonical_name(self, 
                          original_name: str, 
                          scope: TagScope = TagScope.CONTROLLER,
                          scope_qualifier: Optional[str] = None) -> Optional[str]:
        """
        Get canonical name for a tag.
        
        Args:
            original_name: Original tag name
            scope: Tag scope
            scope_qualifier: Program name, module name, etc.
            
        Returns:
            Canonical name if found, None otherwise
        """
        canonicalizer = self.get_canonicalization_system()
        
        # Try to find the tag reference first
        tag_ref = canonicalizer.get_tag_reference(original_name, scope, scope_qualifier)
        if tag_ref:
            return tag_ref.canonical_name
        
        # Fallback to direct canonicalization
        return canonicalizer.get_canonical_name(original_name)
    
    def search_canonical_tags(self, 
                             pattern: str,
                             scope: Optional[TagScope] = None,
                             include_description: bool = True) -> List[TagReference]:
        """
        Search for tags using canonicalization system.
        
        Args:
            pattern: Search pattern (supports wildcards)
            scope: Limit to specific scope
            include_description: Include description in search
            
        Returns:
            List of matching tag references
        """
        canonicalizer = self.get_canonicalization_system()
        return canonicalizer.search_tags(pattern, scope, include_description)
    
    def get_tag_conflicts(self) -> List[TagConflict]:
        """
        Get all tag conflicts found during canonicalization.
        
        Returns:
            List of tag conflicts
        """
        canonicalizer = self.get_canonicalization_system()
        return canonicalizer.conflicts
    
    def get_canonicalization_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive canonicalization statistics.
        
        Returns:
            Dictionary containing canonicalization statistics
        """
        canonicalizer = self.get_canonicalization_system()
        return canonicalizer.get_statistics()
    
    def generate_tag_cross_reference(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate comprehensive cross-reference table for all tags.
        
        Returns:
            Cross-reference dictionary mapping canonical names to tag info
        """
        canonicalizer = self.get_canonicalization_system()
        return canonicalizer.generate_cross_reference_table()
    
    def validate_tag_naming(self) -> Dict[str, Any]:
        """
        Validate tag naming conventions and return detailed report.
        
        Returns:
            Validation report with conflicts and recommendations
        """
        canonicalizer = self.get_canonicalization_system()
        conflicts = canonicalizer.validate_tags()
        
        # Categorize conflicts by severity
        errors = [c for c in conflicts if c.severity == "Error"]
        warnings = [c for c in conflicts if c.severity == "Warning"]
        info = [c for c in conflicts if c.severity == "Info"]
        
        # Generate recommendations
        recommendations = []
        if errors:
            recommendations.append("Fix tag naming errors before proceeding with code generation")
        if warnings:
            recommendations.append("Review tag naming warnings to avoid potential issues")
        if len(conflicts) == 0:
            recommendations.append("All tag names follow good naming conventions")
        
        return {
            'total_conflicts': len(conflicts),
            'errors': len(errors),
            'warnings': len(warnings),
            'info': len(info),
            'conflicts': [
                {
                    'type': c.conflict_type.value,
                    'severity': c.severity,
                    'description': c.description,
                    'tag1': c.tag1.original_name,
                    'tag1_scope': c.tag1.scope.value,
                    'tag2': c.tag2.original_name if c.tag2 else None,
                    'resolution': c.get_resolution_suggestion()
                } for c in conflicts
            ],
            'recommendations': recommendations,
            'statistics': canonicalizer.get_statistics()
        }
    
    def validate_file_only(self, filepath: str) -> Dict[str, Any]:
        """
        Validate an L5X file without loading it into the parser.
        
        Args:
            filepath (str): Path to the L5X file
            
        Returns:
            Dict[str, Any]: Validation results with status and details
        """
        result = {
            'valid': False,
            'file_exists': False,
            'is_xml': False,
            'is_l5x': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            file_path = Path(filepath)
            
            # Check file existence
            if not file_path.exists():
                result['errors'].append(f"File not found: {filepath}")
                return result
            result['file_exists'] = True
            
            # Check file extension
            if file_path.suffix.lower() != '.l5x':
                result['warnings'].append("File does not have .l5x extension")
            
            # Try to parse XML
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                result['is_xml'] = True
            except ET.ParseError as e:
                result['errors'].append(f"XML parsing error: {e}")
                return result
            
            # Check L5X structure
            if root.tag != 'RSLogix5000Content':
                result['errors'].append(f"Invalid root element: {root.tag}")
                return result
            
            # Check for Controller element
            controller = root.find('Controller')
            if controller is None:
                result['errors'].append("Missing Controller element")
                return result
            
            result['is_l5x'] = True
            result['valid'] = True
            
        except Exception as e:
            result['errors'].append(f"Validation error: {e}")
        
        return result

    # ===== LADDER LOGIC EXTRACTION METHODS (STEP 9) =====
    
    def extract_ladder_logic(self) -> Dict[str, Any]:
        """
        Extract and parse all ladder logic from the L5X file.
        
        Returns:
            Dictionary containing ladder logic routines and analysis
        """
        if not self.root:
            logger.error("No L5X file loaded")
            return {}
        
        try:
            # Initialize ladder logic extractor if not already done
            if not self.ladder_logic_extractor:
                self.ladder_logic_extractor = LadderLogicExtractor()
            
            # Extract ladder logic routines
            logger.info("Extracting ladder logic from L5X file")
            routines = self.ladder_logic_extractor.extract_ladder_logic_from_xml(self.root)
            
            # Get comprehensive analysis
            summary = self.ladder_logic_extractor.export_ladder_logic_summary()
            
            logger.info(f"Successfully extracted ladder logic: {len(routines)} routines, "
                       f"{summary['statistics']['total_rungs']} rungs, "
                       f"{summary['statistics']['total_instructions']} instructions")
            
            return {
                'routines': routines,
                'summary': summary,
                'extraction_successful': True
            }
            
        except Exception as e:
            logger.error(f"Error extracting ladder logic: {e}")
            return {
                'routines': {},
                'summary': {},
                'extraction_successful': False,
                'error': str(e)
            }
    
    def get_ladder_logic_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about extracted ladder logic.
        
        Returns:
            Dictionary with ladder logic statistics
        """
        if not self.ladder_logic_extractor:
            return {}
        
        return self.ladder_logic_extractor.get_routine_statistics()
    
    def search_ladder_instructions(self, instruction_type: str = None, 
                                 tag_reference: str = None) -> List[Dict[str, Any]]:
        """
        Search for specific instructions in ladder logic.
        
        Args:
            instruction_type: Filter by instruction type (e.g., 'XIC', 'OTE')
            tag_reference: Filter by tag reference
            
        Returns:
            List of matching instructions with context
        """
        if not self.ladder_logic_extractor:
            return []
        
        from src.models.ladder_logic import InstructionType
        
        # Convert string to InstructionType enum if provided
        inst_type = None
        if instruction_type:
            try:
                inst_type = InstructionType(instruction_type.upper())
            except ValueError:
                logger.warning(f"Unknown instruction type: {instruction_type}")
        
        results = self.ladder_logic_extractor.search_instructions(
            instruction_type=inst_type,
            tag_reference=tag_reference
        )
        
        # Format results for easier consumption
        formatted_results = []
        for routine_name, rung_number, instruction in results:
            formatted_results.append({
                'routine_name': routine_name,
                'rung_number': rung_number,
                'instruction_type': instruction.instruction_type.value,
                'parameters': [p.value for p in instruction.parameters],
                'raw_text': instruction.raw_text,
                'tag_references': instruction.get_tag_references()
            })
        
        return formatted_results
    
    def get_tag_ladder_usage(self) -> Dict[str, Any]:
        """
        Get analysis of how tags are used in ladder logic.
        
        Returns:
            Dictionary mapping tag names to usage information
        """
        if not self.ladder_logic_extractor:
            return {}
        
        return self.ladder_logic_extractor.get_tag_usage_analysis()
    
    def get_ladder_logic_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of ladder logic content.
        
        Returns:
            Complete ladder logic summary
        """
        if not self.ladder_logic_extractor:
            # Try to extract if not done yet
            self.extract_ladder_logic()
        
        if self.ladder_logic_extractor:
            return self.ladder_logic_extractor.export_ladder_logic_summary()
        
        return {}


# Utility functions for external use
def validate_l5x_file(filepath: str) -> bool:
    """
    Quick validation of an L5X file.
    
    Args:
        filepath (str): Path to the L5X file
        
    Returns:
        bool: True if file is valid L5X, False otherwise
    """
    parser = L5XParser()
    try:
        return parser.load_file(filepath)
    except L5XParseError:
        return False


def get_controller_info_quick(filepath: str) -> Optional[ControllerInfo]:
    """
    Quick extraction of controller information.
    
    Args:
        filepath (str): Path to the L5X file
        
    Returns:
        ControllerInfo: Controller information if successful, None otherwise
    """
    parser = L5XParser()
    try:
        if parser.load_file(filepath):
            return parser.get_controller_info()
    except L5XParseError:
        pass
    
    return None
