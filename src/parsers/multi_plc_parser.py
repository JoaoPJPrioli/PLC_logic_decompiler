#!/usr/bin/env python3
"""
Step 36: Multi-PLC Brand Support
Extends the PLC Logic Decompiler to support multiple PLC manufacturers beyond Rockwell/Allen-Bradley.

This module provides parsers and models for:
- Siemens S7/TIA Portal (.awl, .scl, .graph files)
- Schneider Electric Unity/Control Expert (.xef files) 
- Mitsubishi GX Works (.gxw files)
- Omron CX-Programmer (.cxp files)
- Generic extensible parser framework
"""

import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import zipfile
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PLCBrand(Enum):
    """Supported PLC brands"""
    ROCKWELL_AB = "rockwell_ab"
    SIEMENS = "siemens"
    SCHNEIDER = "schneider"
    MITSUBISHI = "mitsubishi"
    OMRON = "omron"
    GENERIC = "generic"

class FileFormat(Enum):
    """Supported file formats"""
    L5X = "l5x"  # Rockwell/Allen-Bradley
    AWL = "awl"  # Siemens AWL (Statement List)
    SCL = "scl"  # Siemens SCL (Structured Control Language)
    GRAPH = "graph"  # Siemens Graph
    XEF = "xef"  # Schneider Electric Unity
    GXW = "gxw"  # Mitsubishi GX Works
    CXP = "cxp"  # Omron CX-Programmer
    XML = "xml"  # Generic XML
    JSON = "json"  # Generic JSON

class InstructionType(Enum):
    """Universal instruction types across PLC brands"""
    CONTACT_NO = "contact_no"  # Normally Open Contact
    CONTACT_NC = "contact_nc"  # Normally Closed Contact
    COIL = "coil"  # Output Coil
    SET_RESET = "set_reset"  # Set/Reset
    TIMER = "timer"  # Timer
    COUNTER = "counter"  # Counter
    MATH = "math"  # Mathematical operations
    COMPARISON = "comparison"  # Comparison operations
    MOVE = "move"  # Data movement
    JUMP = "jump"  # Jump/Call
    FUNCTION = "function"  # Function block
    ANALOG = "analog"  # Analog I/O
    COMMUNICATION = "communication"  # Communication
    STRING = "string"  # String operations
    ARRAY = "array"  # Array operations

@dataclass
class UniversalTag:
    """Universal tag representation across PLC brands"""
    name: str
    data_type: str
    scope: str = "global"
    address: Optional[str] = None
    comment: Optional[str] = None
    initial_value: Optional[Any] = None
    brand_specific: Dict[str, Any] = field(default_factory=dict)
    array_dimensions: List[int] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    is_memory: bool = True

@dataclass
class UniversalInstruction:
    """Universal instruction representation across PLC brands"""
    instruction_type: InstructionType
    mnemonic: str
    operands: List[str] = field(default_factory=list)
    comment: Optional[str] = None
    address: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    brand_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UniversalRung:
    """Universal rung representation across PLC brands"""
    number: int
    instructions: List[UniversalInstruction] = field(default_factory=list)
    comment: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

@dataclass
class UniversalRoutine:
    """Universal routine representation across PLC brands"""
    name: str
    routine_type: str
    rungs: List[UniversalRung] = field(default_factory=list)
    comment: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    local_tags: List[UniversalTag] = field(default_factory=list)

@dataclass
class UniversalProgram:
    """Universal program representation across PLC brands"""
    name: str
    routines: List[UniversalRoutine] = field(default_factory=list)
    tags: List[UniversalTag] = field(default_factory=list)
    comment: Optional[str] = None
    main_routine: Optional[str] = None

@dataclass
class PLCProject:
    """Universal PLC project representation"""
    name: str
    brand: PLCBrand
    file_format: FileFormat
    controller_type: str
    firmware_version: Optional[str] = None
    programs: List[UniversalProgram] = field(default_factory=list)
    global_tags: List[UniversalTag] = field(default_factory=list)
    io_modules: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None

class BasePLCParser(ABC):
    """Abstract base class for PLC parsers"""
    
    def __init__(self):
        self.brand = PLCBrand.GENERIC
        self.supported_formats = []
        self.project = None
        
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file"""
        pass
        
    @abstractmethod
    def parse(self, file_path: Path) -> PLCProject:
        """Parse the PLC file and return a PLCProject"""
        pass
        
    @abstractmethod
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate the file format and structure"""
        pass

class RockwellParser(BasePLCParser):
    """Parser for Rockwell/Allen-Bradley L5X files"""
    
    def __init__(self):
        super().__init__()
        self.brand = PLCBrand.ROCKWELL_AB
        self.supported_formats = [FileFormat.L5X]
        
    def can_parse(self, file_path: Path) -> bool:
        """Check if this is a valid L5X file"""
        return file_path.suffix.lower() == '.l5x'
        
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate L5X file structure"""
        errors = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            if root.tag != 'RSLogix5000Content':
                errors.append("Invalid root element - expected 'RSLogix5000Content'")
            
            controller = root.find('.//Controller')
            if controller is None:
                errors.append("No Controller element found")
                
        except ET.ParseError as e:
            errors.append(f"XML parsing error: {e}")
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return len(errors) == 0, errors
        
    def parse(self, file_path: Path) -> PLCProject:
        """Parse L5X file"""
        logger.info(f"Parsing Rockwell L5X file: {file_path}")
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract controller information
        controller = root.find('.//Controller')
        controller_name = controller.get('Name', 'Unknown')
        controller_type = controller.get('Type', 'Unknown')
        
        project = PLCProject(
            name=controller_name,
            brand=PLCBrand.ROCKWELL_AB,
            file_format=FileFormat.L5X,
            controller_type=controller_type,
            created_date=datetime.now()
        )
        
        # Parse tags
        project.global_tags = self._parse_tags(root)
        
        # Parse programs
        project.programs = self._parse_programs(root)
        
        return project
        
    def _parse_tags(self, root: ET.Element) -> List[UniversalTag]:
        """Parse tags from L5X"""
        tags = []
        
        for tag_elem in root.findall('.//Tag'):
            tag = UniversalTag(
                name=tag_elem.get('Name', ''),
                data_type=tag_elem.get('DataType', ''),
                comment=tag_elem.find('.//Comment')
            )
            
            if tag.comment is not None:
                tag.comment = tag.comment.text
                
            tags.append(tag)
            
        return tags
        
    def _parse_programs(self, root: ET.Element) -> List[UniversalProgram]:
        """Parse programs from L5X"""
        programs = []
        
        for prog_elem in root.findall('.//Program'):
            program = UniversalProgram(
                name=prog_elem.get('Name', ''),
                comment=prog_elem.find('.//Description')
            )
            
            if program.comment is not None:
                program.comment = program.comment.text
                
            # Parse routines
            program.routines = self._parse_routines(prog_elem)
            programs.append(program)
            
        return programs
        
    def _parse_routines(self, program_elem: ET.Element) -> List[UniversalRoutine]:
        """Parse routines from program"""
        routines = []
        
        for routine_elem in program_elem.findall('.//Routine'):
            routine = UniversalRoutine(
                name=routine_elem.get('Name', ''),
                routine_type=routine_elem.get('Type', 'RLL')
            )
            
            # Parse rungs
            routine.rungs = self._parse_rungs(routine_elem)
            routines.append(routine)
            
        return routines
        
    def _parse_rungs(self, routine_elem: ET.Element) -> List[UniversalRung]:
        """Parse rungs from routine"""
        rungs = []
        
        for i, rung_elem in enumerate(routine_elem.findall('.//Rung')):
            rung = UniversalRung(number=i)
            
            # Parse instructions
            rung.instructions = self._parse_instructions(rung_elem)
            rungs.append(rung)
            
        return rungs
        
    def _parse_instructions(self, rung_elem: ET.Element) -> List[UniversalInstruction]:
        """Parse instructions from rung"""
        instructions = []
        
        for instr_elem in rung_elem.findall('.//Instruction'):
            mnemonic = instr_elem.get('Name', '')
            
            # Map L5X instruction types to universal types
            instruction_type = self._map_instruction_type(mnemonic)
            
            instruction = UniversalInstruction(
                instruction_type=instruction_type,
                mnemonic=mnemonic
            )
            
            # Parse operands
            for operand in instr_elem.get('Operand', '').split(','):
                if operand.strip():
                    instruction.operands.append(operand.strip())
                    
            instructions.append(instruction)
            
        return instructions
        
    def _map_instruction_type(self, mnemonic: str) -> InstructionType:
        """Map L5X mnemonics to universal instruction types"""
        mapping = {
            'XIC': InstructionType.CONTACT_NO,
            'XIO': InstructionType.CONTACT_NC,
            'OTE': InstructionType.COIL,
            'OTL': InstructionType.SET_RESET,
            'OTU': InstructionType.SET_RESET,
            'TON': InstructionType.TIMER,
            'TOF': InstructionType.TIMER,
            'RTO': InstructionType.TIMER,
            'CTU': InstructionType.COUNTER,
            'CTD': InstructionType.COUNTER,
            'ADD': InstructionType.MATH,
            'SUB': InstructionType.MATH,
            'MUL': InstructionType.MATH,
            'DIV': InstructionType.MATH,
            'EQU': InstructionType.COMPARISON,
            'NEQ': InstructionType.COMPARISON,
            'LES': InstructionType.COMPARISON,
            'GRT': InstructionType.COMPARISON,
            'MOV': InstructionType.MOVE,
            'JSR': InstructionType.JUMP,
            'SBR': InstructionType.JUMP,
            'RET': InstructionType.JUMP
        }
        
        return mapping.get(mnemonic, InstructionType.FUNCTION)

class SiemensParser(BasePLCParser):
    """Parser for Siemens S7/TIA Portal files"""
    
    def __init__(self):
        super().__init__()
        self.brand = PLCBrand.SIEMENS
        self.supported_formats = [FileFormat.AWL, FileFormat.SCL, FileFormat.GRAPH]
        
    def can_parse(self, file_path: Path) -> bool:
        """Check if this is a valid Siemens file"""
        return file_path.suffix.lower() in ['.awl', '.scl', '.graph', '.xml']
        
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate Siemens file structure"""
        errors = []
        
        try:
            if file_path.suffix.lower() == '.xml':
                # TIA Portal XML format
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                if 'siemens' not in root.tag.lower():
                    errors.append("Not a valid Siemens TIA Portal file")
            else:
                # Text-based formats (AWL, SCL)
                content = file_path.read_text(encoding='utf-8')
                
                if file_path.suffix.lower() == '.awl':
                    if not any(keyword in content.upper() for keyword in ['ORGANIZATION_BLOCK', 'FUNCTION_BLOCK', 'DATA_BLOCK']):
                        errors.append("Not a valid AWL file - missing block declarations")
                elif file_path.suffix.lower() == '.scl':
                    if not any(keyword in content.upper() for keyword in ['FUNCTION', 'FUNCTION_BLOCK', 'DATA_BLOCK']):
                        errors.append("Not a valid SCL file - missing block declarations")
                        
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return len(errors) == 0, errors
        
    def parse(self, file_path: Path) -> PLCProject:
        """Parse Siemens file"""
        logger.info(f"Parsing Siemens file: {file_path}")
        
        project = PLCProject(
            name=file_path.stem,
            brand=PLCBrand.SIEMENS,
            file_format=FileFormat.AWL if file_path.suffix.lower() == '.awl' else FileFormat.SCL,
            controller_type="S7-1500",
            created_date=datetime.now()
        )
        
        if file_path.suffix.lower() == '.awl':
            return self._parse_awl(file_path, project)
        elif file_path.suffix.lower() == '.scl':
            return self._parse_scl(file_path, project)
        else:
            return self._parse_tia_xml(file_path, project)
            
    def _parse_awl(self, file_path: Path, project: PLCProject) -> PLCProject:
        """Parse AWL (Statement List) file"""
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        current_block = None
        current_routine = None
        current_rung = None
        rung_number = 0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Parse block declarations
            if line.upper().startswith('ORGANIZATION_BLOCK') or line.upper().startswith('FUNCTION_BLOCK'):
                block_name = line.split()[-1]
                current_routine = UniversalRoutine(
                    name=block_name,
                    routine_type='AWL'
                )
                
                if current_block is None:
                    current_block = UniversalProgram(name="Main")
                    project.programs.append(current_block)
                    
                current_block.routines.append(current_routine)
                current_rung = UniversalRung(number=rung_number)
                current_routine.rungs.append(current_rung)
                rung_number += 1
                
            # Parse instructions
            elif current_rung is not None:
                instruction = self._parse_awl_instruction(line)
                if instruction:
                    current_rung.instructions.append(instruction)
                    
        return project
        
    def _parse_awl_instruction(self, line: str) -> Optional[UniversalInstruction]:
        """Parse AWL instruction from line"""
        parts = line.split()
        if not parts:
            return None
            
        mnemonic = parts[0].upper()
        operands = parts[1:] if len(parts) > 1 else []
        
        # Map AWL mnemonics to universal types
        instruction_type = self._map_siemens_instruction_type(mnemonic)
        
        return UniversalInstruction(
            instruction_type=instruction_type,
            mnemonic=mnemonic,
            operands=operands
        )
        
    def _parse_scl(self, file_path: Path, project: PLCProject) -> PLCProject:
        """Parse SCL (Structured Control Language) file"""
        content = file_path.read_text(encoding='utf-8')
        
        # Basic SCL parsing - this would need more sophisticated parsing
        # for a production implementation
        routine = UniversalRoutine(
            name="SCL_Main",
            routine_type='SCL'
        )
        
        program = UniversalProgram(name="Main")
        program.routines.append(routine)
        project.programs.append(program)
        
        return project
        
    def _parse_tia_xml(self, file_path: Path, project: PLCProject) -> PLCProject:
        """Parse TIA Portal XML file"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract program information from TIA XML
        # This would need detailed implementation based on TIA XML schema
        
        return project
        
    def _map_siemens_instruction_type(self, mnemonic: str) -> InstructionType:
        """Map Siemens mnemonics to universal instruction types"""
        mapping = {
            'A': InstructionType.CONTACT_NO,  # AND
            'AN': InstructionType.CONTACT_NC,  # AND NOT
            'O': InstructionType.CONTACT_NO,  # OR
            'ON': InstructionType.CONTACT_NC,  # OR NOT
            '=': InstructionType.COIL,  # Assignment
            'S': InstructionType.SET_RESET,  # Set
            'R': InstructionType.SET_RESET,  # Reset
            'TON': InstructionType.TIMER,
            'TOF': InstructionType.TIMER,
            'CTU': InstructionType.COUNTER,
            'CTD': InstructionType.COUNTER,
            'ADD': InstructionType.MATH,
            'SUB': InstructionType.MATH,
            'MUL': InstructionType.MATH,
            'DIV': InstructionType.MATH,
            'MOVE': InstructionType.MOVE,
            'CALL': InstructionType.JUMP
        }
        
        return mapping.get(mnemonic, InstructionType.FUNCTION)

class SchneiderParser(BasePLCParser):
    """Parser for Schneider Electric Unity/Control Expert files"""
    
    def __init__(self):
        super().__init__()
        self.brand = PLCBrand.SCHNEIDER
        self.supported_formats = [FileFormat.XEF]
        
    def can_parse(self, file_path: Path) -> bool:
        """Check if this is a valid XEF file"""
        return file_path.suffix.lower() == '.xef'
        
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate XEF file structure"""
        errors = []
        
        try:
            # XEF files are typically ZIP archives
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                # Check for expected files in XEF archive
                required_files = ['project.stu', 'variables.stu']
                for req_file in required_files:
                    if not any(req_file in f for f in file_list):
                        errors.append(f"Missing required file: {req_file}")
                        
        except zipfile.BadZipFile:
            errors.append("Not a valid ZIP/XEF file")
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return len(errors) == 0, errors
        
    def parse(self, file_path: Path) -> PLCProject:
        """Parse XEF file"""
        logger.info(f"Parsing Schneider XEF file: {file_path}")
        
        project = PLCProject(
            name=file_path.stem,
            brand=PLCBrand.SCHNEIDER,
            file_format=FileFormat.XEF,
            controller_type="M340",
            created_date=datetime.now()
        )
        
        # XEF files are ZIP archives containing Unity project files
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            # Extract and parse project files
            project_files = [f for f in zip_file.namelist() if f.endswith('.stu')]
            
            for proj_file in project_files:
                content = zip_file.read(proj_file).decode('utf-8', errors='ignore')
                self._parse_stu_content(content, project)
                
        return project
        
    def _parse_stu_content(self, content: str, project: PLCProject):
        """Parse Unity STU file content"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Basic Unity parsing - would need more sophisticated implementation
            if line.startswith('PROGRAM'):
                program_name = line.split()[1] if len(line.split()) > 1 else "Main"
                program = UniversalProgram(name=program_name)
                project.programs.append(program)

class MitsubishiParser(BasePLCParser):
    """Parser for Mitsubishi GX Works files"""
    
    def __init__(self):
        super().__init__()
        self.brand = PLCBrand.MITSUBISHI
        self.supported_formats = [FileFormat.GXW]
        
    def can_parse(self, file_path: Path) -> bool:
        """Check if this is a valid GXW file"""
        return file_path.suffix.lower() == '.gxw'
        
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate GXW file structure"""
        errors = []
        
        try:
            # GXW files are typically binary or XML
            content = file_path.read_bytes()
            
            # Check for GX Works file signature
            if not content.startswith(b'GX') and b'Mitsubishi' not in content:
                errors.append("Not a valid GX Works file")
                
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return len(errors) == 0, errors
        
    def parse(self, file_path: Path) -> PLCProject:
        """Parse GXW file"""
        logger.info(f"Parsing Mitsubishi GXW file: {file_path}")
        
        project = PLCProject(
            name=file_path.stem,
            brand=PLCBrand.MITSUBISHI,
            file_format=FileFormat.GXW,
            controller_type="FX3U",
            created_date=datetime.now()
        )
        
        # Basic implementation - would need detailed GXW format parsing
        program = UniversalProgram(name="Main")
        project.programs.append(program)
        
        return project

class OmronParser(BasePLCParser):
    """Parser for Omron CX-Programmer files"""
    
    def __init__(self):
        super().__init__()
        self.brand = PLCBrand.OMRON
        self.supported_formats = [FileFormat.CXP]
        
    def can_parse(self, file_path: Path) -> bool:
        """Check if this is a valid CXP file"""
        return file_path.suffix.lower() == '.cxp'
        
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate CXP file structure"""
        errors = []
        
        try:
            # CXP files are typically ZIP archives
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                # Check for expected Omron files
                if not any('.cxp' in f or 'omron' in f.lower() for f in file_list):
                    errors.append("Not a valid Omron CX-Programmer file")
                    
        except zipfile.BadZipFile:
            errors.append("Not a valid ZIP/CXP file")
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return len(errors) == 0, errors
        
    def parse(self, file_path: Path) -> PLCProject:
        """Parse CXP file"""
        logger.info(f"Parsing Omron CXP file: {file_path}")
        
        project = PLCProject(
            name=file_path.stem,
            brand=PLCBrand.OMRON,
            file_format=FileFormat.CXP,
            controller_type="CP1E",
            created_date=datetime.now()
        )
        
        # Basic implementation - would need detailed CXP format parsing
        program = UniversalProgram(name="Main")
        project.programs.append(program)
        
        return project

class MultiPLCParser:
    """Universal PLC parser supporting multiple brands"""
    
    def __init__(self):
        self.parsers = {
            PLCBrand.ROCKWELL_AB: RockwellParser(),
            PLCBrand.SIEMENS: SiemensParser(),
            PLCBrand.SCHNEIDER: SchneiderParser(),
            PLCBrand.MITSUBISHI: MitsubishiParser(),
            PLCBrand.OMRON: OmronParser()
        }
        
    def detect_brand(self, file_path: Path) -> Optional[PLCBrand]:
        """Auto-detect PLC brand from file"""
        for brand, parser in self.parsers.items():
            if parser.can_parse(file_path):
                return brand
        return None
        
    def parse(self, file_path: Path, brand: Optional[PLCBrand] = None) -> PLCProject:
        """Parse PLC file with auto-detection or specified brand"""
        if brand is None:
            brand = self.detect_brand(file_path)
            
        if brand is None:
            raise ValueError(f"Cannot determine PLC brand for file: {file_path}")
            
        if brand not in self.parsers:
            raise ValueError(f"Unsupported PLC brand: {brand}")
            
        parser = self.parsers[brand]
        
        # Validate file first
        is_valid, errors = parser.validate_file(file_path)
        if not is_valid:
            raise ValueError(f"File validation failed: {'; '.join(errors)}")
            
        return parser.parse(file_path)
        
    def get_supported_brands(self) -> List[PLCBrand]:
        """Get list of supported PLC brands"""
        return list(self.parsers.keys())
        
    def get_supported_formats(self) -> List[FileFormat]:
        """Get list of supported file formats"""
        formats = []
        for parser in self.parsers.values():
            formats.extend(parser.supported_formats)
        return list(set(formats))

class PLCConverter:
    """Convert between different PLC formats"""
    
    def __init__(self):
        self.parser = MultiPLCParser()
        
    def convert_to_universal(self, file_path: Path, brand: Optional[PLCBrand] = None) -> PLCProject:
        """Convert any PLC file to universal format"""
        return self.parser.parse(file_path, brand)
        
    def export_to_json(self, project: PLCProject) -> str:
        """Export project to JSON format"""
        def serialize_project(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                # Convert dataclass to dict
                result = {}
                for key, value in obj.__dict__.items():
                    if isinstance(value, list):
                        result[key] = [serialize_project(item) for item in value]
                    elif isinstance(value, dict):
                        result[key] = {k: serialize_project(v) for k, v in value.items()}
                    else:
                        result[key] = serialize_project(value)
                return result
            else:
                return obj
                
        project_dict = serialize_project(project)
        return json.dumps(project_dict, indent=2)
        
    def export_to_xml(self, project: PLCProject) -> str:
        """Export project to XML format"""
        root = ET.Element("PLCProject")
        root.set("name", project.name)
        root.set("brand", project.brand.value)
        root.set("format", project.file_format.value)
        
        # Add programs
        programs_elem = ET.SubElement(root, "Programs")
        for program in project.programs:
            prog_elem = ET.SubElement(programs_elem, "Program")
            prog_elem.set("name", program.name)
            
            # Add routines
            routines_elem = ET.SubElement(prog_elem, "Routines")
            for routine in program.routines:
                routine_elem = ET.SubElement(routines_elem, "Routine")
                routine_elem.set("name", routine.name)
                routine_elem.set("type", routine.routine_type)
                
        return ET.tostring(root, encoding='unicode')

# Convenience functions
def parse_plc_file(file_path: Union[str, Path], brand: Optional[PLCBrand] = None) -> PLCProject:
    """Parse any supported PLC file"""
    parser = MultiPLCParser()
    return parser.parse(Path(file_path), brand)

def detect_plc_brand(file_path: Union[str, Path]) -> Optional[PLCBrand]:
    """Detect PLC brand from file"""
    parser = MultiPLCParser()
    return parser.detect_brand(Path(file_path))

def get_supported_brands() -> List[PLCBrand]:
    """Get list of supported PLC brands"""
    parser = MultiPLCParser()
    return parser.get_supported_brands()

def convert_plc_project(file_path: Union[str, Path], output_format: str = 'json') -> str:
    """Convert PLC project to specified format"""
    converter = PLCConverter()
    project = converter.convert_to_universal(Path(file_path))
    
    if output_format.lower() == 'json':
        return converter.export_to_json(project)
    elif output_format.lower() == 'xml':
        return converter.export_to_xml(project)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

# Example usage and testing
async def main():
    """Example usage of multi-PLC support"""
    print("🏭 Multi-PLC Brand Support - Step 36")
    print("=" * 50)
    
    # Initialize parser
    parser = MultiPLCParser()
    
    print(f"Supported brands: {[brand.value for brand in parser.get_supported_brands()]}")
    print(f"Supported formats: {[fmt.value for fmt in parser.get_supported_formats()]}")
    
    # Test with existing L5X file
    try:
        l5x_file = Path("Assembly_Controls_Robot.L5X")
        if l5x_file.exists():
            print(f"\n📁 Testing with: {l5x_file}")
            
            # Detect brand
            detected_brand = parser.detect_brand(l5x_file)
            print(f"Detected brand: {detected_brand.value if detected_brand else 'Unknown'}")
            
            # Parse file
            project = parser.parse(l5x_file)
            print(f"Project name: {project.name}")
            print(f"Controller type: {project.controller_type}")
            print(f"Programs: {len(project.programs)}")
            print(f"Global tags: {len(project.global_tags)}")
            
            # Convert to JSON
            converter = PLCConverter()
            json_output = converter.export_to_json(project)
            print(f"JSON export length: {len(json_output)} characters")
            
            # Save sample output
            output_file = Path("multi_plc_sample_output.json")
            output_file.write_text(json_output)
            print(f"Sample output saved to: {output_file}")
            
        else:
            print("⚠️  No L5X file found for testing")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
    print("\n✅ Multi-PLC Brand Support demonstration complete!")

if __name__ == "__main__":
    asyncio.run(main())
