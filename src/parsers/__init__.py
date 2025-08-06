# Multi-PLC Parser Module
# Step 36: Multi-PLC Brand Support

from .multi_plc_parser import (
    PLCBrand,
    FileFormat,
    InstructionType,
    UniversalTag,
    UniversalInstruction,
    UniversalRung,
    UniversalRoutine,
    UniversalProgram,
    PLCProject,
    MultiPLCParser,
    PLCConverter,
    parse_plc_file,
    detect_plc_brand,
    get_supported_brands,
    convert_plc_project
)

__version__ = "1.0.0"
__author__ = "PLC Logic Decompiler Team"
