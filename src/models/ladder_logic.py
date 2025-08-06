#!/usr/bin/env python3
"""
Ladder Logic Models for PLC Code Generator

This module defines the data models for ladder logic components including
rungs, instructions, and logic elements.

Author: GitHub Copilot
Date: July 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import re


class InstructionType(Enum):
    """Types of ladder logic instructions"""
    # Contact Instructions
    XIC = "XIC"  # Examine If Closed (normally open contact)
    XIO = "XIO"  # Examine If Open (normally closed contact)
    
    # Output Instructions
    OTE = "OTE"  # Output Energize
    OTL = "OTL"  # Output Latch
    OTU = "OTU"  # Output Unlatch
    
    # Timer Instructions
    TON = "TON"  # Timer On Delay
    TOF = "TOF"  # Timer Off Delay
    RTO = "RTO"  # Retentive Timer On
    
    # Counter Instructions
    CTU = "CTU"  # Count Up
    CTD = "CTD"  # Count Down
    RES = "RES"  # Reset
    
    # Program Control
    JSR = "JSR"  # Jump to Subroutine
    RET = "RET"  # Return from Subroutine
    JMP = "JMP"  # Jump
    LBL = "LBL"  # Label
    
    # Math Instructions
    ADD = "ADD"  # Add
    SUB = "SUB"  # Subtract
    MUL = "MUL"  # Multiply
    DIV = "DIV"  # Divide
    MOV = "MOV"  # Move
    
    # Comparison Instructions
    EQU = "EQU"  # Equal
    NEQ = "NEQ"  # Not Equal
    LES = "LES"  # Less Than
    LEQ = "LEQ"  # Less Than or Equal
    GRT = "GRT"  # Greater Than
    GEQ = "GEQ"  # Greater Than or Equal
    
    # Logic Instructions
    AND = "AND"  # Bitwise AND
    OR = "OR"   # Bitwise OR
    XOR = "XOR"  # Bitwise XOR
    NOT = "NOT"  # Bitwise NOT
    
    # Unknown or custom instructions
    UNKNOWN = "UNKNOWN"


class RungType(Enum):
    """Types of ladder logic rungs"""
    NORMAL = "N"      # Normal rung
    UNCONDITIONAL = "U"  # Unconditional rung
    COMMENT = "C"     # Comment only rung


@dataclass
class InstructionParameter:
    """Represents a parameter in an instruction"""
    name: Optional[str]          # Parameter name (if named)
    value: str                   # Parameter value/tag reference
    parameter_type: str = "TAG"  # Type: TAG, LITERAL, EXPRESSION
    data_type: Optional[str] = None  # Data type if known


@dataclass
class LadderInstruction:
    """Represents a single ladder logic instruction"""
    instruction_type: InstructionType
    parameters: List[InstructionParameter] = field(default_factory=list)
    raw_text: str = ""           # Original instruction text
    position: int = 0            # Position in rung
    
    def get_tag_references(self) -> List[str]:
        """Get all tag references from this instruction"""
        tags = []
        for param in self.parameters:
            if param.parameter_type == "TAG":
                # Clean up tag reference (remove array indices, member access)
                tag_name = param.value.split('[')[0].split('.')[0]
                if tag_name and tag_name not in tags:
                    tags.append(tag_name)
        return tags
    
    def __str__(self) -> str:
        params = ",".join([p.value for p in self.parameters])
        return f"{self.instruction_type.value}({params})"


@dataclass
class LadderRung:
    """Represents a complete ladder logic rung"""
    number: int
    rung_type: RungType
    instructions: List[LadderInstruction] = field(default_factory=list)
    comment: Optional[str] = None
    raw_text: Optional[str] = None
    routine_name: Optional[str] = None
    
    def get_all_tag_references(self) -> List[str]:
        """Get all unique tag references from all instructions in this rung"""
        all_tags = []
        for instruction in self.instructions:
            for tag in instruction.get_tag_references():
                if tag not in all_tags:
                    all_tags.append(tag)
        return all_tags
    
    def has_output_instructions(self) -> bool:
        """Check if rung has any output instructions"""
        output_types = {InstructionType.OTE, InstructionType.OTL, InstructionType.OTU}
        return any(inst.instruction_type in output_types for inst in self.instructions)
    
    def get_output_tags(self) -> List[str]:
        """Get tags that are outputs in this rung"""
        output_types = {InstructionType.OTE, InstructionType.OTL, InstructionType.OTU}
        output_tags = []
        for instruction in self.instructions:
            if instruction.instruction_type in output_types:
                output_tags.extend(instruction.get_tag_references())
        return output_tags
    
    def get_input_tags(self) -> List[str]:
        """Get tags that are inputs (contacts) in this rung"""
        input_types = {InstructionType.XIC, InstructionType.XIO}
        input_tags = []
        for instruction in self.instructions:
            if instruction.instruction_type in input_types:
                input_tags.extend(instruction.get_tag_references())
        return input_tags


@dataclass
class LadderRoutine:
    """Represents a complete ladder logic routine"""
    name: str
    program_name: str
    routine_type: str = "RLL"
    rungs: List[LadderRung] = field(default_factory=list)
    
    def get_all_tag_references(self) -> List[str]:
        """Get all unique tag references from all rungs"""
        all_tags = []
        for rung in self.rungs:
            for tag in rung.get_all_tag_references():
                if tag not in all_tags:
                    all_tags.append(tag)
        return all_tags
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routine statistics"""
        total_instructions = sum(len(rung.instructions) for rung in self.rungs)
        instruction_counts = {}
        
        for rung in self.rungs:
            for instruction in rung.instructions:
                inst_type = instruction.instruction_type.value
                instruction_counts[inst_type] = instruction_counts.get(inst_type, 0) + 1
        
        return {
            'total_rungs': len(self.rungs),
            'total_instructions': total_instructions,
            'instruction_types': instruction_counts,
            'unique_tags': len(self.get_all_tag_references()),
            'rungs_with_comments': len([r for r in self.rungs if r.comment]),
        }


class LadderLogicParser:
    """Parser for converting text-based ladder logic to structured format"""
    
    # Regex patterns for different instruction types
    INSTRUCTION_PATTERNS = {
        InstructionType.XIC: r'XIC\(([^)]+)\)',
        InstructionType.XIO: r'XIO\(([^)]+)\)',
        InstructionType.OTE: r'OTE\(([^)]+)\)',
        InstructionType.OTL: r'OTL\(([^)]+)\)',
        InstructionType.OTU: r'OTU\(([^)]+)\)',
        InstructionType.TON: r'TON\(([^)]+)\)',
        InstructionType.TOF: r'TOF\(([^)]+)\)',
        InstructionType.RTO: r'RTO\(([^)]+)\)',
        InstructionType.CTU: r'CTU\(([^)]+)\)',
        InstructionType.CTD: r'CTD\(([^)]+)\)',
        InstructionType.RES: r'RES\(([^)]+)\)',
        InstructionType.JSR: r'JSR\(([^)]+)\)',
        InstructionType.JMP: r'JMP\(([^)]+)\)',
        InstructionType.LBL: r'LBL\(([^)]+)\)',
        InstructionType.MOV: r'MOV\(([^)]+)\)',
        InstructionType.ADD: r'ADD\(([^)]+)\)',
        InstructionType.SUB: r'SUB\(([^)]+)\)',
        InstructionType.MUL: r'MUL\(([^)]+)\)',
        InstructionType.DIV: r'DIV\(([^)]+)\)',
        InstructionType.EQU: r'EQU\(([^)]+)\)',
        InstructionType.NEQ: r'NEQ\(([^)]+)\)',
        InstructionType.LES: r'LES\(([^)]+)\)',
        InstructionType.LEQ: r'LEQ\(([^)]+)\)',
        InstructionType.GRT: r'GRT\(([^)]+)\)',
        InstructionType.GEQ: r'GEQ\(([^)]+)\)',
    }
    
    def __init__(self):
        """Initialize the parser"""
        self.compiled_patterns = {}
        for inst_type, pattern in self.INSTRUCTION_PATTERNS.items():
            self.compiled_patterns[inst_type] = re.compile(pattern, re.IGNORECASE)
    
    def parse_rung_text(self, text: str) -> List[LadderInstruction]:
        """
        Parse ladder logic text into structured instructions.
        
        Args:
            text: Raw ladder logic text (e.g., "XIC(Tag1)XIO(Tag2)OTE(Output)")
            
        Returns:
            List of LadderInstruction objects
        """
        instructions = []
        position = 0
        
        # Remove whitespace and normalize
        text = text.strip()
        
        # Handle bracket expressions like [XIC(Tag1),XIC(Tag2)]
        # These represent parallel branches
        if text.startswith('[') and ']' in text:
            bracket_end = text.find(']')
            bracket_content = text[1:bracket_end]
            remaining_text = text[bracket_end+1:]
            
            # Parse instructions within brackets
            branch_instructions = self._parse_bracket_content(bracket_content)
            instructions.extend(branch_instructions)
            
            # Parse remaining instructions
            remaining_instructions = self.parse_rung_text(remaining_text)
            instructions.extend(remaining_instructions)
            
            return instructions
        
        # Find all instructions in the text
        for inst_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                parameters = self._parse_parameters(match.group(1))
                instruction = LadderInstruction(
                    instruction_type=inst_type,
                    parameters=parameters,
                    raw_text=match.group(0),
                    position=match.start()
                )
                instructions.append(instruction)
        
        # Sort by position in text
        instructions.sort(key=lambda x: x.position)
        
        return instructions
    
    def _parse_bracket_content(self, content: str) -> List[LadderInstruction]:
        """Parse instructions within bracket expressions (parallel branches)"""
        instructions = []
        
        # Split by commas, but be careful of commas within instruction parameters
        parts = self._smart_split(content, ',')
        
        for part in parts:
            part = part.strip()
            if part:
                part_instructions = self.parse_rung_text(part)
                instructions.extend(part_instructions)
        
        return instructions
    
    def _smart_split(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter, but ignore delimiters within parentheses"""
        parts = []
        current_part = ""
        paren_depth = 0
        
        for char in text:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == delimiter and paren_depth == 0:
                parts.append(current_part)
                current_part = ""
                continue
            
            current_part += char
        
        if current_part:
            parts.append(current_part)
        
        return parts
    
    def _parse_parameters(self, param_string: str) -> List[InstructionParameter]:
        """Parse instruction parameters from parameter string"""
        if not param_string.strip():
            return []
        
        # Split parameters by commas, respecting nested structures
        param_parts = self._smart_split(param_string, ',')
        parameters = []
        
        for i, part in enumerate(param_parts):
            part = part.strip()
            if not part:
                continue
            
            # Determine parameter type
            param_type = "TAG"
            if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                param_type = "LITERAL"
            elif part == "?":
                param_type = "PLACEHOLDER"
            elif any(op in part for op in ['+', '-', '*', '/', '(', ')']):
                param_type = "EXPRESSION"
            
            parameters.append(InstructionParameter(
                name=None,  # Could be enhanced to handle named parameters
                value=part,
                parameter_type=param_type
            ))
        
        return parameters


def create_instruction_from_text(text: str, position: int = 0) -> Optional[LadderInstruction]:
    """
    Utility function to create a single instruction from text.
    
    Args:
        text: Instruction text (e.g., "XIC(Tag1)")
        position: Position in the rung
        
    Returns:
        LadderInstruction object or None if parsing fails
    """
    parser = LadderLogicParser()
    instructions = parser.parse_rung_text(text)
    
    if instructions:
        instruction = instructions[0]  # Take first instruction
        instruction.position = position
        return instruction
    
    return None
