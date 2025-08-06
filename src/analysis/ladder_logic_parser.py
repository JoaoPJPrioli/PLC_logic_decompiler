"""
Ladder Logic Parser Module
Parses and analyzes ladder logic from L5X files

This module provides comprehensive ladder logic analysis including:
- Rung parsing and instruction extraction
- Contact and coil analysis
- Logic flow mapping
- Instruction categorization and analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import xml.etree.ElementTree as ET
import re
import logging

logger = logging.getLogger(__name__)

class InstructionType(Enum):
    """Types of ladder logic instructions"""
    CONTACT_NO = "XIC"  # Examine If Closed
    CONTACT_NC = "XIO"  # Examine If Open
    COIL = "OTE"        # Output Energize
    COIL_LATCH = "OTL"  # Output Latch
    COIL_UNLATCH = "OTU" # Output Unlatch
    TIMER_ON = "TON"    # Timer On Delay
    TIMER_OFF = "TOF"   # Timer Off Delay
    COUNTER_UP = "CTU"  # Count Up
    COUNTER_DOWN = "CTD" # Count Down
    MATH = "MATH"       # Math operations
    MOVE = "MOV"        # Move
    COMPARE = "CMP"     # Compare
    FUNCTION_BLOCK = "FB" # Function Block

@dataclass
class LadderInstruction:
    """Represents a single ladder logic instruction"""
    type: InstructionType
    operand: str
    description: str = ""
    rung_number: int = 0
    position: Tuple[int, int] = (0, 0)  # (row, col) in rung
    
@dataclass
class LadderRung:
    """Represents a complete ladder logic rung"""
    number: int
    instructions: List[LadderInstruction] = field(default_factory=list)
    comment: str = ""
    routine_name: str = ""
    
    def get_inputs(self) -> List[LadderInstruction]:
        """Get all input instructions (contacts)"""
        return [inst for inst in self.instructions 
                if inst.type in [InstructionType.CONTACT_NO, InstructionType.CONTACT_NC]]
    
    def get_outputs(self) -> List[LadderInstruction]:
        """Get all output instructions (coils)"""
        return [inst for inst in self.instructions 
                if inst.type in [InstructionType.COIL, InstructionType.COIL_LATCH, InstructionType.COIL_UNLATCH]]

@dataclass
class RoutineLadderLogic:
    """Complete ladder logic for a routine"""
    routine_name: str
    program_name: str
    rungs: List[LadderRung] = field(default_factory=list)
    
    def get_all_tags_used(self) -> Set[str]:
        """Get all tag names used in this routine"""
        tags = set()
        for rung in self.rungs:
            for instruction in rung.instructions:
                tags.add(instruction.operand)
        return tags

class LadderLogicParser:
    """
    Parser for ladder logic content in L5X files
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_routine_ladder_logic(self, routine_elem: ET.Element, 
                                 routine_name: str, program_name: str) -> RoutineLadderLogic:
        """
        Parse ladder logic from a routine element
        
        Args:
            routine_elem: XML element for the routine
            routine_name: Name of the routine
            program_name: Name of the parent program
            
        Returns:
            RoutineLadderLogic object with parsed rungs and instructions
        """
        routine_logic = RoutineLadderLogic(
            routine_name=routine_name,
            program_name=program_name
        )
        
        # Find RLL (Relay Ladder Logic) content
        rll_content = routine_elem.find('RLLContent')
        if rll_content is None:
            self.logger.warning(f"No RLL content found in routine {routine_name}")
            return routine_logic
        
        # Parse each rung
        for rung_elem in rll_content.findall('Rung'):
            rung = self._parse_rung(rung_elem, routine_name)
            if rung:
                routine_logic.rungs.append(rung)
        
        self.logger.info(f"Parsed {len(routine_logic.rungs)} rungs from routine {routine_name}")
        return routine_logic
    
    def _parse_rung(self, rung_elem: ET.Element, routine_name: str) -> Optional[LadderRung]:
        """Parse a single rung element"""
        try:
            rung_number = int(rung_elem.get('Number', 0))
            rung_type = rung_elem.get('Type', 'N')
            
            # Skip non-normal rungs for now
            if rung_type != 'N':
                return None
            
            rung = LadderRung(
                number=rung_number,
                routine_name=routine_name
            )
            
            # Get rung comment
            comment_elem = rung_elem.find('Comment')
            if comment_elem is not None:
                rung.comment = comment_elem.text or ""
            
            # Parse rung text (the actual ladder logic)
            text_elem = rung_elem.find('Text')
            if text_elem is not None:
                rung_text = text_elem.text or ""
                instructions = self._parse_rung_text(rung_text, rung_number)
                rung.instructions.extend(instructions)
            
            return rung
            
        except Exception as e:
            self.logger.error(f"Error parsing rung: {e}")
            return None
    
    def _parse_rung_text(self, rung_text: str, rung_number: int) -> List[LadderInstruction]:
        """
        Parse ladder logic text into instructions
        
        This is a simplified parser for common instructions.
        A full implementation would need to handle the complete
        Rockwell ladder logic syntax.
        """
        instructions = []
        
        try:
            # Simple patterns for common instructions
            patterns = {
                InstructionType.CONTACT_NO: r'XIC\(([^)]+)\)',
                InstructionType.CONTACT_NC: r'XIO\(([^)]+)\)',
                InstructionType.COIL: r'OTE\(([^)]+)\)',
                InstructionType.COIL_LATCH: r'OTL\(([^)]+)\)',
                InstructionType.COIL_UNLATCH: r'OTU\(([^)]+)\)',
                InstructionType.TIMER_ON: r'TON\(([^,]+)',
                InstructionType.TIMER_OFF: r'TOF\(([^,]+)',
                InstructionType.COUNTER_UP: r'CTU\(([^,]+)',
                InstructionType.COUNTER_DOWN: r'CTD\(([^,]+)',
                InstructionType.MOVE: r'MOV\(([^,]+)',
            }
            
            for instruction_type, pattern in patterns.items():
                matches = re.finditer(pattern, rung_text)
                for match in matches:
                    operand = match.group(1).strip()
                    
                    # Clean up operand (remove quotes, etc.)
                    operand = operand.replace('"', '').replace("'", "")
                    
                    instruction = LadderInstruction(
                        type=instruction_type,
                        operand=operand,
                        rung_number=rung_number,
                        description=f"{instruction_type.value} instruction for {operand}"
                    )
                    instructions.append(instruction)
            
        except Exception as e:
            self.logger.error(f"Error parsing rung text: {e}")
        
        return instructions
    
    def analyze_ladder_patterns(self, routine_logic: RoutineLadderLogic) -> Dict[str, Any]:
        """
        Analyze patterns in ladder logic
        
        Args:
            routine_logic: Parsed ladder logic
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'routine_name': routine_logic.routine_name,
            'total_rungs': len(routine_logic.rungs),
            'instruction_counts': {},
            'tags_used': list(routine_logic.get_all_tags_used()),
            'patterns': {
                'timer_usage': [],
                'counter_usage': [],
                'latch_circuits': [],
                'seal_circuits': []
            },
            'complexity_score': 0
        }
        
        # Count instruction types
        for rung in routine_logic.rungs:
            for instruction in rung.instructions:
                inst_type = instruction.type.value
                analysis['instruction_counts'][inst_type] = analysis['instruction_counts'].get(inst_type, 0) + 1
        
        # Analyze patterns
        self._analyze_timer_patterns(routine_logic, analysis)
        self._analyze_counter_patterns(routine_logic, analysis)
        self._analyze_latch_patterns(routine_logic, analysis)
        
        # Calculate complexity score
        analysis['complexity_score'] = self._calculate_complexity_score(analysis)
        
        return analysis
    
    def _analyze_timer_patterns(self, routine_logic: RoutineLadderLogic, analysis: Dict[str, Any]):
        """Analyze timer usage patterns"""
        for rung in routine_logic.rungs:
            timer_instructions = [inst for inst in rung.instructions 
                                if inst.type in [InstructionType.TIMER_ON, InstructionType.TIMER_OFF]]
            
            for timer_inst in timer_instructions:
                analysis['patterns']['timer_usage'].append({
                    'rung': rung.number,
                    'type': timer_inst.type.value,
                    'timer': timer_inst.operand
                })
    
    def _analyze_counter_patterns(self, routine_logic: RoutineLadderLogic, analysis: Dict[str, Any]):
        """Analyze counter usage patterns"""
        for rung in routine_logic.rungs:
            counter_instructions = [inst for inst in rung.instructions 
                                  if inst.type in [InstructionType.COUNTER_UP, InstructionType.COUNTER_DOWN]]
            
            for counter_inst in counter_instructions:
                analysis['patterns']['counter_usage'].append({
                    'rung': rung.number,
                    'type': counter_inst.type.value,
                    'counter': counter_inst.operand
                })
    
    def _analyze_latch_patterns(self, routine_logic: RoutineLadderLogic, analysis: Dict[str, Any]):
        """Analyze latching circuit patterns"""
        for rung in routine_logic.rungs:
            latch_instructions = [inst for inst in rung.instructions 
                                if inst.type in [InstructionType.COIL_LATCH, InstructionType.COIL_UNLATCH]]
            
            for latch_inst in latch_instructions:
                analysis['patterns']['latch_circuits'].append({
                    'rung': rung.number,
                    'type': latch_inst.type.value,
                    'operand': latch_inst.operand
                })
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate a complexity score for the routine"""
        base_score = analysis['total_rungs'] * 2
        
        # Add complexity for different instruction types
        instruction_weights = {
            'XIC': 1, 'XIO': 1,     # Simple contacts
            'OTE': 1,                # Simple output
            'OTL': 2, 'OTU': 2,     # Latching outputs
            'TON': 3, 'TOF': 3,     # Timers
            'CTU': 3, 'CTD': 3,     # Counters
            'MOV': 2,                # Move operations
        }
        
        complexity_bonus = 0
        for inst_type, count in analysis['instruction_counts'].items():
            weight = instruction_weights.get(inst_type, 1)
            complexity_bonus += count * weight
        
        return min(100, base_score + complexity_bonus)  # Cap at 100

# Instruction analysis for detailed instruction breakdown
class InstructionAnalyzer:
    """Analyzer for individual PLC instructions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_instruction_usage(self, routines: List[RoutineLadderLogic]) -> Dict[str, Any]:
        """
        Analyze instruction usage across multiple routines
        
        Args:
            routines: List of parsed ladder logic routines
            
        Returns:
            Comprehensive instruction usage analysis
        """
        analysis = {
            'total_routines': len(routines),
            'instruction_summary': {},
            'tag_usage': {},
            'routine_complexity': {},
            'recommendations': []
        }
        
        all_instructions = []
        
        # Collect all instructions
        for routine in routines:
            routine_instructions = []
            for rung in routine.rungs:
                routine_instructions.extend(rung.instructions)
                all_instructions.extend(rung.instructions)
            
            # Calculate per-routine metrics
            analysis['routine_complexity'][routine.routine_name] = {
                'instruction_count': len(routine_instructions),
                'rung_count': len(routine.rungs),
                'unique_tags': len(routine.get_all_tags_used())
            }
        
        # Analyze instruction types
        for instruction in all_instructions:
            inst_type = instruction.type.value
            analysis['instruction_summary'][inst_type] = analysis['instruction_summary'].get(inst_type, 0) + 1
            
            # Track tag usage
            tag = instruction.operand
            if tag not in analysis['tag_usage']:
                analysis['tag_usage'][tag] = {'count': 0, 'instructions': []}
            analysis['tag_usage'][tag]['count'] += 1
            analysis['tag_usage'][tag]['instructions'].append(inst_type)
        
        # Generate recommendations
        self._generate_instruction_recommendations(analysis)
        
        return analysis
    
    def _generate_instruction_recommendations(self, analysis: Dict[str, Any]):
        """Generate recommendations based on instruction analysis"""
        recommendations = []
        
        # Check for excessive latch usage
        latch_count = analysis['instruction_summary'].get('OTL', 0) + analysis['instruction_summary'].get('OTU', 0)
        if latch_count > 10:
            recommendations.append("Consider reviewing latch circuit usage - high count may indicate complex state management")
        
        # Check for timer usage
        timer_count = analysis['instruction_summary'].get('TON', 0) + analysis['instruction_summary'].get('TOF', 0)
        if timer_count > 20:
            recommendations.append("High timer usage detected - consider consolidating timing logic")
        
        # Check for tag reuse
        heavily_used_tags = {tag: data for tag, data in analysis['tag_usage'].items() if data['count'] > 10}
        if heavily_used_tags:
            recommendations.append(f"High-usage tags detected: {list(heavily_used_tags.keys())[:5]} - ensure proper documentation")
        
        analysis['recommendations'] = recommendations
