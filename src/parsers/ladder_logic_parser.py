#!/usr/bin/env python3
"""
Ladder Logic Parser Extension for L5X Parser

This module extends the L5X parser with ladder logic parsing capabilities.
It extracts and parses ladder logic rungs from routine XML content.

Author: GitHub Copilot
Date: July 2025
"""

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..models.ladder_logic import (
    LadderRoutine, LadderRung, LadderInstruction, LadderLogicParser,
    InstructionType, RungType
)


class LadderLogicExtractor:
    """
    Extension class for extracting ladder logic from L5X files.
    
    This class works with the existing L5X parser to extract and parse
    ladder logic rungs, instructions, and tag relationships.
    """
    
    def __init__(self):
        """Initialize the ladder logic extractor"""
        self.logger = logging.getLogger(__name__)
        self.parser = LadderLogicParser()
        self.routines = {}  # Dict[str, LadderRoutine]
        
    def extract_ladder_logic_from_xml(self, root: ET.Element) -> Dict[str, LadderRoutine]:
        """
        Extract all ladder logic routines from L5X XML root.
        
        Args:
            root: Root XML element of L5X file
            
        Returns:
            Dictionary mapping routine names to LadderRoutine objects
        """
        self.routines = {}
        
        # Find all programs
        programs = root.findall('.//Program')
        self.logger.info(f"Found {len(programs)} programs to process for ladder logic")
        
        for program in programs:
            program_name = program.get('Name', 'Unknown')
            self.logger.debug(f"Processing program: {program_name}")
            
            # Find routines within the program
            routines = program.findall('.//Routine')
            for routine_elem in routines:
                routine_name = routine_elem.get('Name', 'Unknown')
                routine_type = routine_elem.get('Type', 'Unknown')
                
                if routine_type == 'RLL':  # Relay Ladder Logic
                    self.logger.debug(f"Processing RLL routine: {routine_name}")
                    ladder_routine = self._extract_routine_content(
                        routine_elem, routine_name, program_name
                    )
                    if ladder_routine:
                        full_routine_name = f"{program_name}.{routine_name}"
                        self.routines[full_routine_name] = ladder_routine
                else:
                    self.logger.debug(f"Skipping non-RLL routine: {routine_name} (Type: {routine_type})")
        
        self.logger.info(f"Successfully extracted {len(self.routines)} ladder logic routines")
        return self.routines
    
    def _extract_routine_content(self, routine_elem: ET.Element, 
                                routine_name: str, program_name: str) -> Optional[LadderRoutine]:
        """
        Extract ladder logic content from a routine XML element.
        
        Args:
            routine_elem: XML element for the routine
            routine_name: Name of the routine
            program_name: Name of the parent program
            
        Returns:
            LadderRoutine object or None if extraction fails
        """
        try:
            # Find RLLContent element
            rll_content = routine_elem.find('RLLContent')
            if rll_content is None:
                self.logger.warning(f"No RLLContent found in routine {routine_name}")
                return None
            
            # Create ladder routine
            ladder_routine = LadderRoutine(
                name=routine_name,
                program_name=program_name,
                routine_type='RLL'
            )
            
            # Extract rungs
            rungs = rll_content.findall('Rung')
            self.logger.debug(f"Found {len(rungs)} rungs in routine {routine_name}")
            
            for rung_elem in rungs:
                ladder_rung = self._extract_rung(rung_elem, routine_name)
                if ladder_rung:
                    ladder_routine.rungs.append(ladder_rung)
            
            self.logger.info(f"Extracted {len(ladder_routine.rungs)} rungs from routine {routine_name}")
            return ladder_routine
            
        except Exception as e:
            self.logger.error(f"Error extracting routine {routine_name}: {e}")
            return None
    
    def _extract_rung(self, rung_elem: ET.Element, routine_name: str) -> Optional[LadderRung]:
        """
        Extract a single rung from XML element.
        
        Args:
            rung_elem: XML element for the rung
            routine_name: Name of the parent routine
            
        Returns:
            LadderRung object or None if extraction fails
        """
        try:
            # Get rung attributes
            number = int(rung_elem.get('Number', 0))
            rung_type_str = rung_elem.get('Type', 'N')
            
            # Convert rung type
            rung_type = RungType.NORMAL
            if rung_type_str == 'U':
                rung_type = RungType.UNCONDITIONAL
            elif rung_type_str == 'C':
                rung_type = RungType.COMMENT
            
            # Create rung
            ladder_rung = LadderRung(
                number=number,
                rung_type=rung_type,
                routine_name=routine_name
            )
            
            # Extract text content (ladder logic)
            text_elem = rung_elem.find('Text')
            if text_elem is not None and text_elem.text:
                ladder_rung.raw_text = text_elem.text.strip()
                
                # Parse instructions from text
                try:
                    instructions = self.parser.parse_rung_text(ladder_rung.raw_text)
                    ladder_rung.instructions = instructions
                    self.logger.debug(f"Parsed {len(instructions)} instructions from rung {number}")
                except Exception as parse_error:
                    self.logger.warning(f"Error parsing rung {number} text: {parse_error}")
                    # Continue with empty instructions list
            
            # Extract comment
            comment_elem = rung_elem.find('Comment')
            if comment_elem is not None and comment_elem.text:
                ladder_rung.comment = comment_elem.text.strip()
            
            return ladder_rung
            
        except Exception as e:
            self.logger.error(f"Error extracting rung: {e}")
            return None
    
    def get_routine_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about extracted ladder logic.
        
        Returns:
            Dictionary with statistics
        """
        if not self.routines:
            return {}
        
        total_rungs = 0
        total_instructions = 0
        instruction_type_counts = {}
        all_tags = set()
        routines_with_comments = 0
        
        routine_stats = {}
        
        for routine_name, routine in self.routines.items():
            stats = routine.get_statistics()
            routine_stats[routine_name] = stats
            
            total_rungs += stats['total_rungs']
            total_instructions += stats['total_instructions']
            
            # Accumulate instruction type counts
            for inst_type, count in stats['instruction_types'].items():
                instruction_type_counts[inst_type] = instruction_type_counts.get(inst_type, 0) + count
            
            # Collect unique tags
            routine_tags = routine.get_all_tag_references()
            all_tags.update(routine_tags)
            
            if stats['rungs_with_comments'] > 0:
                routines_with_comments += 1
        
        return {
            'total_routines': len(self.routines),
            'total_rungs': total_rungs,
            'total_instructions': total_instructions,
            'instruction_type_counts': instruction_type_counts,
            'unique_tags_referenced': len(all_tags),
            'routines_with_comments': routines_with_comments,
            'routine_details': routine_stats,
            'most_common_instructions': sorted(
                instruction_type_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def search_instructions(self, instruction_type: InstructionType = None, 
                          tag_reference: str = None) -> List[Tuple[str, int, LadderInstruction]]:
        """
        Search for instructions across all routines.
        
        Args:
            instruction_type: Filter by instruction type
            tag_reference: Filter by tag reference
            
        Returns:
            List of tuples (routine_name, rung_number, instruction)
        """
        results = []
        
        for routine_name, routine in self.routines.items():
            for rung in routine.rungs:
                for instruction in rung.instructions:
                    # Filter by instruction type
                    if instruction_type and instruction.instruction_type != instruction_type:
                        continue
                    
                    # Filter by tag reference
                    if tag_reference:
                        tag_refs = instruction.get_tag_references()
                        if tag_reference not in tag_refs:
                            continue
                    
                    results.append((routine_name, rung.number, instruction))
        
        return results
    
    def get_tag_usage_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze how tags are used across ladder logic.
        
        Returns:
            Dictionary mapping tag names to usage statistics
        """
        tag_usage = {}
        
        for routine_name, routine in self.routines.items():
            for rung in routine.rungs:
                # Analyze input tags (contacts)
                input_tags = rung.get_input_tags()
                for tag in input_tags:
                    if tag not in tag_usage:
                        tag_usage[tag] = {
                            'input_count': 0,
                            'output_count': 0,
                            'routines': set(),
                            'instruction_types': set()
                        }
                    tag_usage[tag]['input_count'] += 1
                    tag_usage[tag]['routines'].add(routine_name)
                
                # Analyze output tags
                output_tags = rung.get_output_tags()
                for tag in output_tags:
                    if tag not in tag_usage:
                        tag_usage[tag] = {
                            'input_count': 0,
                            'output_count': 0,
                            'routines': set(),
                            'instruction_types': set()
                        }
                    tag_usage[tag]['output_count'] += 1
                    tag_usage[tag]['routines'].add(routine_name)
                
                # Track instruction types for each tag
                for instruction in rung.instructions:
                    for tag in instruction.get_tag_references():
                        if tag in tag_usage:
                            tag_usage[tag]['instruction_types'].add(instruction.instruction_type.value)
        
        # Convert sets to lists for JSON serialization
        for tag, usage in tag_usage.items():
            usage['routines'] = list(usage['routines'])
            usage['instruction_types'] = list(usage['instruction_types'])
            usage['total_references'] = usage['input_count'] + usage['output_count']
        
        return tag_usage
    
    def export_ladder_logic_summary(self) -> Dict[str, Any]:
        """
        Export comprehensive summary of ladder logic analysis.
        
        Returns:
            Complete summary dictionary
        """
        return {
            'statistics': self.get_routine_statistics(),
            'tag_usage': self.get_tag_usage_analysis(),
            'routines': {
                name: {
                    'program': routine.program_name,
                    'rungs': len(routine.rungs),
                    'instructions': sum(len(r.instructions) for r in routine.rungs),
                    'unique_tags': len(routine.get_all_tag_references()),
                    'commented_rungs': len([r for r in routine.rungs if r.comment])
                }
                for name, routine in self.routines.items()
            }
        }
