"""
Step 14: UDT (User Defined Type) Support
Advanced analysis of complex data structures and nested tags in PLC systems.

This module provides comprehensive UDT analysis including:
- UDT definition parsing and structure analysis
- Member access pattern detection and analysis
- Nested UDT relationship mapping
- Member dependency tracking
- UDT usage analysis across routines
- Built-in type handling (TIMER, COUNTER, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from enum import Enum
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class UDTType(Enum):
    """Types of UDT structures supported."""
    TIMER = "TIMER"
    COUNTER = "COUNTER"
    CUSTOM = "CUSTOM"
    IO_MODULE = "IO_MODULE"
    ARRAY = "ARRAY"
    BUILT_IN = "BUILT_IN"


class MemberType(Enum):
    """Types of UDT members."""
    BOOL = "BOOL"
    INT = "INT"
    DINT = "DINT"
    REAL = "REAL"
    STRING = "STRING"
    UDT = "UDT"
    ARRAY = "ARRAY"


@dataclass
class UDTMember:
    """Represents a member of a UDT structure."""
    name: str
    data_type: MemberType
    description: str = ""
    default_value: Any = None
    nested_udt: Optional[str] = None  # Name of nested UDT if applicable
    array_dimensions: List[int] = field(default_factory=list)
    is_required: bool = True
    access_pattern: str = ""  # How this member is typically accessed
    
    def get_full_type(self) -> str:
        """Get the full type description including arrays."""
        base_type = self.nested_udt if self.nested_udt else self.data_type.value
        if self.array_dimensions:
            dims = ','.join(str(d) for d in self.array_dimensions)
            return f"{base_type}[{dims}]"
        return base_type


@dataclass
class UDTDefinition:
    """Represents a UDT definition with all its members."""
    name: str
    udt_type: UDTType
    members: Dict[str, UDTMember] = field(default_factory=dict)
    description: str = ""
    size_bytes: int = 0
    inherits_from: Optional[str] = None
    is_built_in: bool = False
    usage_count: int = 0
    
    def add_member(self, member: UDTMember) -> None:
        """Add a member to this UDT definition."""
        self.members[member.name] = member
        
    def get_member_path(self, member_name: str) -> Optional[str]:
        """Get the full path to a member (handles nested access)."""
        if member_name in self.members:
            return f"{self.name}.{member_name}"
        
        # Check for nested member access (e.g., "SubUDT.Member")
        parts = member_name.split('.', 1)
        if len(parts) == 2 and parts[0] in self.members:
            nested_member = self.members[parts[0]]
            if nested_member.nested_udt:
                return f"{self.name}.{member_name}"
        
        return None


@dataclass
class MemberAccess:
    """Represents an access to a UDT member in code."""
    tag_name: str
    member_path: str
    access_type: str  # READ, WRITE, READ_WRITE
    instruction: str  # The instruction that accesses this member
    routine_name: str
    rung_number: int
    context: str = ""  # Additional context about the access
    
    @property
    def full_path(self) -> str:
        """Get the full access path."""
        return f"{self.tag_name}.{self.member_path}"


@dataclass
class UDTInstance:
    """Represents an instance of a UDT in the program."""
    name: str
    udt_definition: str
    tag_type: str = "Base"
    scope: str = "Controller"  # Controller, Program, or specific scope
    initial_values: Dict[str, Any] = field(default_factory=dict)
    member_accesses: List[MemberAccess] = field(default_factory=list)
    
    def add_member_access(self, access: MemberAccess) -> None:
        """Add a member access to this instance."""
        self.member_accesses.append(access)
        
    def get_accessed_members(self) -> Set[str]:
        """Get all members accessed in this instance."""
        return {access.member_path for access in self.member_accesses}


@dataclass
class UDTRelationship:
    """Represents a relationship between UDTs or UDT instances."""
    source: str
    target: str
    relationship_type: str  # CONTAINS, REFERENCES, INHERITS, ACCESSES
    strength: float = 1.0  # Relationship strength (0-1)
    context: str = ""
    

class UDTAnalyzer:
    """Comprehensive analyzer for UDT structures and usage patterns."""
    
    def __init__(self):
        self.udt_definitions: Dict[str, UDTDefinition] = {}
        self.udt_instances: Dict[str, UDTInstance] = {}
        self.member_accesses: List[MemberAccess] = []
        self.udt_relationships: List[UDTRelationship] = []
        self._initialize_built_in_udts()
        
    def _initialize_built_in_udts(self) -> None:
        """Initialize built-in UDT definitions (TIMER, COUNTER, etc.)."""
        # Timer UDT definition
        timer_udt = UDTDefinition(
            name="TIMER",
            udt_type=UDTType.TIMER,
            description="Built-in Timer data type",
            size_bytes=12,
            is_built_in=True
        )
        
        timer_members = [
            UDTMember("PRE", MemberType.DINT, "Preset value", 0),
            UDTMember("ACC", MemberType.DINT, "Accumulated value", 0),
            UDTMember("EN", MemberType.BOOL, "Enable bit", False),
            UDTMember("TT", MemberType.BOOL, "Timing bit", False),
            UDTMember("DN", MemberType.BOOL, "Done bit", False)
        ]
        
        for member in timer_members:
            timer_udt.add_member(member)
        
        self.udt_definitions["TIMER"] = timer_udt
        
        # Counter UDT definition
        counter_udt = UDTDefinition(
            name="COUNTER",
            udt_type=UDTType.COUNTER,
            description="Built-in Counter data type",
            size_bytes=12,
            is_built_in=True
        )
        
        counter_members = [
            UDTMember("PRE", MemberType.DINT, "Preset value", 0),
            UDTMember("ACC", MemberType.DINT, "Accumulated value", 0),
            UDTMember("CU", MemberType.BOOL, "Count-up enable bit", False),
            UDTMember("CD", MemberType.BOOL, "Count-down enable bit", False),
            UDTMember("DN", MemberType.BOOL, "Done bit", False),
            UDTMember("OV", MemberType.BOOL, "Overflow bit", False),
            UDTMember("UN", MemberType.BOOL, "Underflow bit", False)
        ]
        
        for member in counter_members:
            counter_udt.add_member(member)
            
        self.udt_definitions["COUNTER"] = counter_udt
        
    def analyze_udt_definitions(self, data_types_xml) -> None:
        """Analyze UDT definitions from XML data types section."""
        if not data_types_xml:
            logger.info("No DataTypes section found in L5X file")
            return
            
        try:
            for data_type in data_types_xml.findall('.//DataType'):
                udt_name = data_type.get('Name', '')
                if not udt_name:
                    continue
                    
                logger.info(f"Analyzing UDT definition: {udt_name}")
                
                udt_def = UDTDefinition(
                    name=udt_name,
                    udt_type=UDTType.CUSTOM,
                    description=data_type.get('Description', ''),
                    is_built_in=False
                )
                
                # Analyze UDT members
                for member in data_type.findall('.//Member'):
                    member_name = member.get('Name', '')
                    member_data_type = member.get('DataType', '')
                    
                    if not member_name or not member_data_type:
                        continue
                        
                    # Determine member type
                    if member_data_type in ['BOOL']:
                        member_type = MemberType.BOOL
                    elif member_data_type in ['INT']:
                        member_type = MemberType.INT
                    elif member_data_type in ['DINT']:
                        member_type = MemberType.DINT
                    elif member_data_type in ['REAL']:
                        member_type = MemberType.REAL
                    elif member_data_type in ['STRING']:
                        member_type = MemberType.STRING
                    else:
                        member_type = MemberType.UDT
                        
                    udt_member = UDTMember(
                        name=member_name,
                        data_type=member_type,
                        description=member.get('Description', ''),
                        nested_udt=member_data_type if member_type == MemberType.UDT else None
                    )
                    
                    # Handle array dimensions
                    dimensions = member.get('Dimensions')
                    if dimensions:
                        try:
                            udt_member.array_dimensions = [int(d.strip()) for d in dimensions.split(',')]
                            udt_member.data_type = MemberType.ARRAY
                        except ValueError:
                            logger.warning(f"Invalid array dimensions for {member_name}: {dimensions}")
                    
                    udt_def.add_member(udt_member)
                    
                self.udt_definitions[udt_name] = udt_def
                logger.info(f"Added UDT definition '{udt_name}' with {len(udt_def.members)} members")
                
        except Exception as e:
            logger.error(f"Error analyzing UDT definitions: {e}")
            
    def analyze_tag_instances(self, tags_xml) -> None:
        """Analyze UDT instances from XML tags section."""
        if not tags_xml:
            logger.info("No Tags section found in L5X file")
            return
            
        try:
            for tag in tags_xml.findall('.//Tag'):
                tag_name = tag.get('Name', '')
                data_type = tag.get('DataType', '')
                tag_type = tag.get('TagType', 'Base')
                
                if not tag_name or not data_type:
                    continue
                
                # Check if this is a UDT instance or built-in type we care about
                if data_type in self.udt_definitions or data_type in ['TIMER', 'COUNTER']:
                    logger.debug(f"Found UDT instance: {tag_name} of type {data_type}")
                    
                    instance = UDTInstance(
                        name=tag_name,
                        udt_definition=data_type,
                        tag_type=tag_type,
                        scope="Controller"  # Could be enhanced to detect actual scope
                    )
                    
                    # Parse initial values if present
                    data_element = tag.find('.//Data')
                    if data_element is not None:
                        self._parse_initial_values(instance, data_element)
                    
                    self.udt_instances[tag_name] = instance
                    
                    # Update usage count in definition
                    if data_type in self.udt_definitions:
                        self.udt_definitions[data_type].usage_count += 1
                        
                # Handle array tags with UDT elements
                dimensions = tag.get('Dimensions')
                if dimensions and data_type in self.udt_definitions:
                    logger.debug(f"Found UDT array: {tag_name}[{dimensions}] of type {data_type}")
                    # Could create multiple instances for array elements
                    
        except Exception as e:
            logger.error(f"Error analyzing UDT instances: {e}")
            
    def _parse_initial_values(self, instance: UDTInstance, data_element) -> None:
        """Parse initial values from XML data element."""
        try:
            # This is a simplified parser - real implementation would handle
            # the complex data format used in L5X files
            for child in data_element:
                if child.tag == 'DataValue':
                    member = child.get('Member')
                    value = child.get('Value')  
                    if member and value is not None:
                        instance.initial_values[member] = value
        except Exception as e:
            logger.error(f"Error parsing initial values for {instance.name}: {e}")
            
    def analyze_member_accesses(self, routine_analysis) -> None:
        """Analyze member access patterns from routine analysis."""
        if not routine_analysis:
            logger.info("No routine analysis provided for member access analysis")
            return
            
        logger.info("Analyzing UDT member access patterns...")
        
        try:
            # Access routine analysis through integration layer
            for routine_name, routine_data in routine_analysis.analyzed_routines.items():
                for rung_idx, rung in enumerate(routine_data.rungs):
                    self._analyze_rung_member_accesses(rung, routine_name, rung_idx)
                    
            logger.info(f"Found {len(self.member_accesses)} member accesses")
            
        except Exception as e:
            logger.error(f"Error analyzing member accesses: {e}")
            
    def _analyze_rung_member_accesses(self, rung, routine_name: str, rung_number: int) -> None:
        """Analyze member accesses in a single rung."""
        rung_text = rung.get('Text', '')
        if not rung_text:
            return
            
        # Pattern to match UDT member accesses (Tag.Member format)
        member_access_pattern = r'([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?)\.((?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)|(?:Data\.[0-9]+))'
        
        matches = re.finditer(member_access_pattern, rung_text)
        
        for match in matches:
            tag_name = match.group(1)
            member_path = match.group(2)
            
            # Determine access type based on instruction context
            access_type = self._determine_access_type(rung_text, match.start())
            
            # Find the instruction that contains this access
            instruction = self._find_containing_instruction(rung_text, match.start())
            
            access = MemberAccess(
                tag_name=tag_name,
                member_path=member_path,
                access_type=access_type,
                instruction=instruction,
                routine_name=routine_name,
                rung_number=rung_number,
                context=rung_text[max(0, match.start()-20):match.end()+20]
            )
            
            self.member_accesses.append(access)
            
            # Add to instance if it exists
            if tag_name in self.udt_instances:
                self.udt_instances[tag_name].add_member_access(access)
                
    def _determine_access_type(self, rung_text: str, position: int) -> str:
        """Determine if a member access is READ, WRITE, or READ_WRITE."""
        # Look for instruction patterns around the access
        before_text = rung_text[max(0, position-30):position]
        after_text = rung_text[position:position+30]
        
        # Write operations
        if any(instr in before_text or instr in after_text for instr in ['OTE', 'OTL', 'OTU', 'MOV', 'CPT']):
            return "WRITE"
        
        # Read operations  
        if any(instr in before_text or instr in after_text for instr in ['XIC', 'XIO', 'EQU', 'NEQ', 'GRT', 'LES']):
            return "READ"
            
        # Timer/Counter instructions might read and write different members
        if any(instr in before_text or instr in after_text for instr in ['TON', 'TOF', 'RTO', 'CTU', 'CTD']):
            return "READ_WRITE"
            
        return "READ"  # Default assumption
        
    def _find_containing_instruction(self, rung_text: str, position: int) -> str:
        """Find the instruction that contains this member access."""
        # Look for instruction patterns around the position
        instruction_pattern = r'([A-Z]{2,4})\s*\('
        
        # Search backwards and forwards for instructions
        before_text = rung_text[max(0, position-50):position+50]
        
        matches = list(re.finditer(instruction_pattern, before_text))
        if matches:
            # Find the closest instruction
            closest_match = min(matches, key=lambda m: abs(m.start() - (position - max(0, position-50))))
            return closest_match.group(1)
        
        return "UNKNOWN"
        
    def analyze_udt_relationships(self) -> None:
        """Analyze relationships between UDTs and instances."""
        logger.info("Analyzing UDT relationships...")
        
        # Inheritance relationships
        for udt_name, udt_def in self.udt_definitions.items():
            if udt_def.inherits_from:
                relationship = UDTRelationship(
                    source=udt_name,
                    target=udt_def.inherits_from,
                    relationship_type="INHERITS",
                    strength=1.0,
                    context=f"UDT {udt_name} inherits from {udt_def.inherits_from}"
                )
                self.udt_relationships.append(relationship)
                
        # Containment relationships (nested UDTs)
        for udt_name, udt_def in self.udt_definitions.items():
            for member_name, member in udt_def.members.items():
                if member.nested_udt and member.nested_udt in self.udt_definitions:
                    relationship = UDTRelationship(
                        source=udt_name,
                        target=member.nested_udt,
                        relationship_type="CONTAINS",
                        strength=0.8,
                        context=f"UDT {udt_name} contains {member.nested_udt} as member {member_name}"
                    )
                    self.udt_relationships.append(relationship)
                    
        # Usage relationships (instances accessing other instances)
        access_counts = defaultdict(int)
        for access in self.member_accesses:
            access_counts[(access.tag_name, access.routine_name)] += 1
            
        # Create relationships based on co-occurrence in routines
        routine_instances = defaultdict(set)
        for access in self.member_accesses:
            routine_instances[access.routine_name].add(access.tag_name)
            
        for routine_name, instances in routine_instances.items():
            instances_list = list(instances)
            for i in range(len(instances_list)):
                for j in range(i + 1, len(instances_list)):
                    # Calculate relationship strength based on access frequency
                    strength = min(1.0, (access_counts[(instances_list[i], routine_name)] + 
                                       access_counts[(instances_list[j], routine_name)]) / 10.0)
                    
                    relationship = UDTRelationship(
                        source=instances_list[i],
                        target=instances_list[j],
                        relationship_type="ACCESSES",
                        strength=strength,
                        context=f"Both accessed in routine {routine_name}"
                    )
                    self.udt_relationships.append(relationship)
                    
        logger.info(f"Found {len(self.udt_relationships)} UDT relationships")
        
    def get_udt_usage_summary(self) -> Dict[str, Dict]:
        """Get a summary of UDT usage across the program."""
        summary = {}
        
        for udt_name, udt_def in self.udt_definitions.items():
            instances = [inst for inst in self.udt_instances.values() 
                        if inst.udt_definition == udt_name]
            
            member_usage = defaultdict(int)
            for instance in instances:
                for access in instance.member_accesses:
                    member_usage[access.member_path] += 1
                    
            summary[udt_name] = {
                'definition': udt_def,
                'instance_count': len(instances),
                'total_accesses': sum(len(inst.member_accesses) for inst in instances),
                'member_usage': dict(member_usage),
                'most_used_member': max(member_usage.items(), key=lambda x: x[1])[0] if member_usage else None,
                'instances': [inst.name for inst in instances]
            }
            
        return summary
        
    def get_member_access_patterns(self) -> Dict[str, List[MemberAccess]]:
        """Get member access patterns organized by tag."""
        patterns = defaultdict(list)
        
        for access in self.member_accesses:
            patterns[access.tag_name].append(access)
            
        return dict(patterns)
        
    def find_unused_udt_members(self) -> Dict[str, List[str]]:
        """Find UDT members that are never accessed."""
        unused = {}
        
        for udt_name, udt_def in self.udt_definitions.items():
            if udt_def.is_built_in:
                continue  # Skip built-in types
                
            used_members = set()
            
            # Collect all used members from instances
            for instance in self.udt_instances.values():
                if instance.udt_definition == udt_name:
                    used_members.update(instance.get_accessed_members())
                    
            # Find unused members
            all_members = set(udt_def.members.keys())
            unused_members = all_members - used_members
            
            if unused_members:
                unused[udt_name] = list(unused_members)
                
        return unused
        
    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get comprehensive analysis metrics."""
        return {
            'total_udt_definitions': len(self.udt_definitions),
            'custom_udts': len([udt for udt in self.udt_definitions.values() if not udt.is_built_in]),
            'built_in_udts': len([udt for udt in self.udt_definitions.values() if udt.is_built_in]),
            'total_instances': len(self.udt_instances),
            'total_member_accesses': len(self.member_accesses),
            'total_relationships': len(self.udt_relationships),
            'average_members_per_udt': sum(len(udt.members) for udt in self.udt_definitions.values()) / len(self.udt_definitions) if self.udt_definitions else 0,
            'most_used_udt': max(self.udt_definitions.items(), key=lambda x: x[1].usage_count)[0] if self.udt_definitions else None,
            'access_patterns': len(self.get_member_access_patterns()),
            'unused_members': sum(len(members) for members in self.find_unused_udt_members().values())
        }
