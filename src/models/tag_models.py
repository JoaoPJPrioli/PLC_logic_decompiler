"""
Data models for L5X parser components.

This module defines the core data structures used throughout the L5X parsing
and processing pipeline, including Tag, IOTag, and related data classes.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class TagType(Enum):
    """Enumeration of PLC tag types."""
    CONTROLLER = "controller"
    PROGRAM = "program"
    IO_INPUT = "io_input"
    IO_OUTPUT = "io_output"
    UDT = "udt"
    ARRAY = "array"


class DataType(Enum):
    """Enumeration of PLC data types."""
    BOOL = "BOOL"
    SINT = "SINT"
    INT = "INT"
    DINT = "DINT"
    LINT = "LINT"
    REAL = "REAL"
    LREAL = "LREAL"
    STRING = "STRING"
    TIMER = "TIMER"
    COUNTER = "COUNTER"
    UDT = "UDT"  # User Defined Type
    UNKNOWN = "UNKNOWN"


@dataclass
class TagDimension:
    """Represents array dimensions for a tag."""
    size: int
    index_start: int = 0
    
    def __str__(self) -> str:
        if self.index_start == 0:
            return f"[{self.size}]"
        else:
            return f"[{self.index_start}..{self.index_start + self.size - 1}]"


@dataclass
class Tag:
    """
    Base class for PLC tags.
    
    Represents a tag in the PLC with all its metadata including name,
    data type, description, and scope information.
    """
    name: str
    data_type: str
    tag_type: TagType = TagType.CONTROLLER
    description: Optional[str] = None
    dimensions: Optional[List[TagDimension]] = None
    scope: Optional[str] = None  # Program name for program-scoped tags
    constant: bool = False
    external_access: str = "Read/Write"
    radix: Optional[str] = None
    value: Optional[Any] = None
    comments: Dict[str, str] = field(default_factory=dict)  # For array element comments
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.dimensions is None:
            self.dimensions = []
    
    @property
    def is_array(self) -> bool:
        """Check if this tag is an array."""
        return len(self.dimensions) > 0
    
    @property
    def is_multidimensional_array(self) -> bool:
        """Check if this is a multi-dimensional array."""
        return len(self.dimensions) > 1
    
    @property
    def total_elements(self) -> int:
        """Calculate total number of elements in array."""
        if not self.is_array:
            return 1
        
        total = 1
        for dim in self.dimensions:
            total *= dim.size
        return total
    
    @property
    def canonical_name(self) -> str:
        """Get the canonicalized tag name."""
        if self.scope and self.tag_type == TagType.PROGRAM:
            return f"{self.scope}.{self.name}"
        return self.name
    
    def get_array_element_name(self, indices: List[int]) -> str:
        """
        Get the name for a specific array element.
        
        Args:
            indices: List of array indices
            
        Returns:
            str: Formatted array element name
        """
        if not self.is_array:
            return self.canonical_name
        
        if len(indices) != len(self.dimensions):
            raise ValueError(f"Expected {len(self.dimensions)} indices, got {len(indices)}")
        
        index_str = ",".join(str(idx) for idx in indices)
        return f"{self.canonical_name}[{index_str}]"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tag to dictionary representation."""
        return {
            'name': self.name,
            'canonical_name': self.canonical_name,
            'data_type': self.data_type,
            'tag_type': self.tag_type.value,
            'description': self.description,
            'dimensions': [{'size': d.size, 'index_start': d.index_start} for d in self.dimensions],
            'scope': self.scope,
            'constant': self.constant,
            'external_access': self.external_access,
            'radix': self.radix,
            'value': self.value,
            'is_array': self.is_array,
            'total_elements': self.total_elements,
            'comments': self.comments
        }


@dataclass
class IOTag(Tag):
    """
    I/O tag with connection information.
    
    Extends the base Tag class with I/O-specific information like
    module name, connection details, and bit-level information.
    """
    module_name: Optional[str] = None
    connection_name: Optional[str] = None
    rpi: Optional[int] = None  # Requested Packet Interval
    connection_type: Optional[str] = None  # Input/Output
    bit_number: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization for IOTag."""
        super().__post_init__()
        
        # Set appropriate tag type based on connection
        if self.connection_type:
            if self.connection_type.lower() == "input":
                self.tag_type = TagType.IO_INPUT
            elif self.connection_type.lower() == "output":
                self.tag_type = TagType.IO_OUTPUT
    
    @property
    def full_io_address(self) -> str:
        """Get the full I/O address string."""
        base_name = self.canonical_name
        if self.bit_number is not None:
            return f"{base_name}.{self.bit_number}"
        return base_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert IOTag to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'module_name': self.module_name,
            'connection_name': self.connection_name,
            'rpi': self.rpi,
            'connection_type': self.connection_type,
            'bit_number': self.bit_number,
            'full_io_address': self.full_io_address
        })
        return base_dict


@dataclass
class UDTMember:
    """Represents a member of a User Defined Type."""
    name: str
    data_type: str
    description: Optional[str] = None
    dimensions: Optional[List[TagDimension]] = None
    hidden: bool = False
    external_access: str = "Read/Write"
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.dimensions is None:
            self.dimensions = []
    
    @property
    def is_array(self) -> bool:
        """Check if this member is an array."""
        return len(self.dimensions) > 0


@dataclass  
class UDTTag(Tag):
    """
    User Defined Type tag.
    
    Represents a UDT with its member structure and provides methods
    for expanding UDT members into individual tags.
    """
    members: List[UDTMember] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization for UDTTag."""
        super().__post_init__()
        self.tag_type = TagType.UDT
    
    def get_member_by_name(self, member_name: str) -> Optional[UDTMember]:
        """Get a UDT member by name."""
        for member in self.members:
            if member.name == member_name:
                return member
        return None
    
    def get_member_tag_name(self, member_name: str) -> str:
        """Get the full tag name for a UDT member."""
        return f"{self.canonical_name}.{member_name}"
    
    def expand_members(self) -> List[Tag]:
        """
        Expand UDT members into individual Tag objects.
        
        Returns:
            List[Tag]: List of Tag objects for each UDT member
        """
        expanded_tags = []
        
        for member in self.members:
            member_tag = Tag(
                name=f"{self.name}.{member.name}",
                data_type=member.data_type,
                tag_type=self.tag_type,
                description=member.description,
                dimensions=member.dimensions.copy() if member.dimensions else [],
                scope=self.scope,
                constant=self.constant,
                external_access=member.external_access
            )
            expanded_tags.append(member_tag)
        
        return expanded_tags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert UDTTag to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'members': [
                {
                    'name': m.name,
                    'data_type': m.data_type,
                    'description': m.description,
                    'dimensions': [{'size': d.size, 'index_start': d.index_start} for d in m.dimensions],
                    'hidden': m.hidden,
                    'external_access': m.external_access,
                    'is_array': m.is_array
                }
                for m in self.members
            ]
        })
        return base_dict


def normalize_data_type(data_type: str) -> str:
    """
    Normalize data type string to standard format.
    
    Args:
        data_type: Raw data type string from L5X
        
    Returns:
        str: Normalized data type string
    """
    if not data_type:
        return DataType.UNKNOWN.value
    
    # Convert to uppercase and strip whitespace
    normalized = data_type.upper().strip()
    
    # Handle common variations
    type_mappings = {
        'BOOLEAN': 'BOOL',
        'INTEGER': 'INT',
        'DOUBLE': 'LREAL',
        'FLOAT': 'REAL'
    }
    
    return type_mappings.get(normalized, normalized)


def parse_dimensions_string(dimensions_str: str) -> List[TagDimension]:
    """
    Parse dimension string into TagDimension objects.
    
    Args:
        dimensions_str: String like "10" or "5,3" for multi-dimensional arrays
        
    Returns:
        List[TagDimension]: List of parsed dimensions
    """
    if not dimensions_str:
        return []
    
    dimensions = []
    for dim_str in dimensions_str.split(','):
        dim_str = dim_str.strip()
        if dim_str.isdigit():
            dimensions.append(TagDimension(size=int(dim_str)))
        else:
            # Handle ranges like "0..9"
            if '..' in dim_str:
                start, end = dim_str.split('..')
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                size = end_idx - start_idx + 1
                dimensions.append(TagDimension(size=size, index_start=start_idx))
            else:
                # Default to treating as size
                try:
                    size = int(dim_str)
                    dimensions.append(TagDimension(size=size))
                except ValueError:
                    # Invalid dimension, skip
                    continue
    
    return dimensions


def create_tag_from_xml_element(tag_element, tag_type: TagType = TagType.CONTROLLER, scope: Optional[str] = None) -> Tag:
    """
    Create a Tag object from an XML element.
    
    Args:
        tag_element: XML element containing tag data
        tag_type: Type of tag being created
        scope: Scope for program-scoped tags
        
    Returns:
        Tag: Created tag object
    """
    name = tag_element.attrib.get('Name', '')
    data_type = normalize_data_type(tag_element.attrib.get('DataType', ''))
    
    # Parse dimensions if present
    dimensions = []
    dimensions_str = tag_element.attrib.get('Dimensions', '')
    if dimensions_str:
        dimensions = parse_dimensions_string(dimensions_str)
    
    # Extract description
    description = None
    desc_element = tag_element.find('Description')
    if desc_element is not None and desc_element.text:
        description = desc_element.text.strip()
    
    # Extract other attributes
    constant = tag_element.attrib.get('Constant', 'false').lower() == 'true'
    external_access = tag_element.attrib.get('ExternalAccess', 'Read/Write')
    radix = tag_element.attrib.get('Radix')
    
    # Extract comments for array elements
    comments = {}
    comments_element = tag_element.find('Comments')
    if comments_element is not None:
        for comment in comments_element.findall('Comment'):
            operand = comment.attrib.get('Operand', '')
            comment_text = comment.text.strip() if comment.text else ''
            if operand and comment_text:
                comments[operand] = comment_text
    
    return Tag(
        name=name,
        data_type=data_type,
        tag_type=tag_type,
        description=description,
        dimensions=dimensions,
        scope=scope,
        constant=constant,
        external_access=external_access,
        radix=radix,
        comments=comments
    )
