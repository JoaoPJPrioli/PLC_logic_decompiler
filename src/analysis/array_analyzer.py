"""
Step 15: Array Handling
Advanced analysis of array tags and indexed access patterns in PLC systems.

This module provides comprehensive array analysis including:
- Array tag definition parsing and structure analysis
- Multi-dimensional array support with bounds checking
- Indexed access pattern detection and analysis
- Array element relationship mapping
- Array usage optimization analysis
- Dynamic vs static index analysis
- Array boundary analysis and safety checking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from enum import Enum
import re
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ArrayType(Enum):
    """Types of arrays supported."""
    BOOL_ARRAY = "BOOL_ARRAY"
    INT_ARRAY = "INT_ARRAY"
    DINT_ARRAY = "DINT_ARRAY"
    REAL_ARRAY = "REAL_ARRAY"
    STRING_ARRAY = "STRING_ARRAY"
    UDT_ARRAY = "UDT_ARRAY"
    IO_ARRAY = "IO_ARRAY"
    NESTED_ARRAY = "NESTED_ARRAY"


class IndexType(Enum):
    """Types of array indexing."""
    STATIC = "STATIC"  # Constant index like [5]
    DYNAMIC = "DYNAMIC"  # Variable index like [i]
    EXPRESSION = "EXPRESSION"  # Computed index like [i+1]
    RANGE = "RANGE"  # Range access like [0..7]
    UNKNOWN = "UNKNOWN"


@dataclass
class ArrayDimension:
    """Represents a single dimension of an array."""
    size: int
    lower_bound: int = 0
    upper_bound: Optional[int] = None
    is_dynamic: bool = False
    
    def __post_init__(self):
        """Calculate upper bound if not provided."""
        if self.upper_bound is None:
            self.upper_bound = self.lower_bound + self.size - 1
            
    def is_valid_index(self, index: int) -> bool:
        """Check if an index is valid for this dimension."""
        return self.lower_bound <= index <= self.upper_bound
        
    def get_range(self) -> Tuple[int, int]:
        """Get the valid range for this dimension."""
        return (self.lower_bound, self.upper_bound)


@dataclass
class ArrayIndex:
    """Represents an array index access."""
    dimension: int  # Which dimension (0-based)
    index_value: Union[int, str]  # The index value or expression
    index_type: IndexType
    source_instruction: str = ""
    routine_name: str = ""
    rung_number: int = 0
    is_bounds_checked: bool = False
    
    def is_static(self) -> bool:
        """Check if this is a static index."""
        return self.index_type == IndexType.STATIC
        
    def get_static_value(self) -> Optional[int]:
        """Get the static index value if available."""
        if self.is_static() and isinstance(self.index_value, int):
            return self.index_value
        return None


@dataclass
class ArrayAccess:
    """Represents an access to an array element."""
    array_name: str
    indices: List[ArrayIndex]
    access_type: str  # READ, WRITE, READ_WRITE
    instruction: str
    routine_name: str
    rung_number: int
    context: str = ""
    
    @property
    def full_path(self) -> str:
        """Get the full array access path."""
        index_str = "][".join([str(idx.index_value) for idx in self.indices])
        return f"{self.array_name}[{index_str}]"
        
    @property
    def dimension_count(self) -> int:
        """Get the number of dimensions accessed."""
        return len(self.indices)
        
    def is_fully_static(self) -> bool:
        """Check if all indices are static."""
        return all(idx.is_static() for idx in self.indices)


@dataclass
class ArrayDefinition:
    """Represents an array tag definition."""
    name: str
    array_type: ArrayType
    element_data_type: str
    dimensions: List[ArrayDimension]
    description: str = ""
    scope: str = "Controller"
    initial_values: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    
    @property
    def total_elements(self) -> int:
        """Calculate total number of elements in the array."""
        total = 1
        for dim in self.dimensions:
            total *= dim.size
        return total
        
    @property
    def dimension_count(self) -> int:
        """Get the number of dimensions."""
        return len(self.dimensions)
        
    def is_valid_access(self, indices: List[int]) -> bool:
        """Check if a set of indices represents a valid access."""
        if len(indices) != len(self.dimensions):
            return False
        return all(
            dim.is_valid_index(idx) 
            for dim, idx in zip(self.dimensions, indices)
        )
        
    def get_element_count_by_dimension(self) -> List[int]:
        """Get element count for each dimension."""
        return [dim.size for dim in self.dimensions]
        
    def get_bounds_by_dimension(self) -> List[Tuple[int, int]]:
        """Get bounds for each dimension."""
        return [dim.get_range() for dim in self.dimensions]


@dataclass
class ArrayUsagePattern:
    """Represents usage patterns for an array."""
    array_name: str
    total_accesses: int = 0
    static_accesses: int = 0
    dynamic_accesses: int = 0
    read_accesses: int = 0
    write_accesses: int = 0
    unique_indices: Set[Tuple[int, ...]] = field(default_factory=set)
    accessed_elements: Set[str] = field(default_factory=set)
    unused_elements: Set[str] = field(default_factory=set)
    bounds_violations: List[ArrayAccess] = field(default_factory=list)
    
    @property
    def usage_ratio(self) -> float:
        """Calculate the ratio of accessed elements to total elements."""
        if not hasattr(self, '_total_elements') or self._total_elements == 0:
            return 0.0
        return len(self.accessed_elements) / self._total_elements
        
    @property
    def static_ratio(self) -> float:
        """Calculate the ratio of static accesses to total accesses."""
        if self.total_accesses == 0:
            return 0.0
        return self.static_accesses / self.total_accesses


@dataclass
class ArrayRelationship:
    """Represents a relationship between arrays."""
    source_array: str
    target_array: str
    relationship_type: str  # INDEXED_BY, PARALLEL_ACCESS, LOOP_COUPLED
    strength: float = 1.0
    context: str = ""
    routine_names: Set[str] = field(default_factory=set)


class ArrayAnalyzer:
    """Comprehensive analyzer for array structures and usage patterns."""
    
    def __init__(self):
        self.array_definitions: Dict[str, ArrayDefinition] = {}
        self.array_accesses: List[ArrayAccess] = []
        self.array_usage_patterns: Dict[str, ArrayUsagePattern] = {}
        self.array_relationships: List[ArrayRelationship] = []
        
    def analyze_array_definitions(self, tags_xml) -> None:
        """Analyze array definitions from XML tags section."""
        if not tags_xml:
            logger.info("No Tags section found in L5X file")
            return
            
        try:
            for tag in tags_xml.findall('.//Tag'):
                tag_name = tag.get('Name', '')
                data_type = tag.get('DataType', '')
                dimensions = tag.get('Dimensions', '')
                
                if not tag_name or not dimensions:
                    continue
                    
                logger.debug(f"Found array tag: {tag_name} with dimensions {dimensions}")
                
                # Parse dimensions
                dimension_list = self._parse_dimensions(dimensions)
                if not dimension_list:
                    continue
                    
                # Determine array type
                array_type = self._determine_array_type(data_type)
                
                # Create array definition
                array_def = ArrayDefinition(
                    name=tag_name,
                    array_type=array_type,
                    element_data_type=data_type,
                    dimensions=dimension_list,
                    description=tag.get('Description', ''),
                    scope="Controller"  # Could be enhanced to detect actual scope
                )
                
                # Parse initial values if present
                data_element = tag.find('.//Data')
                if data_element is not None:
                    self._parse_array_initial_values(array_def, data_element)
                    
                self.array_definitions[tag_name] = array_def
                logger.info(f"Added array definition '{tag_name}' with {array_def.dimension_count}D, {array_def.total_elements} elements")
                
        except Exception as e:
            logger.error(f"Error analyzing array definitions: {e}")
            
    def _parse_dimensions(self, dimensions_str: str) -> List[ArrayDimension]:
        """Parse dimension string into ArrayDimension objects."""
        try:
            dimensions = []
            
            # Handle single dimension (most common case)
            if ',' not in dimensions_str:
                size = int(dimensions_str.strip())
                dimensions.append(ArrayDimension(size=size))
            else:
                # Handle multi-dimensional arrays
                for dim_str in dimensions_str.split(','):
                    size = int(dim_str.strip())
                    dimensions.append(ArrayDimension(size=size))
                    
            return dimensions
            
        except ValueError as e:
            logger.warning(f"Invalid dimensions format: {dimensions_str} - {e}")
            return []
            
    def _determine_array_type(self, data_type: str) -> ArrayType:
        """Determine the array type based on element data type."""
        type_mapping = {
            'BOOL': ArrayType.BOOL_ARRAY,
            'INT': ArrayType.INT_ARRAY,
            'DINT': ArrayType.DINT_ARRAY,
            'REAL': ArrayType.REAL_ARRAY,
            'STRING': ArrayType.STRING_ARRAY,
            'SINT': ArrayType.INT_ARRAY,  # Treat as INT array
        }
        
        if data_type in type_mapping:
            return type_mapping[data_type]
        elif 'Local:' in data_type or 'INPUT' in data_type or 'OUTPUT' in data_type:
            return ArrayType.IO_ARRAY
        else:
            return ArrayType.UDT_ARRAY  # Assume UDT for unknown types
            
    def _parse_array_initial_values(self, array_def: ArrayDefinition, data_element) -> None:
        """Parse initial values from XML data element."""
        try:
            # Look for Element tags with Index and Value attributes
            for element in data_element.findall('.//Element'):
                index = element.get('Index', '')
                value = element.get('Value', '')
                
                if index and value is not None:
                    # Clean up index format (remove brackets)
                    clean_index = index.strip('[]')
                    array_def.initial_values[clean_index] = value
                    
        except Exception as e:
            logger.error(f"Error parsing initial values for {array_def.name}: {e}")
            
    def analyze_array_accesses(self, routine_analysis) -> None:
        """Analyze array access patterns from routine analysis."""
        if not routine_analysis:
            logger.info("No routine analysis provided for array access analysis")
            return
            
        logger.info("Analyzing array access patterns...")
        
        try:
            # Access routine analysis through integration layer
            for routine_name, routine_data in routine_analysis.analyzed_routines.items():
                for rung_idx, rung in enumerate(routine_data.rungs):
                    self._analyze_rung_array_accesses(rung, routine_name, rung_idx)
                    
            logger.info(f"Found {len(self.array_accesses)} array accesses")
            
            # Build usage patterns
            self._build_usage_patterns()
            
        except Exception as e:
            logger.error(f"Error analyzing array accesses: {e}")
            
    def _analyze_rung_array_accesses(self, rung, routine_name: str, rung_number: int) -> None:
        """Analyze array accesses in a single rung."""
        rung_text = rung.get('Text', '')
        if not rung_text:
            return
            
        # Pattern to match array accesses (Tag[index] or Tag[index1,index2] format)
        array_access_pattern = r'([A-Za-z_][A-Za-z0-9_.:]*)\[([^\]]+)\]'
        
        matches = re.finditer(array_access_pattern, rung_text)
        
        for match in matches:
            array_name = match.group(1)
            indices_str = match.group(2)
            
            # Parse indices
            indices = self._parse_access_indices(indices_str, rung_text, match.start())
            
            if not indices:
                continue
                
            # Determine access type
            access_type = self._determine_access_type(rung_text, match.start())
            
            # Find containing instruction
            instruction = self._find_containing_instruction(rung_text, match.start())
            
            # Create array access
            access = ArrayAccess(
                array_name=array_name,
                indices=indices,
                access_type=access_type,
                instruction=instruction,
                routine_name=routine_name,
                rung_number=rung_number,
                context=rung_text[max(0, match.start()-20):match.end()+20]
            )
            
            self.array_accesses.append(access)
            
    def _parse_access_indices(self, indices_str: str, rung_text: str, position: int) -> List[ArrayIndex]:
        """Parse array access indices from string."""
        indices = []
        
        try:
            # Split by comma for multi-dimensional access
            index_parts = [part.strip() for part in indices_str.split(',')]
            
            for dim, index_part in enumerate(index_parts):
                # Determine index type and value
                if index_part.isdigit():
                    # Static integer index
                    index_value = int(index_part)
                    index_type = IndexType.STATIC
                elif self._is_variable_name(index_part):
                    # Dynamic variable index
                    index_value = index_part
                    index_type = IndexType.DYNAMIC
                elif any(op in index_part for op in ['+', '-', '*', '/', '(', ')']):
                    # Expression index
                    index_value = index_part
                    index_type = IndexType.EXPRESSION
                else:
                    # Unknown index type
                    index_value = index_part
                    index_type = IndexType.UNKNOWN
                    
                # Find source instruction
                instruction = self._find_containing_instruction(rung_text, position)
                
                array_index = ArrayIndex(
                    dimension=dim,
                    index_value=index_value,
                    index_type=index_type,
                    source_instruction=instruction
                )
                
                indices.append(array_index)
                
        except Exception as e:
            logger.warning(f"Error parsing indices '{indices_str}': {e}")
            
        return indices
        
    def _is_variable_name(self, name: str) -> bool:
        """Check if a string looks like a variable name."""
        return re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name) is not None
        
    def _determine_access_type(self, rung_text: str, position: int) -> str:
        """Determine if an array access is READ, WRITE, or READ_WRITE."""
        # Look for instruction patterns around the access
        before_text = rung_text[max(0, position-30):position]
        after_text = rung_text[position:position+30]
        
        # Write operations
        if any(instr in before_text or instr in after_text for instr in ['OTE', 'OTL', 'OTU', 'MOV', 'CPT', 'COP']):
            return "WRITE"
        
        # Read operations  
        if any(instr in before_text or instr in after_text for instr in ['XIC', 'XIO', 'EQU', 'NEQ', 'GRT', 'LES']):
            return "READ"
            
        # Array-specific instructions
        if any(instr in before_text or instr in after_text for instr in ['COP', 'FLL', 'FAL']):
            return "READ_WRITE"
            
        return "READ"  # Default assumption
        
    def _find_containing_instruction(self, rung_text: str, position: int) -> str:
        """Find the instruction that contains this array access."""
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
        
    def _build_usage_patterns(self) -> None:
        """Build usage patterns for all arrays."""
        # Initialize patterns for all defined arrays
        for array_name, array_def in self.array_definitions.items():
            pattern = ArrayUsagePattern(array_name=array_name)
            pattern._total_elements = array_def.total_elements
            self.array_usage_patterns[array_name] = pattern
            
        # Process all accesses
        for access in self.array_accesses:
            array_name = access.array_name
            
            # Create pattern if not exists (for arrays not in definitions)
            if array_name not in self.array_usage_patterns:
                pattern = ArrayUsagePattern(array_name=array_name)
                self.array_usage_patterns[array_name] = pattern
            else:
                pattern = self.array_usage_patterns[array_name]
                
            # Update pattern statistics
            pattern.total_accesses += 1
            
            if access.is_fully_static():
                pattern.static_accesses += 1
                # Add to unique indices if all static
                static_indices = tuple(idx.get_static_value() for idx in access.indices)
                if all(idx is not None for idx in static_indices):
                    pattern.unique_indices.add(static_indices)
            else:
                pattern.dynamic_accesses += 1
                
            # Count access types
            if access.access_type.upper() in ['READ', 'read']:
                pattern.read_accesses += 1
            elif access.access_type.upper() == 'WRITE':
                pattern.write_accesses += 1
                
            # Track accessed elements
            pattern.accessed_elements.add(access.full_path)
            
        # Identify unused elements
        self._identify_unused_elements()
        
        logger.info(f"Built usage patterns for {len(self.array_usage_patterns)} arrays")
        
    def _identify_unused_elements(self) -> None:
        """Identify unused array elements."""
        for array_name, pattern in self.array_usage_patterns.items():
            if array_name not in self.array_definitions:
                continue
                
            array_def = self.array_definitions[array_name]
            
            # For single-dimension arrays, check individual elements
            if array_def.dimension_count == 1:
                dim = array_def.dimensions[0]
                all_elements = set(f"{array_name}[{i}]" for i in range(dim.lower_bound, dim.upper_bound + 1))
                pattern.unused_elements = all_elements - pattern.accessed_elements
                
    def analyze_array_relationships(self) -> None:
        """Analyze relationships between arrays."""
        logger.info("Analyzing array relationships...")
        
        # Group accesses by routine
        routine_arrays = defaultdict(set)
        for access in self.array_accesses:
            routine_arrays[access.routine_name].add(access.array_name)
            
        # Find co-occurrence relationships
        for routine_name, arrays in routine_arrays.items():
            arrays_list = list(arrays)
            for i in range(len(arrays_list)):
                for j in range(i + 1, len(arrays_list)):
                    array1, array2 = arrays_list[i], arrays_list[j]
                    
                    # Calculate relationship strength based on co-occurrence frequency
                    co_occurrences = sum(1 for access in self.array_accesses 
                                       if access.routine_name == routine_name and 
                                       access.array_name in [array1, array2])
                    
                    strength = min(1.0, co_occurrences / 10.0)
                    
                    relationship = ArrayRelationship(
                        source_array=array1,
                        target_array=array2,
                        relationship_type="PARALLEL_ACCESS",
                        strength=strength,
                        context=f"Co-accessed in routine {routine_name}",
                        routine_names={routine_name}
                    )
                    
                    self.array_relationships.append(relationship)
                    
        # Find index relationships (arrays used as indices for other arrays)
        self._find_index_relationships()
        
        logger.info(f"Found {len(self.array_relationships)} array relationships")
        
    def _find_index_relationships(self) -> None:
        """Find arrays that are used as indices for other arrays."""
        for access in self.array_accesses:
            for index in access.indices:
                if index.index_type == IndexType.DYNAMIC:
                    # Check if the index variable is an array element
                    index_var = str(index.index_value)
                    
                    # Look for array access pattern in index
                    if '[' in index_var and ']' in index_var:
                        # Extract array name from index
                        index_array_match = re.match(r'([A-Za-z_][A-Za-z0-9_.:]*)\[', index_var)
                        if index_array_match:
                            index_array = index_array_match.group(1)
                            
                            relationship = ArrayRelationship(
                                source_array=index_array,
                                target_array=access.array_name,
                                relationship_type="INDEXED_BY",
                                strength=1.0,
                                context=f"Array {index_array} used as index for {access.array_name}",
                                routine_names={access.routine_name}
                            )
                            
                            self.array_relationships.append(relationship)
                            
    def get_array_analysis_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of array analysis."""
        return {
            'total_arrays': len(self.array_definitions),
            'total_accesses': len(self.array_accesses),
            'total_relationships': len(self.array_relationships),
            'arrays_by_type': self._count_arrays_by_type(),
            'access_patterns': self._analyze_access_patterns(),
            'usage_efficiency': self._calculate_usage_efficiency(),
            'bounds_safety': self._analyze_bounds_safety()
        }
        
    def _count_arrays_by_type(self) -> Dict[str, int]:
        """Count arrays by type."""
        type_counts = defaultdict(int)
        for array_def in self.array_definitions.values():
            type_counts[array_def.array_type.value] += 1
        return dict(type_counts)
        
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze access patterns across all arrays."""
        total_static = sum(pattern.static_accesses for pattern in self.array_usage_patterns.values())
        total_dynamic = sum(pattern.dynamic_accesses for pattern in self.array_usage_patterns.values())
        total_accesses = total_static + total_dynamic
        
        return {
            'static_accesses': total_static,
            'dynamic_accesses': total_dynamic,
            'static_ratio': total_static / total_accesses if total_accesses > 0 else 0,
            'most_accessed_array': max(self.array_usage_patterns.items(), 
                                     key=lambda x: x[1].total_accesses)[0] if self.array_usage_patterns else None
        }
        
    def _calculate_usage_efficiency(self) -> Dict[str, Any]:
        """Calculate usage efficiency metrics."""
        usage_ratios = [pattern.usage_ratio for pattern in self.array_usage_patterns.values() 
                       if hasattr(pattern, '_total_elements') and pattern._total_elements > 0]
        
        if not usage_ratios:
            return {'average_usage_ratio': 0.0, 'arrays_with_unused_elements': 0}
            
        return {
            'average_usage_ratio': sum(usage_ratios) / len(usage_ratios),
            'arrays_with_unused_elements': sum(1 for pattern in self.array_usage_patterns.values() 
                                             if pattern.unused_elements)
        }
        
    def _analyze_bounds_safety(self) -> Dict[str, Any]:
        """Analyze bounds safety for array accesses."""
        safe_accesses = 0
        potentially_unsafe = 0
        
        for access in self.array_accesses:
            if access.array_name in self.array_definitions:
                array_def = self.array_definitions[access.array_name]
                
                if access.is_fully_static():
                    # Check bounds for static accesses
                    indices = [idx.get_static_value() for idx in access.indices]
                    if all(idx is not None for idx in indices) and array_def.is_valid_access(indices):
                        safe_accesses += 1
                    else:
                        potentially_unsafe += 1
                        access.is_bounds_checked = False
                else:
                    # Dynamic accesses are potentially unsafe
                    potentially_unsafe += 1
                    
        return {
            'safe_accesses': safe_accesses,
            'potentially_unsafe_accesses': potentially_unsafe,
            'safety_ratio': safe_accesses / len(self.array_accesses) if self.array_accesses else 1.0
        }
