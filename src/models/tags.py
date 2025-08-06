"""
Tag models for PLC analysis
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

class TagScope(Enum):
    """Tag scope enumeration"""
    CONTROLLER = "controller"
    PROGRAM = "program"
    IO = "io"

class DataType(Enum):
    """Common PLC data types"""
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
    CONTROL = "CONTROL"
    UDT = "UDT"  # User Defined Type
    UNKNOWN = "UNKNOWN"

@dataclass
class Tag:
    """
    Represents a PLC tag with all its properties
    """
    name: str
    data_type: str
    scope: TagScope
    description: str = ""
    value: Any = None
    program_name: str = ""
    external_access: str = "Read/Write"
    constant: bool = False
    array_dimensions: List[int] = field(default_factory=list)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.modified_date is None:
            self.modified_date = datetime.now()
    
    @property
    def is_array(self) -> bool:
        """Check if tag is an array"""
        return len(self.array_dimensions) > 0
    
    @property
    def array_size(self) -> int:
        """Calculate total array size"""
        if not self.is_array:
            return 1
        
        size = 1
        for dim in self.array_dimensions:
            size *= dim
        return size
    
    @property
    def full_name(self) -> str:
        """Get full qualified tag name"""
        if self.scope == TagScope.PROGRAM and self.program_name:
            return f"{self.program_name}.{self.name}"
        return self.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tag to dictionary"""
        return {
            'name': self.name,
            'data_type': self.data_type,
            'scope': self.scope.value,
            'description': self.description,
            'value': self.value,
            'program_name': self.program_name,
            'external_access': self.external_access,
            'constant': self.constant,
            'array_dimensions': self.array_dimensions,
            'is_array': self.is_array,
            'array_size': self.array_size,
            'full_name': self.full_name,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'modified_date': self.modified_date.isoformat() if self.modified_date else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tag':
        """Create tag from dictionary"""
        scope_str = data.get('scope', 'controller')
        scope = TagScope.CONTROLLER
        try:
            scope = TagScope(scope_str)
        except ValueError:
            pass
        
        created_date = None
        if data.get('created_date'):
            try:
                created_date = datetime.fromisoformat(data['created_date'])
            except (ValueError, TypeError):
                pass
        
        modified_date = None
        if data.get('modified_date'):
            try:
                modified_date = datetime.fromisoformat(data['modified_date'])
            except (ValueError, TypeError):
                pass
        
        return cls(
            name=data.get('name', ''),
            data_type=data.get('data_type', 'UNKNOWN'),
            scope=scope,
            description=data.get('description', ''),
            value=data.get('value'),
            program_name=data.get('program_name', ''),
            external_access=data.get('external_access', 'Read/Write'),
            constant=data.get('constant', False),
            array_dimensions=data.get('array_dimensions', []),
            created_date=created_date,
            modified_date=modified_date
        )

@dataclass
class TagCollection:
    """
    Collection of tags with search and filter capabilities
    """
    tags: List[Tag] = field(default_factory=list)
    name: str = "Tag Collection"
    created_date: datetime = field(default_factory=datetime.now)
    
    def add_tag(self, tag: Tag) -> None:
        """Add a tag to the collection"""
        self.tags.append(tag)
    
    def remove_tag(self, tag_name: str) -> bool:
        """Remove a tag by name"""
        for i, tag in enumerate(self.tags):
            if tag.name == tag_name:
                del self.tags[i]
                return True
        return False
    
    def find_tag(self, tag_name: str) -> Optional[Tag]:
        """Find a tag by name"""
        for tag in self.tags:
            if tag.name == tag_name:
                return tag
        return None
    
    def filter_by_scope(self, scope: TagScope) -> List[Tag]:
        """Filter tags by scope"""
        return [tag for tag in self.tags if tag.scope == scope]
    
    def filter_by_data_type(self, data_type: str) -> List[Tag]:
        """Filter tags by data type"""
        return [tag for tag in self.tags if tag.data_type == data_type]
    
    def filter_by_program(self, program_name: str) -> List[Tag]:
        """Filter tags by program name"""
        return [tag for tag in self.tags if tag.program_name == program_name]
    
    def get_array_tags(self) -> List[Tag]:
        """Get all array tags"""
        return [tag for tag in self.tags if tag.is_array]
    
    def get_constant_tags(self) -> List[Tag]:
        """Get all constant tags"""
        return [tag for tag in self.tags if tag.constant]
    
    def search_tags(self, search_term: str) -> List[Tag]:
        """Search tags by name or description"""
        search_term = search_term.lower()
        results = []
        
        for tag in self.tags:
            if (search_term in tag.name.lower() or 
                search_term in tag.description.lower()):
                results.append(tag)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        data_type_counts = {}
        scope_counts = {}
        
        for tag in self.tags:
            # Count data types
            data_type_counts[tag.data_type] = data_type_counts.get(tag.data_type, 0) + 1
            
            # Count scopes
            scope_str = tag.scope.value
            scope_counts[scope_str] = scope_counts.get(scope_str, 0) + 1
        
        return {
            'total_tags': len(self.tags),
            'array_tags': len(self.get_array_tags()),
            'constant_tags': len(self.get_constant_tags()),
            'data_type_distribution': data_type_counts,
            'scope_distribution': scope_counts,
            'collection_name': self.name,
            'created_date': self.created_date.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary"""
        return {
            'name': self.name,
            'created_date': self.created_date.isoformat(),
            'tags': [tag.to_dict() for tag in self.tags],
            'statistics': self.get_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TagCollection':
        """Create collection from dictionary"""
        created_date = datetime.now()
        if data.get('created_date'):
            try:
                created_date = datetime.fromisoformat(data['created_date'])
            except (ValueError, TypeError):
                pass
        
        collection = cls(
            name=data.get('name', 'Tag Collection'),
            created_date=created_date
        )
        
        # Add tags
        for tag_data in data.get('tags', []):
            tag = Tag.from_dict(tag_data)
            collection.add_tag(tag)
        
        return collection

@dataclass 
class TagAnalysisResult:
    """
    Result of tag analysis operations
    """
    collection: TagCollection
    analysis_type: str
    results: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary"""
        return {
            'analysis_type': self.analysis_type,
            'results': self.results,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'collection_statistics': self.collection.get_statistics()
        }

class TagAnalyzer:
    """
    Analyzer for tag collections
    """
    
    @staticmethod
    def analyze_usage_patterns(collection: TagCollection) -> TagAnalysisResult:
        """Analyze tag usage patterns"""
        start_time = datetime.now()
        
        results = {
            'most_common_data_types': {},
            'scope_distribution': {},
            'array_usage': {
                'total_arrays': 0,
                'total_array_elements': 0,
                'largest_array': 0
            },
            'naming_patterns': {
                'prefixes': {},
                'suffixes': {}
            }
        }
        
        # Analyze data types
        stats = collection.get_statistics()
        results['most_common_data_types'] = stats['data_type_distribution']
        results['scope_distribution'] = stats['scope_distribution']
        
        # Analyze arrays
        array_tags = collection.get_array_tags()
        results['array_usage']['total_arrays'] = len(array_tags)
        
        total_elements = 0
        largest_array = 0
        for tag in array_tags:
            size = tag.array_size
            total_elements += size
            if size > largest_array:
                largest_array = size
        
        results['array_usage']['total_array_elements'] = total_elements
        results['array_usage']['largest_array'] = largest_array
        
        # Analyze naming patterns
        prefixes = {}
        suffixes = {}
        
        for tag in collection.tags:
            name = tag.name
            if '_' in name:
                parts = name.split('_')
                if len(parts) >= 2:
                    prefix = parts[0]
                    suffix = parts[-1]
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1
                    suffixes[suffix] = suffixes.get(suffix, 0) + 1
        
        results['naming_patterns']['prefixes'] = prefixes
        results['naming_patterns']['suffixes'] = suffixes
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return TagAnalysisResult(
            collection=collection,
            analysis_type="usage_patterns",
            results=results,
            execution_time=execution_time
        )
    
    @staticmethod
    def find_unused_tags(collection: TagCollection) -> TagAnalysisResult:
        """Find potentially unused tags (simplified analysis)"""
        start_time = datetime.now()
        
        # This is a simplified implementation
        # In a real scenario, you'd need to analyze the actual PLC logic
        results = {
            'potentially_unused': [],
            'analysis_method': 'simplified_heuristic',
            'note': 'This analysis uses simplified heuristics and may not be completely accurate'
        }
        
        for tag in collection.tags:
            # Simple heuristic: tags without descriptions might be unused
            # This is just an example - real analysis would be much more complex
            if not tag.description and not tag.constant:
                results['potentially_unused'].append({
                    'name': tag.name,
                    'data_type': tag.data_type,
                    'scope': tag.scope.value,
                    'reason': 'No description provided'
                })
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return TagAnalysisResult(
            collection=collection,
            analysis_type="unused_tags",
            results=results,
            execution_time=execution_time
        )
