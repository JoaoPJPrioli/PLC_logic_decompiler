"""
Tag Canonicalization System

This module provides comprehensive tag name canonicalization and validation
for PLC tags across different scopes (controller, program, I/O).

Author: GitHub Copilot
Date: December 2024
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field

from .tag_models import Tag
from .io_models import IOModule, IOPoint


class TagScope(Enum):
    """Tag scope types"""
    CONTROLLER = "Controller"
    PROGRAM = "Program" 
    IO = "IO"
    LOCAL = "Local"
    GLOBAL = "Global"


class TagConflictType(Enum):
    """Types of tag name conflicts"""
    DUPLICATE_NAME = "Duplicate"
    SCOPE_COLLISION = "ScopeCollision"
    RESERVED_WORD = "Reserved"
    INVALID_SYNTAX = "InvalidSyntax"
    CASE_MISMATCH = "CaseMismatch"


@dataclass
class TagReference:
    """Represents a tag reference with full context"""
    original_name: str          # Original tag name as found
    canonical_name: str         # Standardized canonical name
    scope: TagScope            # Tag scope
    scope_qualifier: Optional[str]  # Program name, module name, etc.
    data_type: Optional[str]    # Tag data type
    description: Optional[str]  # Tag description
    is_array: bool = False     # Whether tag is an array
    array_dimensions: List[int] = field(default_factory=list)
    source_location: Optional[str] = None  # Where the tag was found
    
    def get_qualified_name(self) -> str:
        """Get fully qualified tag name"""
        if self.scope_qualifier:
            if self.scope == TagScope.PROGRAM:
                return f"Program:{self.scope_qualifier}.{self.original_name}"
            elif self.scope == TagScope.IO:
                return f"{self.scope_qualifier}:{self.original_name}"
        return self.original_name
    
    def get_display_name(self) -> str:
        """Get human-readable display name"""
        qualified = self.get_qualified_name()
        if self.is_array and self.array_dimensions:
            dims = "x".join(str(d) for d in self.array_dimensions)
            return f"{qualified}[{dims}]"
        return qualified


@dataclass
class TagConflict:
    """Represents a tag name conflict"""
    conflict_type: TagConflictType
    tag1: TagReference
    tag2: Optional[TagReference] = None
    description: str = ""
    severity: str = "Warning"  # Warning, Error, Info
    
    def get_resolution_suggestion(self) -> str:
        """Get suggested resolution for the conflict"""
        if self.conflict_type == TagConflictType.DUPLICATE_NAME:
            return f"Consider renaming one of the conflicting tags or use scope qualifiers"
        elif self.conflict_type == TagConflictType.SCOPE_COLLISION:
            return f"Use fully qualified names to avoid scope ambiguity"
        elif self.conflict_type == TagConflictType.RESERVED_WORD:
            return f"Rename tag to avoid reserved word '{self.tag1.original_name}'"
        elif self.conflict_type == TagConflictType.INVALID_SYNTAX:
            return f"Fix invalid characters in tag name '{self.tag1.original_name}'"
        elif self.conflict_type == TagConflictType.CASE_MISMATCH:
            return f"Standardize case for consistency"
        return "Review tag naming convention"


class TagCanonicalizer:
    """Main tag canonicalization and validation system"""
    
    # Reserved words that shouldn't be used as tag names
    RESERVED_WORDS = {
        'and', 'or', 'not', 'xor', 'true', 'false', 'if', 'then', 'else',
        'for', 'while', 'repeat', 'until', 'case', 'of', 'begin', 'end',
        'function', 'procedure', 'var', 'const', 'type', 'program',
        'controller', 'module', 'routine', 'rung', 'tag', 'local', 'global'
    }
    
    # Valid tag name pattern (letters, numbers, underscore, must start with letter/underscore)
    VALID_TAG_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    
    def __init__(self):
        self.tag_references: Dict[str, TagReference] = {}
        self.canonical_map: Dict[str, str] = {}  # original -> canonical
        self.reverse_map: Dict[str, List[str]] = {}  # canonical -> [originals]
        self.conflicts: List[TagConflict] = []
        self.scope_prefixes: Dict[TagScope, str] = {
            TagScope.CONTROLLER: "",
            TagScope.PROGRAM: "Program",
            TagScope.IO: "IO",
            TagScope.LOCAL: "Local",
            TagScope.GLOBAL: "Global"
        }
    
    def canonicalize_tag_name(self, 
                             original_name: str, 
                             scope: TagScope = TagScope.CONTROLLER,
                             scope_qualifier: Optional[str] = None) -> str:
        """
        Convert a tag name to its canonical form.
        
        Args:
            original_name: Original tag name
            scope: Tag scope
            scope_qualifier: Program name, module name, etc.
            
        Returns:
            Canonical tag name
        """
        # Remove any existing scope qualifiers
        clean_name = original_name
        if ':' in clean_name:
            clean_name = clean_name.split(':')[-1]
        if '.' in clean_name and scope == TagScope.PROGRAM:
            # Handle Program:ProgramName.TagName format
            parts = clean_name.split('.')
            if len(parts) > 1:
                clean_name = parts[-1]
        
        # Basic canonicalization rules
        canonical = clean_name.strip()
        
        # Remove invalid characters (keep only letters, numbers, underscores)
        canonical = re.sub(r'[^a-zA-Z0-9_]', '_', canonical)
        
        # Ensure starts with letter or underscore
        if canonical and not canonical[0].isalpha() and canonical[0] != '_':
            canonical = f"_{canonical}"
        
        # Handle empty names
        if not canonical:
            canonical = "UnnamedTag"
        
        # Add scope prefix if needed for disambiguation
        if scope != TagScope.CONTROLLER and scope_qualifier:
            scope_prefix = self.scope_prefixes.get(scope, "")
            if scope_prefix:
                canonical = f"{scope_prefix}_{scope_qualifier}_{canonical}"
        
        # Ensure uniqueness by adding suffix if needed
        base_canonical = canonical
        counter = 1
        while canonical in self.reverse_map and original_name not in self.reverse_map[canonical]:
            canonical = f"{base_canonical}_{counter}"
            counter += 1
        
        return canonical
    
    def add_tag_reference(self, 
                         original_name: str,
                         scope: TagScope = TagScope.CONTROLLER,
                         scope_qualifier: Optional[str] = None,
                         data_type: Optional[str] = None,
                         description: Optional[str] = None,
                         is_array: bool = False,
                         array_dimensions: List[int] = None,
                         source_location: Optional[str] = None) -> TagReference:
        """
        Add a tag reference to the canonicalization system.
        
        Returns:
            TagReference object for the added tag
        """
        if array_dimensions is None:
            array_dimensions = []
        
        canonical_name = self.canonicalize_tag_name(original_name, scope, scope_qualifier)
        
        tag_ref = TagReference(
            original_name=original_name,
            canonical_name=canonical_name,
            scope=scope,
            scope_qualifier=scope_qualifier,
            data_type=data_type,
            description=description,
            is_array=is_array,
            array_dimensions=array_dimensions,
            source_location=source_location
        )
        
        # Store in our mappings
        full_key = f"{scope.value}:{scope_qualifier or ''}:{original_name}"
        self.tag_references[full_key] = tag_ref
        self.canonical_map[original_name] = canonical_name
        
        if canonical_name not in self.reverse_map:
            self.reverse_map[canonical_name] = []
        if original_name not in self.reverse_map[canonical_name]:
            self.reverse_map[canonical_name].append(original_name)
        
        return tag_ref
    
    def add_controller_tags(self, tags: Dict[str, Tag]):
        """Add controller-scoped tags"""
        for tag_name, tag in tags.items():
            self.add_tag_reference(
                original_name=tag_name,
                scope=TagScope.CONTROLLER,
                data_type=tag.data_type,
                description=tag.description,
                is_array=tag.is_array,
                array_dimensions=[d.size for d in tag.dimensions] if tag.dimensions else [],
                source_location="Controller"
            )
    
    def add_program_tags(self, tags: Dict[str, Tag]):
        """Add program-scoped tags"""
        for tag_name, tag in tags.items():
            # Extract program name from canonical name
            program_name = None
            if tag.scope and tag.canonical_name.startswith("Program:"):
                parts = tag.canonical_name.split(".")
                if len(parts) > 0:
                    program_part = parts[0].replace("Program:", "")
                    program_name = program_part
            
            self.add_tag_reference(
                original_name=tag.name,
                scope=TagScope.PROGRAM,
                scope_qualifier=program_name,
                data_type=tag.data_type,
                description=tag.description,
                is_array=tag.is_array,
                array_dimensions=[d.size for d in tag.dimensions] if tag.dimensions else [],
                source_location=f"Program:{program_name}" if program_name else "Program"
            )
    
    def add_io_modules(self, modules: Dict[str, IOModule]):
        """Add I/O module tags"""
        for module_name, module in modules.items():
            for connection_name, connection in module.connections.items():
                for operand, io_point in connection.io_points.items():
                    # Create tag name from I/O point
                    tag_name = f"{module_name}_{connection_name}_{operand.replace('.', '_')}"
                    
                    self.add_tag_reference(
                        original_name=tag_name,
                        scope=TagScope.IO,
                        scope_qualifier=module_name,
                        data_type=io_point.data_type or "BOOL",
                        description=io_point.comment,
                        is_array=False,
                        source_location=f"IO:{module_name}:{connection_name}"
                    )
    
    def validate_tags(self) -> List[TagConflict]:
        """
        Validate all tags and find conflicts.
        
        Returns:
            List of tag conflicts found
        """
        self.conflicts.clear()
        
        # Check for reserved words
        for tag_ref in self.tag_references.values():
            if tag_ref.original_name.lower() in self.RESERVED_WORDS:
                conflict = TagConflict(
                    conflict_type=TagConflictType.RESERVED_WORD,
                    tag1=tag_ref,
                    description=f"Tag name '{tag_ref.original_name}' is a reserved word",
                    severity="Warning"
                )
                self.conflicts.append(conflict)
        
        # Check for invalid syntax
        for tag_ref in self.tag_references.values():
            if not self.VALID_TAG_PATTERN.match(tag_ref.original_name):
                conflict = TagConflict(
                    conflict_type=TagConflictType.INVALID_SYNTAX,
                    tag1=tag_ref,
                    description=f"Tag name '{tag_ref.original_name}' contains invalid characters",
                    severity="Error"
                )
                self.conflicts.append(conflict)
        
        # Check for duplicates across different scopes
        name_to_refs = {}
        for tag_ref in self.tag_references.values():
            name = tag_ref.original_name.lower()
            if name not in name_to_refs:
                name_to_refs[name] = []
            name_to_refs[name].append(tag_ref)
        
        for name, refs in name_to_refs.items():
            if len(refs) > 1:
                # Check if they're in different scopes
                scopes = set(ref.scope for ref in refs)
                if len(scopes) > 1:
                    conflict = TagConflict(
                        conflict_type=TagConflictType.SCOPE_COLLISION,  
                        tag1=refs[0],
                        tag2=refs[1],
                        description=f"Tag name '{name}' exists in multiple scopes",
                        severity="Warning"
                    )
                    self.conflicts.append(conflict)
                else:
                    # Same scope, true duplicate
                    conflict = TagConflict(
                        conflict_type=TagConflictType.DUPLICATE_NAME,
                        tag1=refs[0],
                        tag2=refs[1],
                        description=f"Duplicate tag name '{name}' in same scope",
                        severity="Error"
                    )
                    self.conflicts.append(conflict)
        
        # Check for case mismatches
        canonical_to_cases = {}
        for tag_ref in self.tag_references.values():
            canonical = tag_ref.original_name.lower()
            if canonical not in canonical_to_cases:
                canonical_to_cases[canonical] = set()
            canonical_to_cases[canonical].add(tag_ref.original_name)
        
        for canonical, cases in canonical_to_cases.items():
            if len(cases) > 1:
                # Multiple case variations
                refs = [ref for ref in self.tag_references.values() 
                       if ref.original_name.lower() == canonical]
                if len(refs) >= 2:
                    conflict = TagConflict(
                        conflict_type=TagConflictType.CASE_MISMATCH,
                        tag1=refs[0],
                        tag2=refs[1],
                        description=f"Case variations of tag name: {', '.join(cases)}",
                        severity="Info"
                    )
                    self.conflicts.append(conflict)
        
        return self.conflicts
    
    def get_canonical_name(self, original_name: str) -> Optional[str]:
        """Get canonical name for an original tag name"""
        return self.canonical_map.get(original_name)
    
    def get_tag_reference(self, 
                         original_name: str, 
                         scope: TagScope = TagScope.CONTROLLER,
                         scope_qualifier: Optional[str] = None) -> Optional[TagReference]:
        """Get tag reference by original name and scope"""
        full_key = f"{scope.value}:{scope_qualifier or ''}:{original_name}"
        return self.tag_references.get(full_key)
    
    def search_tags(self, 
                   pattern: str, 
                   scope: Optional[TagScope] = None,
                   include_description: bool = True) -> List[TagReference]:
        """
        Search for tags matching a pattern.
        
        Args:
            pattern: Search pattern (supports wildcards with *)
            scope: Limit search to specific scope
            include_description: Include description in search
            
        Returns:
            List of matching tag references
        """
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        regex = re.compile(regex_pattern, re.IGNORECASE)
        
        results = []
        for tag_ref in self.tag_references.values():
            if scope and tag_ref.scope != scope:
                continue
            
            # Check name match
            if regex.search(tag_ref.original_name) or regex.search(tag_ref.canonical_name):
                results.append(tag_ref)
                continue
            
            # Check description match if requested
            if include_description and tag_ref.description:
                if regex.search(tag_ref.description):
                    results.append(tag_ref)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get canonicalization statistics"""
        scope_counts = {}
        for tag_ref in self.tag_references.values():
            scope = tag_ref.scope.value
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
        
        conflict_counts = {}
        for conflict in self.conflicts:
            conflict_type = conflict.conflict_type.value
            conflict_counts[conflict_type] = conflict_counts.get(conflict_type, 0) + 1
        
        return {
            'total_tags': len(self.tag_references),
            'canonical_mappings': len(self.canonical_map),
            'conflicts_found': len(self.conflicts),
            'tags_by_scope': scope_counts,
            'conflicts_by_type': conflict_counts,
            'unique_canonical_names': len(self.reverse_map),
            'reserved_word_violations': len([c for c in self.conflicts 
                                           if c.conflict_type == TagConflictType.RESERVED_WORD]),
            'syntax_errors': len([c for c in self.conflicts 
                                if c.conflict_type == TagConflictType.INVALID_SYNTAX])
        }
    
    def generate_cross_reference_table(self) -> Dict[str, Dict[str, Any]]:
        """Generate cross-reference table for all tags"""
        cross_ref = {}
        
        for tag_ref in self.tag_references.values():
            canonical = tag_ref.canonical_name
            if canonical not in cross_ref:
                cross_ref[canonical] = {
                    'canonical_name': canonical,
                    'references': [],
                    'scopes': set(),
                    'data_types': set(),
                    'descriptions': []
                }
            
            cross_ref[canonical]['references'].append({
                'original_name': tag_ref.original_name,
                'qualified_name': tag_ref.get_qualified_name(),
                'scope': tag_ref.scope.value,
                'scope_qualifier': tag_ref.scope_qualifier,
                'source_location': tag_ref.source_location
            })
            
            cross_ref[canonical]['scopes'].add(tag_ref.scope.value)
            if tag_ref.data_type:
                cross_ref[canonical]['data_types'].add(tag_ref.data_type)
            if tag_ref.description:
                cross_ref[canonical]['descriptions'].append(tag_ref.description)
        
        # Convert sets to lists for JSON serialization
        for entry in cross_ref.values():
            entry['scopes'] = list(entry['scopes'])
            entry['data_types'] = list(entry['data_types'])
        
        return cross_ref
