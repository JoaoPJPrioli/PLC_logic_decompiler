"""
Step 30: Enhanced Search and Discovery
Advanced search capabilities for PLC tags, logic patterns, and system analysis

This module provides comprehensive search and discovery functionality that builds
upon the ChromaDB integration to offer intelligent analysis, pattern recognition,
and advanced query capabilities for PLC systems.
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
import re
import math
from collections import defaultdict, Counter
import hashlib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from semantic.chromadb_integration import (
        PLCSemanticSearchEngine,
        SemanticSearchResult,
        SearchQuery,
        create_semantic_search_engine
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    print("Semantic search not available - using mock implementations")
    SEMANTIC_AVAILABLE = False

try:
    # Core PLC components
    from src.core.l5x_parser import L5XParser
    from src.analysis.ladder_logic_parser import LadderLogicParser
    from src.models.tags import Tag
    from src.models.knowledge_graph import PLCKnowledgeGraph
    CORE_IMPORTS_AVAILABLE = True
except ImportError:
    print("Core imports not available - using mock implementations")
    CORE_IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Advanced search filter configuration"""
    tag_types: Optional[List[str]] = None  # ['BOOL', 'REAL', 'DINT', etc.]
    scopes: Optional[List[str]] = None  # ['controller', 'program', etc.]
    instruction_types: Optional[List[str]] = None  # ['XIC', 'TON', 'MOV', etc.]
    complexity_range: Optional[Tuple[float, float]] = None  # (min, max)
    date_range: Optional[Tuple[datetime, datetime]] = None
    safety_related: Optional[bool] = None
    has_comments: Optional[bool] = None
    array_only: Optional[bool] = None
    udt_only: Optional[bool] = None


@dataclass
class DiscoveryResult:
    """Result from discovery analysis"""
    result_id: str
    discovery_type: str  # 'pattern', 'anomaly', 'optimization', 'relationship'
    title: str
    description: str
    confidence: float
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    related_components: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class PatternMatch:
    """Result from pattern matching analysis"""
    pattern_id: str
    pattern_name: str
    match_confidence: float
    components: List[str]
    locations: List[Dict[str, str]]  # program, routine, rung info
    pattern_description: str
    best_practices: List[str]
    potential_issues: List[str]


@dataclass
class TagRelationship:
    """Relationship between tags"""
    source_tag: str
    target_tag: str
    relationship_type: str  # 'writes_to', 'reads_from', 'enables', 'triggers'
    strength: float  # 0.0 to 1.0
    context: str  # instruction or logic context
    frequency: int  # how often this relationship occurs


class EnhancedSearchEngine:
    """Enhanced search engine with advanced discovery capabilities"""
    
    def __init__(self, semantic_engine: Optional[Any] = None, db_path: str = "./enhanced_search_db"):
        """Initialize enhanced search engine"""
        self.semantic_engine = semantic_engine
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Search indices and caches
        self.tag_index = {}  # Fast tag lookup
        self.instruction_index = {}  # Instruction pattern index
        self.relationship_cache = {}  # Cached relationships
        self.pattern_cache = {}  # Cached patterns
        
        # Discovery patterns and rules
        self.pattern_library = self._initialize_pattern_library()
        self.analysis_rules = self._initialize_analysis_rules()
        
        # Search statistics and optimization
        self.search_stats = defaultdict(int)
        self.query_cache = {}
        
        logger.info("Enhanced search engine initialized")
    
    def _initialize_pattern_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize library of known PLC patterns"""
        return {
            'start_stop_station': {
                'description': 'Start/Stop station with maintained output',
                'indicators': ['start', 'stop', 'run', 'motor'],
                'instructions': ['XIC', 'XIO', 'OTE', 'OTL'],
                'confidence_threshold': 0.7,
                'best_practices': [
                    'Use normally closed stop buttons',
                    'Implement proper seal-in logic',
                    'Include emergency stop in circuit'
                ],
                'potential_issues': [
                    'Missing emergency stop integration',
                    'Improper stop button wiring',
                    'No seal-in logic protection'
                ]
            },
            'timer_sequence': {
                'description': 'Sequential timing operations',
                'indicators': ['timer', 'delay', 'sequence', 'step'],
                'instructions': ['TON', 'TOF', 'RTO', 'XIC', 'OTE'],
                'confidence_threshold': 0.8,
                'best_practices': [
                    'Use consistent timing base',
                    'Document timing requirements',
                    'Implement timing bypass for maintenance'
                ],
                'potential_issues': [
                    'Timing conflicts in sequence',
                    'Missing reset conditions',
                    'Inadequate timing documentation'
                ]
            },
            'alarm_management': {
                'description': 'Alarm generation and management',
                'indicators': ['alarm', 'fault', 'error', 'warning'],
                'instructions': ['XIC', 'XIO', 'OTE', 'OSR', 'MOV'],
                'confidence_threshold': 0.6,
                'best_practices': [
                    'Categorize alarms by priority',
                    'Implement alarm acknowledgment',
                    'Log alarm history'
                ],
                'potential_issues': [
                    'Alarm flooding potential',
                    'Missing alarm priorities',
                    'No alarm acknowledgment logic'
                ]
            },
            'safety_interlock': {
                'description': 'Safety interlock and protection logic',
                'indicators': ['safety', 'interlock', 'guard', 'estop'],
                'instructions': ['XIC', 'XIO', 'AFI', 'OTE'],
                'confidence_threshold': 0.9,
                'best_practices': [
                    'Use hardwired safety circuits',
                    'Implement dual-channel monitoring',
                    'Regular safety system testing'
                ],
                'potential_issues': [
                    'Software-only safety implementation',
                    'Missing safety redundancy',
                    'Inadequate safety documentation'
                ]
            },
            'pid_control': {
                'description': 'PID control loop implementation',
                'indicators': ['pid', 'control', 'setpoint', 'process'],
                'instructions': ['PID', 'MOV', 'MUL', 'ADD', 'SUB'],
                'confidence_threshold': 0.8,
                'best_practices': [
                    'Tune PID parameters properly',
                    'Implement anti-windup protection',
                    'Monitor control loop performance'
                ],
                'potential_issues': [
                    'Poor PID tuning',
                    'Missing anti-windup logic',
                    'No control loop monitoring'
                ]
            }
        }
    
    def _initialize_analysis_rules(self) -> List[Dict[str, Any]]:
        """Initialize analysis rules for discovery"""
        return [
            {
                'rule_name': 'unused_tags',
                'description': 'Identify tags that are defined but never used',
                'severity': 'medium',
                'check_function': self._check_unused_tags
            },
            {
                'rule_name': 'missing_comments',
                'description': 'Find critical tags without documentation',
                'severity': 'low',
                'check_function': self._check_missing_comments
            },
            {
                'rule_name': 'naming_inconsistencies',
                'description': 'Detect inconsistent tag naming patterns',
                'severity': 'medium',
                'check_function': self._check_naming_inconsistencies
            },
            {
                'rule_name': 'complex_rungs',
                'description': 'Identify overly complex ladder rungs',
                'severity': 'medium',
                'check_function': self._check_complex_rungs
            },
            {
                'rule_name': 'safety_concerns',
                'description': 'Identify potential safety implementation issues',
                'severity': 'high',
                'check_function': self._check_safety_concerns
            },
            {
                'rule_name': 'performance_bottlenecks',
                'description': 'Find potential performance issues',
                'severity': 'medium',
                'check_function': self._check_performance_bottlenecks
            }
        ]
    
    async def advanced_search(self, query_text: str, 
                            filters: Optional[SearchFilter] = None,
                            search_types: List[str] = ['all'],
                            max_results: int = 20,
                            include_related: bool = True) -> List[Dict[str, Any]]:
        """Perform advanced search with filtering and relationship analysis"""
        logger.info(f"Performing advanced search: {query_text}")
        
        # Generate cache key
        cache_key = self._generate_cache_key(query_text, filters, search_types, max_results)
        
        # Check cache first
        if cache_key in self.query_cache:
            logger.info("Returning cached search results")
            return self.query_cache[cache_key]
        
        # Perform semantic search if available
        semantic_results = []
        if self.semantic_engine and SEMANTIC_AVAILABLE:
            try:
                semantic_query = SearchQuery(
                    query_text=query_text,
                    search_types=search_types,
                    max_results=max_results * 2,  # Get more for filtering
                    similarity_threshold=0.5
                )
                semantic_results = await self.semantic_engine.semantic_search(semantic_query)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # Apply filters to results
        filtered_results = self._apply_search_filters(semantic_results, filters)
        
        # Enhance results with additional analysis
        enhanced_results = []
        for result in filtered_results[:max_results]:
            enhanced_result = await self._enhance_search_result(result, include_related)
            enhanced_results.append(enhanced_result)
        
        # Cache results
        self.query_cache[cache_key] = enhanced_results
        self.search_stats['advanced_searches'] += 1
        
        logger.info(f"Advanced search completed: {len(enhanced_results)} results")
        return enhanced_results
    
    async def discover_patterns(self, context_data: Dict[str, Any] = None) -> List[PatternMatch]:
        """Discover known patterns in PLC logic"""
        logger.info("Starting pattern discovery analysis")
        
        pattern_matches = []
        
        # Get context data if not provided
        if context_data is None:
            context_data = await self._gather_context_data()
        
        # Analyze each pattern in the library
        for pattern_id, pattern_def in self.pattern_library.items():
            try:
                matches = await self._analyze_pattern(pattern_id, pattern_def, context_data)
                pattern_matches.extend(matches)
            except Exception as e:
                logger.error(f"Error analyzing pattern {pattern_id}: {e}")
        
        # Sort by confidence
        pattern_matches.sort(key=lambda x: x.match_confidence, reverse=True)
        
        logger.info(f"Pattern discovery completed: {len(pattern_matches)} patterns found")
        return pattern_matches
    
    async def analyze_tag_relationships(self, context_data: Dict[str, Any] = None) -> List[TagRelationship]:
        """Analyze relationships between tags"""
        logger.info("Starting tag relationship analysis")
        
        if context_data is None:
            context_data = await self._gather_context_data()
        
        relationships = []
        
        # Extract tags and instructions from context
        tags = context_data.get('tags', [])
        instructions = context_data.get('instructions', [])
        
        # Build tag usage map
        tag_usage = defaultdict(list)
        for instruction in instructions:
            for operand in instruction.get('operands', []):
                if any(tag['name'] == operand for tag in tags):
                    tag_usage[operand].append(instruction)
        
        # Find relationships between tags
        for tag1_name, tag1_instructions in tag_usage.items():
            for tag2_name, tag2_instructions in tag_usage.items():
                if tag1_name != tag2_name:
                    relationship = self._analyze_tag_pair(
                        tag1_name, tag1_instructions,
                        tag2_name, tag2_instructions
                    )
                    if relationship:
                        relationships.append(relationship)
        
        # Sort by relationship strength
        relationships.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"Tag relationship analysis completed: {len(relationships)} relationships found")
        return relationships
    
    async def discovery_analysis(self, analysis_types: List[str] = ['all']) -> List[DiscoveryResult]:
        """Perform comprehensive discovery analysis"""
        logger.info(f"Starting discovery analysis: {analysis_types}")
        
        discovery_results = []
        
        # Gather context data
        context_data = await self._gather_context_data()
        
        # Run analysis rules
        analysis_rules = self.analysis_rules
        if 'all' not in analysis_types:
            analysis_rules = [rule for rule in analysis_rules if rule['rule_name'] in analysis_types]
        
        for rule in analysis_rules:
            try:
                rule_results = await rule['check_function'](context_data)
                for result in rule_results:
                    result.metadata['rule_name'] = rule['rule_name']
                    result.metadata['severity'] = rule['severity']
                discovery_results.extend(rule_results)
            except Exception as e:
                logger.error(f"Error running analysis rule {rule['rule_name']}: {e}")
        
        # Sort by impact level and confidence
        impact_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        discovery_results.sort(
            key=lambda x: (impact_order.get(x.impact_level, 0), x.confidence),
            reverse=True
        )
        
        logger.info(f"Discovery analysis completed: {len(discovery_results)} findings")
        return discovery_results
    
    async def intelligent_recommendations(self, context_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate intelligent recommendations for PLC improvements"""
        logger.info("Generating intelligent recommendations")
        
        if context_data is None:
            context_data = await self._gather_context_data()
        
        recommendations = []
        
        # Pattern-based recommendations
        patterns = await self.discover_patterns(context_data)
        for pattern in patterns:
            if pattern.match_confidence > 0.7:
                recommendations.extend([
                    {
                        'type': 'best_practice',
                        'title': f"Improve {pattern.pattern_name} implementation",
                        'description': f"Based on detected {pattern.pattern_name} pattern",
                        'priority': 'medium',
                        'recommendations': pattern.best_practices,
                        'components': pattern.components
                    }
                ])
        
        # Discovery-based recommendations
        discoveries = await self.discovery_analysis(['safety_concerns', 'performance_bottlenecks'])
        for discovery in discoveries:
            if discovery.confidence > 0.6:
                recommendations.append({
                    'type': 'improvement',
                    'title': discovery.title,
                    'description': discovery.description,
                    'priority': discovery.impact_level,
                    'recommendations': discovery.recommendations,
                    'components': discovery.related_components
                })
        
        # Tag relationship recommendations
        relationships = await self.analyze_tag_relationships(context_data)
        strong_relationships = [r for r in relationships if r.strength > 0.8]
        if strong_relationships:
            recommendations.append({
                'type': 'optimization',
                'title': 'Optimize tag relationships',
                'description': f'Found {len(strong_relationships)} strong tag relationships that could be optimized',
                'priority': 'low',
                'recommendations': [
                    'Consider grouping related tags in UDTs',
                    'Optimize instruction placement for better performance',
                    'Document tag relationships for maintenance'
                ],
                'components': [r.source_tag for r in strong_relationships[:10]]
            })
        
        logger.info(f"Generated {len(recommendations)} intelligent recommendations")
        return recommendations
    
    def _apply_search_filters(self, results: List[Any], filters: Optional[SearchFilter]) -> List[Any]:
        """Apply search filters to results"""
        if not filters or not results:
            return results
        
        filtered_results = []
        
        for result in results:
            # Apply tag type filter
            if filters.tag_types and hasattr(result, 'metadata'):
                tag_type = result.metadata.get('tag_type') or result.metadata.get('data_type')
                if tag_type and tag_type not in filters.tag_types:
                    continue
            
            # Apply scope filter
            if filters.scopes and hasattr(result, 'metadata'):
                scope = result.metadata.get('scope')
                if scope and scope not in filters.scopes:
                    continue
            
            # Apply safety filter
            if filters.safety_related is not None and hasattr(result, 'content'):
                is_safety = any(keyword in result.content.lower() 
                              for keyword in ['safety', 'emergency', 'interlock', 'guard'])
                if filters.safety_related != is_safety:
                    continue
            
            # Apply comments filter
            if filters.has_comments is not None and hasattr(result, 'metadata'):
                has_comment = bool(result.metadata.get('description') or result.metadata.get('comment_text'))
                if filters.has_comments != has_comment:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    async def _enhance_search_result(self, result: Any, include_related: bool) -> Dict[str, Any]:
        """Enhance search result with additional analysis"""
        enhanced = {
            'original_result': result,
            'enhanced_metadata': {},
            'related_components': [],
            'analysis': {}
        }
        
        if hasattr(result, 'content'):
            # Analyze content complexity
            enhanced['analysis']['complexity_score'] = self._calculate_complexity_score(result.content)
            
            # Extract key concepts
            enhanced['analysis']['key_concepts'] = self._extract_key_concepts(result.content)
        
        if include_related and hasattr(result, 'metadata'):
            # Find related components
            related = await self._find_related_components(result.metadata)
            enhanced['related_components'] = related
        
        return enhanced
    
    async def _gather_context_data(self) -> Dict[str, Any]:
        """Gather context data for analysis"""
        # Mock context data - in real implementation this would gather from L5X parsing
        return {
            'tags': [
                {'name': 'Emergency_Stop', 'data_type': 'BOOL', 'scope': 'controller', 'description': 'Emergency stop button'},
                {'name': 'Motor_Run', 'data_type': 'BOOL', 'scope': 'controller', 'description': 'Motor run command'},
                {'name': 'Timer_Delay', 'data_type': 'TIMER', 'scope': 'program', 'description': 'Startup delay timer'},
                {'name': 'Production_Count', 'data_type': 'DINT', 'scope': 'controller', 'description': 'Production counter'},
                {'name': 'Safety_Gate', 'data_type': 'BOOL', 'scope': 'controller', 'description': 'Safety gate status'}
            ],
            'instructions': [
                {'type': 'XIC', 'operands': ['Emergency_Stop'], 'program': 'MainProgram', 'routine': 'SafetyRoutine'},
                {'type': 'XIC', 'operands': ['Safety_Gate'], 'program': 'MainProgram', 'routine': 'SafetyRoutine'},
                {'type': 'OTE', 'operands': ['Motor_Run'], 'program': 'MainProgram', 'routine': 'MotorControl'},
                {'type': 'TON', 'operands': ['Timer_Delay'], 'program': 'MainProgram', 'routine': 'StartupSequence'},
                {'type': 'CTU', 'operands': ['Production_Count'], 'program': 'MainProgram', 'routine': 'ProductionLogic'}
            ],
            'programs': [
                {'name': 'MainProgram', 'routines': ['SafetyRoutine', 'MotorControl', 'StartupSequence', 'ProductionLogic']}
            ]
        }
    
    async def _analyze_pattern(self, pattern_id: str, pattern_def: Dict[str, Any], 
                             context_data: Dict[str, Any]) -> List[PatternMatch]:
        """Analyze a specific pattern in the context data"""
        matches = []
        
        # Look for pattern indicators in tags and instructions
        indicator_matches = 0
        instruction_matches = 0
        components = []
        locations = []
        
        # Check tag names for indicators
        for tag in context_data.get('tags', []):
            tag_name_lower = tag['name'].lower()
            for indicator in pattern_def['indicators']:
                if indicator.lower() in tag_name_lower:
                    indicator_matches += 1
                    components.append(tag['name'])
        
        # Check instructions for pattern
        for instruction in context_data.get('instructions', []):
            if instruction['type'] in pattern_def['instructions']:
                instruction_matches += 1
                locations.append({
                    'program': instruction.get('program', 'Unknown'),
                    'routine': instruction.get('routine', 'Unknown'),
                    'instruction': instruction['type']
                })
        
        # Calculate confidence
        max_possible_matches = len(pattern_def['indicators']) + len(pattern_def['instructions'])
        actual_matches = indicator_matches + instruction_matches
        confidence = actual_matches / max_possible_matches if max_possible_matches > 0 else 0
        
        # Create match if confidence meets threshold
        if confidence >= pattern_def['confidence_threshold']:
            match = PatternMatch(
                pattern_id=pattern_id,
                pattern_name=pattern_def['description'],
                match_confidence=confidence,
                components=list(set(components)),
                locations=locations,
                pattern_description=pattern_def['description'],
                best_practices=pattern_def['best_practices'],
                potential_issues=pattern_def['potential_issues']
            )
            matches.append(match)
        
        return matches
    
    def _analyze_tag_pair(self, tag1_name: str, tag1_instructions: List[Dict],
                         tag2_name: str, tag2_instructions: List[Dict]) -> Optional[TagRelationship]:
        """Analyze relationship between two tags"""
        
        # Look for read/write patterns
        tag1_writes = [i for i in tag1_instructions if i.get('type') in ['OTE', 'OTL', 'OTU', 'MOV']]
        tag1_reads = [i for i in tag1_instructions if i.get('type') in ['XIC', 'XIO']]
        
        tag2_writes = [i for i in tag2_instructions if i.get('type') in ['OTE', 'OTL', 'OTU', 'MOV']]
        tag2_reads = [i for i in tag2_instructions if i.get('type') in ['XIC', 'XIO']]
        
        relationship_type = None
        strength = 0.0
        context = ""
        frequency = 0
        
        # Analyze relationship patterns
        if tag1_writes and tag2_reads:
            relationship_type = "enables"
            strength = min(len(tag1_writes), len(tag2_reads)) / max(len(tag1_writes), len(tag2_reads))
            context = "Output to input logic"
            frequency = min(len(tag1_writes), len(tag2_reads))
        elif tag1_reads and tag2_writes:
            relationship_type = "triggers"  
            strength = min(len(tag1_reads), len(tag2_writes)) / max(len(tag1_reads), len(tag2_writes))
            context = "Input to output logic"
            frequency = min(len(tag1_reads), len(tag2_writes))
        
        # Only return relationships with meaningful strength
        if relationship_type and strength > 0.3:
            return TagRelationship(
                source_tag=tag1_name,
                target_tag=tag2_name,
                relationship_type=relationship_type,
                strength=strength,
                context=context,
                frequency=frequency
            )
        
        return None
    
    async def _check_unused_tags(self, context_data: Dict[str, Any]) -> List[DiscoveryResult]:
        """Check for unused tags"""
        results = []
        
        tags = {tag['name']: tag for tag in context_data.get('tags', [])}
        used_tags = set()
        
        # Find used tags in instructions
        for instruction in context_data.get('instructions', []):
            for operand in instruction.get('operands', []):
                used_tags.add(operand)
        
        # Find unused tags
        unused_tags = set(tags.keys()) - used_tags
        
        if unused_tags:
            results.append(DiscoveryResult(
                result_id=f"unused_tags_{len(unused_tags)}",
                discovery_type="optimization",
                title=f"Found {len(unused_tags)} unused tags",
                description=f"Tags defined but never referenced in logic: {', '.join(list(unused_tags)[:5])}{'...' if len(unused_tags) > 5 else ''}",
                confidence=0.9,
                impact_level="medium",
                related_components=list(unused_tags),
                recommendations=[
                    "Remove unused tags to reduce memory usage",
                    "Review tag definitions for correctness",
                    "Consider if tags are used in other programs"
                ],
                metadata={'unused_count': len(unused_tags)},
                timestamp=datetime.now()
            ))
        
        return results
    
    async def _check_missing_comments(self, context_data: Dict[str, Any]) -> List[DiscoveryResult]:
        """Check for missing comments on important tags"""
        results = []
        
        tags_without_comments = []
        safety_tags_without_comments = []
        
        for tag in context_data.get('tags', []):
            if not tag.get('description'):
                tags_without_comments.append(tag['name'])
                
                # Check if it's safety-related
                if any(keyword in tag['name'].lower() 
                      for keyword in ['safety', 'emergency', 'interlock', 'stop']):
                    safety_tags_without_comments.append(tag['name'])
        
        if safety_tags_without_comments:
            results.append(DiscoveryResult(
                result_id="missing_safety_comments",
                discovery_type="documentation",
                title=f"Safety tags without comments: {len(safety_tags_without_comments)}",
                description=f"Critical safety tags lacking documentation: {', '.join(safety_tags_without_comments[:3])}{'...' if len(safety_tags_without_comments) > 3 else ''}",
                confidence=0.95,
                impact_level="high",
                related_components=safety_tags_without_comments,
                recommendations=[
                    "Add descriptive comments to all safety-related tags",
                    "Document safety interlock requirements",
                    "Include operational context in descriptions"
                ],
                metadata={'safety_tags_count': len(safety_tags_without_comments)},
                timestamp=datetime.now()
            ))
        
        if tags_without_comments and len(tags_without_comments) > 5:
            results.append(DiscoveryResult(
                result_id="missing_general_comments", 
                discovery_type="documentation",
                title=f"Many tags without comments: {len(tags_without_comments)}",
                description=f"Large number of tags lacking documentation may impact maintainability",
                confidence=0.8,
                impact_level="medium",
                related_components=tags_without_comments[:10],
                recommendations=[
                    "Implement tag documentation standards",
                    "Add comments during development",
                    "Review and document existing tags"
                ],
                metadata={'total_uncommented': len(tags_without_comments)},
                timestamp=datetime.now()
            ))
        
        return results
    
    async def _check_naming_inconsistencies(self, context_data: Dict[str, Any]) -> List[DiscoveryResult]:
        """Check for inconsistent naming patterns"""
        results = []
        
        tag_names = [tag['name'] for tag in context_data.get('tags', [])]
        
        # Analyze naming patterns
        patterns = {
            'underscore_style': 0,
            'camel_case': 0,
            'all_caps': 0,
            'mixed_style': 0
        }
        
        for name in tag_names:
            if '_' in name and name.islower():
                patterns['underscore_style'] += 1
            elif name[0].isupper() and any(c.isupper() for c in name[1:]):
                patterns['camel_case'] += 1
            elif name.isupper():
                patterns['all_caps'] += 1
            else:
                patterns['mixed_style'] += 1
        
        # Check for inconsistency
        total_tags = len(tag_names)
        if total_tags > 0:
            dominant_pattern = max(patterns, key=patterns.get)
            dominant_count = patterns[dominant_pattern]
            inconsistent_count = total_tags - dominant_count
            
            if inconsistent_count > total_tags * 0.3:  # More than 30% inconsistent
                results.append(DiscoveryResult(
                    result_id="naming_inconsistency",
                    discovery_type="standards",
                    title="Inconsistent tag naming patterns",
                    description=f"Mixed naming styles detected: {inconsistent_count}/{total_tags} tags don't follow dominant pattern",
                    confidence=0.7,
                    impact_level="medium",
                    related_components=tag_names[:5],
                    recommendations=[
                        "Establish consistent naming conventions",
                        "Refactor existing tags to follow standard",
                        "Document naming guidelines for team"
                    ],
                    metadata={
                        'patterns': patterns,
                        'dominant_pattern': dominant_pattern,
                        'inconsistent_count': inconsistent_count
                    },
                    timestamp=datetime.now()
                ))
        
        return results
    
    async def _check_complex_rungs(self, context_data: Dict[str, Any]) -> List[DiscoveryResult]:
        """Check for overly complex rungs"""
        results = []
        
        # Group instructions by routine to estimate rung complexity
        routine_instructions = defaultdict(list)
        for instruction in context_data.get('instructions', []):
            routine_key = f"{instruction.get('program', 'Unknown')}:{instruction.get('routine', 'Unknown')}"
            routine_instructions[routine_key].append(instruction)
        
        complex_routines = []
        for routine_key, instructions in routine_instructions.items():
            # Simple complexity metric: number of instructions per routine
            complexity = len(instructions)
            if complexity > 10:  # Arbitrary threshold
                complex_routines.append((routine_key, complexity))
        
        if complex_routines:
            complex_routines.sort(key=lambda x: x[1], reverse=True)
            
            results.append(DiscoveryResult(
                result_id="complex_routines",
                discovery_type="complexity",
                title=f"Complex routines detected: {len(complex_routines)}",
                description=f"Routines with high instruction count may be difficult to maintain",
                confidence=0.6,
                impact_level="medium",
                related_components=[routine for routine, _ in complex_routines],
                recommendations=[
                    "Break complex routines into smaller functions",
                    "Use subroutines for repeated logic",
                    "Add documentation for complex logic"
                ],
                metadata={
                    'complex_routines': complex_routines,
                    'max_complexity': complex_routines[0][1] if complex_routines else 0
                },
                timestamp=datetime.now()
            ))
        
        return results
    
    async def _check_safety_concerns(self, context_data: Dict[str, Any]) -> List[DiscoveryResult]:
        """Check for potential safety implementation issues"""
        results = []
        
        # Look for safety-related tags and instructions
        safety_tags = []
        safety_instructions = []
        
        for tag in context_data.get('tags', []):
            if any(keyword in tag['name'].lower() 
                  for keyword in ['safety', 'emergency', 'estop', 'interlock', 'guard']):
                safety_tags.append(tag)
        
        for instruction in context_data.get('instructions', []):
            for operand in instruction.get('operands', []):
                if any(keyword in operand.lower() 
                      for keyword in ['safety', 'emergency', 'estop', 'interlock']):
                    safety_instructions.append(instruction)
        
        # Check for safety implementation concerns
        if safety_tags and not safety_instructions:
            results.append(DiscoveryResult(
                result_id="unused_safety_tags",
                discovery_type="safety",
                title="Safety tags not used in logic",
                description=f"Found {len(safety_tags)} safety tags but no corresponding logic",
                confidence=0.8,
                impact_level="high",
                related_components=[tag['name'] for tag in safety_tags],
                recommendations=[
                    "Implement safety logic for defined safety tags",
                    "Review safety system requirements",
                    "Ensure proper safety circuit implementation"
                ],
                metadata={'safety_tags': [tag['name'] for tag in safety_tags]},
                timestamp=datetime.now()
            ))
        
        # Check for software-only safety implementations
        software_safety_count = len([i for i in safety_instructions if i.get('type') in ['OTE', 'OTL']])
        if software_safety_count > 0:
            results.append(DiscoveryResult(
                result_id="software_safety_concern",
                discovery_type="safety",
                title="Potential software-only safety implementation",
                description=f"Found {software_safety_count} safety outputs that may need hardwired backup",
                confidence=0.7,
                impact_level="critical",
                related_components=[i.get('operands', ['Unknown'])[0] for i in safety_instructions if i.get('type') in ['OTE', 'OTL']],
                recommendations=[
                    "Implement hardwired safety circuits",
                    "Use safety-rated PLC modules",
                    "Add redundant safety monitoring",
                    "Follow applicable safety standards (IEC 61508, ISO 13849)"
                ],
                metadata={'software_safety_outputs': software_safety_count},
                timestamp=datetime.now()
            ))
        
        return results
    
    async def _check_performance_bottlenecks(self, context_data: Dict[str, Any]) -> List[DiscoveryResult]:
        """Check for potential performance bottlenecks"""
        results = []
        
        # Analyze instruction distribution
        instruction_counts = Counter()
        for instruction in context_data.get('instructions', []):
            instruction_counts[instruction['type']] += 1
        
        # Look for performance-impacting patterns
        math_instructions = sum(instruction_counts[instr] for instr in ['ADD', 'SUB', 'MUL', 'DIV', 'SQR'] if instr in instruction_counts)
        file_instructions = sum(instruction_counts[instr] for instr in ['FAL', 'FSC', 'COP'] if instr in instruction_counts)
        
        if math_instructions > 20:  # Arbitrary threshold
            results.append(DiscoveryResult(
                result_id="high_math_usage",
                discovery_type="performance",
                title=f"High mathematical instruction usage: {math_instructions}",
                description="Large number of math instructions may impact scan time",
                confidence=0.6,
                impact_level="medium",
                related_components=list(instruction_counts.keys()),
                recommendations=[
                    "Consider optimizing mathematical calculations",
                    "Use lookup tables for complex calculations",
                    "Distribute math operations across scan cycles"
                ],
                metadata={'math_instruction_count': math_instructions},
                timestamp=datetime.now()
            ))
        
        return results
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score for content"""
        # Simple complexity metrics
        word_count = len(content.split())
        unique_words = len(set(content.lower().split()))
        
        # More unique words relative to total = higher complexity
        complexity = unique_words / word_count if word_count > 0 else 0
        return min(complexity * 10, 10.0)  # Scale to 0-10
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Simple keyword extraction
        technical_keywords = [
            'timer', 'counter', 'motor', 'safety', 'emergency', 'interlock',
            'alarm', 'fault', 'enable', 'disable', 'start', 'stop', 'run',
            'production', 'sequence', 'control', 'monitor', 'status'
        ]
        
        content_lower = content.lower()
        found_concepts = []
        
        for keyword in technical_keywords:
            if keyword in content_lower:
                found_concepts.append(keyword)
        
        return found_concepts[:5]  # Return top 5 concepts
    
    async def _find_related_components(self, metadata: Dict[str, Any]) -> List[str]:
        """Find components related to search result"""
        related = []
        
        # Mock implementation - in real system would analyze actual relationships
        tag_name = metadata.get('tag_name') or metadata.get('routine_name')
        if tag_name:
            # Find similar named components
            if 'motor' in tag_name.lower():
                related.extend(['Motor_Start', 'Motor_Stop', 'Motor_Status'])
            elif 'safety' in tag_name.lower():
                related.extend(['Emergency_Stop', 'Safety_Gate', 'Interlock_Status'])
            elif 'timer' in tag_name.lower():
                related.extend(['Timer_Enable', 'Timer_Reset', 'Timer_Status'])
        
        return related[:3]  # Limit to 3 related components
    
    def _generate_cache_key(self, query_text: str, filters: Optional[SearchFilter], 
                          search_types: List[str], max_results: int) -> str:
        """Generate cache key for search results"""
        key_data = {
            'query': query_text,
            'filters': asdict(filters) if filters else None,
            'types': sorted(search_types),
            'max_results': max_results
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'total_searches': dict(self.search_stats),
            'cache_size': len(self.query_cache),
            'pattern_library_size': len(self.pattern_library),
            'analysis_rules_count': len(self.analysis_rules),
            'index_sizes': {
                'tags': len(self.tag_index),
                'instructions': len(self.instruction_index)
            }
        }


# Convenience functions
async def create_enhanced_search_engine(semantic_engine: Optional[Any] = None) -> EnhancedSearchEngine:
    """Create enhanced search engine"""
    return EnhancedSearchEngine(semantic_engine)


async def perform_intelligent_search(query_text: str, 
                                    semantic_engine: Optional[Any] = None,
                                    filters: Optional[SearchFilter] = None) -> List[Dict[str, Any]]:
    """Perform intelligent search with enhanced capabilities"""
    enhanced_engine = await create_enhanced_search_engine(semantic_engine)
    return await enhanced_engine.advanced_search(query_text, filters)


async def discover_system_patterns(semantic_engine: Optional[Any] = None) -> List[PatternMatch]:
    """Discover patterns in PLC system"""
    enhanced_engine = await create_enhanced_search_engine(semantic_engine)
    return await enhanced_engine.discover_patterns()


async def analyze_system_relationships(semantic_engine: Optional[Any] = None) -> List[TagRelationship]:
    """Analyze tag relationships in PLC system"""
    enhanced_engine = await create_enhanced_search_engine(semantic_engine)
    return await enhanced_engine.analyze_tag_relationships()


# Export main classes and functions
__all__ = [
    'EnhancedSearchEngine',
    'SearchFilter',
    'DiscoveryResult',
    'PatternMatch',
    'TagRelationship',
    'create_enhanced_search_engine',
    'perform_intelligent_search',
    'discover_system_patterns',
    'analyze_system_relationships'
]
