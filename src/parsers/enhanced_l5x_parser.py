#!/usr/bin/env python3
"""
L5X Parser Extension for Step 10: Instruction Analysis

This module extends the L5X parser with advanced instruction analysis capabilities
for enhanced tag relationship extraction and complex instruction parsing.

Author: GitHub Copilot
Date: July 2025
"""

import logging
from typing import Dict, List, Any, Optional, Set
from xml.etree import ElementTree as ET

from ..analysis.instruction_analysis import (
    InstructionAnalyzer, ExpressionParser, TagRelationship,
    InstructionAnalysis, ParameterAnalysis
)
from ..models.ladder_logic import LadderInstruction, LadderRung, LadderRoutine
from ..parsers.l5x_parser import L5XParser

logger = logging.getLogger(__name__)


class EnhancedL5XParser(L5XParser):
    """Enhanced L5X parser with Step 10 instruction analysis capabilities"""
    
    def __init__(self):
        super().__init__()
        self.instruction_analyzer = InstructionAnalyzer()
        self.expression_parser = ExpressionParser()
        self._analysis_cache: Dict[str, Any] = {}
    
    def extract_advanced_ladder_analysis(self) -> Dict[str, Any]:
        """
        Extract ladder logic with advanced instruction analysis
        
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if not self.tree:
            return {'error': 'No L5X file loaded', 'analysis_successful': False}
        
        try:
            # First extract basic ladder logic using Step 9 functionality
            basic_ladder = self.extract_ladder_logic()
            
            if not basic_ladder.get('extraction_successful', False):
                return basic_ladder
            
            # Enhance with Step 10 analysis
            analysis_result = {
                'analysis_successful': True,
                'basic_extraction': basic_ladder,
                'instruction_analysis': {},
                'routine_analyses': {},
                'tag_relationships': [],
                'complexity_metrics': {},
                'expression_analysis': {},
                'summary': {}
            }
            
            # Analyze each routine with enhanced capabilities
            routines = basic_ladder.get('routines', {})
            
            for routine_name, routine_data in routines.items():
                logger.info(f"Performing advanced analysis on routine: {routine_name}")
                
                # Convert to LadderRoutine object for analysis
                ladder_routine = self._convert_to_ladder_routine(routine_name, routine_data)
                
                # Perform comprehensive analysis
                routine_analysis = self.instruction_analyzer.analyze_routine(ladder_routine)
                analysis_result['routine_analyses'][routine_name] = routine_analysis
                
                # Analyze complex expressions in this routine
                expression_analysis = self._analyze_routine_expressions(ladder_routine)
                analysis_result['expression_analysis'][routine_name] = expression_analysis
            
            # Compile global tag relationships
            analysis_result['tag_relationships'] = self.instruction_analyzer.get_instruction_relationships()
            
            # Generate global complexity metrics
            analysis_result['complexity_metrics'] = self._generate_global_complexity_metrics(
                analysis_result['routine_analyses']
            )
            
            # Create comprehensive summary
            analysis_result['summary'] = self._create_analysis_summary(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Advanced ladder analysis failed: {e}")
            return {
                'error': str(e),
                'analysis_successful': False,
                'basic_extraction': basic_ladder if 'basic_ladder' in locals() else None
            }
    
    def _convert_to_ladder_routine(self, routine_name: str, routine_data: Dict[str, Any]) -> LadderRoutine:
        """Convert routine data dictionary to LadderRoutine object"""
        ladder_routine = LadderRoutine(
            name=routine_name,
            program_name=routine_data.get('program_name', 'Unknown'),
            routine_type=routine_data.get('type', 'RLL')
        )
        
        # Convert rungs
        for rung_data in routine_data.get('rungs', []):
            ladder_rung = LadderRung(
                number=rung_data.get('number', 0),
                rung_type=rung_data.get('type', 'N'),
                comment=rung_data.get('comment'),
                raw_text=rung_data.get('text'),
                routine_name=routine_name
            )
            
            # Convert instructions
            for inst_data in rung_data.get('instructions', []):
                ladder_instruction = LadderInstruction(
                    instruction_type=inst_data.get('type'),
                    parameters=inst_data.get('parameters', []),
                    raw_text=inst_data.get('raw_text', ''),
                    position=inst_data.get('position', 0)
                )
                ladder_rung.instructions.append(ladder_instruction)
            
            ladder_routine.rungs.append(ladder_rung)
        
        return ladder_routine
    
    def _analyze_routine_expressions(self, routine: LadderRoutine) -> Dict[str, Any]:
        """Analyze mathematical and logical expressions in a routine"""
        expression_analysis = {
            'total_expressions': 0,
            'complex_expressions': [],
            'expression_dependencies': {},
            'expression_complexity_scores': [],
            'common_patterns': {}
        }
        
        for rung in routine.rungs:
            for instruction in rung.instructions:
                for param in instruction.parameters:
                    if param.parameter_type == "EXPRESSION":
                        expr_result = self.expression_parser.parse_expression(param.value)
                        
                        expression_analysis['total_expressions'] += 1
                        expression_analysis['expression_complexity_scores'].append(
                            expr_result['complexity_score']
                        )
                        
                        if expr_result['complexity_score'] > 5:  # Complex threshold
                            expression_analysis['complex_expressions'].append({
                                'expression': param.value,
                                'routine': routine.name,
                                'rung': rung.number,
                                'instruction_type': instruction.instruction_type.value,
                                'analysis': expr_result
                            })
                        
                        # Track dependencies
                        for tag in expr_result['tags']:
                            if tag not in expression_analysis['expression_dependencies']:
                                expression_analysis['expression_dependencies'][tag] = []
                            expression_analysis['expression_dependencies'][tag].append({
                                'expression': param.value,
                                'routine': routine.name,
                                'rung': rung.number
                            })
        
        # Calculate summary statistics
        if expression_analysis['expression_complexity_scores']:
            scores = expression_analysis['expression_complexity_scores']
            expression_analysis['average_complexity'] = sum(scores) / len(scores)
            expression_analysis['max_complexity'] = max(scores)
            expression_analysis['min_complexity'] = min(scores)
        else:
            expression_analysis['average_complexity'] = 0
            expression_analysis['max_complexity'] = 0
            expression_analysis['min_complexity'] = 0
        
        return expression_analysis
    
    def _generate_global_complexity_metrics(self, routine_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate global complexity metrics across all routines"""
        global_metrics = {
            'total_routines': len(routine_analyses),
            'total_complexity': 0,
            'average_routine_complexity': 0,
            'most_complex_routine': None,
            'instruction_category_distribution': {},
            'tag_interaction_summary': {
                'total_unique_input_tags': set(),
                'total_unique_output_tags': set(),
                'most_connected_tags': {}
            },
            'relationship_summary': {
                'total_relationships': 0,
                'relationship_types': {},
                'most_connected_routines': {}
            }
        }
        
        max_complexity = 0
        
        for routine_name, analysis in routine_analyses.items():
            complexity = analysis['complexity_metrics']['total_complexity']
            global_metrics['total_complexity'] += complexity
            
            if complexity > max_complexity:
                max_complexity = complexity
                global_metrics['most_complex_routine'] = {
                    'name': routine_name,
                    'complexity': complexity
                }
            
            # Aggregate instruction categories
            for category, count in analysis['complexity_metrics']['instruction_categories'].items():
                global_metrics['instruction_category_distribution'][category] = (
                    global_metrics['instruction_category_distribution'].get(category, 0) + count
                )
            
            # Aggregate tag interactions
            for inst_analysis in analysis['instruction_analyses']:
                global_metrics['tag_interaction_summary']['total_unique_input_tags'].update(
                    inst_analysis.input_tags
                )
                global_metrics['tag_interaction_summary']['total_unique_output_tags'].update(
                    inst_analysis.output_tags
                )
            
            # Count relationships
            relationship_count = len(analysis['tag_relationships'])
            global_metrics['relationship_summary']['total_relationships'] += relationship_count
            global_metrics['relationship_summary']['most_connected_routines'][routine_name] = relationship_count
            
            # Aggregate relationship types
            for relationship in analysis['tag_relationships']:
                rel_type = relationship.relationship_type
                global_metrics['relationship_summary']['relationship_types'][rel_type] = (
                    global_metrics['relationship_summary']['relationship_types'].get(rel_type, 0) + 1
                )
        
        # Calculate averages
        if global_metrics['total_routines'] > 0:
            global_metrics['average_routine_complexity'] = (
                global_metrics['total_complexity'] / global_metrics['total_routines']
            )
        
        # Convert sets to counts for JSON serialization
        global_metrics['tag_interaction_summary']['total_unique_input_tags'] = len(
            global_metrics['tag_interaction_summary']['total_unique_input_tags']
        )
        global_metrics['tag_interaction_summary']['total_unique_output_tags'] = len(
            global_metrics['tag_interaction_summary']['total_unique_output_tags']
        )
        
        # Find most connected tags across all routines
        tag_connections = {}
        for routine_name, analysis in routine_analyses.items():
            for relationship in analysis['tag_relationships']:
                source_tag = relationship.source_tag
                target_tag = relationship.target_tag
                
                tag_connections[source_tag] = tag_connections.get(source_tag, 0) + 1
                tag_connections[target_tag] = tag_connections.get(target_tag, 0) + 1
        
        # Get top 10 most connected tags
        sorted_tags = sorted(tag_connections.items(), key=lambda x: x[1], reverse=True)
        global_metrics['tag_interaction_summary']['most_connected_tags'] = dict(sorted_tags[:10])
        
        return global_metrics
    
    def _create_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive summary of the analysis"""
        summary = {
            'analysis_timestamp': None,  # Would be set to current time in real implementation
            'file_info': {
                'controller_info': self.get_controller_info().__dict__ if self.get_controller_info() else None,
                'programs': len(self.get_programs() or {}),
                'total_tags': len(self.extract_controller_tags()) + len(self.extract_program_tags())
            },
            'ladder_logic_summary': {},
            'instruction_analysis_summary': {},
            'complexity_assessment': {},
            'recommendations': []
        }
        
        # Basic ladder logic summary
        basic_ladder = analysis_result['basic_extraction']
        summary['ladder_logic_summary'] = {
            'total_routines': len(basic_ladder.get('routines', {})),
            'total_rungs': basic_ladder.get('summary', {}).get('statistics', {}).get('total_rungs', 0),
            'total_instructions': basic_ladder.get('summary', {}).get('statistics', {}).get('total_instructions', 0),
            'unique_tags_referenced': basic_ladder.get('summary', {}).get('statistics', {}).get('unique_tags_referenced', 0)
        }
        
        # Instruction analysis summary
        complexity_metrics = analysis_result['complexity_metrics']
        summary['instruction_analysis_summary'] = {
            'total_analyzed_instructions': sum(
                len(analysis['instruction_analyses']) 
                for analysis in analysis_result['routine_analyses'].values()
            ),
            'total_tag_relationships': complexity_metrics.get('relationship_summary', {}).get('total_relationships', 0),
            'most_complex_routine': complexity_metrics.get('most_complex_routine', {}),
            'instruction_distribution': complexity_metrics.get('instruction_category_distribution', {}),
            'expression_analysis': {
                'total_expressions': sum(
                    analysis.get('total_expressions', 0)
                    for analysis in analysis_result['expression_analysis'].values()
                ),
                'complex_expressions': sum(
                    len(analysis.get('complex_expressions', []))
                    for analysis in analysis_result['expression_analysis'].values()
                )
            }
        }
        
        # Complexity assessment
        total_complexity = complexity_metrics.get('total_complexity', 0)
        avg_complexity = complexity_metrics.get('average_routine_complexity', 0)
        
        if avg_complexity < 20:
            complexity_level = "Low"
        elif avg_complexity < 50:
            complexity_level = "Medium"
        elif avg_complexity < 100:
            complexity_level = "High"
        else:
            complexity_level = "Very High"
        
        summary['complexity_assessment'] = {
            'overall_level': complexity_level,
            'total_complexity_score': total_complexity,
            'average_routine_complexity': avg_complexity,
            'complexity_distribution': self._calculate_complexity_distribution(analysis_result['routine_analyses'])
        }
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(analysis_result)
        
        return summary
    
    def _calculate_complexity_distribution(self, routine_analyses: Dict[str, Any]) -> Dict[str, int]:
        """Calculate distribution of routines by complexity level"""
        distribution = {"Low": 0, "Medium": 0, "High": 0, "Very High": 0}
        
        for analysis in routine_analyses.values():
            complexity = analysis['complexity_metrics']['total_complexity']
            
            if complexity < 20:
                distribution["Low"] += 1
            elif complexity < 50:
                distribution["Medium"] += 1
            elif complexity < 100:
                distribution["High"] += 1
            else:
                distribution["Very High"] += 1
        
        return distribution
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        complexity_metrics = analysis_result['complexity_metrics']
        
        # High complexity routine recommendation
        if complexity_metrics.get('most_complex_routine', {}).get('complexity', 0) > 100:
            recommendations.append({
                'type': 'complexity',
                'priority': 'high',
                'title': 'High Complexity Routine Detected',
                'description': f"Routine '{complexity_metrics['most_complex_routine']['name']}' has very high complexity. Consider breaking it into smaller subroutines.",
                'routine': complexity_metrics['most_complex_routine']['name']
            })
        
        # Many complex expressions recommendation
        total_complex_expressions = sum(
            len(analysis.get('complex_expressions', []))
            for analysis in analysis_result['expression_analysis'].values()
        )
        
        if total_complex_expressions > 5:
            recommendations.append({
                'type': 'expressions',
                'priority': 'medium',
                'title': 'Complex Mathematical Expressions',
                'description': f"Found {total_complex_expressions} complex expressions. Consider using intermediate tags for readability.",
                'count': total_complex_expressions
            })
        
        # Tag usage recommendations
        tag_interaction_summary = complexity_metrics.get('tag_interaction_summary', {})
        most_connected_tags = tag_interaction_summary.get('most_connected_tags', {})
        
        if most_connected_tags:
            top_tag, connection_count = next(iter(most_connected_tags.items()))
            if connection_count > 10:
                recommendations.append({
                    'type': 'tag_usage',
                    'priority': 'low',
                    'title': 'Heavily Used Tag Detected',
                    'description': f"Tag '{top_tag}' is used in {connection_count} instruction relationships. Monitor for potential performance impact.",
                    'tag': top_tag,
                    'connection_count': connection_count
                })
        
        # Relationship type recommendations
        relationship_types = complexity_metrics.get('relationship_summary', {}).get('relationship_types', {})
        
        if relationship_types.get('latches', 0) > relationship_types.get('unlatches', 0):
            recommendations.append({
                'type': 'logic_pattern',
                'priority': 'medium',
                'title': 'Unbalanced Latch/Unlatch Logic',
                'description': f"Found {relationship_types.get('latches', 0)} latch operations but only {relationship_types.get('unlatches', 0)} unlatch operations. Verify proper reset logic.",
                'latches': relationship_types.get('latches', 0),
                'unlatches': relationship_types.get('unlatches', 0)
            })
        
        return recommendations
    
    def search_advanced_instructions(self, 
                                   instruction_type: Optional[str] = None,
                                   tag_reference: Optional[str] = None,
                                   parameter_role: Optional[str] = None,
                                   complexity_threshold: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Advanced instruction search with Step 10 capabilities
        
        Args:
            instruction_type: Type of instruction to search for
            tag_reference: Tag reference to search for
            parameter_role: Parameter role to filter by
            complexity_threshold: Minimum complexity score
            
        Returns:
            List of matching instructions with analysis data
        """
        if not hasattr(self, '_advanced_analysis_cache'):
            self._advanced_analysis_cache = self.extract_advanced_ladder_analysis()
        
        results = []
        analysis_data = self._advanced_analysis_cache
        
        if not analysis_data.get('analysis_successful', False):
            return results
        
        # Search through all instruction analyses
        for routine_name, routine_analysis in analysis_data['routine_analyses'].items():
            for inst_analysis in routine_analysis['instruction_analyses']:
                match = True
                
                # Filter by instruction type
                if instruction_type and inst_analysis.instruction.instruction_type.value != instruction_type:
                    match = False
                
                # Filter by tag reference
                if tag_reference and match:
                    tag_found = (
                        tag_reference in inst_analysis.input_tags or
                        tag_reference in inst_analysis.output_tags
                    )
                    if not tag_found:
                        match = False
                
                # Filter by parameter role
                if parameter_role and match:
                    role_found = any(
                        param.role.value == parameter_role
                        for param in inst_analysis.parameters
                    )
                    if not role_found:
                        match = False
                
                # Filter by complexity threshold
                if complexity_threshold and match:
                    if inst_analysis.complexity_score < complexity_threshold:
                        match = False
                
                if match:
                    results.append({
                        'routine_name': routine_name,
                        'instruction_type': inst_analysis.instruction.instruction_type.value,
                        'raw_text': inst_analysis.instruction.raw_text,
                        'complexity_score': inst_analysis.complexity_score,
                        'input_tags': list(inst_analysis.input_tags),
                        'output_tags': list(inst_analysis.output_tags),
                        'parameter_roles': [param.role.value for param in inst_analysis.parameters],
                        'execution_conditions': inst_analysis.execution_conditions,
                        'side_effects': inst_analysis.side_effects
                    })
        
        return results
    
    def get_tag_relationship_graph(self) -> Dict[str, Any]:
        """
        Generate a tag relationship graph for visualization
        
        Returns:
            Graph data structure with nodes and edges
        """
        if not hasattr(self, '_advanced_analysis_cache'):
            self._advanced_analysis_cache = self.extract_advanced_ladder_analysis()
        
        analysis_data = self._advanced_analysis_cache
        
        if not analysis_data.get('analysis_successful', False):
            return {'nodes': [], 'edges': [], 'error': 'Analysis not available'}
        
        graph = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_relationships': len(analysis_data['tag_relationships']),
                'relationship_types': set()
            }
        }
        
        # Collect all unique tags
        all_tags = set()
        for relationship in analysis_data['tag_relationships']:
            all_tags.add(relationship.source_tag)
            all_tags.add(relationship.target_tag)
            graph['metadata']['relationship_types'].add(relationship.relationship_type)
        
        # Create nodes
        for tag in all_tags:
            graph['nodes'].append({
                'id': tag,
                'label': tag,
                'type': 'tag'
            })
        
        # Create edges
        for relationship in analysis_data['tag_relationships']:
            graph['edges'].append({
                'source': relationship.source_tag,
                'target': relationship.target_tag,
                'relationship': relationship.relationship_type,
                'instruction_type': relationship.instruction_type.value,
                'routine': relationship.routine_name,
                'rung': relationship.rung_number
            })
        
        # Convert set to list for JSON serialization
        graph['metadata']['relationship_types'] = list(graph['metadata']['relationship_types'])
        
        return graph
    
    def clear_analysis_cache(self):
        """Clear the analysis cache to force re-analysis"""
        if hasattr(self, '_advanced_analysis_cache'):
            delattr(self, '_advanced_analysis_cache')
        self.instruction_analyzer.clear_cache()
        self._analysis_cache.clear()
