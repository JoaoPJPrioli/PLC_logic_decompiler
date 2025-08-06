"""
Instruction Analysis Module
Advanced analysis of PLC instructions and their relationships

This module provides:
- Detailed instruction breakdown and categorization
- Instruction dependency analysis
- Performance impact assessment
- Safety instruction analysis
- Optimization recommendations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import logging
from collections import defaultdict, Counter

from .ladder_logic_parser import LadderInstruction, RoutineLadderLogic, InstructionType

logger = logging.getLogger(__name__)

class InstructionCategory(Enum):
    """Categories of PLC instructions"""
    BASIC_LOGIC = "BASIC_LOGIC"         # XIC, XIO, OTE
    LATCH_LOGIC = "LATCH_LOGIC"         # OTL, OTU
    TIMING = "TIMING"                   # TON, TOF, RTO
    COUNTING = "COUNTING"               # CTU, CTD, RES
    COMPARISON = "COMPARISON"           # EQU, NEQ, GRT, LES
    MATH = "MATH"                       # ADD, SUB, MUL, DIV
    DATA_MOVEMENT = "DATA_MOVEMENT"     # MOV, COP
    PROGRAM_CONTROL = "PROGRAM_CONTROL" # JSR, RET, JMP, LBL
    COMMUNICATION = "COMMUNICATION"     # MSG, communication blocks
    SAFETY = "SAFETY"                   # Safety-related instructions
    ADVANCED = "ADVANCED"               # Function blocks, AOI

@dataclass
class InstructionMetrics:
    """Metrics for instruction analysis"""
    total_count: int = 0
    unique_count: int = 0
    most_used: List[Tuple[str, int]] = field(default_factory=list)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    complexity_score: float = 0.0

@dataclass
class TagDependency:
    """Represents dependency between tags through instructions"""
    source_tag: str
    target_tag: str
    instruction_type: str
    rung_number: int
    routine_name: str
    dependency_strength: float = 1.0

@dataclass
class ParameterAnalysis:
    """Analysis results for an instruction parameter"""
    parameter: str
    role: str = "unknown"
    is_tag_reference: bool = False
    base_tag: Optional[str] = None      # Base tag name without array/member access
    array_index: Optional[str] = None   # Array index if applicable
    member_path: List[str] = field(default_factory=list)  # Member access path
    data_type_hint: Optional[str] = None
    is_constant: bool = False
    constant_value: Optional[Any] = None
    dependencies: Set[str] = field(default_factory=set)  # Other tags this depends on

@dataclass
class InstructionAnalysis:
    """Comprehensive analysis of a ladder logic instruction"""
    instruction: 'LadderInstruction'
    category: InstructionCategory
    parameters: List[ParameterAnalysis] = field(default_factory=list)
    input_tags: Set[str] = field(default_factory=set)
    output_tags: Set[str] = field(default_factory=set)
    tag_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    complexity_score: int = 1
    execution_conditions: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)

@dataclass
class TagRelationship:
    """Represents a relationship between tags through instruction logic"""
    source_tag: str
    target_tag: str
    relationship_type: str  # "enables", "disables", "sets", "resets", "calculates", etc.
    instruction_type: str
    routine_name: str
    rung_number: int
    conditions: List[str] = field(default_factory=list)  # Conditions under which relationship applies

class InstructionAnalyzer:
    """
    Comprehensive analyzer for PLC instructions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.instruction_categories = self._build_instruction_categories()
        self.complexity_weights = self._build_complexity_weights()
    
    def _build_instruction_categories(self) -> Dict[str, InstructionCategory]:
        """Build mapping of instruction types to categories"""
        return {
            # Basic Logic
            'XIC': InstructionCategory.BASIC_LOGIC,
            'XIO': InstructionCategory.BASIC_LOGIC,
            'OTE': InstructionCategory.BASIC_LOGIC,
            'ONS': InstructionCategory.BASIC_LOGIC,
            
            # Latch Logic
            'OTL': InstructionCategory.LATCH_LOGIC,
            'OTU': InstructionCategory.LATCH_LOGIC,
            
            # Timing
            'TON': InstructionCategory.TIMING,
            'TOF': InstructionCategory.TIMING,
            'RTO': InstructionCategory.TIMING,
            
            # Counting
            'CTU': InstructionCategory.COUNTING,
            'CTD': InstructionCategory.COUNTING,
            'RES': InstructionCategory.COUNTING,
            
            # Comparison
            'EQU': InstructionCategory.COMPARISON,
            'NEQ': InstructionCategory.COMPARISON,
            'GRT': InstructionCategory.COMPARISON,
            'LES': InstructionCategory.COMPARISON,
            'GEQ': InstructionCategory.COMPARISON,
            'LEQ': InstructionCategory.COMPARISON,
            'LIM': InstructionCategory.COMPARISON,
            'MEQ': InstructionCategory.COMPARISON,
            
            # Math
            'ADD': InstructionCategory.MATH,
            'SUB': InstructionCategory.MATH,
            'MUL': InstructionCategory.MATH,
            'DIV': InstructionCategory.MATH,
            'MOD': InstructionCategory.MATH,
            'SQR': InstructionCategory.MATH,
            'NEG': InstructionCategory.MATH,
            'ABS': InstructionCategory.MATH,
            
            # Data Movement
            'MOV': InstructionCategory.DATA_MOVEMENT,
            'COP': InstructionCategory.DATA_MOVEMENT,
            'FLL': InstructionCategory.DATA_MOVEMENT,
            'CLR': InstructionCategory.DATA_MOVEMENT,
            
            # Program Control
            'JSR': InstructionCategory.PROGRAM_CONTROL,
            'RET': InstructionCategory.PROGRAM_CONTROL,
            'JMP': InstructionCategory.PROGRAM_CONTROL,
            'LBL': InstructionCategory.PROGRAM_CONTROL,
            'MCR': InstructionCategory.PROGRAM_CONTROL,
            'TND': InstructionCategory.PROGRAM_CONTROL,
            
            # Communication
            'MSG': InstructionCategory.COMMUNICATION,
        }
    
    def _build_complexity_weights(self) -> Dict[str, float]:
        """Build complexity weights for different instruction types"""
        return {
            # Basic instructions have low complexity
            'XIC': 1.0, 'XIO': 1.0, 'OTE': 1.0,
            
            # Latch instructions are more complex
            'OTL': 2.0, 'OTU': 2.0,
            
            # Timers and counters add complexity
            'TON': 3.0, 'TOF': 3.0, 'RTO': 3.5,
            'CTU': 3.0, 'CTD': 3.0,
            
            # Comparison instructions
            'EQU': 2.0, 'NEQ': 2.0, 'GRT': 2.0, 'LES': 2.0,
            'GEQ': 2.0, 'LEQ': 2.0, 'LIM': 3.0, 'MEQ': 2.5,
            
            # Math instructions
            'ADD': 2.5, 'SUB': 2.5, 'MUL': 3.0, 'DIV': 3.5,
            'MOD': 3.5, 'SQR': 3.0, 'NEG': 2.0, 'ABS': 2.0,
            
            # Data movement
            'MOV': 2.0, 'COP': 3.0, 'FLL': 2.5, 'CLR': 2.0,
            
            # Program control adds significant complexity
            'JSR': 4.0, 'RET': 2.0, 'JMP': 3.0, 'LBL': 1.0,
            'MCR': 3.5, 'TND': 2.0,
            
            # Communication is complex
            'MSG': 5.0,
        }
    
    def analyze_instructions(self, routines: List[RoutineLadderLogic]) -> Dict[str, Any]:
        """
        Perform comprehensive instruction analysis
        
        Args:
            routines: List of parsed ladder logic routines
            
        Returns:
            Detailed instruction analysis results
        """
        analysis = {
            'summary': self._create_instruction_summary(routines),
            'metrics': self._calculate_instruction_metrics(routines),
            'dependencies': self._analyze_tag_dependencies(routines),
            'patterns': self._analyze_instruction_patterns(routines),
            'safety_analysis': self._analyze_safety_instructions(routines),
            'optimization_suggestions': []
        }
        
        # Generate optimization suggestions
        analysis['optimization_suggestions'] = self._generate_optimization_suggestions(analysis)
        
        return analysis
    
    def _create_instruction_summary(self, routines: List[RoutineLadderLogic]) -> Dict[str, Any]:
        """Create high-level summary of instructions"""
        summary = {
            'total_routines': len(routines),
            'total_rungs': 0,
            'total_instructions': 0,
            'instruction_counts': Counter(),
            'category_counts': Counter(),
            'routines_breakdown': {}
        }
        
        for routine in routines:
            routine_info = {
                'rungs': len(routine.rungs),
                'instructions': 0,
                'instruction_types': Counter()
            }
            
            for rung in routine.rungs:
                for instruction in rung.instructions:
                    summary['instruction_counts'][instruction.type.value] += 1
                    routine_info['instruction_types'][instruction.type.value] += 1
                    routine_info['instructions'] += 1
                    
                    # Categorize instruction
                    category = self.instruction_categories.get(instruction.type.value, InstructionCategory.ADVANCED)
                    summary['category_counts'][category.value] += 1
            
            summary['total_rungs'] += routine_info['rungs']
            summary['total_instructions'] += routine_info['instructions']
            summary['routines_breakdown'][routine.routine_name] = routine_info
        
        return summary
    
    def _calculate_instruction_metrics(self, routines: List[RoutineLadderLogic]) -> InstructionMetrics:
        """Calculate detailed instruction metrics"""
        all_instructions = []
        instruction_counter = Counter()
        
        # Collect all instructions
        for routine in routines:
            for rung in routine.rungs:
                for instruction in rung.instructions:
                    all_instructions.append(instruction)
                    instruction_counter[instruction.type.value] += 1
        
        # Calculate complexity score
        complexity_score = 0.0
        for instruction in all_instructions:
            weight = self.complexity_weights.get(instruction.type.value, 1.0)
            complexity_score += weight
        
        # Normalize complexity score
        if all_instructions:
            complexity_score = min(100, complexity_score / len(all_instructions) * 10)
        
        # Category distribution
        category_dist = {}
        for instruction_type, count in instruction_counter.items():
            category = self.instruction_categories.get(instruction_type, InstructionCategory.ADVANCED)
            category_dist[category.value] = category_dist.get(category.value, 0) + count
        
        return InstructionMetrics(
            total_count=len(all_instructions),
            unique_count=len(instruction_counter),
            most_used=instruction_counter.most_common(10),
            category_distribution=category_dist,
            complexity_score=complexity_score
        )
    
    def _analyze_tag_dependencies(self, routines: List[RoutineLadderLogic]) -> List[TagDependency]:
        """Analyze dependencies between tags through instructions"""
        dependencies = []
        
        for routine in routines:
            for rung in routine.rungs:
                # Look for patterns where one tag influences another
                inputs = [inst for inst in rung.instructions 
                         if inst.type in [InstructionType.CONTACT_NO, InstructionType.CONTACT_NC]]
                outputs = [inst for inst in rung.instructions 
                          if inst.type in [InstructionType.COIL, InstructionType.COIL_LATCH, InstructionType.COIL_UNLATCH]]
                
                # Create dependencies from inputs to outputs
                for input_inst in inputs:
                    for output_inst in outputs:
                        dependency = TagDependency(
                            source_tag=input_inst.operand,
                            target_tag=output_inst.operand,
                            instruction_type=f"{input_inst.type.value}->{output_inst.type.value}",
                            rung_number=rung.number,
                            routine_name=routine.routine_name,
                            dependency_strength=self._calculate_dependency_strength(input_inst, output_inst)
                        )
                        dependencies.append(dependency)
        
        return dependencies
    
    def _calculate_dependency_strength(self, input_inst: LadderInstruction, 
                                     output_inst: LadderInstruction) -> float:
        """Calculate strength of dependency between two instructions"""
        base_strength = 1.0
        
        # Stronger dependency for direct logic
        if input_inst.type == InstructionType.CONTACT_NO and output_inst.type == InstructionType.COIL:
            base_strength = 1.0
        elif input_inst.type == InstructionType.CONTACT_NC and output_inst.type == InstructionType.COIL:
            base_strength = 0.9  # Slightly weaker for NC logic
        elif output_inst.type in [InstructionType.COIL_LATCH, InstructionType.COIL_UNLATCH]:
            base_strength = 1.2  # Stronger for latching logic
        
        return base_strength
    
    def _analyze_instruction_patterns(self, routines: List[RoutineLadderLogic]) -> Dict[str, Any]:
        """Analyze common instruction patterns"""
        patterns = {
            'seal_circuits': [],
            'safety_interlocks': [],
            'timer_sequences': [],
            'counter_applications': [],
            'state_machines': []
        }
        
        for routine in routines:
            # Look for seal circuit patterns (output feeding back as input)
            self._find_seal_circuits(routine, patterns)
            
            # Look for safety interlocks (multiple inputs to critical outputs)
            self._find_safety_interlocks(routine, patterns)
            
            # Look for timer sequences
            self._find_timer_sequences(routine, patterns)
            
            # Look for counter applications
            self._find_counter_applications(routine, patterns)
        
        return patterns
    
    def _find_seal_circuits(self, routine: RoutineLadderLogic, patterns: Dict[str, Any]):
        """Find seal circuit patterns"""
        for rung in routine.rungs:
            outputs = [inst for inst in rung.instructions if inst.type == InstructionType.COIL]
            inputs = [inst for inst in rung.instructions 
                     if inst.type in [InstructionType.CONTACT_NO, InstructionType.CONTACT_NC]]
            
            for output in outputs:
                for input_inst in inputs:
                    if output.operand == input_inst.operand:
                        patterns['seal_circuits'].append({
                            'routine': routine.routine_name,
                            'rung': rung.number,
                            'tag': output.operand,
                            'type': 'self_seal'
                        })
    
    def _find_safety_interlocks(self, routine: RoutineLadderLogic, patterns: Dict[str, Any]):
        """Find safety interlock patterns"""
        safety_keywords = ['ESTOP', 'EMERGENCY', 'SAFETY', 'GUARD', 'DOOR', 'LIGHT_CURTAIN']
        
        for rung in routine.rungs:
            safety_inputs = []
            for instruction in rung.instructions:
                if any(keyword in instruction.operand.upper() for keyword in safety_keywords):
                    safety_inputs.append(instruction)
            
            if len(safety_inputs) >= 2:  # Multiple safety inputs
                patterns['safety_interlocks'].append({
                    'routine': routine.routine_name,
                    'rung': rung.number,
                    'safety_tags': [inst.operand for inst in safety_inputs],
                    'input_count': len(safety_inputs)
                })
    
    def _find_timer_sequences(self, routine: RoutineLadderLogic, patterns: Dict[str, Any]):
        """Find timer sequence patterns"""
        timer_rungs = []
        
        for rung in routine.rungs:
            timer_instructions = [inst for inst in rung.instructions 
                                if inst.type in [InstructionType.TIMER_ON, InstructionType.TIMER_OFF]]
            if timer_instructions:
                timer_rungs.append({
                    'rung': rung.number,
                    'timers': [inst.operand for inst in timer_instructions]
                })
        
        # Look for sequences (timers that might be chained)
        if len(timer_rungs) >= 2:
            patterns['timer_sequences'].append({
                'routine': routine.routine_name,
                'sequence': timer_rungs,
                'length': len(timer_rungs)
            })
    
    def _find_counter_applications(self, routine: RoutineLadderLogic, patterns: Dict[str, Any]):
        """Find counter application patterns"""
        for rung in routine.rungs:
            counter_instructions = [inst for inst in rung.instructions 
                                  if inst.type in [InstructionType.COUNTER_UP, InstructionType.COUNTER_DOWN]]
            
            for counter_inst in counter_instructions:
                patterns['counter_applications'].append({
                    'routine': routine.routine_name,
                    'rung': rung.number,
                    'counter': counter_inst.operand,
                    'type': counter_inst.type.value
                })
    
    def _analyze_safety_instructions(self, routines: List[RoutineLadderLogic]) -> Dict[str, Any]:
        """Analyze safety-related instructions and patterns"""
        safety_analysis = {
            'safety_tags_found': [],
            'safety_circuits': [],
            'safety_recommendations': [],
            'emergency_stops': [],
            'safety_score': 0
        }
        
        safety_keywords = [
            'ESTOP', 'EMERGENCY', 'SAFETY', 'GUARD', 'DOOR', 'LIGHT_CURTAIN',
            'MUTE', 'BYPASS', 'RESET', 'FAULT', 'ALARM', 'INTERLOCK'
        ]
        
        safety_tag_count = 0
        total_outputs = 0
        
        for routine in routines:
            for rung in routine.rungs:
                for instruction in rung.instructions:
                    # Check for safety-related tags
                    if any(keyword in instruction.operand.upper() for keyword in safety_keywords):
                        safety_analysis['safety_tags_found'].append({
                            'tag': instruction.operand,
                            'instruction_type': instruction.type.value,
                            'routine': routine.routine_name,
                            'rung': rung.number
                        })
                        safety_tag_count += 1
                    
                    # Count total outputs for safety ratio
                    if instruction.type == InstructionType.COIL:
                        total_outputs += 1
                        
                        # Check for emergency stop patterns
                        if 'ESTOP' in instruction.operand.upper():
                            safety_analysis['emergency_stops'].append({
                                'tag': instruction.operand,
                                'routine': routine.routine_name,
                                'rung': rung.number
                            })
        
        # Calculate safety score
        if total_outputs > 0:
            safety_ratio = safety_tag_count / total_outputs
            safety_analysis['safety_score'] = min(100, safety_ratio * 100)
        
        # Generate safety recommendations
        if safety_analysis['safety_score'] < 20:
            safety_analysis['safety_recommendations'].append("Consider adding more safety interlocks")
        
        if not safety_analysis['emergency_stops']:
            safety_analysis['safety_recommendations'].append("No emergency stop logic detected - verify safety implementation")
        
        return safety_analysis
    
    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on analysis"""
        suggestions = []
        
        metrics = analysis['metrics']
        summary = analysis['summary']
        
        # Check complexity
        if metrics.complexity_score > 70:
            suggestions.append("High complexity detected - consider breaking down large routines")
        
        # Check instruction distribution
        if metrics.category_distribution.get('BASIC_LOGIC', 0) < metrics.total_count * 0.3:
            suggestions.append("Low basic logic percentage - verify program structure")
        
        # Check for excessive latching
        latch_count = summary['instruction_counts'].get('OTL', 0) + summary['instruction_counts'].get('OTU', 0)
        if latch_count > metrics.total_count * 0.1:
            suggestions.append("High latching instruction usage - review state management approach")
        
        # Check for timer usage
        timer_count = (summary['instruction_counts'].get('TON', 0) + 
                      summary['instruction_counts'].get('TOF', 0) + 
                      summary['instruction_counts'].get('RTO', 0))
        if timer_count > 20:
            suggestions.append("Consider consolidating timer logic for better maintainability")
        
        # Check safety analysis
        safety_score = analysis['safety_analysis']['safety_score']
        if safety_score < 30:
            suggestions.append("Consider adding more safety interlocks and protective logic")
        
        return suggestions
