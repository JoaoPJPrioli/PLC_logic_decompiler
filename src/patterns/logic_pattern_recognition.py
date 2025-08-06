"""
Step 31: Logic Pattern Recognition
Advanced pattern recognition for common PLC logic structures with optimization insights

This module provides sophisticated pattern recognition capabilities that analyze PLC logic
to identify common patterns, detect anti-patterns, suggest optimizations, and provide
insights for better system design and maintenance.
"""

import os
import sys
import asyncio
import json
import logging
import re
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set, NamedTuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter, deque
from enum import Enum
import hashlib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from search.enhanced_search import (
        EnhancedSearchEngine,
        SearchFilter,
        PatternMatch
    )
    ENHANCED_SEARCH_AVAILABLE = True
except ImportError:
    print("Enhanced search not available - using mock implementations")
    ENHANCED_SEARCH_AVAILABLE = False

try:
    from semantic.chromadb_integration import PLCSemanticSearchEngine
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


class PatternType(Enum):
    """Types of logic patterns"""
    CONTROL = "control"
    SAFETY = "safety"
    TIMING = "timing" 
    SEQUENCING = "sequencing"
    COMMUNICATION = "communication"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization"


class PatternComplexity(Enum):
    """Pattern complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class PatternCategory(Enum):
    """Pattern categories for classification"""
    BEST_PRACTICE = "best_practice"
    COMMON_PATTERN = "common_pattern"
    ANTI_PATTERN = "anti_pattern"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    MAINTENANCE_CONCERN = "maintenance_concern"
    SAFETY_CRITICAL = "safety_critical"
    UTILITY = "utility"
    AUTOMATION = "automation"
    PROCESS = "process"
    COMMUNICATION = "communication"
    MONITORING = "monitoring"


@dataclass
class PatternElement:
    """Individual element within a pattern"""
    element_type: str  # 'instruction', 'tag', 'condition', 'action'
    element_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class PatternTemplate:
    """Template definition for pattern matching"""
    pattern_id: str
    name: str
    description: str
    pattern_type: PatternType
    complexity: PatternComplexity
    category: PatternCategory
    elements: List[PatternElement]
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PatternInstance:
    """Detected instance of a pattern"""
    template_id: str
    instance_id: str
    confidence: float
    location: Dict[str, str]  # program, routine, rung info
    matched_elements: Dict[str, str]  # element_id -> actual_component
    metrics: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    optimization_potential: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PatternAnalysisResult:
    """Result of pattern analysis"""
    analysis_id: str
    patterns_detected: List[PatternInstance]
    anti_patterns: List[PatternInstance]
    optimization_opportunities: List[Dict[str, Any]]
    system_metrics: Dict[str, float]
    recommendations: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class LogicPatternRecognizer:
    """Advanced logic pattern recognition engine"""
    
    def __init__(self, enhanced_search_engine: Optional[Any] = None):
        """Initialize pattern recognizer"""
        self.enhanced_search_engine = enhanced_search_engine
        self.pattern_templates = {}
        self.recognition_cache = {}
        self.analysis_history = []
        
        # Pattern matching configuration
        self.fuzzy_threshold = 0.8
        self.min_confidence = 0.5
        self.max_patterns_per_routine = 50
        
        # Initialize pattern library
        self._initialize_pattern_library()
        
        # Analysis metrics
        self.recognition_stats = defaultdict(int)
        
        logger.info("Logic pattern recognizer initialized")
    
    def _initialize_pattern_library(self):
        """Initialize comprehensive pattern library"""
        
        # Control Patterns
        self._add_motor_control_patterns()
        self._add_valve_control_patterns()
        self._add_pump_control_patterns()
        
        # Safety Patterns
        self._add_safety_interlock_patterns()
        self._add_emergency_stop_patterns()
        self._add_guard_monitoring_patterns()
        
        # Timing Patterns
        self._add_timer_patterns()
        self._add_delay_patterns()
        self._add_pulse_patterns()
        
        # Sequencing Patterns
        self._add_step_sequence_patterns()
        self._add_state_machine_patterns()
        self._add_batch_control_patterns()
        
        # Communication Patterns
        self._add_message_handling_patterns()
        self._add_handshake_patterns()
        
        # Monitoring Patterns
        self._add_alarm_patterns()
        self._add_diagnostic_patterns()
        
        # Anti-patterns
        self._add_anti_patterns()
        
        logger.info(f"Initialized {len(self.pattern_templates)} pattern templates")
    
    def _add_motor_control_patterns(self):
        """Add motor control patterns"""
        
        # Standard Motor Control with Overload Protection
        self.pattern_templates['motor_standard'] = PatternTemplate(
            pattern_id='motor_standard',
            name='Standard Motor Control',
            description='Standard 3-wire motor control with overload protection',
            pattern_type=PatternType.CONTROL,
            complexity=PatternComplexity.SIMPLE,
            category=PatternCategory.BEST_PRACTICE,
            elements=[
                PatternElement('instruction', 'start_contact', {'type': 'XIC'}),
                PatternElement('instruction', 'stop_contact', {'type': 'XIO'}),
                PatternElement('instruction', 'overload_contact', {'type': 'XIC'}),
                PatternElement('instruction', 'seal_contact', {'type': 'XIC'}),
                PatternElement('instruction', 'motor_output', {'type': 'OTE'})
            ],
            relationships={
                'parallel': ['start_contact', 'seal_contact'],
                'series': ['stop_contact', 'overload_contact'],
                'output': ['motor_output']
            },
            conditions=['start_contact.tag != stop_contact.tag'],
            benefits=[
                'Reliable motor starting and stopping',
                'Overload protection integrated',
                'Maintained output with seal-in logic'
            ],
            optimizations=[
                'Add run status indication',
                'Include motor runtime tracking',
                'Add maintenance counters'
            ]
        )
        
        # Variable Frequency Drive Control
        self.pattern_templates['vfd_control'] = PatternTemplate(
            pattern_id='vfd_control',
            name='VFD Motor Control',
            description='Variable frequency drive control with speed reference',
            pattern_type=PatternType.CONTROL,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.COMMON_PATTERN,
            elements=[
                PatternElement('instruction', 'run_command', {'type': 'OTE'}),
                PatternElement('instruction', 'speed_reference', {'type': 'MOV'}),
                PatternElement('instruction', 'fault_reset', {'type': 'ONS'}),
                PatternElement('tag', 'speed_feedback', {'data_type': 'REAL'})
            ],
            benefits=[
                'Variable speed control',
                'Energy efficiency',
                'Soft starting capability'
            ],
            optimizations=[
                'Implement PID speed control',
                'Add acceleration/deceleration ramping',
                'Monitor drive diagnostics'
            ]
        )
    
    def _add_valve_control_patterns(self):
        """Add valve control patterns"""
        
        # Two-Position Valve Control
        self.pattern_templates['valve_two_position'] = PatternTemplate(
            pattern_id='valve_two_position',
            name='Two-Position Valve Control',
            description='Open/close valve control with position feedback',
            pattern_type=PatternType.CONTROL,
            complexity=PatternComplexity.SIMPLE,
            category=PatternCategory.COMMON_PATTERN,
            elements=[
                PatternElement('instruction', 'open_command', {'type': 'OTE'}),
                PatternElement('instruction', 'close_command', {'type': 'OTE'}),
                PatternElement('tag', 'open_feedback', {'data_type': 'BOOL'}),
                PatternElement('tag', 'closed_feedback', {'data_type': 'BOOL'})
            ],
            relationships={
                'interlocked': ['open_command', 'close_command']
            },
            conditions=['open_command.tag != close_command.tag'],
            benefits=[
                'Positive valve position control',
                'Position feedback verification',
                'Interlocked operation'
            ],
            risks=[
                'Both outputs active simultaneously',
                'Missing position feedback handling'
            ]
        )
    
    def _add_pump_control_patterns(self):
        """Add pump control patterns"""
        
        # Standard Pump Control
        self.pattern_templates['pump_standard'] = PatternTemplate(
            pattern_id='pump_standard',
            name='Standard Pump Control',
            description='Standard on/off pump control with status feedback',
            pattern_type=PatternType.CONTROL,
            complexity=PatternComplexity.SIMPLE,
            category=PatternCategory.COMMON_PATTERN,
            elements=[
                PatternElement('instruction', 'pump_start', {'type': 'XIC'}),
                PatternElement('instruction', 'pump_stop', {'type': 'XIO'}),
                PatternElement('instruction', 'pump_run', {'type': 'OTE'}),
                PatternElement('instruction', 'pump_feedback', {'type': 'XIC'})
            ],
            relationships={
                'series': ['pump_start', 'pump_stop'],
                'parallel': ['pump_feedback']
            },
            benefits=[
                'Simple pump control',
                'Status feedback available',
                'Standard implementation'
            ],
            optimizations=[
                'Add pump protection logic',
                'Include runtime monitoring',
                'Add maintenance scheduling'
            ]
        )
    
    def _add_emergency_stop_patterns(self):
        """Add emergency stop patterns"""
        
        # Emergency Stop Chain
        self.pattern_templates['emergency_stop'] = PatternTemplate(
            pattern_id='emergency_stop',
            name='Emergency Stop Chain',
            description='Emergency stop with multiple E-stop buttons',
            pattern_type=PatternType.SAFETY,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.SAFETY_CRITICAL,
            elements=[
                PatternElement('instruction', 'estop_1', {'type': 'XIC'}),
                PatternElement('instruction', 'estop_2', {'type': 'XIC'}),
                PatternElement('instruction', 'estop_output', {'type': 'OTE'})
            ],
            relationships={
                'series': ['estop_1', 'estop_2']
            },
            benefits=[
                'Multiple emergency stop points',
                'Fail-safe emergency shutdown',
                'Clear safety shutdown logic'
            ],
            risks=[
                'Software-only E-stop implementation',
                'No bypass monitoring',
                'Missing diagnostics'
            ]
        )
    
    def _add_guard_monitoring_patterns(self):
        """Add guard monitoring patterns"""
        
        # Guard Door Monitoring
        self.pattern_templates['guard_monitor'] = PatternTemplate(
            pattern_id='guard_monitor',
            name='Guard Door Monitoring',
            description='Safety guard door position monitoring',
            pattern_type=PatternType.SAFETY,
            complexity=PatternComplexity.SIMPLE,
            category=PatternCategory.SAFETY_CRITICAL,
            elements=[
                PatternElement('instruction', 'guard_closed', {'type': 'XIC'}),
                PatternElement('instruction', 'guard_locked', {'type': 'XIC'}),
                PatternElement('instruction', 'guard_ok', {'type': 'OTE'})
            ],
            relationships={
                'series': ['guard_closed', 'guard_locked']
            },
            benefits=[
                'Guard position monitoring',
                'Lock verification',
                'Safety interlock function'
            ]
        )
    
    def _add_delay_patterns(self):
        """Add delay patterns"""
        
        # Sequential Delay
        self.pattern_templates['sequential_delay'] = PatternTemplate(
            pattern_id='sequential_delay',
            name='Sequential Delay Pattern',
            description='Sequential timing with multiple delays',
            pattern_type=PatternType.TIMING,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.COMMON_PATTERN,
            elements=[
                PatternElement('instruction', 'delay_trigger', {'type': 'XIC'}),
                PatternElement('instruction', 'delay_timer_1', {'type': 'TON'}),
                PatternElement('instruction', 'delay_timer_2', {'type': 'TON'}),
                PatternElement('instruction', 'delay_output', {'type': 'OTE'})
            ],
            benefits=[
                'Sequential timing control',
                'Flexible delay configuration',
                'Clear timing logic'
            ]
        )
    
    def _add_pulse_patterns(self):
        """Add pulse patterns"""
        
        # Pulse Generation
        self.pattern_templates['pulse_generator'] = PatternTemplate(
            pattern_id='pulse_generator',
            name='Pulse Generator Pattern',
            description='Generates timed pulses for control',
            pattern_type=PatternType.TIMING,
            complexity=PatternComplexity.SIMPLE,
            category=PatternCategory.UTILITY,
            elements=[
                PatternElement('instruction', 'pulse_trigger', {'type': 'XIC'}),
                PatternElement('instruction', 'pulse_timer', {'type': 'TON'}),
                PatternElement('instruction', 'pulse_output', {'type': 'OTE'})
            ],
            benefits=[
                'Precise pulse generation',
                'Configurable pulse width',
                'Reliable timing control'
            ]
        )
    
    def _add_step_sequence_patterns(self):
        """Add step sequence patterns"""
        
        # Step Sequence
        self.pattern_templates['step_sequence'] = PatternTemplate(
            pattern_id='step_sequence',
            name='Step Sequence Pattern',
            description='Sequential step control with conditions',
            pattern_type=PatternType.SEQUENCING,
            complexity=PatternComplexity.COMPLEX,
            category=PatternCategory.AUTOMATION,
            elements=[
                PatternElement('instruction', 'step_1_active', {'type': 'XIC'}),
                PatternElement('instruction', 'step_1_complete', {'type': 'XIC'}),
                PatternElement('instruction', 'step_2_active', {'type': 'OTE'}),
                PatternElement('instruction', 'step_timer', {'type': 'TON'})
            ],
            benefits=[
                'Structured sequential control',
                'Clear step transitions',
                'Easy troubleshooting'
            ]
        )
    
    def _add_batch_control_patterns(self):
        """Add batch control patterns"""
        
        # Batch Phase Control
        self.pattern_templates['batch_phase'] = PatternTemplate(
            pattern_id='batch_phase',
            name='Batch Phase Control',
            description='Batch recipe phase control logic',
            pattern_type=PatternType.SEQUENCING,
            complexity=PatternComplexity.COMPLEX,
            category=PatternCategory.PROCESS,
            elements=[
                PatternElement('instruction', 'phase_start', {'type': 'XIC'}),
                PatternElement('instruction', 'phase_conditions', {'type': 'XIC'}),
                PatternElement('instruction', 'phase_timer', {'type': 'TON'}),
                PatternElement('instruction', 'phase_complete', {'type': 'OTE'})
            ],
            benefits=[
                'Structured batch control',
                'Recipe-based operation',
                'Phase tracking'
            ]
        )
    
    def _add_message_handling_patterns(self):
        """Add message handling patterns"""
        
        # Message Queue
        self.pattern_templates['message_queue'] = PatternTemplate(
            pattern_id='message_queue',
            name='Message Queue Pattern',
            description='Message handling and queuing logic',
            pattern_type=PatternType.COMMUNICATION,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.COMMUNICATION,
            elements=[
                PatternElement('instruction', 'message_received', {'type': 'XIC'}),
                PatternElement('instruction', 'queue_available', {'type': 'XIC'}),
                PatternElement('instruction', 'message_processed', {'type': 'OTE'})
            ],
            benefits=[
                'Reliable message handling',
                'Queue management',
                'Communication control'
            ]
        )
    
    def _add_handshake_patterns(self):
        """Add handshake patterns"""
        
        # Communication Handshake
        self.pattern_templates['comm_handshake'] = PatternTemplate(
            pattern_id='comm_handshake',
            name='Communication Handshake',
            description='Two-way communication handshake protocol',
            pattern_type=PatternType.COMMUNICATION,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.COMMUNICATION,
            elements=[
                PatternElement('instruction', 'send_request', {'type': 'OTE'}),
                PatternElement('instruction', 'receive_ack', {'type': 'XIC'}),
                PatternElement('instruction', 'send_complete', {'type': 'OTE'})
            ],
            benefits=[
                'Reliable communication',
                'Handshake verification',
                'Error detection'
            ]
        )
    
    def _add_diagnostic_patterns(self):
        """Add diagnostic patterns"""
        
        # System Diagnostics
        self.pattern_templates['system_diagnostics'] = PatternTemplate(
            pattern_id='system_diagnostics',
            name='System Diagnostics',
            description='System health and diagnostic monitoring',
            pattern_type=PatternType.MONITORING,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.MONITORING,
            elements=[
                PatternElement('instruction', 'health_check', {'type': 'XIC'}),
                PatternElement('instruction', 'diagnostic_timer', {'type': 'TON'}),
                PatternElement('instruction', 'system_healthy', {'type': 'OTE'})
            ],
            benefits=[
                'System health monitoring',
                'Predictive maintenance',
                'Early fault detection'
            ]
        )
    
    def _add_safety_interlock_patterns(self):
        """Add safety interlock patterns"""
        
        # Multi-Input Safety Chain
        self.pattern_templates['safety_chain'] = PatternTemplate(
            pattern_id='safety_chain',
            name='Safety Interlock Chain',
            description='Series safety interlock with multiple inputs',
            pattern_type=PatternType.SAFETY,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.SAFETY_CRITICAL,
            elements=[
                PatternElement('instruction', 'safety_input_1', {'type': 'XIC'}),
                PatternElement('instruction', 'safety_input_2', {'type': 'XIC'}),
                PatternElement('instruction', 'safety_input_3', {'type': 'XIC'}),
                PatternElement('instruction', 'safety_output', {'type': 'OTE'})
            ],
            relationships={
                'series': ['safety_input_1', 'safety_input_2', 'safety_input_3']
            },
            benefits=[
                'Multiple safety inputs monitored',
                'Fail-safe operation',
                'Clear safety logic flow'
            ],
            risks=[
                'Software-only safety implementation',
                'No redundancy in safety chain',
                'Missing diagnostic monitoring'
            ],
            optimizations=[
                'Add safety input diagnostics',
                'Implement dual-channel monitoring',
                'Add safety system testing'
            ]
        )
    
    def _add_state_machine_patterns(self):
        """Add state machine patterns"""
        
        # Simple State Machine
        self.pattern_templates['state_machine'] = PatternTemplate(
            pattern_id='state_machine',
            name='State Machine Pattern',
            description='State-based control logic with transitions',
            pattern_type=PatternType.SEQUENCING,
            complexity=PatternComplexity.COMPLEX,
            category=PatternCategory.AUTOMATION,
            elements=[
                PatternElement('instruction', 'current_state', {'type': 'EQU'}),
                PatternElement('instruction', 'state_condition', {'type': 'XIC'}),
                PatternElement('instruction', 'next_state', {'type': 'MOV'}),
                PatternElement('instruction', 'state_action', {'type': 'OTE'})
            ],
            relationships={
                'state_transition': ['current_state', 'state_condition', 'next_state']
            },
            benefits=[
                'Structured state control',
                'Clear state transitions',
                'Scalable control logic'
            ],
            optimizations=[
                'Add state transition logging',
                'Implement state timeout handling',
                'Add state validation'
            ]
        )
    
    def _add_timer_patterns(self):
        """Add timer-based patterns"""
        
        # Timer Delay Pattern
        self.pattern_templates['timer_delay'] = PatternTemplate(
            pattern_id='timer_delay',
            name='Timer Delay Pattern',
            description='Time delay for equipment sequencing',
            pattern_type=PatternType.TIMING,
            complexity=PatternComplexity.SIMPLE,
            category=PatternCategory.COMMON_PATTERN,
            elements=[
                PatternElement('instruction', 'timer_enable', {'type': 'XIC'}),
                PatternElement('instruction', 'timer', {'type': 'TON'}),
                PatternElement('instruction', 'timer_done', {'type': 'XIC'}),
                PatternElement('instruction', 'delayed_output', {'type': 'OTE'})
            ],
            relationships={
                'enables': ['timer_enable', 'timer'],
                'result': ['timer_done', 'delayed_output']
            },
            benefits=[
                'Controlled timing sequence',
                'Equipment protection delay',
                'Sequence coordination'
            ],
            optimizations=[
                'Add timer bypass for maintenance',
                'Monitor timer performance',
                'Document timing requirements'
            ]
        )
        
        # Cascaded Timer Sequence
        self.pattern_templates['timer_cascade'] = PatternTemplate(
            pattern_id='timer_cascade',
            name='Cascaded Timer Sequence',
            description='Sequential timer operations for multi-step processes',
            pattern_type=PatternType.SEQUENCING,
            complexity=PatternComplexity.COMPLEX,
            category=PatternCategory.COMMON_PATTERN,
            elements=[
                PatternElement('instruction', 'sequence_start', {'type': 'XIC'}),
                PatternElement('instruction', 'timer_1', {'type': 'TON'}),
                PatternElement('instruction', 'timer_2', {'type': 'TON'}),
                PatternElement('instruction', 'timer_3', {'type': 'TON'}),
                PatternElement('instruction', 'step_1_output', {'type': 'OTE'}),
                PatternElement('instruction', 'step_2_output', {'type': 'OTE'}),
                PatternElement('instruction', 'step_3_output', {'type': 'OTE'})
            ],
            relationships={
                'sequence': ['timer_1', 'timer_2', 'timer_3'],
                'outputs': ['step_1_output', 'step_2_output', 'step_3_output']
            },
            benefits=[
                'Coordinated sequential operation',
                'Predictable timing behavior',
                'Clear step progression'
            ],
            optimizations=[
                'Add sequence reset capability',
                'Implement step skip functionality',
                'Add sequence monitoring'
            ]
        )
    
    def _add_step_sequence_patterns(self):
        """Add step sequencing patterns"""
        
        # State Machine Pattern
        self.pattern_templates['state_machine'] = PatternTemplate(
            pattern_id='state_machine',
            name='State Machine Control',
            description='State-based control with transitions',
            pattern_type=PatternType.SEQUENCING,
            complexity=PatternComplexity.COMPLEX,
            category=PatternCategory.BEST_PRACTICE,
            elements=[
                PatternElement('tag', 'current_state', {'data_type': 'DINT'}),
                PatternElement('instruction', 'state_1_logic', {'type': 'EQU'}),
                PatternElement('instruction', 'state_2_logic', {'type': 'EQU'}),
                PatternElement('instruction', 'state_transition', {'type': 'MOV'})
            ],
            benefits=[
                'Clear state-based operation',
                'Predictable state transitions',
                'Easy troubleshooting'
            ],
            optimizations=[
                'Add state transition logging',
                'Implement state timeout monitoring',
                'Add invalid state detection'
            ]
        )
    
    def _add_alarm_patterns(self):
        """Add alarm handling patterns"""
        
        # Alarm with Acknowledgment
        self.pattern_templates['alarm_ack'] = PatternTemplate(
            pattern_id='alarm_ack',
            name='Alarm with Acknowledgment',
            description='Alarm generation with operator acknowledgment',
            pattern_type=PatternType.MONITORING,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.BEST_PRACTICE,
            elements=[
                PatternElement('instruction', 'alarm_condition', {'type': 'XIC'}),
                PatternElement('instruction', 'alarm_latch', {'type': 'OTL'}),
                PatternElement('instruction', 'alarm_ack', {'type': 'XIC'}),
                PatternElement('instruction', 'alarm_reset', {'type': 'OTU'}),
                PatternElement('instruction', 'alarm_output', {'type': 'XIC'})
            ],
            relationships={
                'latches': ['alarm_condition', 'alarm_latch'],
                'resets': ['alarm_ack', 'alarm_reset']
            },
            benefits=[
                'Persistent alarm indication',
                'Operator acknowledgment required',
                'Clear alarm state management'
            ],
            optimizations=[
                'Add alarm priority levels',
                'Implement alarm logging',
                'Add alarm statistics'
            ]
        )
    
    def _add_anti_patterns(self):
        """Add anti-patterns to detect poor practices"""
        
        # Nested TON Timers (Anti-pattern)
        self.pattern_templates['nested_timers_anti'] = PatternTemplate(
            pattern_id='nested_timers_anti',
            name='Nested Timer Anti-Pattern',
            description='Nested TON timers creating timing dependencies',
            pattern_type=PatternType.TIMING,
            complexity=PatternComplexity.MODERATE,
            category=PatternCategory.ANTI_PATTERN,
            elements=[
                PatternElement('instruction', 'outer_timer', {'type': 'TON'}),
                PatternElement('instruction', 'inner_timer', {'type': 'TON'}),
                PatternElement('instruction', 'dependency', {'type': 'XIC'})
            ],
            relationships={
                'nested': ['outer_timer', 'inner_timer']
            },
            risks=[
                'Complex timing dependencies',
                'Difficult troubleshooting',
                'Unpredictable behavior'
            ],
            optimizations=[
                'Use sequential timing instead',
                'Implement state machine approach',
                'Simplify timing logic'
            ]
        )
        
        # Hardcoded Values Anti-pattern
        self.pattern_templates['hardcoded_values_anti'] = PatternTemplate(
            pattern_id='hardcoded_values_anti',
            name='Hardcoded Values Anti-Pattern',
            description='Hardcoded constants in logic instructions',
            pattern_type=PatternType.MAINTENANCE,
            complexity=PatternComplexity.SIMPLE,
            category=PatternCategory.ANTI_PATTERN,
            elements=[
                PatternElement('instruction', 'comparison', {'type': ['EQU', 'GEQ', 'LEQ']}),
                PatternElement('parameter', 'hardcoded_value', {'type': 'constant'})
            ],
            risks=[
                'Difficult to modify parameters',
                'No centralized configuration',
                'Maintenance complications'
            ],
            optimizations=[
                'Use tag-based parameters',
                'Create parameter data blocks',
                'Implement configuration management'
            ]
        )
    
    async def analyze_patterns(self, context_data: Dict[str, Any] = None) -> PatternAnalysisResult:
        """Perform comprehensive pattern analysis"""
        logger.info("Starting comprehensive pattern analysis")
        
        analysis_id = hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        if context_data is None:
            context_data = await self._gather_context_data()
        
        # Detect pattern instances
        detected_patterns = []
        anti_patterns = []
        
        for template_id, template in self.pattern_templates.items():
            try:
                instances = await self._match_pattern_template(template, context_data)
                
                for instance in instances:
                    if template.category == PatternCategory.ANTI_PATTERN:
                        anti_patterns.append(instance)
                    else:
                        detected_patterns.append(instance)
                        
            except Exception as e:
                logger.error(f"Error matching pattern {template_id}: {e}")
        
        # Generate optimization opportunities
        optimization_opportunities = await self._identify_optimization_opportunities(
            detected_patterns, anti_patterns, context_data
        )
        
        # Calculate system metrics
        system_metrics = self._calculate_system_metrics(
            detected_patterns, anti_patterns, context_data
        )
        
        # Generate recommendations
        recommendations = self._generate_pattern_recommendations(
            detected_patterns, anti_patterns, optimization_opportunities
        )
        
        result = PatternAnalysisResult(
            analysis_id=analysis_id,
            patterns_detected=detected_patterns,
            anti_patterns=anti_patterns,
            optimization_opportunities=optimization_opportunities,
            system_metrics=system_metrics,
            recommendations=recommendations
        )
        
        # Cache result
        self.analysis_history.append(result)
        self.recognition_stats['analyses_performed'] += 1
        
        logger.info(f"Pattern analysis completed: {len(detected_patterns)} patterns, {len(anti_patterns)} anti-patterns")
        return result
    
    async def _match_pattern_template(self, template: PatternTemplate, 
                                    context_data: Dict[str, Any]) -> List[PatternInstance]:
        """Match a pattern template against context data"""
        instances = []
        
        # Extract relevant data
        instructions = context_data.get('instructions', [])
        tags = context_data.get('tags', [])
        routines = context_data.get('routines', [])
        
        # Group instructions by routine for context
        routine_instructions = defaultdict(list)
        for instruction in instructions:
            routine_key = f"{instruction.get('program', 'Unknown')}:{instruction.get('routine', 'Unknown')}"
            routine_instructions[routine_key].append(instruction)
        
        # Try to match pattern in each routine
        for routine_key, routine_instr in routine_instructions.items():
            try:
                routine_instances = await self._match_pattern_in_routine(
                    template, routine_instr, tags, routine_key
                )
                instances.extend(routine_instances)
            except Exception as e:
                logger.debug(f"Pattern matching error in {routine_key}: {e}")
        
        return instances
    
    async def _match_pattern_in_routine(self, template: PatternTemplate,
                                      instructions: List[Dict[str, Any]],
                                      tags: List[Dict[str, Any]],
                                      routine_key: str) -> List[PatternInstance]:
        """Match pattern within a specific routine"""
        instances = []
        
        # Simple pattern matching based on instruction types and relationships
        element_matches = {}
        
        # Find potential matches for each pattern element
        for element in template.elements:
            matches = []
            
            if element.element_type == 'instruction':
                # Look for instruction type matches
                expected_type = element.parameters.get('type')
                if isinstance(expected_type, list):
                    matches = [instr for instr in instructions if instr.get('type') in expected_type]
                else:
                    matches = [instr for instr in instructions if instr.get('type') == expected_type]
            
            elif element.element_type == 'tag':
                # Look for tag matches
                expected_data_type = element.parameters.get('data_type')
                if expected_data_type:
                    matches = [tag for tag in tags if tag.get('data_type') == expected_data_type]
                else:
                    matches = tags
            
            element_matches[element.element_id] = matches
        
        # Try to find valid combinations
        if all(element_matches.values()):  # All elements have potential matches
            # Simplified matching - create instance if we have matches for all elements
            confidence = self._calculate_pattern_confidence(template, element_matches, instructions)
            
            if confidence >= template.confidence_threshold:
                instance_id = hashlib.md5(f"{template.pattern_id}_{routine_key}_{datetime.now()}".encode()).hexdigest()[:8]
                
                # Select best matches for each element
                matched_elements = {}
                for element_id, matches in element_matches.items():
                    if matches:
                        matched_elements[element_id] = matches[0].get('type', str(matches[0]))
                
                # Calculate metrics
                metrics = self._calculate_instance_metrics(template, matched_elements, instructions)
                
                # Generate recommendations
                recommendations = self._generate_instance_recommendations(template, matched_elements)
                
                instance = PatternInstance(
                    template_id=template.pattern_id,
                    instance_id=instance_id,
                    confidence=confidence,
                    location={'routine': routine_key},
                    matched_elements=matched_elements,
                    metrics=metrics,
                    recommendations=recommendations,
                    optimization_potential=self._calculate_optimization_potential(template, metrics)
                )
                
                instances.append(instance)
        
        return instances
    
    def _calculate_pattern_confidence(self, template: PatternTemplate,
                                    element_matches: Dict[str, List[Any]],
                                    instructions: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for pattern match"""
        
        # Base confidence from element matches
        total_elements = len(template.elements)
        matched_elements = sum(1 for matches in element_matches.values() if matches)
        base_confidence = matched_elements / total_elements if total_elements > 0 else 0
        
        # Bonus for relationship satisfaction
        relationship_bonus = 0.0
        if template.relationships:
            satisfied_relationships = 0
            total_relationships = len(template.relationships)
            
            # Simple relationship checking (could be enhanced)
            for rel_type, element_ids in template.relationships.items():
                if all(element_id in element_matches and element_matches[element_id] 
                      for element_id in element_ids):
                    satisfied_relationships += 1
            
            relationship_bonus = (satisfied_relationships / total_relationships) * 0.2
        
        # Complexity adjustment
        complexity_adjustment = {
            PatternComplexity.SIMPLE: 0.1,
            PatternComplexity.MODERATE: 0.0,
            PatternComplexity.COMPLEX: -0.1,
            PatternComplexity.ADVANCED: -0.2
        }.get(template.complexity, 0.0)
        
        final_confidence = min(1.0, base_confidence + relationship_bonus + complexity_adjustment)
        return max(0.0, final_confidence)
    
    def _calculate_instance_metrics(self, template: PatternTemplate,
                                  matched_elements: Dict[str, str],
                                  instructions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for pattern instance"""
        metrics = {
            'instruction_count': len([e for e in matched_elements.values() if e in [i.get('type') for i in instructions]]),
            'complexity_score': len(matched_elements) * template.complexity.value.__len__(),
            'maintainability_score': 8.0 if template.category == PatternCategory.BEST_PRACTICE else 5.0,
            'performance_impact': 3.0 if template.pattern_type == PatternType.TIMING else 1.0
        }
        
        # Add pattern-specific metrics
        if template.pattern_type == PatternType.SAFETY:
            metrics['safety_criticality'] = 9.0
        elif template.pattern_type == PatternType.CONTROL:
            metrics['control_complexity'] = len(matched_elements) * 2.0
        
        return metrics
    
    def _calculate_optimization_potential(self, template: PatternTemplate, 
                                        metrics: Dict[str, float]) -> float:
        """Calculate optimization potential for pattern instance"""
        base_potential = 0.5
        
        # Anti-patterns have high optimization potential
        if template.category == PatternCategory.ANTI_PATTERN:
            base_potential = 0.9
        elif template.category == PatternCategory.OPTIMIZATION_OPPORTUNITY:
            base_potential = 0.8
        elif template.category == PatternCategory.MAINTENANCE_CONCERN:
            base_potential = 0.7
        elif template.category == PatternCategory.BEST_PRACTICE:
            base_potential = 0.2  # Already optimized
        
        # Adjust based on complexity
        complexity_factor = metrics.get('complexity_score', 5.0) / 10.0
        
        return min(1.0, base_potential + complexity_factor * 0.3)
    
    def _generate_instance_recommendations(self, template: PatternTemplate,
                                         matched_elements: Dict[str, str]) -> List[str]:
        """Generate recommendations for pattern instance"""
        recommendations = []
        
        # Template-based recommendations
        recommendations.extend(template.optimizations[:3])  # Limit to top 3
        
        # Category-specific recommendations
        if template.category == PatternCategory.ANTI_PATTERN:
            recommendations.extend([
                "Consider refactoring this pattern for better maintainability",
                "Review best practices for this type of logic"
            ])
        elif template.category == PatternCategory.SAFETY_CRITICAL:
            recommendations.extend([
                "Verify safety system compliance",
                "Consider adding redundancy for critical safety functions"
            ])
        
        return recommendations[:5]  # Limit total recommendations
    
    async def _identify_optimization_opportunities(self, patterns: List[PatternInstance],
                                                 anti_patterns: List[PatternInstance],
                                                 context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""
        opportunities = []
        
        # Anti-pattern optimization opportunities
        for anti_pattern in anti_patterns:
            opportunities.append({
                'type': 'anti_pattern_removal',
                'title': f'Refactor {anti_pattern.template_id}',
                'description': f'Anti-pattern detected with {anti_pattern.confidence:.1%} confidence',
                'priority': 'high',
                'estimated_effort': 'medium',
                'benefits': ['Improved maintainability', 'Reduced complexity'],
                'pattern_instance': anti_pattern
            })
        
        # Pattern consolidation opportunities
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.template_id].append(pattern)
        
        for template_id, instances in pattern_groups.items():
            if len(instances) > 3:  # Multiple instances of same pattern
                opportunities.append({
                    'type': 'pattern_consolidation',
                    'title': f'Consolidate {template_id} patterns',
                    'description': f'Found {len(instances)} instances that could be consolidated',
                    'priority': 'medium',
                    'estimated_effort': 'high',
                    'benefits': ['Reduced code duplication', 'Centralized logic'],
                    'instance_count': len(instances)
                })
        
        # Performance optimization opportunities
        timing_patterns = [p for p in patterns if p.template_id.startswith('timer_')]
        if len(timing_patterns) > 10:
            opportunities.append({
                'type': 'timing_optimization',
                'title': 'Optimize timing logic',
                'description': f'Found {len(timing_patterns)} timing patterns that could be optimized',
                'priority': 'medium',
                'estimated_effort': 'medium',
                'benefits': ['Improved scan time', 'Better timing accuracy'],
                'pattern_count': len(timing_patterns)
            })
        
        return opportunities
    
    def _calculate_system_metrics(self, patterns: List[PatternInstance],
                                anti_patterns: List[PatternInstance],
                                context_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall system metrics"""
        total_instructions = len(context_data.get('instructions', []))
        total_patterns = len(patterns) + len(anti_patterns)
        
        metrics = {
            'pattern_density': total_patterns / max(1, total_instructions) * 100,
            'best_practice_ratio': len([p for p in patterns if p.template_id.endswith('_standard')]) / max(1, len(patterns)),
            'anti_pattern_ratio': len(anti_patterns) / max(1, total_patterns),
            'average_confidence': sum(p.confidence for p in patterns + anti_patterns) / max(1, total_patterns),
            'optimization_potential': sum(p.optimization_potential for p in patterns + anti_patterns) / max(1, total_patterns),
            'complexity_score': sum(p.metrics.get('complexity_score', 0) for p in patterns) / max(1, len(patterns)),
            'maintainability_score': sum(p.metrics.get('maintainability_score', 5) for p in patterns) / max(1, len(patterns)),
            'safety_pattern_count': len([p for p in patterns if 'safety' in p.template_id])
        }
        
        return metrics
    
    def _generate_pattern_recommendations(self, patterns: List[PatternInstance],
                                        anti_patterns: List[PatternInstance],
                                        opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        # Anti-pattern recommendations
        if anti_patterns:
            recommendations.append(
                f"Address {len(anti_patterns)} anti-patterns to improve code quality"
            )
        
        # Pattern consolidation recommendations
        pattern_counts = Counter(p.template_id for p in patterns)
        frequent_patterns = [pid for pid, count in pattern_counts.items() if count > 2]
        
        if frequent_patterns:
            recommendations.append(
                f"Consider creating reusable function blocks for {len(frequent_patterns)} frequently used patterns"  
            )
        
        # Safety recommendations
        safety_patterns = [p for p in patterns if 'safety' in p.template_id]
        if safety_patterns:
            recommendations.append(
                f"Review {len(safety_patterns)} safety patterns for compliance with safety standards"
            )
        
        # Optimization recommendations
        high_potential = [p for p in patterns if p.optimization_potential > 0.7]
        if high_potential:
            recommendations.append(
                f"Focus optimization efforts on {len(high_potential)} high-potential patterns"
            )
        
        # General recommendations
        if len(opportunities) > 5:
            recommendations.append("Prioritize optimization opportunities based on business impact")
        
        recommendations.append("Document pattern implementations for knowledge sharing")
        recommendations.append("Consider establishing coding standards based on detected patterns")
        
        return recommendations[:8]  # Limit to top recommendations
    
    async def _gather_context_data(self) -> Dict[str, Any]:
        """Gather context data for pattern analysis"""
        # Mock context data - in real implementation would integrate with L5X parsing
        return {
            'instructions': [
                # Motor control pattern
                {'type': 'XIC', 'operands': ['Start_Button'], 'program': 'MainProgram', 'routine': 'MotorControl'},
                {'type': 'XIO', 'operands': ['Stop_Button'], 'program': 'MainProgram', 'routine': 'MotorControl'},
                {'type': 'XIC', 'operands': ['Overload_OK'], 'program': 'MainProgram', 'routine': 'MotorControl'},
                {'type': 'XIC', 'operands': ['Motor_Run'], 'program': 'MainProgram', 'routine': 'MotorControl'},  # Seal contact
                {'type': 'OTE', 'operands': ['Motor_Run'], 'program': 'MainProgram', 'routine': 'MotorControl'},
                
                # Timer delay pattern
                {'type': 'XIC', 'operands': ['Delay_Enable'], 'program': 'MainProgram', 'routine': 'TimerLogic'},
                {'type': 'TON', 'operands': ['Delay_Timer'], 'program': 'MainProgram', 'routine': 'TimerLogic'},
                {'type': 'XIC', 'operands': ['Delay_Timer.DN'], 'program': 'MainProgram', 'routine': 'TimerLogic'},
                {'type': 'OTE', 'operands': ['Delayed_Output'], 'program': 'MainProgram', 'routine': 'TimerLogic'},
                
                # Safety chain pattern
                {'type': 'XIC', 'operands': ['Safety_Gate_1'], 'program': 'SafetyProgram', 'routine': 'SafetyChain'},
                {'type': 'XIC', 'operands': ['Safety_Gate_2'], 'program': 'SafetyProgram', 'routine': 'SafetyChain'},
                {'type': 'XIC', 'operands': ['Emergency_Stop'], 'program': 'SafetyProgram', 'routine': 'SafetyChain'},
                {'type': 'OTE', 'operands': ['Safety_OK'], 'program': 'SafetyProgram', 'routine': 'SafetyChain'},
                
                # Anti-pattern: Nested timers
                {'type': 'TON', 'operands': ['Outer_Timer'], 'program': 'MainProgram', 'routine': 'BadTiming'},
                {'type': 'XIC', 'operands': ['Outer_Timer.DN'], 'program': 'MainProgram', 'routine': 'BadTiming'},
                {'type': 'TON', 'operands': ['Inner_Timer'], 'program': 'MainProgram', 'routine': 'BadTiming'},
                
                # Alarm pattern
                {'type': 'XIC', 'operands': ['High_Temp'], 'program': 'AlarmProgram', 'routine': 'TempAlarms'},
                {'type': 'OTL', 'operands': ['Temp_Alarm'], 'program': 'AlarmProgram', 'routine': 'TempAlarms'},
                {'type': 'XIC', 'operands': ['Alarm_Ack'], 'program': 'AlarmProgram', 'routine': 'TempAlarms'},
                {'type': 'OTU', 'operands': ['Temp_Alarm'], 'program': 'AlarmProgram', 'routine': 'TempAlarms'},
            ],
            'tags': [
                {'name': 'Start_Button', 'data_type': 'BOOL', 'scope': 'controller'},
                {'name': 'Stop_Button', 'data_type': 'BOOL', 'scope': 'controller'},
                {'name': 'Motor_Run', 'data_type': 'BOOL', 'scope': 'controller'},
                {'name': 'Overload_OK', 'data_type': 'BOOL', 'scope': 'controller'},
                {'name': 'Delay_Timer', 'data_type': 'TIMER', 'scope': 'program'},
                {'name': 'Safety_Gate_1', 'data_type': 'BOOL', 'scope': 'controller'},
                {'name': 'Safety_Gate_2', 'data_type': 'BOOL', 'scope': 'controller'},
                {'name': 'Emergency_Stop', 'data_type': 'BOOL', 'scope': 'controller'},
                {'name': 'High_Temp', 'data_type': 'BOOL', 'scope': 'controller'},
                {'name': 'Temp_Alarm', 'data_type': 'BOOL', 'scope': 'controller'},
            ],
            'routines': [
                {'name': 'MotorControl', 'program': 'MainProgram'},
                {'name': 'TimerLogic', 'program': 'MainProgram'},
                {'name': 'SafetyChain', 'program': 'SafetyProgram'},
                {'name': 'TempAlarms', 'program': 'AlarmProgram'},
                {'name': 'BadTiming', 'program': 'MainProgram'}
            ]
        }
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern recognition statistics"""
        return {
            'total_templates': len(self.pattern_templates),
            'template_categories': {
                category.value: len([t for t in self.pattern_templates.values() if t.category == category])
                for category in PatternCategory
            },
            'template_types': {
                ptype.value: len([t for t in self.pattern_templates.values() if t.pattern_type == ptype])
                for ptype in PatternType
            },
            'recognition_stats': dict(self.recognition_stats),
            'cache_size': len(self.recognition_cache),
            'analysis_history_count': len(self.analysis_history)
        }


# Convenience functions
async def create_logic_pattern_recognizer(enhanced_search_engine: Optional[Any] = None) -> LogicPatternRecognizer:
    """Create logic pattern recognizer"""
    return LogicPatternRecognizer(enhanced_search_engine)


async def analyze_plc_patterns(context_data: Optional[Dict[str, Any]] = None,
                              enhanced_search_engine: Optional[Any] = None) -> PatternAnalysisResult:
    """Analyze PLC patterns in context data"""
    recognizer = await create_logic_pattern_recognizer(enhanced_search_engine)
    return await recognizer.analyze_patterns(context_data)


async def detect_anti_patterns(context_data: Optional[Dict[str, Any]] = None) -> List[PatternInstance]:
    """Detect anti-patterns in PLC logic"""
    result = await analyze_plc_patterns(context_data)
    return result.anti_patterns


async def find_optimization_opportunities(context_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Find optimization opportunities in PLC logic"""
    result = await analyze_plc_patterns(context_data)
    return result.optimization_opportunities


# Export main classes and functions
__all__ = [
    'LogicPatternRecognizer',
    'PatternTemplate',
    'PatternInstance',
    'PatternAnalysisResult',
    'PatternType',
    'PatternComplexity',
    'PatternCategory',
    'PatternElement',
    'create_logic_pattern_recognizer',
    'analyze_plc_patterns',
    'detect_anti_patterns',
    'find_optimization_opportunities'
]
