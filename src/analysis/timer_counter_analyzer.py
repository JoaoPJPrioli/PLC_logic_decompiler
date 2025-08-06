"""
Timer and Counter Analysis Module for Step 13

This module provides specialized analysis capabilities for timer and counter instructions
in PLC ladder logic systems, building on the routine analysis foundation from Step 12.

Features:
- TON (Timer On Delay) analysis with timing parameters
- TOF (Timer Off Delay) analysis with timing parameters  
- RTO (Retentive Timer On) analysis with retentive behavior
- CTU (Count Up) analysis with counting logic
- CTD (Count Down) analysis with counting logic
- Timer/Counter state tracking and lifecycle analysis
- Timing relationship analysis and dependency mapping
- Performance metrics for timing operations
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class TimerType(Enum):
    """Timer instruction types"""
    TON = "TON"  # Timer On Delay
    TOF = "TOF"  # Timer Off Delay
    RTO = "RTO"  # Retentive Timer On


class CounterType(Enum):
    """Counter instruction types"""
    CTU = "CTU"  # Count Up
    CTD = "CTD"  # Count Down


class TimerState(Enum):
    """Timer operational states"""
    IDLE = "idle"
    TIMING = "timing"
    DONE = "done"
    RESET = "reset"


class CounterState(Enum):
    """Counter operational states"""
    AT_ZERO = "at_zero"
    COUNTING = "counting"
    AT_PRESET = "at_preset"
    OVERFLOW = "overflow"


@dataclass
class TimerInfo:
    """Information about a timer instruction"""
    name: str
    timer_type: TimerType
    tag_name: str
    preset_value: Optional[int] = None
    preset_tag: Optional[str] = None
    accumulator_tag: Optional[str] = None
    enable_bit: Optional[str] = None
    timer_timing_bit: Optional[str] = None
    done_bit: Optional[str] = None
    routine_name: str = ""
    rung_number: int = 0
    instruction_index: int = 0
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    timing_relationships: List[str] = field(default_factory=list)
    estimated_cycle_time: float = 0.0


@dataclass
class CounterInfo:
    """Information about a counter instruction"""
    name: str
    counter_type: CounterType
    tag_name: str
    preset_value: Optional[int] = None
    preset_tag: Optional[str] = None
    accumulator_tag: Optional[str] = None
    count_up_bit: Optional[str] = None
    count_down_bit: Optional[str] = None
    done_bit: Optional[str] = None
    overflow_bit: Optional[str] = None
    underflow_bit: Optional[str] = None
    routine_name: str = ""
    rung_number: int = 0
    instruction_index: int = 0
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    counting_relationships: List[str] = field(default_factory=list)
    estimated_count_rate: float = 0.0


@dataclass
class TimingChain:
    """Represents a chain of related timers"""
    chain_id: str
    timers: List[str]
    total_time: float
    chain_type: str  # "sequential", "parallel", "cascaded"
    trigger_conditions: List[str] = field(default_factory=list)
    completion_actions: List[str] = field(default_factory=list)


@dataclass
class CountingChain:
    """Represents a chain of related counters"""
    chain_id: str
    counters: List[str]
    total_count: int
    chain_type: str  # "sequential", "parallel", "cascaded"
    trigger_conditions: List[str] = field(default_factory=list)
    completion_actions: List[str] = field(default_factory=list)


class TimerCounterAnalyzer:
    """Advanced analyzer for timer and counter instructions"""
    
    def __init__(self):
        """Initialize the timer/counter analyzer"""
        self.timers: Dict[str, TimerInfo] = {}
        self.counters: Dict[str, CounterInfo] = {}
        self.timing_chains: List[TimingChain] = []
        self.counting_chains: List[CountingChain] = []
        
        # Pattern matching for timer/counter instructions
        self.timer_patterns = {
            TimerType.TON: re.compile(r'TON\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', re.IGNORECASE),
            TimerType.TOF: re.compile(r'TOF\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', re.IGNORECASE),
            TimerType.RTO: re.compile(r'RTO\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', re.IGNORECASE)
        }
        
        self.counter_patterns = {
            CounterType.CTU: re.compile(r'CTU\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', re.IGNORECASE),
            CounterType.CTD: re.compile(r'CTD\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', re.IGNORECASE)
        }
        
        # Analysis cache
        self._analysis_cache: Dict[str, Any] = {}
    
    def analyze_timers_and_counters(self, ladder_routines: List[Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of timers and counters in ladder logic
        
        Args:
            ladder_routines: List of ladder logic routines to analyze
            
        Returns:
            Dictionary containing comprehensive timer/counter analysis
        """
        logger.info("Starting comprehensive timer and counter analysis...")
        
        start_time = time.time()
        
        # Step 1: Extract and analyze all timers and counters
        self._extract_timers_and_counters(ladder_routines)
        
        # Step 2: Analyze timing and counting relationships
        self._analyze_timing_relationships()
        self._analyze_counting_relationships()
        
        # Step 3: Identify timing and counting chains
        self._identify_timing_chains()
        self._identify_counting_chains()
        
        # Step 4: Perform performance analysis
        performance_analysis = self._analyze_performance()
        
        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations()
        
        analysis_time = time.time() - start_time
        
        # Compile results
        results = {
            'success': True,
            'analysis_time': analysis_time,
            'summary': {
                'total_timers': len(self.timers),
                'total_counters': len(self.counters),
                'timing_chains': len(self.timing_chains),
                'counting_chains': len(self.counting_chains)
            },
            'timers': {name: self._timer_to_dict(timer) for name, timer in self.timers.items()},
            'counters': {name: self._counter_to_dict(counter) for name, counter in self.counters.items()},
            'timing_chains': [self._timing_chain_to_dict(chain) for chain in self.timing_chains],
            'counting_chains': [self._counting_chain_to_dict(chain) for chain in self.counting_chains],
            'performance_analysis': performance_analysis,
            'recommendations': recommendations
        }
        
        logger.info(f"Timer/Counter analysis completed in {analysis_time:.3f}s")
        logger.info(f"Found {len(self.timers)} timers and {len(self.counters)} counters")
        
        return results
    
    def _extract_timers_and_counters(self, ladder_routines: List[Any]):
        """Extract timer and counter instructions from ladder logic"""
        logger.info("Extracting timer and counter instructions...")
        
        for routine in ladder_routines:
            if not hasattr(routine, 'rungs'):
                continue
                
            for rung_idx, rung in enumerate(routine.rungs):
                if not hasattr(rung, 'instructions'):
                    continue
                    
                for inst_idx, instruction in enumerate(rung.instructions):
                    # Check for timer instructions
                    timer_info = self._parse_timer_instruction(instruction, routine.name, rung_idx, inst_idx)
                    if timer_info:
                        self.timers[timer_info.name] = timer_info
                    
                    # Check for counter instructions
                    counter_info = self._parse_counter_instruction(instruction, routine.name, rung_idx, inst_idx)
                    if counter_info:
                        self.counters[counter_info.name] = counter_info
    
    def _parse_timer_instruction(self, instruction, routine_name: str, rung_idx: int, inst_idx: int) -> Optional[TimerInfo]:
        """Parse a timer instruction"""
        if not hasattr(instruction, 'raw_text') or not instruction.raw_text:
            return None
        
        raw_text = instruction.raw_text.strip()
        
        for timer_type, pattern in self.timer_patterns.items():
            match = pattern.search(raw_text)
            if match:
                tag_name = match.group(1).strip()
                preset_param = match.group(2).strip()
                # Third parameter might be accumulator or additional settings
                
                timer_info = TimerInfo(
                    name=f"{routine_name}_R{rung_idx}_{timer_type.value}_{inst_idx}",
                    timer_type=timer_type,
                    tag_name=tag_name,
                    routine_name=routine_name,
                    rung_number=rung_idx,
                    instruction_index=inst_idx
                )
                
                # Parse preset value
                if preset_param.isdigit():
                    timer_info.preset_value = int(preset_param)
                else:
                    timer_info.preset_tag = preset_param
                
                # Generate standard bit references
                timer_info.enable_bit = f"{tag_name}.EN"
                timer_info.timer_timing_bit = f"{tag_name}.TT"
                timer_info.done_bit = f"{tag_name}.DN"
                timer_info.accumulator_tag = f"{tag_name}.ACC"
                
                return timer_info
        
        return None
    
    def _parse_counter_instruction(self, instruction, routine_name: str, rung_idx: int, inst_idx: int) -> Optional[CounterInfo]:
        """Parse a counter instruction"""
        if not hasattr(instruction, 'raw_text') or not instruction.raw_text:
            return None
        
        raw_text = instruction.raw_text.strip()
        
        for counter_type, pattern in self.counter_patterns.items():
            match = pattern.search(raw_text)
            if match:
                tag_name = match.group(1).strip()
                preset_param = match.group(2).strip()
                
                counter_info = CounterInfo(
                    name=f"{routine_name}_R{rung_idx}_{counter_type.value}_{inst_idx}",
                    counter_type=counter_type,
                    tag_name=tag_name,
                    routine_name=routine_name,
                    rung_number=rung_idx,
                    instruction_index=inst_idx
                )
                
                # Parse preset value
                if preset_param.isdigit():
                    counter_info.preset_value = int(preset_param)
                else:
                    counter_info.preset_tag = preset_param
                
                # Generate standard bit references
                counter_info.count_up_bit = f"{tag_name}.CU"
                counter_info.count_down_bit = f"{tag_name}.CD"
                counter_info.done_bit = f"{tag_name}.DN"
                counter_info.overflow_bit = f"{tag_name}.OV"
                counter_info.underflow_bit = f"{tag_name}.UN"
                counter_info.accumulator_tag = f"{tag_name}.ACC"
                
                return counter_info
        
        return None
    
    def _analyze_timing_relationships(self):
        """Analyze relationships between timers"""
        logger.info("Analyzing timing relationships...")
        
        # Simple relationship analysis based on tag usage
        timer_names = list(self.timers.keys())
        
        for i, timer1_name in enumerate(timer_names):
            timer1 = self.timers[timer1_name]
            
            for j, timer2_name in enumerate(timer_names[i+1:], i+1):
                timer2 = self.timers[timer2_name]
                
                # Check if timer1 done bit might trigger timer2
                if self._check_timer_relationship(timer1, timer2):
                    timer1.timing_relationships.append(timer2_name)
                    timer2.dependencies.add(timer1_name)
                    timer1.dependents.add(timer2_name)
    
    def _analyze_counting_relationships(self):
        """Analyze relationships between counters"""
        logger.info("Analyzing counting relationships...")
        
        # Simple relationship analysis
        counter_names = list(self.counters.keys())
        
        for i, counter1_name in enumerate(counter_names):
            counter1 = self.counters[counter1_name]
            
            for j, counter2_name in enumerate(counter_names[i+1:], i+1):
                counter2 = self.counters[counter2_name]
                
                # Check if counter1 done bit might trigger counter2
                if self._check_counter_relationship(counter1, counter2):
                    counter1.counting_relationships.append(counter2_name)
                    counter2.dependencies.add(counter1_name)
                    counter1.dependents.add(counter2_name)
    
    def _check_timer_relationship(self, timer1: TimerInfo, timer2: TimerInfo) -> bool:
        """Check if two timers have a relationship"""
        # Simple heuristic: if they're in the same routine and close together
        return (timer1.routine_name == timer2.routine_name and 
                abs(timer1.rung_number - timer2.rung_number) <= 3)
    
    def _check_counter_relationship(self, counter1: CounterInfo, counter2: CounterInfo) -> bool:
        """Check if two counters have a relationship"""
        # Simple heuristic: if they're in the same routine and close together
        return (counter1.routine_name == counter2.routine_name and 
                abs(counter1.rung_number - counter2.rung_number) <= 3)
    
    def _identify_timing_chains(self):
        """Identify chains of related timers"""
        logger.info("Identifying timing chains...")
        
        visited = set()
        chain_id = 0
        
        for timer_name, timer_info in self.timers.items():
            if timer_name in visited:
                continue
            
            # Find connected timers
            chain_timers = self._find_connected_timers(timer_name, visited)
            
            if len(chain_timers) > 1:
                total_time = sum(
                    self.timers[t].preset_value or 0 
                    for t in chain_timers 
                    if self.timers[t].preset_value
                )
                
                chain = TimingChain(
                    chain_id=f"timing_chain_{chain_id}",
                    timers=chain_timers,
                    total_time=total_time,
                    chain_type="sequential"  # Simplified assumption
                )
                
                self.timing_chains.append(chain)
                chain_id += 1
    
    def _identify_counting_chains(self):
        """Identify chains of related counters"""
        logger.info("Identifying counting chains...")
        
        visited = set()
        chain_id = 0
        
        for counter_name, counter_info in self.counters.items():
            if counter_name in visited:
                continue
            
            # Find connected counters
            chain_counters = self._find_connected_counters(counter_name, visited)
            
            if len(chain_counters) > 1:
                total_count = sum(
                    self.counters[c].preset_value or 0 
                    for c in chain_counters 
                    if self.counters[c].preset_value
                )
                
                chain = CountingChain(
                    chain_id=f"counting_chain_{chain_id}",
                    counters=chain_counters,
                    total_count=total_count,
                    chain_type="sequential"  # Simplified assumption
                )
                
                self.counting_chains.append(chain)
                chain_id += 1
    
    def _find_connected_timers(self, start_timer: str, visited: Set[str]) -> List[str]:
        """Find all timers connected to the start timer"""
        connected = []
        queue = [start_timer]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            connected.append(current)
            
            # Add related timers to queue
            if current in self.timers:
                timer_info = self.timers[current]
                for related_timer in timer_info.timing_relationships:
                    if related_timer not in visited:
                        queue.append(related_timer)
        
        return connected
    
    def _find_connected_counters(self, start_counter: str, visited: Set[str]) -> List[str]:
        """Find all counters connected to the start counter"""
        connected = []
        queue = [start_counter]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            connected.append(current)
            
            # Add related counters to queue
            if current in self.counters:
                counter_info = self.counters[current]
                for related_counter in counter_info.counting_relationships:
                    if related_counter not in visited:
                        queue.append(related_counter)
        
        return connected
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        logger.info("Analyzing performance characteristics...")
        
        # Calculate timing statistics
        timer_presets = [
            timer.preset_value for timer in self.timers.values() 
            if timer.preset_value is not None
        ]
        
        counter_presets = [
            counter.preset_value for counter in self.counters.values() 
            if counter.preset_value is not None
        ]
        
        return {
            'timing_analysis': {
                'total_timers': len(self.timers),
                'average_preset_time': sum(timer_presets) / len(timer_presets) if timer_presets else 0,
                'max_preset_time': max(timer_presets) if timer_presets else 0,
                'min_preset_time': min(timer_presets) if timer_presets else 0,
                'total_timing_chains': len(self.timing_chains)
            },
            'counting_analysis': {
                'total_counters': len(self.counters),
                'average_preset_count': sum(counter_presets) / len(counter_presets) if counter_presets else 0,
                'max_preset_count': max(counter_presets) if counter_presets else 0,
                'min_preset_count': min(counter_presets) if counter_presets else 0,
                'total_counting_chains': len(self.counting_chains)
            }
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check for very long timers
        long_timers = [
            (name, timer.preset_value) 
            for name, timer in self.timers.items() 
            if timer.preset_value and timer.preset_value > 10000  # > 10 seconds
        ]
        
        if long_timers:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'title': 'Long Duration Timers',
                'description': f'Found {len(long_timers)} timers with very long durations. Consider if these are appropriate.',
                'affected_timers': long_timers[:5]
            })
        
        # Check for complex timing chains
        complex_chains = [chain for chain in self.timing_chains if len(chain.timers) > 5]
        
        if complex_chains:
            recommendations.append({
                'type': 'complexity',
                'priority': 'low',
                'title': 'Complex Timing Chains',
                'description': f'Found {len(complex_chains)} complex timing chains. Consider simplifying if possible.',
                'complex_chains': [chain.chain_id for chain in complex_chains]
            })
        
        return recommendations
    
    def _timer_to_dict(self, timer: TimerInfo) -> Dict[str, Any]:
        """Convert TimerInfo to dictionary"""
        return {
            'name': timer.name,
            'type': timer.timer_type.value,
            'tag_name': timer.tag_name,
            'preset_value': timer.preset_value,
            'preset_tag': timer.preset_tag,
            'routine': timer.routine_name,
            'rung': timer.rung_number,
            'dependencies': list(timer.dependencies),
            'dependents': list(timer.dependents),
            'relationships': timer.timing_relationships
        }
    
    def _counter_to_dict(self, counter: CounterInfo) -> Dict[str, Any]:
        """Convert CounterInfo to dictionary"""
        return {
            'name': counter.name,
            'type': counter.counter_type.value,
            'tag_name': counter.tag_name,
            'preset_value': counter.preset_value,
            'preset_tag': counter.preset_tag,
            'routine': counter.routine_name,
            'rung': counter.rung_number,
            'dependencies': list(counter.dependencies),
            'dependents': list(counter.dependents),
            'relationships': counter.counting_relationships
        }
    
    def _timing_chain_to_dict(self, chain: TimingChain) -> Dict[str, Any]:
        """Convert TimingChain to dictionary"""
        return {
            'chain_id': chain.chain_id,
            'timers': chain.timers,
            'total_time': chain.total_time,
            'chain_type': chain.chain_type,
            'timer_count': len(chain.timers)
        }
    
    def _counting_chain_to_dict(self, chain: CountingChain) -> Dict[str, Any]:
        """Convert CountingChain to dictionary"""
        return {
            'chain_id': chain.chain_id,
            'counters': chain.counters,
            'total_count': chain.total_count,
            'chain_type': chain.chain_type,
            'counter_count': len(chain.counters)
        }
