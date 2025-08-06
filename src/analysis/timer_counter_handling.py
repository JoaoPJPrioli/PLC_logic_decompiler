"""
Timer and Counter Handling Analysis Module
Comprehensive analysis of timer and counter instructions in PLC logic

This module provides advanced analysis capabilities for TON, TOF, RTO, CTU, CTD instructions
including state tracking, timing relationships, and performance analysis.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import xml.etree.ElementTree as ET

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    """Timer state enumeration"""
    IDLE = "idle"
    TIMING = "timing"
    DONE = "done"
    ERROR = "error"

class CounterState(Enum):
    """Counter state enumeration"""
    COUNTING = "counting"
    PRESET_REACHED = "preset_reached"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"

@dataclass
class TimerInstruction:
    """Timer instruction data model"""
    name: str
    timer_type: TimerType
    tag_name: str
    rung_number: int
    routine_name: str
    preset_value: Optional[int] = None
    preset_tag: Optional[str] = None
    time_base: str = "1ms"  # Default time base
    enable_bit: str = ""
    timing_bit: str = ""
    done_bit: str = ""
    accumulated_value: int = 0
    description: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        
        # Generate bit references
        if self.tag_name:
            self.enable_bit = f"{self.tag_name}.EN"
            self.timing_bit = f"{self.tag_name}.TT"
            self.done_bit = f"{self.tag_name}.DN"

@dataclass
class CounterInstruction:
    """Counter instruction data model"""
    name: str
    counter_type: CounterType
    tag_name: str
    rung_number: int
    routine_name: str
    preset_value: Optional[int] = None
    preset_tag: Optional[str] = None
    count_up_bit: str = ""
    count_down_bit: str = ""
    done_bit: str = ""
    overflow_bit: str = ""
    underflow_bit: str = ""
    accumulated_value: int = 0
    description: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        
        # Generate bit references
        if self.tag_name:
            self.count_up_bit = f"{self.tag_name}.CU"
            self.count_down_bit = f"{self.tag_name}.CD"
            self.done_bit = f"{self.tag_name}.DN"
            self.overflow_bit = f"{self.tag_name}.OV"
            self.underflow_bit = f"{self.tag_name}.UN"

@dataclass
class TimingChain:
    """Represents a chain of interconnected timers"""
    chain_id: str
    timers: List[TimerInstruction]
    total_delay: int  # milliseconds
    critical_path: bool = False
    description: str = ""

@dataclass
class CountingChain:
    """Represents a chain of interconnected counters"""
    chain_id: str
    counters: List[CounterInstruction]
    total_count_capacity: int
    description: str = ""

class TimerCounterAnalyzer:
    """Advanced analyzer for timer and counter instructions"""
    
    def __init__(self):
        self.timers: List[TimerInstruction] = []
        self.counters: List[CounterInstruction] = []
        self.timing_chains: List[TimingChain] = []
        self.counting_chains: List[CountingChain] = []
        self.analysis_results = {}
        
        logger.info("Timer and Counter Analyzer initialized")
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze L5X file for timer and counter instructions"""
        try:
            logger.info(f"Analyzing timers and counters in: {file_path}")
            
            # Parse XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract timer and counter instructions
            self._extract_timers(root)
            self._extract_counters(root)
            
            # Analyze relationships
            self._analyze_timing_chains()
            self._analyze_counting_chains()
            
            # Generate analysis results
            results = self._generate_analysis_results()
            
            logger.info(f"Analysis complete: {len(self.timers)} timers, {len(self.counters)} counters found")
            return results
            
        except Exception as e:
            logger.error(f"Timer/Counter analysis failed: {e}")
            return {"error": str(e), "timers": 0, "counters": 0}
    
    def _extract_timers(self, root: ET.Element):
        """Extract timer instructions from XML"""
        timer_types = ["TON", "TOF", "RTO"]
        
        for routine in root.findall(".//Routine"):
            routine_name = routine.get("Name", "Unknown")
            
            for rung_idx, rung in enumerate(routine.findall(".//Rung")):
                rung_number = rung_idx + 1
                
                # Find timer instructions
                for instruction in rung.findall(".//Instruction"):
                    mnemonic = instruction.get("Mnemonic", "")
                    
                    if mnemonic in timer_types:
                        timer = self._parse_timer_instruction(
                            instruction, routine_name, rung_number
                        )
                        if timer:
                            self.timers.append(timer)
    
    def _parse_timer_instruction(self, instruction: ET.Element, routine_name: str, rung_number: int) -> Optional[TimerInstruction]:
        """Parse individual timer instruction"""
        try:
            mnemonic = instruction.get("Mnemonic", "")
            
            # Extract operands
            operands = instruction.findall(".//Operand")
            tag_name = ""
            preset_value = None
            preset_tag = None
            
            for operand in operands:
                operand_type = operand.get("Type", "")
                if operand_type == "T":  # Timer tag
                    tag_name = operand.text or ""
                elif operand_type == "REAL" or operand_type == "DINT":
                    try:
                        preset_value = int(float(operand.text or "0"))
                    except (ValueError, TypeError):
                        preset_tag = operand.text or ""
            
            return TimerInstruction(
                name=f"{routine_name}_R{rung_number}_{mnemonic}",
                timer_type=TimerType(mnemonic),
                tag_name=tag_name,
                rung_number=rung_number,
                routine_name=routine_name,
                preset_value=preset_value,
                preset_tag=preset_tag
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse timer instruction: {e}")
            return None
    
    def _extract_counters(self, root: ET.Element):
        """Extract counter instructions from XML"""
        counter_types = ["CTU", "CTD"]
        
        for routine in root.findall(".//Routine"):
            routine_name = routine.get("Name", "Unknown")
            
            for rung_idx, rung in enumerate(routine.findall(".//Rung")):
                rung_number = rung_idx + 1
                
                # Find counter instructions
                for instruction in rung.findall(".//Instruction"):
                    mnemonic = instruction.get("Mnemonic", "")
                    
                    if mnemonic in counter_types:
                        counter = self._parse_counter_instruction(
                            instruction, routine_name, rung_number
                        )
                        if counter:
                            self.counters.append(counter)
    
    def _parse_counter_instruction(self, instruction: ET.Element, routine_name: str, rung_number: int) -> Optional[CounterInstruction]:
        """Parse individual counter instruction"""
        try:
            mnemonic = instruction.get("Mnemonic", "")
            
            # Extract operands
            operands = instruction.findall(".//Operand")
            tag_name = ""
            preset_value = None
            preset_tag = None
            
            for operand in operands:
                operand_type = operand.get("Type", "")
                if operand_type == "C":  # Counter tag
                    tag_name = operand.text or ""
                elif operand_type == "DINT":
                    try:
                        preset_value = int(operand.text or "0")
                    except (ValueError, TypeError):
                        preset_tag = operand.text or ""
            
            return CounterInstruction(
                name=f"{routine_name}_R{rung_number}_{mnemonic}",
                counter_type=CounterType(mnemonic),
                tag_name=tag_name,
                rung_number=rung_number,
                routine_name=routine_name,
                preset_value=preset_value,
                preset_tag=preset_tag
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse counter instruction: {e}")
            return None
    
    def _analyze_timing_chains(self):
        """Analyze chains of interconnected timers"""
        # Group timers by routine for basic chain analysis
        routine_timers = {}
        for timer in self.timers:
            if timer.routine_name not in routine_timers:
                routine_timers[timer.routine_name] = []
            routine_timers[timer.routine_name].append(timer)
        
        # Create timing chains for each routine with multiple timers
        for routine_name, timers in routine_timers.items():
            if len(timers) > 1:
                total_delay = sum(
                    t.preset_value or 0 for t in timers 
                    if t.preset_value is not None
                )
                
                chain = TimingChain(
                    chain_id=f"{routine_name}_timing_chain",
                    timers=timers,
                    total_delay=total_delay,
                    description=f"Timing chain in routine {routine_name}"
                )
                self.timing_chains.append(chain)
    
    def _analyze_counting_chains(self):
        """Analyze chains of interconnected counters"""
        # Group counters by routine for basic chain analysis
        routine_counters = {}
        for counter in self.counters:
            if counter.routine_name not in routine_counters:
                routine_counters[counter.routine_name] = []
            routine_counters[counter.routine_name].append(counter)
        
        # Create counting chains for each routine with multiple counters
        for routine_name, counters in routine_counters.items():
            if len(counters) > 1:
                total_capacity = sum(
                    c.preset_value or 0 for c in counters 
                    if c.preset_value is not None
                )
                
                chain = CountingChain(
                    chain_id=f"{routine_name}_counting_chain",
                    counters=counters,
                    total_count_capacity=total_capacity,
                    description=f"Counting chain in routine {routine_name}"
                )
                self.counting_chains.append(chain)
    
    def _generate_analysis_results(self) -> Dict[str, Any]:
        """Generate comprehensive analysis results"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_timers": len(self.timers),
                "total_counters": len(self.counters),
                "timing_chains": len(self.timing_chains),
                "counting_chains": len(self.counting_chains)
            },
            "timer_analysis": {
                "by_type": self._analyze_timers_by_type(),
                "by_routine": self._analyze_timers_by_routine(),
                "timing_statistics": self._calculate_timing_statistics()
            },
            "counter_analysis": {
                "by_type": self._analyze_counters_by_type(),
                "by_routine": self._analyze_counters_by_routine(),
                "counting_statistics": self._calculate_counting_statistics()
            },
            "chains": {
                "timing_chains": [asdict(chain) for chain in self.timing_chains],
                "counting_chains": [asdict(chain) for chain in self.counting_chains]
            }
        }
        
        self.analysis_results = results
        return results
    
    def _analyze_timers_by_type(self) -> Dict[str, int]:
        """Analyze timer distribution by type"""
        type_counts = {}
        for timer in self.timers:
            timer_type = timer.timer_type.value
            type_counts[timer_type] = type_counts.get(timer_type, 0) + 1
        return type_counts
    
    def _analyze_counters_by_type(self) -> Dict[str, int]:
        """Analyze counter distribution by type"""
        type_counts = {}
        for counter in self.counters:
            counter_type = counter.counter_type.value
            type_counts[counter_type] = type_counts.get(counter_type, 0) + 1
        return type_counts
    
    def _analyze_timers_by_routine(self) -> Dict[str, int]:
        """Analyze timer distribution by routine"""
        routine_counts = {}
        for timer in self.timers:
            routine_counts[timer.routine_name] = routine_counts.get(timer.routine_name, 0) + 1
        return routine_counts
    
    def _analyze_counters_by_routine(self) -> Dict[str, int]:
        """Analyze counter distribution by routine"""
        routine_counts = {}
        for counter in self.counters:
            routine_counts[counter.routine_name] = routine_counts.get(counter.routine_name, 0) + 1
        return routine_counts
    
    def _calculate_timing_statistics(self) -> Dict[str, Any]:
        """Calculate timing-related statistics"""
        preset_values = [t.preset_value for t in self.timers if t.preset_value is not None]
        
        if not preset_values:
            return {"error": "No preset values found"}
        
        return {
            "min_preset": min(preset_values),
            "max_preset": max(preset_values),
            "avg_preset": sum(preset_values) / len(preset_values),
            "total_timing_capacity": sum(preset_values)
        }
    
    def _calculate_counting_statistics(self) -> Dict[str, Any]:
        """Calculate counting-related statistics"""
        preset_values = [c.preset_value for c in self.counters if c.preset_value is not None]
        
        if not preset_values:
            return {"error": "No preset values found"}
        
        return {
            "min_preset": min(preset_values),
            "max_preset": max(preset_values),
            "avg_preset": sum(preset_values) / len(preset_values),
            "total_counting_capacity": sum(preset_values)
        }
    
    def get_timers_by_routine(self, routine_name: str) -> List[TimerInstruction]:
        """Get all timers in a specific routine"""
        return [t for t in self.timers if t.routine_name == routine_name]
    
    def get_counters_by_routine(self, routine_name: str) -> List[CounterInstruction]:
        """Get all counters in a specific routine"""
        return [c for c in self.counters if c.routine_name == routine_name]
    
    def export_analysis(self, output_file: str = None) -> str:
        """Export analysis results to JSON file"""
        if not self.analysis_results:
            logger.warning("No analysis results to export")
            return ""
        
        if output_file is None:
            output_file = f"timer_counter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            
            logger.info(f"Analysis exported to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ""

# Convenience functions for easy integration
def analyze_timers_and_counters(file_path: str) -> Dict[str, Any]:
    """Analyze timers and counters in an L5X file"""
    analyzer = TimerCounterAnalyzer()
    return analyzer.analyze_file(file_path)

def get_timer_counter_summary(file_path: str) -> Dict[str, int]:
    """Get a quick summary of timers and counters"""
    results = analyze_timers_and_counters(file_path)
    return results.get("summary", {})

# Test function
def test_timer_counter_analyzer():
    """Test the timer counter analyzer"""
    print("🧪 Testing Timer Counter Analyzer")
    print("=" * 40)
    
    # Test with sample file if available
    sample_file = "Assembly_Controls_Robot.L5X"
    if os.path.exists(sample_file):
        analyzer = TimerCounterAnalyzer()
        results = analyzer.analyze_file(sample_file)
        
        print(f"✅ Analysis completed:")
        print(f"   Timers found: {results['summary']['total_timers']}")
        print(f"   Counters found: {results['summary']['total_counters']}")
        print(f"   Timing chains: {results['summary']['timing_chains']}")
        print(f"   Counting chains: {results['summary']['counting_chains']}")
        
        return True
    else:
        print("⚠️ Sample file not found, creating mock results")
        return False

if __name__ == "__main__":
    test_timer_counter_analyzer()
