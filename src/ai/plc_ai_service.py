"""
PLC AI Service
AI service specifically designed for PLC code analysis and generation.

This module provides:
- PLC-specific AI prompting and context management
- Integration with PLC analysis results for AI input
- Code generation pipeline with validation
- Specialized AI operations for PLC systems
- Context optimization for PLC-specific tasks
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .ai_interface import (
    AIInterfaceManager, AIMessage, AIResponse, 
    ModelCapability, AIProvider
)

logger = logging.getLogger(__name__)


@dataclass
class PLCContext:
    """PLC-specific context for AI operations."""
    controller_name: str = ""
    tags: List[Dict[str, Any]] = field(default_factory=list)
    routines: List[Dict[str, Any]] = field(default_factory=list)
    udts: List[Dict[str, Any]] = field(default_factory=list)
    arrays: List[Dict[str, Any]] = field(default_factory=list)
    timers: List[Dict[str, Any]] = field(default_factory=list)
    counters: List[Dict[str, Any]] = field(default_factory=list)
    io_modules: List[Dict[str, Any]] = field(default_factory=list)
    logic_patterns: List[Dict[str, Any]] = field(default_factory=list)
    critical_paths: List[str] = field(default_factory=list)
    safety_concerns: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)


@dataclass
class CodeGenerationRequest:
    """Request for AI code generation."""
    task_description: str
    target_language: str = "python"
    include_comments: bool = True
    include_error_handling: bool = True
    code_style: str = "pep8"
    max_complexity: int = 10
    specific_tags: List[str] = field(default_factory=list)
    specific_routines: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class CodeGenerationResult:
    """Result from AI code generation."""
    generated_code: str
    explanation: str
    validation_notes: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    confidence_score: float = 0.0
    ai_response: Optional[AIResponse] = None


class PLCAIService:
    """AI service for PLC code analysis and generation."""
    
    def __init__(self, ai_manager: Optional[AIInterfaceManager] = None):
        self.ai_manager = ai_manager or AIInterfaceManager()
        self.plc_context = PLCContext()
        
        # PLC-specific templates and prompts
        self.system_prompts = self._load_system_prompts()
        self.code_templates = self._load_code_templates()
        
        # Generation history
        self.generation_history: List[CodeGenerationResult] = []
        
        logger.info("PLC AI Service initialized")
        
    def _load_system_prompts(self) -> Dict[str, str]:
        """Load PLC-specific system prompts."""
        return {
            "code_generation": """You are an expert PLC programmer and Python developer specializing in industrial automation.
Your task is to generate clean, efficient Python code for PLC data acquisition and control using pycomm3.

Key principles:
- Generate production-ready, well-documented code
- Follow PEP 8 style guidelines
- Include proper error handling and logging
- Use appropriate data types and validation
- Optimize for performance and reliability
- Consider industrial safety requirements
- Provide clear explanations and comments

You have access to detailed PLC analysis including tags, routines, I/O configuration, and logic patterns.
Use this information to generate contextually appropriate code.""",

            "analysis": """You are an expert PLC analyst with deep knowledge of ladder logic, industrial automation, and control systems.
Your task is to analyze PLC programs and provide insights, recommendations, and optimization suggestions.

Focus areas:
- Logic flow analysis and optimization
- Safety system evaluation
- Performance bottleneck identification
- Code quality assessment
- Best practice recommendations
- Industrial standard compliance

Provide detailed, actionable analysis based on the PLC program structure and logic.""",

            "validation": """You are an expert code reviewer specializing in industrial automation and PLC systems.
Your task is to validate generated code against PLC specifications and industrial requirements.

Validation criteria:
- Correctness against PLC tag definitions
- Compliance with industrial safety standards
- Performance and efficiency considerations
- Error handling adequacy
- Code maintainability and readability
- Integration compatibility

Provide specific feedback and recommendations for improvement."""
        }
        
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code generation templates."""
        return {
            "pycomm3_basic": '''"""
PLC Data Acquisition using pycomm3
Generated from L5X analysis for {controller_name}
"""

import logging
from pycomm3 import LogixDriver
from typing import Dict, List, Optional, Any, Union
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PLCInterface:
    """Interface for PLC communication and data acquisition."""
    
    def __init__(self, plc_ip: str):
        self.plc_ip = plc_ip
        self.driver = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to PLC."""
        try:
            self.driver = LogixDriver(self.plc_ip)
            self.driver.open()
            self.connected = True
            logger.info(f"Connected to PLC at {{self.plc_ip}}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PLC: {{e}}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from PLC."""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Disconnected from PLC")
            
    def read_tag(self, tag_name: str) -> Optional[Any]:
        """Read single tag value."""
        if not self.connected:
            logger.error("Not connected to PLC")
            return None
            
        try:
            result = self.driver.read(tag_name)
            return result.value if result else None
        except Exception as e:
            logger.error(f"Failed to read tag {{tag_name}}: {{e}}")
            return None
            
    def read_tags(self, tag_names: List[str]) -> Dict[str, Any]:
        """Read multiple tag values."""
        if not self.connected:
            logger.error("Not connected to PLC")
            return {{}}
            
        results = {{}}
        try:
            for tag_name in tag_names:
                result = self.driver.read(tag_name)
                results[tag_name] = result.value if result else None
        except Exception as e:
            logger.error(f"Failed to read tags: {{e}}")
            
        return results
        
    def write_tag(self, tag_name: str, value: Any) -> bool:
        """Write single tag value."""
        if not self.connected:
            logger.error("Not connected to PLC")
            return False
            
        try:
            result = self.driver.write(tag_name, value)
            return result is not None
        except Exception as e:
            logger.error(f"Failed to write tag {{tag_name}}: {{e}}")
            return False


# Example usage
if __name__ == "__main__":
    plc = PLCInterface("192.168.1.100")  # Replace with actual PLC IP
    
    if plc.connect():
        # Add your specific tag operations here
        pass
        
    plc.disconnect()
''',

            "tag_monitor": '''"""
PLC Tag Monitoring System
Continuous monitoring of critical PLC tags
"""

import time
import json
import logging
from datetime import datetime
from pycomm3 import LogixDriver
from typing import Dict, List, Any, Optional
import threading
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TagReading:
    """Single tag reading with timestamp."""
    tag_name: str
    value: Any
    timestamp: float
    quality: str = "GOOD"


class TagMonitor:
    """Continuous tag monitoring system."""
    
    def __init__(self, plc_ip: str, tags: List[str], sample_rate: float = 1.0):
        self.plc_ip = plc_ip
        self.tags = tags
        self.sample_rate = sample_rate
        self.driver = None
        self.monitoring = False
        self.readings: List[TagReading] = []
        self.callbacks: Dict[str, callable] = {{}}
        
    def add_callback(self, tag_name: str, callback: callable) -> None:
        """Add callback for tag value changes."""
        self.callbacks[tag_name] = callback
        
    def start_monitoring(self) -> bool:
        """Start monitoring tags."""
        try:
            self.driver = LogixDriver(self.plc_ip)
            self.driver.open()
            self.monitoring = True
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info(f"Started monitoring {{len(self.tags)}} tags")
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {{e}}")
            return False
            
    def stop_monitoring(self) -> None:
        """Stop monitoring tags."""
        self.monitoring = False
        if self.driver:
            self.driver.close()
        logger.info("Stopped tag monitoring")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        last_values = {{}}
        
        while self.monitoring:
            try:
                current_time = time.time()
                
                for tag_name in self.tags:
                    result = self.driver.read(tag_name)
                    if result:
                        reading = TagReading(
                            tag_name=tag_name,
                            value=result.value,
                            timestamp=current_time
                        )
                        
                        self.readings.append(reading)
                        
                        # Check for value changes and trigger callbacks
                        if tag_name in self.callbacks:
                            if tag_name not in last_values or last_values[tag_name] != result.value:
                                self.callbacks[tag_name](reading)
                                
                        last_values[tag_name] = result.value
                        
                time.sleep(self.sample_rate)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {{e}}")
                time.sleep(self.sample_rate)
                
    def get_latest_readings(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get latest tag readings."""
        return [asdict(reading) for reading in self.readings[-count:]]
        
    def save_readings(self, filename: str) -> None:
        """Save readings to file."""
        with open(filename, 'w') as f:
            json.dump(self.get_latest_readings(len(self.readings)), f, indent=2)


# Example usage
if __name__ == "__main__":
    tags_to_monitor = [
        # Add your specific tags here
    ]
    
    monitor = TagMonitor("192.168.1.100", tags_to_monitor, sample_rate=0.5)
    
    # Add callbacks for critical tags
    def alarm_callback(reading: TagReading):
        if reading.value:
            logger.warning(f"ALARM: {{reading.tag_name}} = {{reading.value}}")
            
    # monitor.add_callback("AlarmTag", alarm_callback)
    
    if monitor.start_monitoring():
        try:
            # Monitor for specified time or until interrupted
            time.sleep(60)  # Monitor for 1 minute
        except KeyboardInterrupt:
            pass
        finally:
            monitor.stop_monitoring()
            monitor.save_readings("tag_readings.json")
'''
        }
        
    def set_plc_context(self, 
                       controller_name: str = "",
                       tags: Optional[List[Dict[str, Any]]] = None,
                       routines: Optional[List[Dict[str, Any]]] = None,
                       **kwargs) -> None:
        """Set PLC context for AI operations."""
        self.plc_context.controller_name = controller_name
        if tags:
            self.plc_context.tags = tags
        if routines:
            self.plc_context.routines = routines
            
        # Set additional context fields
        for key, value in kwargs.items():
            if hasattr(self.plc_context, key) and value is not None:
                setattr(self.plc_context, key, value)
                
        logger.info(f"Updated PLC context for controller: {controller_name}")
        
    def update_context_from_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """Update context from comprehensive PLC analysis results."""
        try:
            # Extract relevant information from analysis results
            if 'parsing_results' in analysis_results:
                parsing = analysis_results['parsing_results']
                self.plc_context.controller_name = parsing.get('controller_name', '')
                
            if 'tag_summary' in analysis_results:
                tag_summary = analysis_results['tag_summary']
                self.plc_context.tags = tag_summary.get('detailed_tags', [])
                
            if 'routine_summary' in analysis_results:
                routine_summary = analysis_results['routine_summary']
                self.plc_context.routines = routine_summary.get('routine_details', [])
                
            if 'udt_analysis' in analysis_results:
                udt_analysis = analysis_results['udt_analysis']
                if 'udt_summary' in udt_analysis:
                    self.plc_context.udts = udt_analysis['udt_summary'].get('udt_definitions', [])
                    
            if 'array_analysis' in analysis_results:
                array_analysis = analysis_results['array_analysis']
                if 'array_summary' in array_analysis:
                    self.plc_context.arrays = array_analysis['array_summary'].get('array_definitions', [])
                    
            if 'timer_analysis' in analysis_results:
                timer_analysis = analysis_results['timer_analysis']
                if 'timer_summary' in timer_analysis:
                    self.plc_context.timers = timer_analysis['timer_summary'].get('timer_details', [])
                    self.plc_context.counters = timer_analysis['timer_summary'].get('counter_details', [])
                    
            if 'flow_analysis' in analysis_results:
                flow_analysis = analysis_results['flow_analysis']
                if 'patterns_detected' in flow_analysis:
                    patterns = flow_analysis['patterns_detected']
                    self.plc_context.logic_patterns = patterns.get('pattern_details', {})
                if 'critical_analysis' in flow_analysis:
                    critical = flow_analysis['critical_analysis']
                    self.plc_context.critical_paths = critical.get('critical_paths', [])
                    self.plc_context.safety_concerns = critical.get('safety_concerns', [])
                    self.plc_context.optimization_opportunities = critical.get('optimization_opportunities', [])
                    
            logger.info("Updated PLC context from comprehensive analysis results")
            
        except Exception as e:
            logger.error(f"Failed to update context from analysis: {e}")
            
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        """Generate code using AI based on PLC context."""
        try:
            # Build context-aware prompt
            prompt = self._build_code_generation_prompt(request)
            
            # Prepare messages for AI
            messages = [
                AIMessage(role="system", content=self.system_prompts["code_generation"]),
                AIMessage(role="user", content=prompt)
            ]
            
            # Generate response
            ai_response = await self.ai_manager.generate_response(messages)
            
            # Parse and validate response
            result = self._parse_code_generation_response(ai_response, request)
            
            # Store in history
            self.generation_history.append(result)
            
            logger.info(f"Generated code for: {request.task_description}")
            return result
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return CodeGenerationResult(
                generated_code="",
                explanation=f"Code generation failed: {e}",
                validation_notes=[f"Error: {e}"]
            )
            
    def _build_code_generation_prompt(self, request: CodeGenerationRequest) -> str:
        """Build comprehensive prompt for code generation."""
        prompt_parts = [
            f"Task: {request.task_description}",
            f"Target Language: {request.target_language}",
            f"Code Style: {request.code_style}",
            ""
        ]
        
        # Add PLC context information
        if self.plc_context.controller_name:
            prompt_parts.extend([
                "PLC CONTEXT:",
                f"Controller: {self.plc_context.controller_name}",
                ""
            ])
            
        # Add relevant tags
        if request.specific_tags or self.plc_context.tags:
            relevant_tags = []
            if request.specific_tags:
                # Filter tags based on request
                for tag in self.plc_context.tags:
                    if tag.get('name', '') in request.specific_tags:
                        relevant_tags.append(tag)
            else:
                # Use first 20 tags to avoid token limits
                relevant_tags = self.plc_context.tags[:20]
                
            if relevant_tags:
                prompt_parts.extend([
                    "RELEVANT TAGS:",
                    json.dumps(relevant_tags, indent=2),
                    ""
                ])
                
        # Add relevant routines
        if request.specific_routines or self.plc_context.routines:
            relevant_routines = []
            if request.specific_routines:
                for routine in self.plc_context.routines:
                    if routine.get('name', '') in request.specific_routines:
                        relevant_routines.append(routine)
            else:
                relevant_routines = self.plc_context.routines[:10]
                
            if relevant_routines:
                prompt_parts.extend([
                    "RELEVANT ROUTINES:",
                    json.dumps(relevant_routines, indent=2),
                    ""
                ])
                
        # Add UDTs if relevant
        if self.plc_context.udts:
            prompt_parts.extend([
                "UDT DEFINITIONS:",
                json.dumps(self.plc_context.udts[:5], indent=2),
                ""
            ])
            
        # Add requirements and constraints
        if request.requirements:
            prompt_parts.extend([
                "REQUIREMENTS:",
                "\n".join(f"- {req}" for req in request.requirements),
                ""
            ])
            
        if request.constraints:
            prompt_parts.extend([
                "CONSTRAINTS:",
                "\n".join(f"- {constraint}" for constraint in request.constraints),
                ""
            ])
            
        # Add generation preferences
        prompt_parts.extend([
            "GENERATION PREFERENCES:",
            f"- Include comments: {request.include_comments}",
            f"- Include error handling: {request.include_error_handling}",
            f"- Maximum complexity: {request.max_complexity}",
            "",
            "Please generate clean, well-documented code that follows best practices.",
            "Include an explanation of the generated code and any important considerations.",
            "Structure your response as:",
            "```python",
            "# Generated code here",
            "```",
            "",
            "EXPLANATION:",
            "Detailed explanation of the code and its functionality."
        ])
        
        return "\n".join(prompt_parts)
        
    def _parse_code_generation_response(self, 
                                      ai_response: AIResponse, 
                                      request: CodeGenerationRequest) -> CodeGenerationResult:
        """Parse AI response into CodeGenerationResult."""
        content = ai_response.content
        
        # Extract code block
        code_start = content.find("```python")
        code_end = content.find("```", code_start + 9)
        
        if code_start >= 0 and code_end >= 0:
            generated_code = content[code_start + 9:code_end].strip()
        else:
            # Fallback: try to find any code block
            code_start = content.find("```")
            if code_start >= 0:
                code_end = content.find("```", code_start + 3)
                if code_end >= 0:
                    generated_code = content[code_start + 3:code_end].strip()
                else:
                    generated_code = content[code_start + 3:].strip()
            else:
                generated_code = content
                
        # Extract explanation
        explanation_start = content.find("EXPLANATION:")
        if explanation_start >= 0:
            explanation = content[explanation_start + 12:].strip()
        else:
            # Try to find explanation after code
            if code_end >= 0:
                explanation = content[code_end + 3:].strip()
            else:
                explanation = "Generated code as requested."
                
        # Basic validation
        validation_notes = []
        if not generated_code:
            validation_notes.append("No code generated")
        elif "import" not in generated_code:
            validation_notes.append("No imports detected - may be incomplete")
            
        # Calculate confidence based on response quality
        confidence_score = 0.8  # Base confidence
        if ai_response.finish_reason == "stop":
            confidence_score += 0.1
        if len(generated_code) > 100:
            confidence_score += 0.1
        confidence_score = min(confidence_score, 1.0)
        
        return CodeGenerationResult(
            generated_code=generated_code,
            explanation=explanation,
            validation_notes=validation_notes,
            confidence_score=confidence_score,
            ai_response=ai_response
        )
        
    async def analyze_plc_system(self, analysis_request: str) -> str:
        """Analyze PLC system using AI."""
        try:
            # Build analysis prompt with context
            prompt = self._build_analysis_prompt(analysis_request)
            
            messages = [
                AIMessage(role="system", content=self.system_prompts["analysis"]),
                AIMessage(role="user", content=prompt)
            ]
            
            response = await self.ai_manager.generate_response(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"PLC analysis failed: {e}")
            return f"Analysis failed: {e}"
            
    def _build_analysis_prompt(self, analysis_request: str) -> str:
        """Build analysis prompt with PLC context."""
        prompt_parts = [
            f"Analysis Request: {analysis_request}",
            "",
            "PLC SYSTEM CONTEXT:"
        ]
        
        if self.plc_context.controller_name:
            prompt_parts.append(f"Controller: {self.plc_context.controller_name}")
            
        if self.plc_context.tags:
            prompt_parts.extend([
                f"Total Tags: {len(self.plc_context.tags)}",
                f"Sample Tags: {json.dumps(self.plc_context.tags[:10], indent=2)}"
            ])
            
        if self.plc_context.routines:
            prompt_parts.extend([
                f"Total Routines: {len(self.plc_context.routines)}",
                f"Routine Names: {[r.get('name', 'Unknown') for r in self.plc_context.routines[:10]]}"
            ])
            
        if self.plc_context.logic_patterns:
            prompt_parts.extend([
                "Detected Logic Patterns:",
                json.dumps(self.plc_context.logic_patterns, indent=2)
            ])
            
        if self.plc_context.safety_concerns:
            prompt_parts.extend([
                "Safety Concerns:",
                "\n".join(f"- {concern}" for concern in self.plc_context.safety_concerns)
            ])
            
        if self.plc_context.optimization_opportunities:
            prompt_parts.extend([
                "Optimization Opportunities:",
                "\n".join(f"- {opp}" for opp in self.plc_context.optimization_opportunities)
            ])
            
        prompt_parts.extend([
            "",
            "Please provide a detailed analysis considering:",
            "- System architecture and design patterns",
            "- Safety and reliability aspects",
            "- Performance and optimization opportunities",
            "- Best practice recommendations",
            "- Potential risks or concerns"
        ])
        
        return "\n".join(prompt_parts)
        
    async def validate_generated_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code using AI."""
        try:
            prompt = f"""
Please validate the following generated code against PLC requirements:

CODE TO VALIDATE:
```python
{code}
```

PLC CONTEXT:
Controller: {self.plc_context.controller_name}
Available Tags: {len(self.plc_context.tags)}
Safety Concerns: {len(self.plc_context.safety_concerns)}

Please check for:
1. Correctness against PLC tag definitions
2. Industrial safety compliance
3. Error handling adequacy
4. Performance considerations
5. Code quality and maintainability

Provide specific feedback and a validation score (0-100).
"""

            messages = [
                AIMessage(role="system", content=self.system_prompts["validation"]),
                AIMessage(role="user", content=prompt)
            ]
            
            response = await self.ai_manager.generate_response(messages)
            
            return {
                "validation_result": response.content,
                "ai_response": response,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return {
                "validation_result": f"Validation failed: {e}",
                "error": str(e),
                "timestamp": time.time()
            }
            
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get code generation history."""
        history = []
        for result in self.generation_history:
            history.append({
                "generated_code": result.generated_code[:200] + "..." if len(result.generated_code) > 200 else result.generated_code,
                "explanation": result.explanation[:200] + "..." if len(result.explanation) > 200 else result.explanation,
                "confidence_score": result.confidence_score,
                "validation_notes": result.validation_notes,
                "timestamp": result.ai_response.metadata.get("timestamp") if result.ai_response else time.time()
            })
        return history
        
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current PLC context."""
        return {
            "controller_name": self.plc_context.controller_name,
            "tags_count": len(self.plc_context.tags),
            "routines_count": len(self.plc_context.routines),
            "udts_count": len(self.plc_context.udts),
            "arrays_count": len(self.plc_context.arrays),
            "timers_count": len(self.plc_context.timers),
            "counters_count": len(self.plc_context.counters),
            "io_modules_count": len(self.plc_context.io_modules),
            "pattern_types": len(self.plc_context.logic_patterns),
            "safety_concerns": len(self.plc_context.safety_concerns),
            "optimization_opportunities": len(self.plc_context.optimization_opportunities)
        }
