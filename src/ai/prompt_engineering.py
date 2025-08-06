"""
Step 18: Advanced Prompt Engineering System
Sophisticated prompt construction for PLC code generation with context-aware templates,
dynamic prompt building, and optimization for different AI models.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts for different tasks."""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_OPTIMIZATION = "code_optimization"
    DOCUMENTATION = "documentation"
    TROUBLESHOOTING = "troubleshooting"
    SAFETY_ANALYSIS = "safety_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    INTEGRATION_GUIDANCE = "integration_guidance"


class PromptComplexity(Enum):
    """Complexity levels for prompts."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class PLCDomain(Enum):
    """PLC-specific domains for specialized prompts."""
    LADDER_LOGIC = "ladder_logic"
    STRUCTURED_TEXT = "structured_text"
    FUNCTION_BLOCKS = "function_blocks"
    SEQUENTIAL_FUNCTION_CHARTS = "sequential_function_charts"
    SAFETY_SYSTEMS = "safety_systems"
    MOTION_CONTROL = "motion_control"
    HMI_INTEGRATION = "hmi_integration"
    COMMUNICATIONS = "communications"


@dataclass
class PromptContext:
    """Context for prompt generation."""
    controller_name: str = ""
    tags: List[Dict[str, Any]] = field(default_factory=list)
    routines: List[Dict[str, Any]] = field(default_factory=list)
    udts: List[Dict[str, Any]] = field(default_factory=list)
    io_modules: List[Dict[str, Any]] = field(default_factory=list)
    safety_concerns: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    logic_patterns: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    domain_focus: Optional[PLCDomain] = None
    user_requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    target_language: str = "python"
    include_comments: bool = True
    include_error_handling: bool = True
    code_style: str = "PEP8"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'controller_name': self.controller_name,
            'tags_count': len(self.tags),
            'routines_count': len(self.routines),
            'udts_count': len(self.udts),
            'io_modules_count': len(self.io_modules),
            'safety_concerns_count': len(self.safety_concerns),
            'optimization_opportunities_count': len(self.optimization_opportunities),
            'logic_patterns_count': len(self.logic_patterns),
            'dependencies_count': len(self.dependencies),
            'domain_focus': self.domain_focus.value if self.domain_focus else None,
            'user_requirements_count': len(self.user_requirements),
            'constraints_count': len(self.constraints),
            'target_language': self.target_language,
            'include_comments': self.include_comments,
            'include_error_handling': self.include_error_handling,
            'code_style': self.code_style
        }


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    prompt_type: PromptType
    complexity: PromptComplexity
    domain: Optional[PLCDomain]
    system_prompt: str
    user_prompt_template: str
    required_context: List[str]
    optional_context: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_format: str = "structured"
    max_tokens: int = 4000
    temperature: float = 0.1
    
    def validate_context(self, context: PromptContext) -> Tuple[bool, List[str]]:
        """Validate that context contains required information."""
        missing_context = []
        
        for req in self.required_context:
            if req == "tags" and not context.tags:
                missing_context.append("PLC tags")
            elif req == "routines" and not context.routines:
                missing_context.append("PLC routines")
            elif req == "controller_name" and not context.controller_name:
                missing_context.append("Controller name")
            elif req == "safety_concerns" and not context.safety_concerns:
                missing_context.append("Safety concerns")
            elif req == "logic_patterns" and not context.logic_patterns:
                missing_context.append("Logic patterns")
        
        return len(missing_context) == 0, missing_context


class PromptOptimizer:
    """Optimizes prompts for different AI models and contexts."""
    
    def __init__(self):
        self.model_preferences = {
            "gpt-4": {
                "max_context": 8000,
                "prefers_structured": True,
                "good_with_examples": True,
                "handles_complexity": "expert"
            },
            "gpt-3.5-turbo": {
                "max_context": 4000,
                "prefers_structured": True,
                "good_with_examples": True,
                "handles_complexity": "moderate"
            },
            "gemini-1.5-flash": {
                "max_context": 1000000,
                "prefers_structured": True,
                "good_with_examples": True,
                "handles_complexity": "complex"
            },
            "gemini-1.5-pro": {
                "max_context": 2000000,
                "prefers_structured": True,
                "good_with_examples": True,
                "handles_complexity": "expert"
            },
            "codellama": {
                "max_context": 4000,
                "prefers_structured": False,
                "good_with_examples": True,
                "handles_complexity": "moderate"
            }
        }
    
    def optimize_for_model(self, template: PromptTemplate, model_name: str, context: PromptContext) -> PromptTemplate:
        """Optimize prompt template for specific AI model."""
        model_prefs = self.model_preferences.get(model_name, {})
        optimized_template = PromptTemplate(
            name=f"{template.name}_optimized_{model_name.replace('-', '_')}",
            prompt_type=template.prompt_type,
            complexity=template.complexity,
            domain=template.domain,
            system_prompt=template.system_prompt,
            user_prompt_template=template.user_prompt_template,
            required_context=template.required_context,
            optional_context=template.optional_context,
            examples=template.examples,
            constraints=template.constraints,
            output_format=template.output_format,
            max_tokens=template.max_tokens,
            temperature=template.temperature
        )
        
        # Adjust for model context limits
        max_context = model_prefs.get("max_context", 4000)
        if template.max_tokens > max_context * 0.7:  # Leave room for response
            optimized_template.max_tokens = int(max_context * 0.7)
        
        # Adjust complexity if model can't handle it
        model_complexity = model_prefs.get("handles_complexity", "moderate")
        if template.complexity.value == "expert" and model_complexity != "expert":
            optimized_template.complexity = PromptComplexity.COMPLEX
        elif template.complexity.value == "complex" and model_complexity == "simple":
            optimized_template.complexity = PromptComplexity.MODERATE
        
        # Adjust structure preference
        if not model_prefs.get("prefers_structured", True):
            optimized_template.output_format = "natural"
        
        # Limit examples if model doesn't handle them well
        if not model_prefs.get("good_with_examples", True):
            optimized_template.examples = optimized_template.examples[:2]
        
        return optimized_template
    
    def estimate_token_usage(self, template: PromptTemplate, context: PromptContext) -> Dict[str, int]:
        """Estimate token usage for prompt."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        system_tokens = len(template.system_prompt) // 4
        
        # Estimate user prompt tokens
        user_prompt = self._build_user_prompt(template, context)
        user_tokens = len(user_prompt) // 4
        
        # Add examples
        examples_tokens = sum(len(ex.get("input", "") + ex.get("output", "")) // 4 
                            for ex in template.examples)
        
        total_input_tokens = system_tokens + user_tokens + examples_tokens
        estimated_output_tokens = template.max_tokens
        
        return {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "examples_tokens": examples_tokens,
            "total_input_tokens": total_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "total_estimated_tokens": total_input_tokens + estimated_output_tokens
        }
    
    def _build_user_prompt(self, template: PromptTemplate, context: PromptContext) -> str:
        """Build user prompt from template and context."""
        # This is a simplified version - the full implementation would be in PromptBuilder
        return template.user_prompt_template.format(
            controller_name=context.controller_name,
            tags_count=len(context.tags),
            routines_count=len(context.routines)
        )


class PromptBuilder:
    """Builds sophisticated prompts from templates and context."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.templates: Dict[str, PromptTemplate] = {}
        self.optimizer = PromptOptimizer()
        self.templates_dir = Path(templates_dir) if templates_dir else Path("prompts")
        
        # Load built-in templates
        self._load_builtin_templates()
        
        # Load custom templates if directory exists
        if self.templates_dir.exists():
            self._load_custom_templates()
    
    def _load_builtin_templates(self):
        """Load built-in prompt templates."""
        
        # Basic PLC Interface Generation Template
        self.templates["basic_plc_interface"] = PromptTemplate(
            name="basic_plc_interface",
            prompt_type=PromptType.CODE_GENERATION,
            complexity=PromptComplexity.SIMPLE,
            domain=PLCDomain.LADDER_LOGIC,
            system_prompt="""You are an expert PLC programmer and Python developer specializing in industrial automation.
Your task is to generate clean, efficient Python code using pycomm3 library for PLC communication.

Key principles:
1. Always include proper error handling and logging
2. Use clear, descriptive variable names
3. Follow PEP8 style guidelines
4. Include comprehensive docstrings and comments
5. Implement connection management and timeout handling
6. Consider industrial environment constraints (reliability, performance)
7. Include data validation and type checking

You have extensive knowledge of:
- Rockwell/Allen-Bradley PLC systems
- pycomm3 library and EtherNet/IP communication
- Industrial communication protocols
- PLC tag structures and data types
- Safety considerations in industrial automation""",
            
            user_prompt_template="""Generate a Python class for interfacing with a {controller_name} PLC system.

PLC System Information:
- Controller: {controller_name}
- Total Tags: {tags_count}
- Key Tags: {key_tags}
- Routines: {routines_count}

Requirements:
{requirements}

Constraints:
{constraints}

Please generate a complete Python class that:
1. Establishes connection to the PLC using pycomm3
2. Implements methods for reading/writing the specified tags
3. Includes proper error handling and logging
4. Follows industrial best practices
5. Is production-ready and maintainable

Focus on the following tags: {specific_tags}""",
            
            required_context=["controller_name", "tags"],
            optional_context=["routines", "safety_concerns"],
            examples=[
                {
                    "input": "Generate interface for reading motor status tags",
                    "output": """class PLCInterface:
    def __init__(self, plc_ip: str):
        self.plc_ip = plc_ip
        self.driver = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        try:
            self.driver = LogixDriver(self.plc_ip)
            return self.driver.open()
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False"""
                }
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        # Advanced Code Analysis Template
        self.templates["code_analysis"] = PromptTemplate(
            name="code_analysis",
            prompt_type=PromptType.CODE_ANALYSIS,
            complexity=PromptComplexity.COMPLEX,
            domain=PLCDomain.LADDER_LOGIC,
            system_prompt="""You are a senior PLC systems analyst with deep expertise in industrial automation, 
safety systems, and code optimization. Your role is to analyze PLC programs and provide comprehensive 
insights about system architecture, performance, safety, and optimization opportunities.

Your analysis should consider:
1. System safety and reliability
2. Performance bottlenecks and optimization opportunities
3. Code maintainability and documentation quality
4. Compliance with industrial standards (IEC 61131, IEC 61508, etc.)
5. Best practices for industrial automation
6. Risk assessment and mitigation strategies
7. Integration considerations with other systems""",
            
            user_prompt_template="""Analyze the following PLC system and provide comprehensive insights:

System Overview:
- Controller: {controller_name}
- Tags: {tags_count} total ({safety_tags_count} safety-critical)
- Routines: {routines_count}
- Logic Patterns: {logic_patterns}
- Safety Concerns: {safety_concerns}
- Performance Metrics: {performance_metrics}

Focus Areas:
{focus_areas}

Please provide a detailed analysis covering:
1. SAFETY ASSESSMENT
   - Critical safety concerns and recommendations
   - Compliance with safety standards
   - Risk mitigation strategies

2. PERFORMANCE ANALYSIS
   - Bottlenecks and optimization opportunities
   - Resource utilization assessment
   - Scalability considerations

3. CODE QUALITY
   - Maintainability assessment
   - Documentation quality
   - Best practice compliance

4. SYSTEM ARCHITECTURE
   - Overall design assessment
   - Integration considerations
   - Modularity and reusability

5. RECOMMENDATIONS
   - Prioritized improvement suggestions
   - Implementation roadmap
   - Cost-benefit analysis""",
            
            required_context=["controller_name", "tags", "routines"],
            optional_context=["safety_concerns", "logic_patterns", "performance_metrics"],
            max_tokens=6000,
            temperature=0.2
        )
        
        # Safety System Analysis Template
        self.templates["safety_analysis"] = PromptTemplate(
            name="safety_analysis",
            prompt_type=PromptType.SAFETY_ANALYSIS,
            complexity=PromptComplexity.EXPERT,
            domain=PLCDomain.SAFETY_SYSTEMS,
            system_prompt="""You are a certified functional safety engineer (TÜV certified) with extensive experience 
in PLC safety systems, IEC 61508, IEC 61511, and machinery safety standards (ISO 13849, IEC 62061).

Your expertise includes:
- Safety Integrity Level (SIL) determination and verification
- Hazard analysis and risk assessment (HAZOP, FMEA)
- Safety instrumented systems (SIS) design and validation
- Proof test procedures and maintenance strategies
- Safety lifecycle management
- Emergency shutdown systems and safety interlocks
- Human factors in safety system design""",
            
            user_prompt_template="""Conduct a comprehensive safety analysis of this PLC system:

System Information:
- Controller: {controller_name}
- Safety-Critical Tags: {safety_tags}
- Emergency Systems: {emergency_systems}
- Safety Interlocks: {safety_interlocks}
- Identified Concerns: {safety_concerns}

Current Safety Implementation:
{current_safety_implementation}

Regulatory Requirements:
{regulatory_requirements}

Please provide:

1. HAZARD IDENTIFICATION
   - Potential hazards and failure modes
   - Risk assessment matrix
   - SIL requirements determination

2. SAFETY SYSTEM EVALUATION
   - Current safety measures assessment
   - Gap analysis against standards
   - Architecture validation

3. COMPLIANCE ASSESSMENT
   - IEC 61508/61511 compliance
   - Machinery safety standards compliance
   - Regional regulatory compliance

4. IMPROVEMENT RECOMMENDATIONS
   - Safety system enhancements
   - Redundancy and diversity requirements
   - Proof testing strategy

5. IMPLEMENTATION ROADMAP
   - Prioritized safety improvements
   - Validation and verification plan
   - Maintenance requirements""",
            
            required_context=["controller_name", "safety_concerns"],
            optional_context=["tags", "routines", "logic_patterns"],
            max_tokens=8000,
            temperature=0.1
        )
        
        # Pattern Recognition Template
        self.templates["pattern_recognition"] = PromptTemplate(
            name="pattern_recognition",
            prompt_type=PromptType.PATTERN_RECOGNITION,
            complexity=PromptComplexity.MODERATE,
            domain=PLCDomain.LADDER_LOGIC,
            system_prompt="""You are an expert in PLC programming patterns and industrial automation best practices.
Your specialty is identifying common logic patterns, anti-patterns, and optimization opportunities in PLC code.

You can recognize and analyze:
- Standard automation patterns (Start/Stop, Interlock, Sequencer, etc.)
- Safety patterns and their implementations
- Communication and data handling patterns
- Performance optimization patterns
- Anti-patterns and code smells in PLC programming""",
            
            user_prompt_template="""Analyze the logic patterns in this PLC system:

System: {controller_name}
Detected Patterns: {logic_patterns}
Pattern Statistics: {pattern_statistics}
Routine Analysis: {routine_analysis}

Logic Structure:
{logic_structure}

Please identify and analyze:

1. RECOGNIZED PATTERNS
   - Standard automation patterns found
   - Pattern implementation quality
   - Pattern consistency across routines

2. OPTIMIZATION OPPORTUNITIES
   - Redundant or inefficient patterns
   - Consolidation possibilities
   - Performance improvements

3. BEST PRACTICE COMPLIANCE
   - Industry standard pattern usage
   - Maintainability assessment
   - Documentation quality

4. RECOMMENDATIONS
   - Pattern standardization suggestions
   - Library development opportunities
   - Training and documentation needs""",
            
            required_context=["logic_patterns"],
            optional_context=["routines", "performance_metrics"],
            max_tokens=5000,
            temperature=0.2
        )
    
    def _load_custom_templates(self):
        """Load custom templates from files."""
        try:
            for template_file in self.templates_dir.glob("*.json"):
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                    template = self._dict_to_template(template_data)
                    self.templates[template.name] = template
                    logger.info(f"Loaded custom template: {template.name}")
        except Exception as e:
            logger.error(f"Failed to load custom templates: {e}")
    
    def _dict_to_template(self, data: Dict[str, Any]) -> PromptTemplate:
        """Convert dictionary to PromptTemplate."""
        return PromptTemplate(
            name=data["name"],
            prompt_type=PromptType(data["prompt_type"]),
            complexity=PromptComplexity(data["complexity"]),
            domain=PLCDomain(data["domain"]) if data.get("domain") else None,
            system_prompt=data["system_prompt"],
            user_prompt_template=data["user_prompt_template"],
            required_context=data["required_context"],
            optional_context=data.get("optional_context", []),
            examples=data.get("examples", []),
            constraints=data.get("constraints", []),
            output_format=data.get("output_format", "structured"),
            max_tokens=data.get("max_tokens", 4000),
            temperature=data.get("temperature", 0.1)
        )
    
    def build_prompt(self, template_name: str, context: PromptContext, 
                    model_name: Optional[str] = None) -> Dict[str, Any]:
        """Build complete prompt from template and context."""
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        
        # Optimize for model if specified
        if model_name:
            template = self.optimizer.optimize_for_model(template, model_name, context)
        
        # Validate context
        is_valid, missing = template.validate_context(context)
        if not is_valid:
            raise ValueError(f"Missing required context: {', '.join(missing)}")
        
        # Build user prompt
        user_prompt = self._build_user_prompt(template, context)
        
        # Prepare examples
        examples = []
        for example in template.examples:
            examples.append({
                "role": "user",
                "content": example["input"]
            })
            examples.append({
                "role": "assistant", 
                "content": example["output"]
            })
        
        # Calculate token estimates
        token_estimates = self.optimizer.estimate_token_usage(template, context)
        
        return {
            "system_prompt": template.system_prompt,
            "user_prompt": user_prompt,
            "examples": examples,
            "template_info": {
                "name": template.name,
                "type": template.prompt_type.value,
                "complexity": template.complexity.value,
                "domain": template.domain.value if template.domain else None,
                "max_tokens": template.max_tokens,
                "temperature": template.temperature
            },
            "token_estimates": token_estimates,
            "context_summary": context.to_dict()
        }
    
    def _build_user_prompt(self, template: PromptTemplate, context: PromptContext) -> str:
        """Build user prompt from template and context."""
        # Extract key information from context
        key_tags = []
        safety_tags = []
        
        for tag in context.tags[:10]:  # Limit to prevent prompt bloat
            key_tags.append(f"- {tag.get('name', 'Unknown')}: {tag.get('type', 'Unknown')} - {tag.get('description', 'No description')}")
            if 'safety' in tag.get('name', '').lower() or tag.get('critical', False):
                safety_tags.append(tag)
        
        # Build logic patterns summary
        pattern_summary = []
        for pattern in context.logic_patterns:
            pattern_summary.append(f"- {pattern.get('pattern', 'Unknown')}: {pattern.get('count', 0)} instances")
        
        # Format template
        format_args = {
            'controller_name': context.controller_name or "Unknown Controller",
            'tags_count': len(context.tags),
            'routines_count': len(context.routines),
            'key_tags': '\n'.join(key_tags) if key_tags else "No tags available",
            'safety_tags_count': len(safety_tags),
            'safety_tags': '\n'.join([f"- {tag.get('name', 'Unknown')}" for tag in safety_tags]) if safety_tags else "No safety tags identified",
            'logic_patterns': '\n'.join(pattern_summary) if pattern_summary else "No patterns detected",
            'safety_concerns': '\n'.join([f"- {concern}" for concern in context.safety_concerns]) if context.safety_concerns else "No safety concerns identified",
            'requirements': '\n'.join([f"- {req}" for req in context.user_requirements]) if context.user_requirements else "No specific requirements provided",
            'constraints': '\n'.join([f"- {constraint}" for constraint in context.constraints]) if context.constraints else "No constraints specified",
            'specific_tags': ', '.join([tag.get('name', 'Unknown') for tag in context.tags[:5]]) if context.tags else "No specific tags",
            'performance_metrics': json.dumps(context.performance_metrics, indent=2) if context.performance_metrics else "No performance metrics available",
            'focus_areas': "Code generation and optimization" if template.prompt_type == PromptType.CODE_GENERATION else "System analysis and improvement",
            'emergency_systems': "E-Stop, Safety Gates" if context.safety_concerns else "None identified",
            'safety_interlocks': "Standard safety interlocks" if context.safety_concerns else "None identified",
            'current_safety_implementation': "Standard PLC safety implementation" if context.safety_concerns else "No safety implementation details available",
            'regulatory_requirements': "General industrial safety standards",
            'pattern_statistics': f"{len(context.logic_patterns)} patterns detected",
            'routine_analysis': f"{len(context.routines)} routines analyzed",
            'logic_structure': "Ladder logic with standard automation patterns"
        }
        
        try:
            return template.user_prompt_template.format(**format_args)
        except KeyError as e:
            logger.warning(f"Missing template variable {e}, using fallback")
            # Try with basic fallback values
            basic_args = {
                'controller_name': context.controller_name or "Unknown Controller",
                'tags_count': len(context.tags),
                'routines_count': len(context.routines)
            }
            # Replace missing variables with placeholders
            prompt = template.user_prompt_template
            for key in format_args:
                if f"{{{key}}}" in prompt and key not in basic_args:
                    prompt = prompt.replace(f"{{{key}}}", f"[{key.replace('_', ' ').title()}]")
            
            return prompt.format(**basic_args)
    
    def list_templates(self, prompt_type: Optional[PromptType] = None, 
                      domain: Optional[PLCDomain] = None) -> List[Dict[str, Any]]:
        """List available templates with filtering options."""
        templates = []
        
        for name, template in self.templates.items():
            if prompt_type and template.prompt_type != prompt_type:
                continue
            if domain and template.domain != domain:
                continue
                
            templates.append({
                "name": name,
                "type": template.prompt_type.value,
                "complexity": template.complexity.value,
                "domain": template.domain.value if template.domain else None,
                "required_context": template.required_context,
                "description": f"{template.prompt_type.value.replace('_', ' ').title()} template for {template.domain.value.replace('_', ' ') if template.domain else 'general'} applications"
            })
        
        return sorted(templates, key=lambda x: (x["type"], x["complexity"]))
    
    def save_template(self, template: PromptTemplate, filename: Optional[str] = None):
        """Save custom template to file."""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True)
        
        filename = filename or f"{template.name}.json"
        filepath = self.templates_dir / filename
        
        template_data = {
            "name": template.name,
            "prompt_type": template.prompt_type.value,
            "complexity": template.complexity.value,
            "domain": template.domain.value if template.domain else None,
            "system_prompt": template.system_prompt,
            "user_prompt_template": template.user_prompt_template,
            "required_context": template.required_context,
            "optional_context": template.optional_context,
            "examples": template.examples,
            "constraints": template.constraints,
            "output_format": template.output_format,
            "max_tokens": template.max_tokens,
            "temperature": template.temperature
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved template: {filename}")


class PromptEngineering:
    """Main class for advanced prompt engineering operations."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.prompt_builder = PromptBuilder(templates_dir)
        self.context_cache: Dict[str, PromptContext] = {}
        
    def create_context_from_analysis(self, analysis_results: Dict[str, Any]) -> PromptContext:
        """Create prompt context from PLC analysis results."""
        context = PromptContext()
        
        # Extract controller information
        if 'controller' in analysis_results:
            controller_info = analysis_results['controller']
            context.controller_name = controller_info.get('name', '')
        
        # Extract tags
        if 'tags' in analysis_results:
            context.tags = analysis_results['tags']
        
        # Extract routines
        if 'routines' in analysis_results:
            context.routines = analysis_results['routines']
        
        # Extract UDTs
        if 'udts' in analysis_results:
            context.udts = analysis_results['udts']
        
        # Extract I/O modules
        if 'io_modules' in analysis_results:
            context.io_modules = analysis_results['io_modules']
        
        # Extract safety concerns
        if 'safety_analysis' in analysis_results:
            safety_data = analysis_results['safety_analysis']
            context.safety_concerns = safety_data.get('concerns', [])
        
        # Extract optimization opportunities
        if 'optimization_analysis' in analysis_results:
            opt_data = analysis_results['optimization_analysis']
            context.optimization_opportunities = opt_data.get('opportunities', [])
        
        # Extract logic patterns
        if 'logic_patterns' in analysis_results:
            context.logic_patterns = analysis_results['logic_patterns']
        
        # Extract dependencies
        if 'dependencies' in analysis_results:
            context.dependencies = analysis_results['dependencies']
        
        # Extract performance metrics
        if 'performance' in analysis_results:
            context.performance_metrics = analysis_results['performance']
        
        return context
    
    def generate_contextual_prompt(self, template_name: str, analysis_results: Dict[str, Any],
                                 user_requirements: List[str] = None,
                                 constraints: List[str] = None,
                                 model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a contextual prompt from analysis results."""
        # Create context from analysis
        context = self.create_context_from_analysis(analysis_results)
        
        # Add user requirements and constraints
        if user_requirements:
            context.user_requirements = user_requirements
        if constraints:
            context.constraints = constraints
        
        # Build prompt
        return self.prompt_builder.build_prompt(template_name, context, model_name)
    
    def create_multi_step_prompt_chain(self, analysis_results: Dict[str, Any],
                                     requirements: List[str]) -> List[Dict[str, Any]]:
        """Create a chain of prompts for complex multi-step code generation."""
        context = self.create_context_from_analysis(analysis_results)
        context.user_requirements = requirements
        
        prompt_chain = []
        
        # Step 1: System Analysis
        analysis_prompt = self.prompt_builder.build_prompt("code_analysis", context)
        prompt_chain.append({
            "step": 1,
            "name": "System Analysis",
            "purpose": "Analyze PLC system and identify key components",
            "prompt": analysis_prompt
        })
        
        # Step 2: Safety Assessment (if safety concerns exist)
        if context.safety_concerns:
            safety_prompt = self.prompt_builder.build_prompt("safety_analysis", context)
            prompt_chain.append({
                "step": 2,
                "name": "Safety Assessment",
                "purpose": "Evaluate safety systems and requirements",
                "prompt": safety_prompt
            })
        
        # Step 3: Pattern Recognition
        if context.logic_patterns:
            pattern_prompt = self.prompt_builder.build_prompt("pattern_recognition", context)
            prompt_chain.append({
                "step": len(prompt_chain) + 1,
                "name": "Pattern Recognition",
                "purpose": "Identify and analyze logic patterns",
                "prompt": pattern_prompt
            })
        
        # Step 4: Code Generation
        code_prompt = self.prompt_builder.build_prompt("basic_plc_interface", context)
        prompt_chain.append({
            "step": len(prompt_chain) + 1,
            "name": "Code Generation",
            "purpose": "Generate Python interface code",
            "prompt": code_prompt
        })
        
        return prompt_chain
    
    def get_template_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommended templates based on analysis results."""
        recommendations = []
        
        # Analyze the content to recommend templates
        has_safety_concerns = bool(analysis_results.get('safety_analysis', {}).get('concerns', []))
        has_patterns = bool(analysis_results.get('logic_patterns', []))
        has_performance_issues = bool(analysis_results.get('optimization_analysis', {}).get('opportunities', []))
        complexity_score = analysis_results.get('complexity', {}).get('average_score', 0)
        
        # Basic interface generation (always recommended)
        recommendations.append({
            "template": "basic_plc_interface",
            "priority": "high",
            "reason": "Essential for PLC communication interface",
            "complexity": "simple"
        })
        
        # Safety analysis if safety concerns exist
        if has_safety_concerns:
            recommendations.append({
                "template": "safety_analysis",
                "priority": "critical",
                "reason": "Safety concerns detected in system",
                "complexity": "expert"
            })
        
        # Pattern recognition if patterns detected
        if has_patterns:
            recommendations.append({
                "template": "pattern_recognition",
                "priority": "medium",
                "reason": "Logic patterns detected for optimization",
                "complexity": "moderate"
            })
        
        # Code analysis for complex systems
        if complexity_score > 5:
            recommendations.append({
                "template": "code_analysis",
                "priority": "high",
                "reason": "High complexity system requires detailed analysis",
                "complexity": "complex"
            })
        
        return sorted(recommendations, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x["priority"]])
    
    def cache_context(self, key: str, context: PromptContext):
        """Cache context for reuse."""
        self.context_cache[key] = context
    
    def get_cached_context(self, key: str) -> Optional[PromptContext]:
        """Get cached context."""
        return self.context_cache.get(key)
    
    def clear_cache(self):
        """Clear context cache."""
        self.context_cache.clear()


# Convenience functions for easy usage
def create_prompt_engineer(templates_dir: Optional[str] = None) -> PromptEngineering:
    """Create a prompt engineering instance."""
    return PromptEngineering(templates_dir)

def quick_prompt(template_name: str, analysis_results: Dict[str, Any], 
                requirements: List[str] = None, model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
    """Quickly generate a prompt from analysis results."""
    engineer = create_prompt_engineer()
    return engineer.generate_contextual_prompt(
        template_name=template_name,
        analysis_results=analysis_results,
        user_requirements=requirements or [],
        model_name=model_name
    )


if __name__ == "__main__":
    # Example usage
    print("🚀 Step 18: Advanced Prompt Engineering System")
    print("=" * 60)
    
    # Create prompt engineer
    engineer = create_prompt_engineer()
    
    # List available templates
    templates = engineer.prompt_builder.list_templates()
    print(f"Available templates: {len(templates)}")
    
    for template in templates:
        print(f"  - {template['name']}: {template['description']}")
    
    print("\n✅ Prompt Engineering System initialized successfully!")
