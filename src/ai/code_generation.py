"""
Step 19: Code Generation Pipeline
Generate Python code using AI with structured input from PLC analysis.
Integrates with Steps 17-18 for comprehensive AI-powered code generation.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import re
import ast
import traceback

# Import from previous steps
try:
    from .ai_interface import AIInterfaceManager, AIMessage, AIResponse
    from .prompt_engineering import PromptEngineering, PromptContext, PLCDomain, PromptType
    from .ai_integration import AnalysisSystemsIntegrator, AnalysisSystemsConfig
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from ai.ai_interface import AIInterfaceManager, AIMessage, AIResponse
    from ai.prompt_engineering import PromptEngineering, PromptContext, PLCDomain, PromptType
    from ai.ai_integration import AnalysisSystemsIntegrator, AnalysisSystemsConfig

logger = logging.getLogger(__name__)


class CodeGenerationType(Enum):
    """Types of code generation requests."""
    FULL_INTERFACE = "full_interface"
    TAG_READER = "tag_reader"
    TAG_WRITER = "tag_writer"
    SAFETY_MONITOR = "safety_monitor"
    DATA_LOGGER = "data_logger"
    ALARM_HANDLER = "alarm_handler"
    DIAGNOSTIC_TOOL = "diagnostic_tool"
    BATCH_PROCESSOR = "batch_processor"


class CodeQuality(Enum):
    """Code quality levels."""
    BASIC = "basic"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"
    SAFETY_CRITICAL = "safety_critical"


class CodeFramework(Enum):
    """Target frameworks for code generation."""
    PYCOMM3 = "pycomm3"
    OPCUA = "opcua"
    MODBUS = "modbus"
    ETHERNET_IP = "ethernet_ip"
    CUSTOM = "custom"


@dataclass
class CodeGenerationRequest:
    """Request specification for code generation."""
    generation_type: CodeGenerationType
    quality_level: CodeQuality = CodeQuality.PRODUCTION
    framework: CodeFramework = CodeFramework.PYCOMM3
    target_language: str = "python"
    
    # PLC-specific parameters
    controller_name: str = ""
    target_tags: List[str] = field(default_factory=list)
    safety_tags: List[str] = field(default_factory=list)
    
    # Code requirements
    include_error_handling: bool = True
    include_logging: bool = True
    include_validation: bool = True
    include_retry_logic: bool = True
    include_monitoring: bool = True
    include_documentation: bool = True
    
    # Performance requirements
    async_support: bool = True
    threading_support: bool = False
    connection_pooling: bool = True
    
    # Additional requirements
    custom_requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'generation_type': self.generation_type.value,
            'quality_level': self.quality_level.value,
            'framework': self.framework.value,
            'target_language': self.target_language,
            'controller_name': self.controller_name,
            'target_tags': self.target_tags,
            'safety_tags': self.safety_tags,
            'include_error_handling': self.include_error_handling,
            'include_logging': self.include_logging,
            'include_validation': self.include_validation,
            'include_retry_logic': self.include_retry_logic,
            'include_monitoring': self.include_monitoring,
            'include_documentation': self.include_documentation,
            'async_support': self.async_support,
            'threading_support': self.threading_support,
            'connection_pooling': self.connection_pooling,
            'custom_requirements': self.custom_requirements,
            'constraints': self.constraints
        }


@dataclass
class GeneratedCode:
    """Container for generated code and metadata."""
    code: str
    language: str
    framework: str
    
    # Generation metadata
    generation_time: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    model_used: str = ""
    
    # Code analysis
    estimated_lines: int = 0
    complexity_score: float = 0.0
    quality_score: float = 0.0
    
    # Dependencies
    imports: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    
    # Validation results
    syntax_valid: bool = False
    lint_score: float = 0.0
    security_score: float = 0.0
    
    # Additional metadata
    features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'code': self.code,
            'language': self.language,
            'framework': self.framework,
            'generation_time': self.generation_time,
            'token_usage': self.token_usage,
            'model_used': self.model_used,
            'estimated_lines': self.estimated_lines,
            'complexity_score': self.complexity_score,
            'quality_score': self.quality_score,
            'imports': self.imports,
            'requirements': self.requirements,
            'syntax_valid': self.syntax_valid,
            'lint_score': self.lint_score,
            'security_score': self.security_score,
            'features': self.features,
            'warnings': self.warnings,
            'suggestions': self.suggestions
        }


class CodeValidator:
    """Validates generated code for syntax, style, and security."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.CodeValidator')
    
    def validate_python_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax."""
        errors = []
        
        try:
            # Parse the code to check syntax
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax error on line {e.lineno}: {e.msg}")
            return False, errors
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
            return False, errors
    
    def analyze_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")
        except:
            # Fallback to regex if AST parsing fails
            import_patterns = [
                r'^import\s+([^\s]+)',
                r'^from\s+([^\s]+)\s+import'
            ]
            
            for line in code.split('\n'):
                line = line.strip()
                for pattern in import_patterns:
                    match = re.match(pattern, line)
                    if match:
                        imports.append(match.group(1))
        
        return list(set(imports))
    
    def estimate_complexity(self, code: str) -> float:
        """Estimate code complexity (simplified McCabe complexity)."""
        complexity = 1  # Base complexity
        
        # Count control flow statements
        control_flow_patterns = [
            r'\bif\b', r'\belse\b', r'\belif\b',
            r'\bfor\b', r'\bwhile\b',
            r'\btry\b', r'\bexcept\b', r'\bfinally\b',
            r'\bwith\b', r'\basync\s+def\b', r'\bdef\b'
        ]
        
        for pattern in control_flow_patterns:
            complexity += len(re.findall(pattern, code, re.IGNORECASE))
        
        # Normalize by lines of code
        lines = len([line for line in code.split('\n') if line.strip()])
        if lines > 0:
            complexity = complexity / lines * 10  # Scale to 0-10 range
        
        return min(complexity, 10.0)
    
    def check_security_issues(self, code: str) -> Tuple[float, List[str]]:
        """Check for potential security issues."""
        issues = []
        score = 10.0  # Start with perfect score
        
        # Security patterns to check
        security_patterns = [
            (r'\beval\s*\(', "Use of eval() function", 2.0),
            (r'\bexec\s*\(', "Use of exec() function", 2.0),
            (r'__import__\s*\(', "Dynamic import usage", 1.0),
            (r'subprocess\.call', "Subprocess execution", 1.5),
            (r'os\.system', "OS system call", 2.0),
            (r'shell\s*=\s*True', "Shell execution enabled", 1.5),
            (r'pickle\.loads?', "Pickle deserialization", 1.0),
            (r'yaml\.load\s*\((?!.*Loader)', "Unsafe YAML loading", 1.0)
        ]
        
        for pattern, description, penalty in security_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                issues.append(f"{description} (found {len(matches)} occurrence(s))")
                score -= penalty
        
        return max(score, 0.0), issues
    
    def validate_code(self, generated_code: GeneratedCode) -> GeneratedCode:
        """Perform comprehensive code validation."""
        start_time = time.time()
        
        # Syntax validation
        syntax_valid, syntax_errors = self.validate_python_syntax(generated_code.code)
        generated_code.syntax_valid = syntax_valid
        
        if syntax_errors:
            generated_code.warnings.extend(syntax_errors)
        
        # Import analysis
        generated_code.imports = self.analyze_imports(generated_code.code)
        
        # Complexity analysis
        generated_code.complexity_score = self.estimate_complexity(generated_code.code)
        
        # Security analysis
        security_score, security_issues = self.check_security_issues(generated_code.code)
        generated_code.security_score = security_score
        
        if security_issues:
            generated_code.warnings.extend(security_issues)
        
        # Line count
        generated_code.estimated_lines = len([
            line for line in generated_code.code.split('\n') 
            if line.strip() and not line.strip().startswith('#')
        ])
        
        # Overall quality score
        quality_factors = [
            generated_code.syntax_valid * 4.0,  # Syntax is critical
            (10.0 - generated_code.complexity_score) * 0.3,  # Lower complexity is better
            generated_code.security_score * 0.5,  # Security matters
            5.0 if generated_code.estimated_lines > 10 else 2.0,  # Reasonable code length
            3.0 if len(generated_code.imports) < 10 else 1.0  # Not too many dependencies
        ]
        
        generated_code.quality_score = min(sum(quality_factors), 10.0)
        
        validation_time = time.time() - start_time
        self.logger.info(f"Code validation completed in {validation_time:.2f}s")
        
        return generated_code


class CodeGenerator:
    """Main code generation engine."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__ + '.CodeGenerator')
        
        # Initialize components
        self.ai_manager = AIInterfaceManager(config_file)
        self.prompt_engineer = PromptEngineering()
        self.validator = CodeValidator()
        
        # Generation statistics
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_tokens_used': 0,
            'average_generation_time': 0.0
        }
    
    def _create_generation_context(self, request: CodeGenerationRequest, 
                                 analysis_results: Optional[Dict[str, Any]] = None) -> PromptContext:
        """Create prompt context from generation request and analysis results."""
        
        # Base context from analysis results if available
        if analysis_results:
            context = self.prompt_engineer.create_context_from_analysis(analysis_results)
        else:
            context = PromptContext(controller_name=request.controller_name)
        
        # Add generation-specific requirements
        context.user_requirements.extend([
            f"Generate {request.generation_type.value} code",
            f"Target quality level: {request.quality_level.value}",
            f"Use {request.framework.value} framework"
        ])
        
        if request.include_error_handling:
            context.user_requirements.append("Include comprehensive error handling")
        
        if request.include_logging:
            context.user_requirements.append("Include structured logging")
        
        if request.include_validation:
            context.user_requirements.append("Include input validation")
        
        if request.include_retry_logic:
            context.user_requirements.append("Include connection retry logic")
        
        if request.include_monitoring:
            context.user_requirements.append("Include performance monitoring")
        
        if request.async_support:
            context.user_requirements.append("Support asynchronous operations")
        
        if request.connection_pooling:
            context.user_requirements.append("Implement connection pooling")
        
        # Add custom requirements and constraints
        context.user_requirements.extend(request.custom_requirements)
        context.constraints.extend(request.constraints)
        
        # Add target tags if specified
        if request.target_tags:
            context.user_requirements.append(f"Focus on tags: {', '.join(request.target_tags)}")
        
        # Add safety considerations
        if request.safety_tags:
            context.safety_concerns.append(f"Monitor safety tags: {', '.join(request.safety_tags)}")
        
        # Set target language
        context.target_language = request.target_language
        
        return context
    
    def _select_template(self, request: CodeGenerationRequest) -> str:
        """Select appropriate template based on generation request."""
        
        # Template selection logic
        if request.generation_type == CodeGenerationType.FULL_INTERFACE:
            if request.quality_level in [CodeQuality.ENTERPRISE, CodeQuality.SAFETY_CRITICAL]:
                return "advanced_plc_interface"
            else:
                return "basic_plc_interface"
        
        elif request.generation_type == CodeGenerationType.SAFETY_MONITOR:
            return "safety_analysis"
        
        elif request.generation_type in [CodeGenerationType.TAG_READER, CodeGenerationType.TAG_WRITER]:
            return "basic_plc_interface"
        
        elif request.generation_type == CodeGenerationType.DATA_LOGGER:
            return "advanced_plc_interface"
        
        elif request.generation_type == CodeGenerationType.DIAGNOSTIC_TOOL:
            return "code_analysis"
        
        else:
            # Default to basic interface
            return "basic_plc_interface"
    
    async def generate_code(self, request: CodeGenerationRequest, 
                          analysis_results: Optional[Dict[str, Any]] = None) -> GeneratedCode:
        """Generate code based on request and analysis results."""
        
        start_time = time.time()
        self.logger.info(f"Starting code generation for {request.generation_type.value}")
        
        try:
            # Create context
            context = self._create_generation_context(request, analysis_results)
            
            # Select template
            template_name = self._select_template(request)
            
            # Generate prompt
            prompt_data = self.prompt_engineer.generate_contextual_prompt(
                template_name=template_name,
                analysis_results=analysis_results or {},
                user_requirements=context.user_requirements,
                constraints=context.constraints
            )
            
            # Create AI messages
            messages = [
                AIMessage(role="system", content=prompt_data["system_prompt"]),
                AIMessage(role="user", content=prompt_data["user_prompt"])
            ]
            
            # Generate response
            self.logger.info(f"Generating code using template: {template_name}")
            response = await self.ai_manager.generate_response(messages)
            
            if not response or not response.content:
                raise ValueError("Empty response from AI model")
            
            # Extract code from response
            code = self._extract_code_from_response(response.content)
            
            # Create generated code object
            generated_code = GeneratedCode(
                code=code,
                language=request.target_language,
                framework=request.framework.value,
                generation_time=time.time() - start_time,
                model_used=response.model or "unknown"
            )
            
            # Add token usage if available
            if hasattr(response, 'tokens_used'):
                generated_code.token_usage = {
                    'total': response.tokens_used,
                    'prompt': getattr(response, 'tokens_prompt', 0),
                    'completion': getattr(response, 'tokens_completion', 0)
                }
            
            # Add features based on request
            generated_code.features = self._identify_features(request, code)
            
            # Validate the generated code
            generated_code = self.validator.validate_code(generated_code)
            
            # Update statistics
            self.generation_stats['total_generations'] += 1
            self.generation_stats['successful_generations'] += 1
            self.generation_stats['total_tokens_used'] += generated_code.token_usage.get('total', 0)
            
            # Update average generation time
            total_time = (self.generation_stats['average_generation_time'] * 
                         (self.generation_stats['total_generations'] - 1) + 
                         generated_code.generation_time)
            self.generation_stats['average_generation_time'] = total_time / self.generation_stats['total_generations']
            
            self.logger.info(f"Code generation completed successfully in {generated_code.generation_time:.2f}s")
            return generated_code
            
        except Exception as e:
            self.generation_stats['total_generations'] += 1
            self.generation_stats['failed_generations'] += 1
            self.logger.error(f"Code generation failed: {str(e)}")
            raise
    
    def _extract_code_from_response(self, response_content: str) -> str:
        """Extract Python code from AI response."""
        
        # Look for code blocks
        code_block_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'```python(.*?)```',
            r'```(.*?)```'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, response_content, re.DOTALL)
            if matches:
                # Return the first (usually largest) code block
                return matches[0].strip()
        
        # If no code blocks found, look for python-like content
        lines = response_content.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start collecting if we see typical Python patterns
            if (re.match(r'^(import|from|def|class|if|for|while|try|with|async)', line.strip()) or
                in_code):
                in_code = True
                code_lines.append(line)
            # Stop if we see non-code content after starting code collection
            elif in_code and line.strip() and not re.match(r'^[\s#]', line):
                break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Fallback: return the entire response
        return response_content.strip()
    
    def _identify_features(self, request: CodeGenerationRequest, code: str) -> List[str]:
        """Identify features present in the generated code."""
        features = []
        
        # Check for requested features
        if request.include_error_handling and ('try:' in code or 'except' in code):
            features.append("Error handling")
        
        if request.include_logging and ('logging' in code or 'logger' in code):
            features.append("Logging")
        
        if request.include_validation and ('validate' in code or 'check' in code):
            features.append("Input validation")
        
        if request.include_retry_logic and ('retry' in code or 'attempt' in code):
            features.append("Retry logic")
        
        if request.async_support and ('async' in code or 'await' in code):
            features.append("Async support")
        
        if request.connection_pooling and ('pool' in code.lower()):
            features.append("Connection pooling")
        
        # Check for common patterns
        if 'class' in code:
            features.append("Object-oriented design")
        
        if 'def ' in code:
            features.append("Modular functions")
        
        if '@' in code:
            features.append("Decorators")
        
        if 'typing' in code or ': str' in code or '-> ' in code:
            features.append("Type hints")
        
        if '"""' in code or "'''" in code:
            features.append("Documentation strings")
        
        return features
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.generation_stats.copy()


class CodeGenerationPipeline:
    """Complete pipeline for code generation from L5X analysis to Python code."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__ + '.CodeGenerationPipeline')
        
        # Initialize components
        self.code_generator = CodeGenerator(config_file)
        self.analysis_integrator = None  # Will be initialized when needed
        
        # Pipeline configuration
        self.output_dir = Path("step19_output")
        self.output_dir.mkdir(exist_ok=True)
    
    async def generate_from_l5x(self, l5x_file_path: str, 
                              request: CodeGenerationRequest) -> Dict[str, Any]:
        """Generate code from L5X file analysis."""
        
        pipeline_start = time.time()
        self.logger.info(f"Starting pipeline for L5X file: {l5x_file_path}")
        
        try:
            # Initialize analysis integrator if needed
            if not self.analysis_integrator:
                config = AnalysisSystemsConfig(l5x_file_path=l5x_file_path)
                self.analysis_integrator = AnalysisSystemsIntegrator(config)
            
            # Run comprehensive analysis
            self.logger.info("Running comprehensive L5X analysis...")
            analysis_results = self.analysis_integrator.run_comprehensive_analysis()
            
            # Generate code
            self.logger.info("Generating code from analysis results...")
            generated_code = await self.code_generator.generate_code(request, analysis_results)
            
            # Create pipeline results
            pipeline_results = {
                'request': request.to_dict(),
                'analysis_summary': self._create_analysis_summary(analysis_results),
                'generated_code': generated_code.to_dict(),
                'pipeline_time': time.time() - pipeline_start,
                'timestamp': time.time()
            }
            
            # Save results
            await self._save_results(pipeline_results, l5x_file_path)
            
            self.logger.info(f"Pipeline completed in {pipeline_results['pipeline_time']:.2f}s")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _create_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of analysis results for reporting."""
        return {
            'controller': analysis_results.get('controller', {}),
            'tags_count': len(analysis_results.get('tags', [])),
            'routines_count': len(analysis_results.get('routines', [])),
            'safety_concerns': len(analysis_results.get('safety_analysis', {}).get('concerns', [])),
            'logic_patterns': len(analysis_results.get('logic_patterns', [])),
            'optimization_opportunities': len(analysis_results.get('optimization_analysis', {}).get('opportunities', [])),
            'analysis_duration': analysis_results.get('metadata', {}).get('analysis_duration', 0)
        }
    
    async def _save_results(self, pipeline_results: Dict[str, Any], l5x_file_path: str):
        """Save pipeline results to output directory."""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        l5x_name = Path(l5x_file_path).stem
        
        # Save complete results as JSON
        json_file = self.output_dir / f"{l5x_name}_generation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        # Save generated code as Python file
        code_file = self.output_dir / f"{l5x_name}_generated_{timestamp}.py"
        with open(code_file, 'w') as f:
            f.write(pipeline_results['generated_code']['code'])
        
        # Save HTML report
        html_file = self.output_dir / f"{l5x_name}_report_{timestamp}.html"
        html_content = self._generate_html_report(pipeline_results)
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_html_report(self, pipeline_results: Dict[str, Any]) -> str:
        """Generate HTML report for pipeline results."""
        
        generated_code = pipeline_results['generated_code']
        analysis_summary = pipeline_results['analysis_summary']
        request = pipeline_results['request']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Step 19: Code Generation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ background: #d4edda; border-color: #c3e6cb; }}
                .warning {{ background: #fff3cd; border-color: #ffeaa7; }}
                .error {{ background: #f8d7da; border-color: #f5c6cb; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; }}
                pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .feature {{ display: inline-block; margin: 5px; padding: 5px 10px; background: #17a2b8; color: white; border-radius: 15px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Step 19: Code Generation Pipeline Report</h1>
                <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>📋 Generation Request</h2>
                <div class="metric"><strong>Type:</strong> {request['generation_type']}</div>
                <div class="metric"><strong>Quality:</strong> {request['quality_level']}</div>
                <div class="metric"><strong>Framework:</strong> {request['framework']}</div>
                <div class="metric"><strong>Language:</strong> {request['target_language']}</div>
                <div class="metric"><strong>Controller:</strong> {request['controller_name']}</div>
            </div>
            
            <div class="section">
                <h2>📊 Analysis Summary</h2>
                <div class="metric"><strong>Tags:</strong> {analysis_summary['tags_count']}</div>
                <div class="metric"><strong>Routines:</strong> {analysis_summary['routines_count']}</div>
                <div class="metric"><strong>Safety Concerns:</strong> {analysis_summary['safety_concerns']}</div>
                <div class="metric"><strong>Logic Patterns:</strong> {analysis_summary['logic_patterns']}</div>
                <div class="metric"><strong>Analysis Time:</strong> {analysis_summary['analysis_duration']:.2f}s</div>
            </div>
            
            <div class="section {'success' if generated_code['syntax_valid'] else 'error'}">
                <h2>🎯 Generation Results</h2>
                <div class="metric"><strong>Generation Time:</strong> {generated_code['generation_time']:.2f}s</div>
                <div class="metric"><strong>Lines of Code:</strong> {generated_code['estimated_lines']}</div>
                <div class="metric"><strong>Quality Score:</strong> {generated_code['quality_score']:.1f}/10</div>
                <div class="metric"><strong>Complexity Score:</strong> {generated_code['complexity_score']:.1f}/10</div>
                <div class="metric"><strong>Security Score:</strong> {generated_code['security_score']:.1f}/10</div>
                <div class="metric"><strong>Syntax Valid:</strong> {'✅ Yes' if generated_code['syntax_valid'] else '❌ No'}</div>
                <div class="metric"><strong>Model Used:</strong> {generated_code['model_used']}</div>
            </div>
            
            <div class="section">
                <h2>⚙️ Features Implemented</h2>
                <div>
                    {"".join([f'<span class="feature">{feature}</span>' for feature in generated_code['features']])}
                </div>
            </div>
            
            <div class="section">
                <h2>📦 Dependencies</h2>
                <div><strong>Imports:</strong> {", ".join(generated_code['imports']) if generated_code['imports'] else 'None detected'}</div>
            </div>
            
            {"".join([f'<div class="section warning"><h2>⚠️ Warnings</h2><ul>{"".join([f"<li>{warning}</li>" for warning in generated_code["warnings"]])}</ul></div>' if generated_code.get('warnings') else ''])}
            
            <div class="section">
                <h2>💻 Generated Code</h2>
                <pre><code>{generated_code['code']}</code></pre>
            </div>
            
            <div class="section">
                <h2>📈 Token Usage</h2>
                <div class="metric"><strong>Total Tokens:</strong> {generated_code['token_usage'].get('total', 'N/A')}</div>
                <div class="metric"><strong>Prompt Tokens:</strong> {generated_code['token_usage'].get('prompt', 'N/A')}</div>
                <div class="metric"><strong>Completion Tokens:</strong> {generated_code['token_usage'].get('completion', 'N/A')}</div>
            </div>
            
            <div class="section">
                <h2>⏱️ Performance Metrics</h2>
                <div class="metric"><strong>Total Pipeline Time:</strong> {pipeline_results['pipeline_time']:.2f}s</div>
                <div class="metric"><strong>Analysis Time:</strong> {analysis_summary['analysis_duration']:.2f}s</div>
                <div class="metric"><strong>Generation Time:</strong> {generated_code['generation_time']:.2f}s</div>
            </div>
        </body>
        </html>
        """
        
        return html_content


# Convenience functions for common operations
async def generate_plc_interface(l5x_file_path: str, controller_name: str = "", 
                               quality: CodeQuality = CodeQuality.PRODUCTION) -> Dict[str, Any]:
    """Generate a basic PLC interface from L5X file."""
    
    request = CodeGenerationRequest(
        generation_type=CodeGenerationType.FULL_INTERFACE,
        quality_level=quality,
        controller_name=controller_name,
        include_error_handling=True,
        include_logging=True,
        include_validation=True,
        async_support=True
    )
    
    pipeline = CodeGenerationPipeline()
    return await pipeline.generate_from_l5x(l5x_file_path, request)


async def generate_safety_monitor(l5x_file_path: str, safety_tags: List[str], 
                                controller_name: str = "") -> Dict[str, Any]:
    """Generate safety monitoring code from L5X file."""
    
    request = CodeGenerationRequest(
        generation_type=CodeGenerationType.SAFETY_MONITOR,
        quality_level=CodeQuality.SAFETY_CRITICAL,
        controller_name=controller_name,
        safety_tags=safety_tags,
        include_error_handling=True,
        include_logging=True,
        include_monitoring=True,
        custom_requirements=[
            "Implement fail-safe behavior",
            "Include safety system diagnostics",
            "Monitor safety tag states continuously"
        ]
    )
    
    pipeline = CodeGenerationPipeline()
    return await pipeline.generate_from_l5x(l5x_file_path, request)


if __name__ == "__main__":
    # Demo/test code
    import asyncio
    
    async def demo():
        """Demonstrate code generation pipeline."""
        print("🚀 Step 19: Code Generation Pipeline Demo")
        print("=" * 50)
        
        # Example L5X file (adjust path as needed)
        l5x_file = "Assembly_Controls_Robot.L5X"
        
        try:
            # Generate basic PLC interface
            print("Generating basic PLC interface...")
            results = await generate_plc_interface(
                l5x_file_path=l5x_file,
                controller_name="Assembly_Controls_Robot",
                quality=CodeQuality.PRODUCTION
            )
            
            print(f"✅ Code generated successfully!")
            print(f"📊 Lines of code: {results['generated_code']['estimated_lines']}")
            print(f"🎯 Quality score: {results['generated_code']['quality_score']:.1f}/10")
            print(f"⏱️ Total time: {results['pipeline_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    # Run demo
    asyncio.run(demo())
