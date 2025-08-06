"""
Step 21: Validation Loop Implementation

This module implements iterative validation and correction loops that automatically
detect issues in generated code and attempt to fix them using AI-powered corrections.

Key Features:
- Iterative validation and correction cycles
- AI-powered automatic issue resolution
- Validation feedback integration
- Code improvement workflows
- Convergence detection and loop termination
- Comprehensive correction tracking and reporting
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

# Import our existing components
from .enhanced_validation import (
    EnhancedPLCValidator, PLCValidationResult, PLCValidationIssue, 
    PLCValidationType, ValidationSeverity, validate_plc_code
)
from .code_generation import (
    CodeGenerator, GeneratedCode, CodeGenerationRequest, CodeGenerationType, 
    CodeQuality, CodeFramework
)
from .prompt_engineering import PromptEngineering, PromptContext, PromptType
from .ai_interface import AIInterfaceManager, AIProvider


class CorrectionStrategy(Enum):
    """Strategies for correcting validation issues"""
    IMMEDIATE = "immediate"          # Fix issues immediately as found
    BATCH = "batch"                  # Collect issues and fix in batches
    PRIORITY = "priority"            # Fix critical/error issues first
    INCREMENTAL = "incremental"      # Make small incremental improvements


class LoopTerminationReason(Enum):
    """Reasons for terminating validation loops"""
    CONVERGED = "converged"          # No more issues found or improvements made
    MAX_ITERATIONS = "max_iterations"  # Reached maximum iteration limit
    DEGRADED = "degraded"            # Code quality is getting worse
    ERROR = "error"                  # Error occurred during validation/correction
    USER_STOPPED = "user_stopped"    # User manually stopped the loop


class CorrectionType(Enum):
    """Types of corrections that can be applied"""
    TAG_MAPPING_FIX = "tag_mapping_fix"
    SECURITY_FIX = "security_fix"
    PERFORMANCE_FIX = "performance_fix"
    SYNTAX_FIX = "syntax_fix"
    PROTOCOL_FIX = "protocol_fix"
    RUNTIME_FIX = "runtime_fix"
    STYLE_FIX = "style_fix"


@dataclass
class CorrectionAttempt:
    """Single correction attempt"""
    attempt_id: str
    correction_type: CorrectionType
    issue: PLCValidationIssue
    original_code: str
    corrected_code: str
    correction_prompt: str
    success: bool
    improvement_score: float
    timestamp: datetime
    ai_response_metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationIteration:
    """Single iteration of the validation loop"""
    iteration_number: int
    timestamp: datetime
    input_code: str
    validation_result: PLCValidationResult
    corrections_attempted: List[CorrectionAttempt] = field(default_factory=list)
    output_code: str = ""
    overall_improvement: float = 0.0
    issues_resolved: int = 0
    issues_introduced: int = 0


@dataclass
class ValidationLoopResult:
    """Complete result of validation loop execution"""
    loop_id: str
    start_time: datetime
    end_time: datetime
    initial_code: str
    final_code: str
    termination_reason: LoopTerminationReason
    total_iterations: int
    
    # Quality progression
    initial_score: float
    final_score: float
    score_improvement: float
    
    # Issue tracking
    initial_issues: int
    final_issues: int
    issues_resolved: int
    issues_introduced: int
    
    # Iterations
    iterations: List[ValidationIteration] = field(default_factory=list)
    
    # Summary statistics
    total_corrections: int = 0
    successful_corrections: int = 0
    correction_success_rate: float = 0.0


class CorrectionGenerator:
    """Generates AI-powered corrections for validation issues"""
    
    def __init__(self, ai_interface: AIInterfaceManager):
        self.ai_interface = ai_interface
        self.prompt_engine = PromptEngineering()
        self.logger = logging.getLogger(__name__)
        
        # Correction templates
        self.correction_templates = {
            CorrectionType.TAG_MAPPING_FIX: self._create_tag_mapping_template(),
            CorrectionType.SECURITY_FIX: self._create_security_fix_template(),
            CorrectionType.PERFORMANCE_FIX: self._create_performance_fix_template(),
            CorrectionType.SYNTAX_FIX: self._create_syntax_fix_template(),
            CorrectionType.PROTOCOL_FIX: self._create_protocol_fix_template(),
            CorrectionType.RUNTIME_FIX: self._create_runtime_fix_template(),
            CorrectionType.STYLE_FIX: self._create_style_fix_template()
        }
    
    def _create_tag_mapping_template(self) -> str:
        """Template for tag mapping corrections"""
        return """
You are a PLC code correction specialist. Fix the tag mapping issues in the following Python code.

ISSUE: {issue_description}
SUGGESTED FIX: {suggested_fix}

ORIGINAL CODE:
{original_code}

AVAILABLE TAGS FROM L5X FILE:
{available_tags}

REQUIREMENTS:
1. Replace unmapped tags with correct L5X tag names
2. Maintain code functionality and structure
3. Use proper tag scoping (controller vs program tags)
4. Preserve code comments and documentation
5. Ensure all tag references are valid

Return ONLY the corrected Python code, no explanations.
"""
    
    def _create_security_fix_template(self) -> str:
        """Template for security corrections"""
        return """
You are a cybersecurity specialist for industrial systems. Fix the security vulnerabilities in this PLC interface code.

SECURITY ISSUE: {issue_description}
RECOMMENDED FIX: {suggested_fix}

VULNERABLE CODE:
{original_code}

SECURITY REQUIREMENTS:
1. Remove hardcoded credentials - use environment variables or secure config
2. Enable SSL/TLS verification for network connections
3. Add input validation for all user inputs
4. Implement proper error handling without exposing sensitive data
5. Use secure logging practices
6. Follow industrial cybersecurity best practices

Return ONLY the security-hardened Python code, no explanations.
"""
    
    def _create_performance_fix_template(self) -> str:
        """Template for performance corrections"""
        return """
You are a PLC communication optimization expert. Improve the performance of this Python code.

PERFORMANCE ISSUE: {issue_description}
OPTIMIZATION SUGGESTION: {suggested_fix}

CURRENT CODE:
{original_code}

OPTIMIZATION REQUIREMENTS:
1. Use batch read/write operations instead of individual operations
2. Implement connection reuse and pooling
3. Add appropriate delays and timing optimizations
4. Optimize data structures and algorithms
5. Implement caching where appropriate
6. Maintain code readability and maintainability

Return ONLY the performance-optimized Python code, no explanations.
"""
    
    def _create_syntax_fix_template(self) -> str:
        """Template for syntax corrections"""
        return """
You are a Python syntax expert. Fix the syntax errors in this PLC interface code.

SYNTAX ERROR: {issue_description}

PROBLEMATIC CODE:
{original_code}

REQUIREMENTS:
1. Fix all Python syntax errors
2. Ensure proper indentation and formatting
3. Add missing imports if needed
4. Fix variable naming and scoping issues
5. Maintain original code functionality
6. Follow Python best practices and PEP 8

Return ONLY the syntax-corrected Python code, no explanations.
"""
    
    def _create_protocol_fix_template(self) -> str:
        """Template for protocol compliance corrections"""
        return """
You are a PLC communication protocol expert. Fix the protocol compliance issues in this code.

PROTOCOL ISSUE: {issue_description}
COMPLIANCE REQUIREMENT: {suggested_fix}

CURRENT CODE:
{original_code}

PROTOCOL REQUIREMENTS:
1. Use proper pycomm3 LogixDriver usage patterns
2. Implement correct connection management with context managers
3. Add proper exception handling for communication errors
4. Follow EtherNet/IP best practices
5. Ensure proper connection cleanup
6. Implement appropriate timeouts and retries

Return ONLY the protocol-compliant Python code, no explanations.
"""
    
    def _create_runtime_fix_template(self) -> str:
        """Template for runtime behavior corrections"""
        return """
You are a real-time systems expert. Fix the runtime behavior issues in this PLC code.

RUNTIME ISSUE: {issue_description}
BEHAVIOR REQUIREMENT: {suggested_fix}

CURRENT CODE:
{original_code}

RUNTIME REQUIREMENTS:
1. Add proper exception handling in loops and real-time operations
2. Implement appropriate sleep/delay patterns
3. Add loop exit conditions and safety checks
4. Prevent infinite loops and resource leaks
5. Implement graceful shutdown procedures
6. Add proper logging for runtime monitoring

Return ONLY the runtime-improved Python code, no explanations.
"""
    
    def _create_style_fix_template(self) -> str:
        """Template for code style corrections"""
        return """
You are a Python code quality expert. Improve the code style and readability of this PLC interface code.

STYLE ISSUE: {issue_description}

CURRENT CODE:
{original_code}

STYLE REQUIREMENTS:
1. Follow PEP 8 guidelines for formatting and naming
2. Add proper docstrings and comments
3. Improve variable and function naming
4. Organize imports properly
5. Add type hints where appropriate
6. Maintain industrial code standards

Return ONLY the style-improved Python code, no explanations.
"""
    
    async def generate_correction(self, issue: PLCValidationIssue, original_code: str, 
                                context: Dict[str, Any] = None) -> CorrectionAttempt:
        """Generate a correction for a validation issue"""
        try:
            # Determine correction type from issue
            correction_type = self._determine_correction_type(issue)
            
            # Build correction prompt
            template = self.correction_templates[correction_type]
            
            correction_prompt = template.format(
                issue_description=issue.message,
                suggested_fix=issue.suggested_fix or "Fix the identified issue",
                original_code=original_code,
                available_tags=context.get('available_tags', '') if context else ''
            )
            
            # Generate correction using AI
            ai_response = await self.ai_interface.generate_response(
                correction_prompt,
                provider=AIProvider.GEMINI,  # Use Gemini for corrections
                max_tokens=2000
            )
            
            # Extract corrected code
            corrected_code = self._extract_code_from_response(ai_response.content)
            
            # Calculate improvement score (basic heuristic)
            improvement_score = self._calculate_improvement_score(
                original_code, corrected_code, issue
            )
            
            attempt = CorrectionAttempt(
                attempt_id=f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                correction_type=correction_type,
                issue=issue,
                original_code=original_code,
                corrected_code=corrected_code,
                correction_prompt=correction_prompt,
                success=len(corrected_code) > 10 and corrected_code != original_code,
                improvement_score=improvement_score,
                timestamp=datetime.now(),
                ai_response_metadata={
                    'token_usage': ai_response.token_usage,
                    'cost': ai_response.cost,
                    'response_time': getattr(ai_response, 'response_time', 0)
                }
            )
            
            self.logger.info(f"Generated correction for {correction_type.value}: "
                           f"{'Success' if attempt.success else 'Failed'}")
            
            return attempt
            
        except Exception as e:
            self.logger.error(f"Error generating correction: {e}")
            
            # Return failed attempt
            return CorrectionAttempt(
                attempt_id=f"correction_failed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                correction_type=CorrectionType.SYNTAX_FIX,
                issue=issue,
                original_code=original_code,
                corrected_code=original_code,
                correction_prompt="",
                success=False,
                improvement_score=0.0,
                timestamp=datetime.now()
            )
    
    def _determine_correction_type(self, issue: PLCValidationIssue) -> CorrectionType:
        """Determine the type of correction needed for an issue"""
        if issue.issue_type == PLCValidationType.TAG_MAPPING:
            return CorrectionType.TAG_MAPPING_FIX
        elif issue.issue_type == PLCValidationType.SECURITY_ANALYSIS:
            return CorrectionType.SECURITY_FIX
        elif issue.issue_type == PLCValidationType.PERFORMANCE_VALIDATION:
            return CorrectionType.PERFORMANCE_FIX
        elif issue.issue_type == PLCValidationType.PROTOCOL_COMPLIANCE:
            return CorrectionType.PROTOCOL_FIX
        elif issue.issue_type == PLCValidationType.RUNTIME_BEHAVIOR:
            return CorrectionType.RUNTIME_FIX
        else:
            return CorrectionType.SYNTAX_FIX
    
    def _extract_code_from_response(self, response_content: str) -> str:
        """Extract Python code from AI response"""
        # Look for code blocks
        import re
        
        # Try to find code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response_content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Try to find code blocks without language specification
        code_blocks = re.findall(r'```\n(.*?)\n```', response_content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, return the entire response (filtered)
        lines = response_content.strip().split('\n')
        
        # Filter out obvious non-code lines
        code_lines = []
        for line in lines:
            if (not line.strip().startswith(('Here', 'The', 'This', 'I', 'You')) and
                not line.strip().endswith((':',)) and
                line.strip()):
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response_content.strip()
    
    def _calculate_improvement_score(self, original_code: str, corrected_code: str, 
                                   issue: PLCValidationIssue) -> float:
        """Calculate basic improvement score for correction"""
        if corrected_code == original_code:
            return 0.0
        
        # Basic heuristics for improvement
        score = 0.0
        
        # Length change (minor factor)
        length_ratio = len(corrected_code) / max(len(original_code), 1)
        if 0.8 <= length_ratio <= 1.5:  # Reasonable length change
            score += 0.1
        
        # Issue-specific improvements
        if issue.issue_type == PLCValidationType.SECURITY_ANALYSIS:
            if 'password' not in corrected_code.lower() and 'password' in original_code.lower():
                score += 0.3
            if 'os.getenv' in corrected_code and 'os.getenv' not in original_code:
                score += 0.2
        
        elif issue.issue_type == PLCValidationType.PERFORMANCE_VALIDATION:
            if corrected_code.count('.read(') < original_code.count('.read('):
                score += 0.3
            if 'batch' in corrected_code.lower():
                score += 0.2
        
        elif issue.issue_type == PLCValidationType.TAG_MAPPING:
            # Hard to measure without re-validation, give moderate score
            score += 0.2
        
        # Code quality indicators
        if 'try:' in corrected_code and 'try:' not in original_code:
            score += 0.1
        
        if 'logging' in corrected_code and 'print(' in original_code:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0


class ValidationLoop:
    """Main validation loop implementation"""
    
    def __init__(self, l5x_file_path: str, ai_interface: AIInterfaceManager):
        self.l5x_file_path = l5x_file_path
        self.ai_interface = ai_interface
        self.validator = EnhancedPLCValidator(l5x_file_path)
        self.correction_generator = CorrectionGenerator(ai_interface)
        self.logger = logging.getLogger(__name__)
        
        # Loop configuration
        self.max_iterations = 5  # Maximum number of iterations
        self.convergence_threshold = 0.1  # Minimum improvement to continue
        self.min_score_threshold = 8.0  # Target quality score
        self.max_corrections_per_iteration = 3  # Limit corrections per iteration
    
    async def run_validation_loop(self, initial_code: GeneratedCode, 
                                strategy: CorrectionStrategy = CorrectionStrategy.PRIORITY) -> ValidationLoopResult:
        """Run the complete validation loop"""
        loop_id = f"validation_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        self.logger.info(f"Starting validation loop {loop_id} with strategy {strategy.value}")
        
        # Initialize result
        result = ValidationLoopResult(
            loop_id=loop_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            initial_code=initial_code.code,
            final_code=initial_code.code,
            termination_reason=LoopTerminationReason.CONVERGED,
            total_iterations=0,
            initial_score=0.0,
            final_score=0.0,
            score_improvement=0.0,
            initial_issues=0,
            final_issues=0,
            issues_resolved=0,
            issues_introduced=0
        )
        
        current_code = initial_code
        previous_score = 0.0
        
        try:
            # Initial validation
            initial_validation = await self.validator.validate_generated_code(current_code)
            result.initial_score = initial_validation.overall_score
            result.initial_issues = initial_validation.total_issues
            previous_score = initial_validation.overall_score
            
            self.logger.info(f"Initial validation: Score {initial_validation.overall_score:.1f}, "
                           f"Issues {initial_validation.total_issues}")
            
            # Main validation loop
            for iteration_num in range(1, self.max_iterations + 1):
                self.logger.info(f"Starting iteration {iteration_num}")
                
                # Create iteration
                iteration = ValidationIteration(
                    iteration_number=iteration_num,
                    timestamp=datetime.now(),
                    input_code=current_code.code,
                    validation_result=initial_validation if iteration_num == 1 else await self.validator.validate_generated_code(current_code)
                )
                
                # Check termination conditions
                if iteration.validation_result.overall_score >= self.min_score_threshold:
                    result.termination_reason = LoopTerminationReason.CONVERGED
                    self.logger.info(f"Converged: Score {iteration.validation_result.overall_score:.1f} "
                                   f"meets threshold {self.min_score_threshold}")
                    result.iterations.append(iteration)
                    break
                
                if iteration.validation_result.total_issues == 0:
                    result.termination_reason = LoopTerminationReason.CONVERGED
                    self.logger.info("Converged: No issues found")
                    result.iterations.append(iteration)
                    break
                
                # Apply correction strategy
                corrected_code = await self._apply_correction_strategy(
                    current_code, iteration.validation_result, strategy
                )
                
                if corrected_code is None or corrected_code.code == current_code.code:
                    result.termination_reason = LoopTerminationReason.CONVERGED
                    self.logger.info("Converged: No corrections could be applied")
                    result.iterations.append(iteration)
                    break
                
                # Update for next iteration
                current_code = corrected_code
                iteration.output_code = corrected_code.code
                
                # Calculate improvement
                score_improvement = iteration.validation_result.overall_score - previous_score
                iteration.overall_improvement = score_improvement
                
                # Check for degradation
                if score_improvement < -0.5:  # Significant degradation
                    result.termination_reason = LoopTerminationReason.DEGRADED
                    self.logger.warning(f"Code quality degraded by {-score_improvement:.1f}")
                    result.iterations.append(iteration)
                    break
                
                # Check for minimal improvement
                if abs(score_improvement) < self.convergence_threshold:
                    result.termination_reason = LoopTerminationReason.CONVERGED
                    self.logger.info(f"Converged: Improvement {score_improvement:.1f} below threshold")
                    result.iterations.append(iteration)
                    break
                
                previous_score = iteration.validation_result.overall_score
                result.iterations.append(iteration)
                
                self.logger.info(f"Iteration {iteration_num} complete: "
                               f"Score {iteration.validation_result.overall_score:.1f}, "
                               f"Issues {iteration.validation_result.total_issues}")
            
            else:
                # Loop completed all iterations
                result.termination_reason = LoopTerminationReason.MAX_ITERATIONS
                self.logger.info(f"Reached maximum iterations ({self.max_iterations})")
            
            # Final validation
            final_validation = await self.validator.validate_generated_code(current_code)
            result.final_code = current_code.code
            result.final_score = final_validation.overall_score
            result.final_issues = final_validation.total_issues
            result.score_improvement = result.final_score - result.initial_score
            result.total_iterations = len(result.iterations)
            
            # Calculate summary statistics
            result.total_corrections = sum(len(iteration.corrections_attempted) for iteration in result.iterations)
            result.successful_corrections = sum(
                sum(1 for correction in iteration.corrections_attempted if correction.success)
                for iteration in result.iterations
            )
            
            if result.total_corrections > 0:
                result.correction_success_rate = result.successful_corrections / result.total_corrections
            
            result.issues_resolved = max(0, result.initial_issues - result.final_issues)
            result.issues_introduced = max(0, result.final_issues - result.initial_issues)
            
        except Exception as e:
            self.logger.error(f"Error in validation loop: {e}")
            result.termination_reason = LoopTerminationReason.ERROR
        
        finally:
            result.end_time = datetime.now()
        
        self.logger.info(f"Validation loop complete: {result.termination_reason.value}, "
                       f"Score improved by {result.score_improvement:.1f}")
        
        return result
    
    async def _apply_correction_strategy(self, current_code: GeneratedCode, 
                                       validation_result: PLCValidationResult,
                                       strategy: CorrectionStrategy) -> Optional[GeneratedCode]:
        """Apply corrections based on strategy"""
        if not validation_result.issues:
            return None
        
        # Select issues to fix based on strategy
        issues_to_fix = self._select_issues_by_strategy(validation_result.issues, strategy)
        
        if not issues_to_fix:
            return None
        
        # Generate corrections
        corrections = []
        current_code_text = current_code.code
        
        for issue in issues_to_fix[:self.max_corrections_per_iteration]:
            correction = await self.correction_generator.generate_correction(
                issue, current_code_text
            )
            corrections.append(correction)
            
            # Apply successful corrections incrementally
            if correction.success and correction.corrected_code != current_code_text:
                current_code_text = correction.corrected_code
                self.logger.info(f"Applied {correction.correction_type.value} correction")
        
        # Return updated code if any corrections were successful
        if current_code_text != current_code.code:
            return GeneratedCode(
                code=current_code_text,
                language=current_code.language,
                framework=current_code.framework,
                quality_level=current_code.quality_level,
                metadata={
                    **current_code.metadata,
                    'corrections_applied': len([c for c in corrections if c.success]),
                    'validation_loop_iteration': True
                }
            )
        
        return None
    
    def _select_issues_by_strategy(self, issues: List[PLCValidationIssue], 
                                 strategy: CorrectionStrategy) -> List[PLCValidationIssue]:
        """Select issues to fix based on correction strategy"""
        if strategy == CorrectionStrategy.PRIORITY:
            # Fix critical and error issues first
            critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
            error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
            warning_issues = [i for i in issues if i.severity == ValidationSeverity.WARNING]
            
            return critical_issues + error_issues + warning_issues
        
        elif strategy == CorrectionStrategy.IMMEDIATE:
            # Fix all issues in order
            return issues
        
        elif strategy == CorrectionStrategy.BATCH:
            # Group by type and fix by type
            return sorted(issues, key=lambda x: x.issue_type.value)
        
        elif strategy == CorrectionStrategy.INCREMENTAL:
            # Fix one issue at a time, starting with highest severity
            sorted_issues = sorted(issues, key=lambda x: (
                x.severity == ValidationSeverity.CRITICAL,
                x.severity == ValidationSeverity.ERROR,
                x.severity == ValidationSeverity.WARNING
            ), reverse=True)
            return sorted_issues[:1]  # Only return first issue
        
        return issues


# Convenience functions

async def run_validation_loop(l5x_file_path: str, generated_code: GeneratedCode,
                            ai_interface: AIInterfaceManager,
                            strategy: CorrectionStrategy = CorrectionStrategy.PRIORITY,
                            max_iterations: int = 5) -> ValidationLoopResult:
    """Convenience function to run validation loop"""
    loop = ValidationLoop(l5x_file_path, ai_interface)
    loop.max_iterations = max_iterations
    return await loop.run_validation_loop(generated_code, strategy)


async def iterative_code_improvement(l5x_file_path: str, initial_code: str,
                                   ai_interface: AIInterfaceManager,
                                   target_score: float = 8.0) -> Tuple[str, ValidationLoopResult]:
    """Convenience function for iterative code improvement"""
    generated_code = GeneratedCode(
        code=initial_code,
        language="python",
        framework="pycomm3",
        quality_level=CodeQuality.PRODUCTION,
        metadata={"source": "iterative_improvement"}
    )
    
    loop = ValidationLoop(l5x_file_path, ai_interface)
    loop.min_score_threshold = target_score
    
    result = await loop.run_validation_loop(generated_code, CorrectionStrategy.PRIORITY)
    return result.final_code, result


if __name__ == "__main__":
    # Example usage
    async def main():
        print("Validation Loop Implementation - Ready for integration")
        print("Use run_validation_loop() for iterative code improvement")
    
    asyncio.run(main())
