"""
Step 18: AI Package Initialization
Advanced Prompt Engineering and AI Integration Package
"""

from .ai_interface import AIInterfaceManager, AIProvider, AIMessage, AIResponse
from .prompt_engineering import (
    PromptEngineering, PromptContext, PromptTemplate, PromptBuilder,
    PromptType, PromptComplexity, PLCDomain, quick_prompt
)
from .ai_integration import (
    AnalysisSystemsIntegrator, AIIntegratedCodeGenerator, AnalysisSystemsConfig
)
from .plc_ai_service import PLCAIService
from .code_generation import (
    CodeGenerator, CodeGenerationPipeline, CodeGenerationRequest, GeneratedCode,
    CodeValidator, CodeGenerationType, CodeQuality, CodeFramework,
    generate_plc_interface, generate_safety_monitor
)
from .enhanced_validation import (
    EnhancedPLCValidator, PLCTagMapper, PLCControllerValidator, PLCSecurityValidator,
    PLCValidationType, ValidationSeverity, PLCValidationIssue, TagMappingResult,
    ControllerCompatibilityResult, PLCValidationResult, validate_plc_code,
    validate_tag_mapping_only, validate_controller_compatibility_only
)
from .validation_loop import (
    ValidationLoop, CorrectionGenerator, CorrectionStrategy, CorrectionType,
    LoopTerminationReason, ValidationLoopResult, CorrectionAttempt, ValidationIteration,
    run_validation_loop, iterative_code_improvement
)
from .advanced_ai_features import (
    AdvancedAIFeatures, ContextType, AIModelRole, LearningStrategy,
    ContextualInformation, MultiModelResponse, LearningRecord,
    AdvancedContextManager, MultiModelCoordinator, LearningEngine,
    generate_code_with_advanced_ai, improve_code_with_learning, analyze_user_ai_patterns
)

__version__ = "1.0.0"
__author__ = "PLC Logic Decompiler Project"

__all__ = [
    # AI Interface
    'AIInterfaceManager',
    'AIProvider', 
    'AIMessage',
    'AIResponse',
    
    # Prompt Engineering
    'PromptEngineering',
    'PromptContext',
    'PromptTemplate',
    'PromptBuilder',
    'PromptType',
    'PromptComplexity',
    'PLCDomain',
    'quick_prompt',
    
    # AI Integration
    'AnalysisSystemsIntegrator',
    'AIIntegratedCodeGenerator',
    'AnalysisSystemsConfig',
    
    # PLC AI Service
    'PLCAIService',
    
    # Code Generation (Step 19)
    'CodeGenerator',
    'CodeGenerationPipeline',
    'CodeGenerationRequest',
    'GeneratedCode',
    'CodeValidator',
    'CodeGenerationType',
    'CodeQuality',
    'CodeFramework',
    'generate_plc_interface',
    'generate_safety_monitor',
    
    # Enhanced Validation (Step 20)
    'EnhancedPLCValidator',
    'PLCTagMapper',
    'PLCControllerValidator', 
    'PLCSecurityValidator',
    'PLCValidationType',
    'ValidationSeverity',
    'PLCValidationIssue',
    'TagMappingResult',
    'ControllerCompatibilityResult',
    'PLCValidationResult',
    'validate_plc_code',
    'validate_tag_mapping_only',
    'validate_controller_compatibility_only',
    
    # Validation Loop (Step 21)
    'ValidationLoop',
    'CorrectionGenerator',
    'CorrectionStrategy',
    'CorrectionType',
    'LoopTerminationReason',
    'ValidationLoopResult',
    'CorrectionAttempt',
    'ValidationIteration',
    'run_validation_loop',
    'iterative_code_improvement',
    
    # Advanced AI Features (Step 22)
    'AdvancedAIFeatures',
    'ContextType',
    'AIModelRole',
    'LearningStrategy',
    'ContextualInformation',
    'MultiModelResponse',
    'LearningRecord',
    'AdvancedContextManager',
    'MultiModelCoordinator',
    'LearningEngine',
    'generate_code_with_advanced_ai',
    'improve_code_with_learning',
    'analyze_user_ai_patterns'
]
