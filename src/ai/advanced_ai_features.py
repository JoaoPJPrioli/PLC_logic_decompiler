"""
Step 22: Advanced AI Features Implementation

This module provides enhanced AI integration with context-aware code generation,
multi-model coordination, learning from correction history, and advanced prompt optimization.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Import our existing AI components
from .ai_interface import AIInterfaceManager, AIProvider, AIMessage, AIResponse
from .prompt_engineering import PromptEngineering, PromptContext, PromptType
from .code_generation import CodeGenerator, GeneratedCode, CodeGenerationType, CodeQuality
from .enhanced_validation import EnhancedPLCValidator, PLCValidationResult
from .validation_loop import ValidationLoop, CorrectionStrategy, ValidationLoopResult

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context that can be used for AI enhancement"""
    L5X_SOURCE = "l5x_source"
    HISTORICAL_CORRECTIONS = "historical_corrections"
    SIMILAR_PROJECTS = "similar_projects"
    INDUSTRY_PATTERNS = "industry_patterns"
    USER_PREFERENCES = "user_preferences"
    VALIDATION_FEEDBACK = "validation_feedback"
    PERFORMANCE_METRICS = "performance_metrics"


class AIModelRole(Enum):
    """Roles for different AI models in multi-model coordination"""
    CODE_GENERATOR = "code_generator"        # Primary code generation
    CODE_REVIEWER = "code_reviewer"         # Code review and validation
    OPTIMIZATION_EXPERT = "optimization_expert"  # Performance optimization
    SECURITY_ANALYST = "security_analyst"   # Security analysis
    DOMAIN_EXPERT = "domain_expert"         # PLC domain knowledge
    QUALITY_ASSESSOR = "quality_assessor"   # Quality assessment


class LearningStrategy(Enum):
    """Strategies for learning from historical data"""
    PATTERN_RECOGNITION = "pattern_recognition"
    SUCCESS_AMPLIFICATION = "success_amplification"
    ERROR_PREVENTION = "error_prevention"
    OPTIMIZATION_LEARNING = "optimization_learning"
    USER_ADAPTATION = "user_adaptation"


@dataclass
class ContextualInformation:
    """Container for contextual information used in AI generation"""
    context_type: ContextType
    content: Dict[str, Any]
    relevance_score: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_recent(self, hours: int = 24) -> bool:
        """Check if context is recent within specified hours"""
        return (datetime.now() - self.timestamp).total_seconds() < (hours * 3600)
    
    def is_relevant(self, threshold: float = 0.7) -> bool:
        """Check if context meets relevance threshold"""
        return self.relevance_score >= threshold


@dataclass
class MultiModelResponse:
    """Response from multi-model AI coordination"""
    primary_response: AIResponse
    review_feedback: Optional[AIResponse] = None
    optimization_suggestions: Optional[AIResponse] = None
    security_analysis: Optional[AIResponse] = None
    quality_assessment: Optional[AIResponse] = None
    consensus_score: float = 0.0
    confidence_score: float = 0.0
    model_agreements: Dict[str, bool] = field(default_factory=dict)
    final_recommendation: Optional[str] = None


@dataclass
class LearningRecord:
    """Record of learning from AI interactions"""
    interaction_id: str
    timestamp: datetime
    context: List[ContextualInformation]
    prompt_used: str
    response_quality: float
    correction_needed: bool
    final_result: str
    user_feedback: Optional[str] = None
    success_metrics: Dict[str, float] = field(default_factory=dict)
    patterns_identified: List[str] = field(default_factory=list)


class AdvancedContextManager:
    """Manages contextual information for AI enhancement"""
    
    def __init__(self, max_context_items: int = 100):
        self.max_context_items = max_context_items
        self.contexts: List[ContextualInformation] = []
        self.context_cache: Dict[str, List[ContextualInformation]] = {}
        
    async def add_context(self, context: ContextualInformation) -> None:
        """Add contextual information"""
        self.contexts.append(context)
        
        # Maintain size limit
        if len(self.contexts) > self.max_context_items:
            self.contexts = self.contexts[-self.max_context_items:]
        
        # Update cache
        context_key = f"{context.context_type.value}_{context.source}"
        if context_key not in self.context_cache:
            self.context_cache[context_key] = []
        self.context_cache[context_key].append(context)
        
        logger.debug(f"Added context: {context.context_type.value} from {context.source}")
    
    async def get_relevant_context(self, context_types: List[ContextType], 
                                 relevance_threshold: float = 0.7,
                                 max_items: int = 10) -> List[ContextualInformation]:
        """Get relevant context for AI generation"""
        relevant_contexts = []
        
        for context in self.contexts:
            if (context.context_type in context_types and 
                context.is_relevant(relevance_threshold)):
                relevant_contexts.append(context)
        
        # Sort by relevance and recency
        relevant_contexts.sort(
            key=lambda x: (x.relevance_score, x.timestamp), 
            reverse=True
        )
        
        return relevant_contexts[:max_items]
    
    async def get_historical_patterns(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Extract patterns from historical context"""
        patterns = []
        
        for context in self.contexts:
            if (context.context_type == ContextType.HISTORICAL_CORRECTIONS and
                pattern_type in context.metadata.get('patterns', [])):
                patterns.append(context.content)
        
        return patterns
    
    async def clear_old_context(self, days: int = 30) -> int:
        """Clear context older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        original_count = len(self.contexts)
        
        self.contexts = [
            ctx for ctx in self.contexts 
            if ctx.timestamp > cutoff_date
        ]
        
        # Update cache
        self.context_cache.clear()
        for context in self.contexts:
            context_key = f"{context.context_type.value}_{context.source}"
            if context_key not in self.context_cache:
                self.context_cache[context_key] = []
            self.context_cache[context_key].append(context)
        
        removed_count = original_count - len(self.contexts)
        logger.info(f"Cleared {removed_count} old context items")
        return removed_count


class MultiModelCoordinator:
    """Coordinates multiple AI models for enhanced code generation"""
    
    def __init__(self, ai_interface: AIInterfaceManager):
        self.ai_interface = ai_interface
        self.model_roles: Dict[AIModelRole, AIProvider] = {}
        self.coordination_strategies = {
            'consensus': self._consensus_strategy,
            'expert_review': self._expert_review_strategy,
            'hierarchical': self._hierarchical_strategy
        }
        
    async def configure_models(self, model_config: Dict[AIModelRole, AIProvider]) -> None:
        """Configure AI models for different roles"""
        self.model_roles = model_config
        logger.info(f"Configured {len(model_config)} AI models for coordination")
    
    async def generate_with_coordination(self, 
                                       prompt: str,
                                       strategy: str = 'consensus',
                                       context: Optional[List[ContextualInformation]] = None) -> MultiModelResponse:
        """Generate response using multi-model coordination"""
        if strategy not in self.coordination_strategies:
            raise ValueError(f"Unknown coordination strategy: {strategy}")
        
        return await self.coordination_strategies[strategy](prompt, context)
    
    async def _consensus_strategy(self, prompt: str, 
                                context: Optional[List[ContextualInformation]] = None) -> MultiModelResponse:
        """Generate response using consensus approach"""
        responses = {}
        
        # Get responses from all available models
        for role, provider in self.model_roles.items():
            try:
                enhanced_prompt = await self._enhance_prompt_for_role(prompt, role, context)
                response = await self.ai_interface.generate_response(
                    enhanced_prompt, provider=provider
                )
                responses[role] = response
            except Exception as e:
                logger.warning(f"Failed to get response from {role.value}: {e}")
        
        if not responses:
            raise RuntimeError("No successful responses from any model")
        
        # Analyze consensus
        primary_response = responses.get(AIModelRole.CODE_GENERATOR)
        if not primary_response:
            primary_response = list(responses.values())[0]
        
        # Calculate consensus score
        consensus_score = await self._calculate_consensus_score(responses)
        
        return MultiModelResponse(
            primary_response=primary_response,
            review_feedback=responses.get(AIModelRole.CODE_REVIEWER),
            optimization_suggestions=responses.get(AIModelRole.OPTIMIZATION_EXPERT),
            security_analysis=responses.get(AIModelRole.SECURITY_ANALYST),
            quality_assessment=responses.get(AIModelRole.QUALITY_ASSESSOR),
            consensus_score=consensus_score,
            confidence_score=primary_response.confidence if primary_response else 0.0,
            model_agreements=await self._analyze_model_agreements(responses)
        )
    
    async def _expert_review_strategy(self, prompt: str,
                                    context: Optional[List[ContextualInformation]] = None) -> MultiModelResponse:
        """Generate response using expert review approach"""
        # Primary generation
        code_generator = self.model_roles.get(AIModelRole.CODE_GENERATOR)
        if not code_generator:
            raise RuntimeError("Code generator model not configured")
        
        enhanced_prompt = await self._enhance_prompt_for_role(
            prompt, AIModelRole.CODE_GENERATOR, context
        )
        primary_response = await self.ai_interface.generate_response(
            enhanced_prompt, provider=code_generator
        )
        
        # Expert reviews
        reviews = {}
        for role in [AIModelRole.CODE_REVIEWER, AIModelRole.SECURITY_ANALYST, 
                    AIModelRole.OPTIMIZATION_EXPERT]:
            if role in self.model_roles:
                review_prompt = await self._create_review_prompt(
                    primary_response.content, role, context
                )
                try:
                    review_response = await self.ai_interface.generate_response(
                        review_prompt, provider=self.model_roles[role]
                    )
                    reviews[role] = review_response
                except Exception as e:
                    logger.warning(f"Failed to get review from {role.value}: {e}")
        
        return MultiModelResponse(
            primary_response=primary_response,
            review_feedback=reviews.get(AIModelRole.CODE_REVIEWER),
            optimization_suggestions=reviews.get(AIModelRole.OPTIMIZATION_EXPERT),
            security_analysis=reviews.get(AIModelRole.SECURITY_ANALYST),
            consensus_score=0.8,  # High confidence in expert review
            confidence_score=primary_response.confidence,
            model_agreements=await self._analyze_model_agreements(reviews)
        )
    
    async def _hierarchical_strategy(self, prompt: str,
                                   context: Optional[List[ContextualInformation]] = None) -> MultiModelResponse:
        """Generate response using hierarchical approach"""
        # Domain expert analysis first
        domain_expert = self.model_roles.get(AIModelRole.DOMAIN_EXPERT)
        domain_analysis = None
        
        if domain_expert:
            domain_prompt = await self._create_domain_analysis_prompt(prompt, context)
            try:
                domain_analysis = await self.ai_interface.generate_response(
                    domain_prompt, provider=domain_expert
                )
            except Exception as e:
                logger.warning(f"Failed to get domain analysis: {e}")
        
        # Enhanced code generation with domain insights
        code_generator = self.model_roles.get(AIModelRole.CODE_GENERATOR)
        if not code_generator:
            raise RuntimeError("Code generator model not configured")
        
        enhanced_context = context or []
        if domain_analysis:
            enhanced_context.append(ContextualInformation(
                context_type=ContextType.INDUSTRY_PATTERNS,
                content={'domain_analysis': domain_analysis.content},
                relevance_score=0.9,
                timestamp=datetime.now(),
                source='domain_expert'
            ))
        
        enhanced_prompt = await self._enhance_prompt_for_role(
            prompt, AIModelRole.CODE_GENERATOR, enhanced_context
        )
        primary_response = await self.ai_interface.generate_response(
            enhanced_prompt, provider=code_generator
        )
        
        return MultiModelResponse(
            primary_response=primary_response,
            consensus_score=0.85,
            confidence_score=primary_response.confidence,
            model_agreements={'domain_expert': True}
        )
    
    async def _enhance_prompt_for_role(self, prompt: str, role: AIModelRole,
                                     context: Optional[List[ContextualInformation]] = None) -> str:
        """Enhance prompt based on model role"""
        role_instructions = {
            AIModelRole.CODE_GENERATOR: "Generate high-quality, production-ready Python code for PLC integration.",
            AIModelRole.CODE_REVIEWER: "Review the code for quality, maintainability, and best practices.",
            AIModelRole.OPTIMIZATION_EXPERT: "Focus on performance optimization and efficiency improvements.",
            AIModelRole.SECURITY_ANALYST: "Analyze security implications and identify potential vulnerabilities.",
            AIModelRole.DOMAIN_EXPERT: "Apply PLC and industrial automation domain expertise.",
            AIModelRole.QUALITY_ASSESSOR: "Assess overall code quality and compliance with standards."
        }
        
        enhanced_prompt = f"{role_instructions.get(role, '')}\n\n{prompt}"
        
        # Add relevant context
        if context:
            context_str = await self._format_context_for_prompt(context, role)
            enhanced_prompt = f"{enhanced_prompt}\n\nRelevant Context:\n{context_str}"
        
        return enhanced_prompt
    
    async def _create_review_prompt(self, code: str, role: AIModelRole,
                                  context: Optional[List[ContextualInformation]] = None) -> str:
        """Create review prompt for specific role"""
        review_instructions = {
            AIModelRole.CODE_REVIEWER: "Review this code for correctness, readability, and maintainability:",
            AIModelRole.SECURITY_ANALYST: "Analyze this code for security vulnerabilities and risks:",
            AIModelRole.OPTIMIZATION_EXPERT: "Analyze this code for performance optimization opportunities:"
        }
        
        instruction = review_instructions.get(role, "Review this code:")
        return f"{instruction}\n\n```python\n{code}\n```"
    
    async def _create_domain_analysis_prompt(self, prompt: str,
                                           context: Optional[List[ContextualInformation]] = None) -> str:
        """Create domain analysis prompt"""
        analysis_prompt = f"""
As a PLC and industrial automation domain expert, analyze the following requirements
and provide insights on best practices, common patterns, and potential challenges:

{prompt}

Please provide:
1. Domain-specific considerations
2. Industry best practices
3. Common patterns that apply
4. Potential pitfalls to avoid
5. Recommendations for implementation
"""
        return analysis_prompt
    
    async def _format_context_for_prompt(self, context: List[ContextualInformation],
                                       role: AIModelRole) -> str:
        """Format context information for prompt inclusion"""
        formatted_context = []
        
        for ctx in context:
            if ctx.is_relevant():
                ctx_str = f"- {ctx.context_type.value}: {json.dumps(ctx.content, indent=2)}"
                formatted_context.append(ctx_str)
        
        return "\n".join(formatted_context)
    
    async def _calculate_consensus_score(self, responses: Dict[AIModelRole, AIResponse]) -> float:
        """Calculate consensus score from multiple responses"""
        if len(responses) < 2:
            return 1.0
        
        # Simple consensus calculation based on response similarity
        # In a real implementation, this would use more sophisticated NLP techniques
        scores = []
        response_list = list(responses.values())
        
        for i in range(len(response_list)):
            for j in range(i + 1, len(response_list)):
                # Simplified similarity score
                similarity = await self._calculate_response_similarity(
                    response_list[i].content, response_list[j].content
                )
                scores.append(similarity)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses"""
        # Simplified similarity calculation
        # In practice, would use proper NLP similarity metrics
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _analyze_model_agreements(self, responses: Dict[AIModelRole, AIResponse]) -> Dict[str, bool]:
        """Analyze agreements between different models"""
        agreements = {}
        
        for role, response in responses.items():
            # Simplified agreement analysis
            # In practice, would analyze specific aspects of responses
            agreements[role.value] = response.confidence > 0.7
        
        return agreements


class LearningEngine:
    """Engine for learning from AI interactions and improving over time"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("learning_data")
        self.storage_path.mkdir(exist_ok=True)
        self.learning_records: List[LearningRecord] = []
        self.patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.success_metrics: Dict[str, float] = {}
        
    async def record_interaction(self, record: LearningRecord) -> None:
        """Record AI interaction for learning"""
        self.learning_records.append(record)
        
        # Extract patterns
        await self._extract_patterns(record)
        
        # Update success metrics
        await self._update_success_metrics(record)
        
        # Persist to storage
        await self._save_learning_record(record)
        
        logger.debug(f"Recorded learning interaction: {record.interaction_id}")
    
    async def get_learning_insights(self, context_type: ContextType) -> Dict[str, Any]:
        """Get learning insights for specific context type"""
        insights = {
            'success_patterns': [],
            'failure_patterns': [],
            'optimization_opportunities': [],
            'user_preferences': []
        }
        
        # Analyze successful patterns
        successful_records = [
            r for r in self.learning_records 
            if r.response_quality > 0.8 and not r.correction_needed
        ]
        
        for record in successful_records:
            if any(ctx.context_type == context_type for ctx in record.context):
                insights['success_patterns'].append({
                    'prompt_pattern': record.prompt_used[:100] + "...",
                    'quality_score': record.response_quality,
                    'patterns': record.patterns_identified
                })
        
        # Analyze failure patterns
        failed_records = [
            r for r in self.learning_records 
            if r.response_quality < 0.5 or r.correction_needed
        ]
        
        for record in failed_records:
            if any(ctx.context_type == context_type for ctx in record.context):
                insights['failure_patterns'].append({
                    'prompt_pattern': record.prompt_used[:100] + "...",
                    'quality_score': record.response_quality,
                    'correction_needed': record.correction_needed
                })
        
        return insights
    
    async def optimize_prompt_template(self, template: str, context_type: ContextType) -> str:
        """Optimize prompt template based on learning"""
        insights = await self.get_learning_insights(context_type)
        
        # Apply learning-based optimizations
        optimized_template = template
        
        # Add successful patterns
        for pattern in insights['success_patterns'][:3]:  # Top 3 successful patterns
            if pattern['quality_score'] > 0.9:
                # Extract key phrases from successful prompts
                # This is simplified - real implementation would use NLP
                optimized_template += f"\nNote: Consider patterns like those in high-quality responses."
        
        # Add failure prevention
        if insights['failure_patterns']:
            optimized_template += f"\nAvoid common issues found in previous generations."
        
        return optimized_template
    
    async def get_user_preferences(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get learned user preferences"""
        preferences = {
            'preferred_code_style': 'pythonic',
            'preferred_quality_level': 'production',
            'preferred_frameworks': ['pycomm3'],
            'common_patterns': []
        }
        
        # Analyze user feedback from learning records
        user_records = self.learning_records
        if user_id:
            user_records = [
                r for r in self.learning_records 
                if r.interaction_id.startswith(user_id)
            ]
        
        # Extract preferences from successful interactions
        successful_records = [r for r in user_records if r.response_quality > 0.8]
        
        if successful_records:
            # Analyze patterns in successful interactions
            for record in successful_records:
                preferences['common_patterns'].extend(record.patterns_identified)
        
        return preferences
    
    async def _extract_patterns(self, record: LearningRecord) -> None:
        """Extract patterns from learning record"""
        # Simplified pattern extraction
        patterns = []
        
        # Check for code quality patterns
        if record.response_quality > 0.8:
            patterns.append('high_quality_response')
        
        if not record.correction_needed:
            patterns.append('no_correction_needed')
        
        # Extract patterns from prompt
        if 'safety' in record.prompt_used.lower():
            patterns.append('safety_focused')
        
        if 'optimization' in record.prompt_used.lower():
            patterns.append('optimization_focused')
        
        record.patterns_identified = patterns
        
        # Update global patterns
        for pattern in patterns:
            if pattern not in self.patterns:
                self.patterns[pattern] = []
            self.patterns[pattern].append({
                'interaction_id': record.interaction_id,
                'quality': record.response_quality,
                'timestamp': record.timestamp
            })
    
    async def _update_success_metrics(self, record: LearningRecord) -> None:
        """Update success metrics from learning record"""
        # Update overall success rate
        total_interactions = len(self.learning_records)
        successful_interactions = len([
            r for r in self.learning_records 
            if r.response_quality > 0.7 and not r.correction_needed
        ])
        
        self.success_metrics['overall_success_rate'] = (
            successful_interactions / total_interactions if total_interactions > 0 else 0.0
        )
        
        # Update average quality
        avg_quality = sum(r.response_quality for r in self.learning_records) / total_interactions
        self.success_metrics['average_quality'] = avg_quality
        
        # Update correction rate
        corrections_needed = len([r for r in self.learning_records if r.correction_needed])
        self.success_metrics['correction_rate'] = corrections_needed / total_interactions
    
    async def _save_learning_record(self, record: LearningRecord) -> None:
        """Save learning record to persistent storage"""
        try:
            record_file = self.storage_path / f"{record.interaction_id}.json"
            record_data = {
                'interaction_id': record.interaction_id,
                'timestamp': record.timestamp.isoformat(),
                'prompt_used': record.prompt_used,
                'response_quality': record.response_quality,
                'correction_needed': record.correction_needed,
                'final_result': record.final_result,
                'user_feedback': record.user_feedback,
                'success_metrics': record.success_metrics,
                'patterns_identified': record.patterns_identified,
                'context': [
                    {
                        'context_type': ctx.context_type.value,
                        'content': ctx.content,
                        'relevance_score': ctx.relevance_score,
                        'timestamp': ctx.timestamp.isoformat(),
                        'source': ctx.source,
                        'metadata': ctx.metadata
                    }
                    for ctx in record.context
                ]
            }
            
            with open(record_file, 'w') as f:
                json.dump(record_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save learning record: {e}")


class AdvancedAIFeatures:
    """Main class coordinating all advanced AI features"""
    
    def __init__(self, ai_interface: AIInterfaceManager, l5x_file_path: str):
        self.ai_interface = ai_interface
        self.l5x_file_path = l5x_file_path
        
        # Initialize components
        self.context_manager = AdvancedContextManager()
        self.multi_model_coordinator = MultiModelCoordinator(ai_interface)
        self.learning_engine = LearningEngine()
        
        # Initialize existing components
        self.code_generator = CodeGenerator(ai_interface, l5x_file_path)
        self.enhanced_validator = EnhancedPLCValidator(l5x_file_path)
        self.validation_loop = ValidationLoop(l5x_file_path, ai_interface)
        
        logger.info("Advanced AI Features initialized")
    
    async def generate_code_with_context(self, 
                                       generation_request: Dict[str, Any],
                                       use_historical_context: bool = True,
                                       use_multi_model: bool = False,
                                       learning_enabled: bool = True) -> Tuple[GeneratedCode, Dict[str, Any]]:
        """Generate code with advanced AI features"""
        interaction_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Build context
            context = []
            if use_historical_context:
                context = await self._build_generation_context(generation_request)
            
            # Create enhanced prompt
            prompt = await self._create_enhanced_prompt(generation_request, context)
            
            # Generate code
            if use_multi_model and len(self.multi_model_coordinator.model_roles) > 1:
                multi_response = await self.multi_model_coordinator.generate_with_coordination(
                    prompt, strategy='expert_review', context=context
                )
                generated_code = GeneratedCode(
                    code=multi_response.primary_response.content,
                    language="python",
                    framework="pycomm3",
                    quality_level="PRODUCTION",
                    metadata={
                        'generation_type': generation_request.get('type', 'FULL_INTERFACE'),
                        'consensus_score': multi_response.consensus_score,
                        'confidence_score': multi_response.confidence_score,
                        'multi_model_used': True
                    }
                )
                generation_metadata = {
                    'multi_model_response': multi_response,
                    'context_used': len(context),
                    'generation_method': 'multi_model'
                }
            else:
                # Single model generation with context
                response = await self.ai_interface.generate_response(prompt)
                generated_code = GeneratedCode(
                    code=response.content,
                    language="python",
                    framework="pycomm3",
                    quality_level="PRODUCTION",
                    metadata={
                        'generation_type': generation_request.get('type', 'FULL_INTERFACE'),
                        'confidence_score': response.confidence,
                        'multi_model_used': False
                    }
                )
                generation_metadata = {
                    'response': response,
                    'context_used': len(context),
                    'generation_method': 'single_model'
                }
            
            # Validate generated code
            validation_result = await self.enhanced_validator.validate_plc_code(
                generated_code.code, l5x_file_path=self.l5x_file_path
            )
            
            # Record learning if enabled
            if learning_enabled:
                await self._record_generation_learning(
                    interaction_id, prompt, generated_code, validation_result, context
                )
            
            # Update context with generation result
            await self._update_context_with_generation(
                generated_code, validation_result, generation_request
            )
            
            return generated_code, {
                **generation_metadata,
                'validation_result': validation_result,
                'interaction_id': interaction_id
            }
            
        except Exception as e:
            logger.error(f"Error in advanced code generation: {e}")
            raise
    
    async def iterative_improvement_with_learning(self,
                                                generated_code: GeneratedCode,
                                                strategy: CorrectionStrategy = CorrectionStrategy.PRIORITY,
                                                max_iterations: int = 5) -> Tuple[ValidationLoopResult, Dict[str, Any]]:
        """Perform iterative improvement with learning integration"""
        interaction_id = f"improve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get learning insights
        insights = await self.learning_engine.get_learning_insights(
            ContextType.HISTORICAL_CORRECTIONS
        )
        
        # Configure validation loop with learning insights
        self.validation_loop.max_iterations = max_iterations
        
        # Add historical context to validation loop
        historical_context = await self.context_manager.get_relevant_context([
            ContextType.HISTORICAL_CORRECTIONS,
            ContextType.VALIDATION_FEEDBACK
        ])
        
        # Run validation loop
        loop_result = await self.validation_loop.run_validation_loop(
            generated_code, strategy
        )
        
        # Record learning from improvement process
        await self._record_improvement_learning(
            interaction_id, generated_code, loop_result, insights
        )
        
        return loop_result, {
            'learning_insights': insights,
            'historical_context_used': len(historical_context),
            'interaction_id': interaction_id
        }
    
    async def analyze_user_patterns(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze user patterns and preferences for personalization"""
        preferences = await self.learning_engine.get_user_preferences(user_id)
        
        # Get context patterns
        context_patterns = {}
        for context_type in ContextType:
            patterns = await self.context_manager.get_historical_patterns(
                context_type.value
            )
            if patterns:
                context_patterns[context_type.value] = len(patterns)
        
        # Get success metrics
        success_metrics = self.learning_engine.success_metrics
        
        return {
            'user_preferences': preferences,
            'context_patterns': context_patterns,
            'success_metrics': success_metrics,
            'total_interactions': len(self.learning_engine.learning_records),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def optimize_for_user(self, user_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize AI features based on user analysis"""
        optimizations = {
            'prompt_templates': {},
            'model_configuration': {},
            'context_preferences': {},
            'quality_thresholds': {}
        }
        
        preferences = user_analysis.get('user_preferences', {})
        
        # Optimize prompt templates
        for context_type in ContextType:
            try:
                template = "Generate high-quality PLC interface code with the following requirements:"
                optimized_template = await self.learning_engine.optimize_prompt_template(
                    template, context_type
                )
                optimizations['prompt_templates'][context_type.value] = optimized_template
            except Exception as e:
                logger.warning(f"Failed to optimize template for {context_type.value}: {e}")
        
        # Configure model preferences
        preferred_quality = preferences.get('preferred_quality_level', 'production')
        optimizations['quality_thresholds'] = {
            'minimum_score': 0.8 if preferred_quality == 'safety_critical' else 0.7,
            'correction_threshold': 0.6,
            'confidence_threshold': 0.7
        }
        
        return optimizations
    
    async def _build_generation_context(self, generation_request: Dict[str, Any]) -> List[ContextualInformation]:
        """Build context for code generation"""
        context = []
        
        # Add L5X source context
        try:
            l5x_context = ContextualInformation(
                context_type=ContextType.L5X_SOURCE,
                content={
                    'file_path': self.l5x_file_path,
                    'generation_type': generation_request.get('type', 'FULL_INTERFACE'),
                    'requirements': generation_request.get('requirements', {})
                },
                relevance_score=0.9,
                timestamp=datetime.now(),
                source='l5x_parser'
            )
            context.append(l5x_context)
        except Exception as e:
            logger.warning(f"Failed to add L5X context: {e}")
        
        # Add historical corrections context
        historical_context = await self.context_manager.get_relevant_context([
            ContextType.HISTORICAL_CORRECTIONS,
            ContextType.SIMILAR_PROJECTS
        ], max_items=5)
        context.extend(historical_context)
        
        return context
    
    async def _create_enhanced_prompt(self, generation_request: Dict[str, Any],
                                    context: List[ContextualInformation]) -> str:
        """Create enhanced prompt with context"""
        base_prompt = f"""
Generate a high-quality Python PLC interface using pycomm3 for the following requirements:

Type: {generation_request.get('type', 'FULL_INTERFACE')}
Quality Level: {generation_request.get('quality_level', 'PRODUCTION')}
Framework: {generation_request.get('framework', 'PYCOMM3')}

Requirements:
{json.dumps(generation_request.get('requirements', {}), indent=2)}
"""
        
        # Add context information
        if context:
            context_str = "\n\nRelevant Context:\n"
            for ctx in context:
                if ctx.is_relevant():
                    context_str += f"- {ctx.context_type.value}: {json.dumps(ctx.content, indent=2)}\n"
            base_prompt += context_str
        
        # Add learning-based optimizations
        try:
            insights = await self.learning_engine.get_learning_insights(
                ContextType.HISTORICAL_CORRECTIONS
            )
            if insights['success_patterns']:
                base_prompt += "\n\nSuccessful patterns to consider:\n"
                for pattern in insights['success_patterns'][:3]:
                    base_prompt += f"- Pattern with quality {pattern['quality_score']:.2f}\n"
        except Exception as e:
            logger.warning(f"Failed to add learning insights: {e}")
        
        return base_prompt
    
    async def _record_generation_learning(self, interaction_id: str, prompt: str,
                                        generated_code: GeneratedCode,
                                        validation_result: PLCValidationResult,
                                        context: List[ContextualInformation]) -> None:
        """Record learning from code generation"""
        try:
            quality_score = validation_result.overall_score / 10.0  # Convert to 0-1 scale
            correction_needed = validation_result.overall_score < 7.0
            
            learning_record = LearningRecord(
                interaction_id=interaction_id,
                timestamp=datetime.now(),
                context=context,
                prompt_used=prompt,
                response_quality=quality_score,
                correction_needed=correction_needed,
                final_result=generated_code.code,
                success_metrics={
                    'validation_score': validation_result.overall_score,
                    'issues_count': len(validation_result.issues),
                    'critical_issues': len([i for i in validation_result.issues if i.severity.value == 'CRITICAL'])
                }
            )
            
            await self.learning_engine.record_interaction(learning_record)
            
        except Exception as e:
            logger.error(f"Failed to record generation learning: {e}")
    
    async def _record_improvement_learning(self, interaction_id: str,
                                         original_code: GeneratedCode,
                                         loop_result: ValidationLoopResult,
                                         insights: Dict[str, Any]) -> None:
        """Record learning from improvement process"""
        try:
            quality_improvement = loop_result.score_improvement
            correction_success = loop_result.correction_success_rate
            
            learning_record = LearningRecord(
                interaction_id=interaction_id,
                timestamp=datetime.now(),
                context=[],  # Context would be built from loop_result
                prompt_used="Iterative improvement process",
                response_quality=correction_success,
                correction_needed=quality_improvement < 1.0,
                final_result=loop_result.final_code,
                success_metrics={
                    'initial_score': loop_result.initial_score,
                    'final_score': loop_result.final_score,
                    'score_improvement': quality_improvement,
                    'iterations': loop_result.total_iterations,
                    'corrections_successful': loop_result.successful_corrections
                }
            )
            
            await self.learning_engine.record_interaction(learning_record)
            
        except Exception as e:
            logger.error(f"Failed to record improvement learning: {e}")
    
    async def _update_context_with_generation(self, generated_code: GeneratedCode,
                                            validation_result: PLCValidationResult,
                                            generation_request: Dict[str, Any]) -> None:
        """Update context with generation results"""
        try:
            # Add generation result as context
            generation_context = ContextualInformation(
                context_type=ContextType.VALIDATION_FEEDBACK,
                content={
                    'generation_type': generation_request.get('type'),
                    'quality_score': validation_result.overall_score,
                    'issues_count': len(validation_result.issues),
                    'successful_generation': validation_result.overall_score >= 7.0
                },
                relevance_score=0.8,
                timestamp=datetime.now(),
                source='code_generation',
                metadata={
                    'framework': generated_code.metadata.get('framework', 'pycomm3'),
                    'quality_level': generated_code.metadata.get('quality_level', 'production')
                }
            )
            
            await self.context_manager.add_context(generation_context)
            
        except Exception as e:
            logger.error(f"Failed to update context: {e}")


# Convenience functions for easy usage
async def generate_code_with_advanced_ai(l5x_file_path: str,
                                       ai_interface: AIInterfaceManager,
                                       generation_request: Dict[str, Any],
                                       **kwargs) -> Tuple[GeneratedCode, Dict[str, Any]]:
    """Convenience function for advanced AI code generation"""
    advanced_ai = AdvancedAIFeatures(ai_interface, l5x_file_path)
    return await advanced_ai.generate_code_with_context(generation_request, **kwargs)


async def improve_code_with_learning(l5x_file_path: str,
                                   ai_interface: AIInterfaceManager,
                                   generated_code: GeneratedCode,
                                   **kwargs) -> Tuple[ValidationLoopResult, Dict[str, Any]]:
    """Convenience function for learning-enhanced code improvement"""
    advanced_ai = AdvancedAIFeatures(ai_interface, l5x_file_path)
    return await advanced_ai.iterative_improvement_with_learning(generated_code, **kwargs)


async def analyze_user_ai_patterns(l5x_file_path: str,
                                 ai_interface: AIInterfaceManager,
                                 user_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for user pattern analysis"""
    advanced_ai = AdvancedAIFeatures(ai_interface, l5x_file_path)
    return await advanced_ai.analyze_user_patterns(user_id)
