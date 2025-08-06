"""
Step 18: AI Integration System
Connects the AI interface and prompt engineering with analysis systems from Steps 9-16.
Provides comprehensive PLC analysis to AI-powered code generation pipeline.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

# Import AI components
from .ai_interface import AIInterfaceManager, AIMessage
from .plc_ai_service import PLCAIService, CodeGenerationRequest, CodeGenerationResult
from .prompt_engineering import (
    PromptEngineering, PromptContext, PromptType, PLCDomain,
    create_prompt_engineer, quick_prompt
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisSystemsConfig:
    """Configuration for analysis systems integration."""
    l5x_file_path: str
    enable_routine_analysis: bool = True
    enable_instruction_analysis: bool = True
    enable_graph_analysis: bool = True
    enable_program_analysis: bool = True
    enable_timer_counter_analysis: bool = True
    enable_udt_analysis: bool = True
    enable_array_analysis: bool = True
    enable_logic_flow_analysis: bool = True
    output_directory: str = "integration_output"
    generate_reports: bool = True
    cache_results: bool = True


class AnalysisSystemsIntegrator:
    """Integrates all analysis systems from Steps 9-16."""
    
    def __init__(self, config: AnalysisSystemsConfig):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis results cache
        self.analysis_cache: Dict[str, Any] = {}
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis using all available systems."""
        start_time = time.time()
        
        analysis_results = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'l5x_file': self.config.l5x_file_path,
                'analysis_duration': 0,
                'systems_used': []
            },
            'controller': {},
            'tags': [],
            'routines': [],
            'instructions': [],
            'graph_analysis': {},
            'program_analysis': {},
            'timer_counter_analysis': {},
            'udt_analysis': {},
            'array_analysis': {},
            'logic_flow_analysis': {},
            'safety_analysis': {},
            'optimization_analysis': {},
            'integration_summary': {}
        }
        
        try:
            # Step 1: Basic L5X parsing and tag extraction (Steps 2-6)
            self.logger.info("Running basic L5X analysis...")
            basic_analysis = self._run_basic_analysis()
            analysis_results.update(basic_analysis)
            analysis_results['metadata']['systems_used'].append('basic_l5x_parser')
            
            # Step 2: Routine analysis (Step 9)
            if self.config.enable_routine_analysis:
                self.logger.info("Running routine analysis...")
                routine_analysis = self._run_routine_analysis()
                analysis_results['routines'] = routine_analysis.get('routines', [])
                analysis_results['metadata']['systems_used'].append('routine_analyzer')
            
            # Step 3: Instruction analysis (Step 10)
            if self.config.enable_instruction_analysis:
                self.logger.info("Running instruction analysis...")
                instruction_analysis = self._run_instruction_analysis()
                analysis_results['instructions'] = instruction_analysis.get('instructions', [])
                analysis_results['metadata']['systems_used'].append('instruction_analyzer')
            
            # Step 4: Graph analysis (Step 11)
            if self.config.enable_graph_analysis:
                self.logger.info("Running graph analysis...")
                graph_analysis = self._run_graph_analysis()
                analysis_results['graph_analysis'] = graph_analysis
                analysis_results['metadata']['systems_used'].append('graph_analyzer')
            
            # Step 5: Program analysis (Step 12)
            if self.config.enable_program_analysis:
                self.logger.info("Running program analysis...")
                program_analysis = self._run_program_analysis()
                analysis_results['program_analysis'] = program_analysis
                analysis_results['metadata']['systems_used'].append('program_analyzer')
            
            # Step 6: Timer and counter analysis (Step 13)
            if self.config.enable_timer_counter_analysis:
                self.logger.info("Running timer and counter analysis...")
                tc_analysis = self._run_timer_counter_analysis()
                analysis_results['timer_counter_analysis'] = tc_analysis
                analysis_results['metadata']['systems_used'].append('timer_counter_analyzer')
            
            # Step 7: UDT analysis (Step 14)
            if self.config.enable_udt_analysis:
                self.logger.info("Running UDT analysis...")
                udt_analysis = self._run_udt_analysis()
                analysis_results['udt_analysis'] = udt_analysis
                analysis_results['metadata']['systems_used'].append('udt_analyzer')
            
            # Step 8: Array analysis (Step 15)
            if self.config.enable_array_analysis:
                self.logger.info("Running array analysis...")
                array_analysis = self._run_array_analysis()
                analysis_results['array_analysis'] = array_analysis
                analysis_results['metadata']['systems_used'].append('array_analyzer')
            
            # Step 9: Logic flow analysis (Step 16)
            if self.config.enable_logic_flow_analysis:
                self.logger.info("Running logic flow analysis...")
                flow_analysis = self._run_logic_flow_analysis()
                analysis_results['logic_flow_analysis'] = flow_analysis
                analysis_results['metadata']['systems_used'].append('logic_flow_analyzer')
            
            # Step 10: Generate integrated insights
            self.logger.info("Generating integrated insights...")
            integrated_insights = self._generate_integrated_insights(analysis_results)
            analysis_results['safety_analysis'] = integrated_insights['safety_analysis']
            analysis_results['optimization_analysis'] = integrated_insights['optimization_analysis']
            analysis_results['integration_summary'] = integrated_insights['integration_summary']
            
            # Calculate total duration
            analysis_results['metadata']['analysis_duration'] = time.time() - start_time
            
            # Cache results if enabled
            if self.config.cache_results:
                self._cache_results(analysis_results)
            
            # Generate reports if enabled
            if self.config.generate_reports:
                self._generate_integration_reports(analysis_results)
            
            self.logger.info(f"Comprehensive analysis completed in {analysis_results['metadata']['analysis_duration']:.2f} seconds")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            analysis_results['error'] = str(e)
            analysis_results['metadata']['analysis_duration'] = time.time() - start_time
            return analysis_results
    
    def _run_basic_analysis(self) -> Dict[str, Any]:
        """Run basic L5X parsing and tag extraction (mock implementation)."""
        # This would integrate with actual Steps 2-6 implementations
        return {
            'controller': {
                'name': 'Assembly_Controls_Robot',
                'type': 'CompactLogix',
                'revision': '33.011',
                'last_edited': '2024-12-20'
            },
            'tags': [
                {'name': 'E_Stop', 'type': 'BOOL', 'scope': 'controller', 'description': 'Emergency stop status', 'critical': True},
                {'name': 'Motor1_Run', 'type': 'BOOL', 'scope': 'controller', 'description': 'Motor 1 run command'},
                {'name': 'Motor1_Speed', 'type': 'REAL', 'scope': 'controller', 'description': 'Motor 1 speed setpoint'},
                {'name': 'Conveyor_Position', 'type': 'DINT', 'scope': 'controller', 'description': 'Conveyor position feedback'},
                {'name': 'Safety_Gate', 'type': 'BOOL', 'scope': 'controller', 'description': 'Safety gate position', 'critical': True},
                {'name': 'Alarm_Active', 'type': 'BOOL', 'scope': 'controller', 'description': 'System alarm status'}
            ],
            'io_modules': [
                {'name': 'Local:1', 'type': 'CompactLogix_Controller', 'slot': 0},
                {'name': 'Local:2', 'type': 'Digital_Input_16', 'slot': 1},
                {'name': 'Local:3', 'type': 'Digital_Output_16', 'slot': 2}
            ]
        }
    
    def _run_routine_analysis(self) -> Dict[str, Any]:
        """Run routine analysis (mock implementation)."""
        return {
            'routines': [
                {
                    'name': 'MainRoutine',
                    'type': 'Ladder',
                    'rungs': 25,
                    'instructions': 142,
                    'complexity_score': 6.5,
                    'tags_used': ['E_Stop', 'Motor1_Run', 'Safety_Gate']
                },
                {
                    'name': 'MotorControl',
                    'type': 'Ladder', 
                    'rungs': 15,
                    'instructions': 87,
                    'complexity_score': 4.2,
                    'tags_used': ['Motor1_Run', 'Motor1_Speed']
                },
                {
                    'name': 'SafetyLogic',
                    'type': 'Ladder',
                    'rungs': 10,
                    'instructions': 45,
                    'complexity_score': 7.8,
                    'tags_used': ['E_Stop', 'Safety_Gate', 'Alarm_Active']
                }
            ]
        }
    
    def _run_instruction_analysis(self) -> Dict[str, Any]:
        """Run instruction analysis (mock implementation)."""
        return {
            'instructions': [
                {'type': 'XIC', 'count': 45, 'tags': ['E_Stop', 'Safety_Gate']},
                {'type': 'XIO', 'count': 23, 'tags': ['Alarm_Active']},
                {'type': 'OTE', 'count': 18, 'tags': ['Motor1_Run']},
                {'type': 'TON', 'count': 8, 'tags': ['Timer_Motor_Start']},
                {'type': 'MOV', 'count': 12, 'tags': ['Motor1_Speed']}
            ],
            'instruction_complexity': {
                'total_instructions': 274,
                'complex_instructions': 45,
                'simple_instructions': 229,
                'average_complexity': 3.2
            }
        }
    
    def _run_graph_analysis(self) -> Dict[str, Any]:
        """Run graph analysis (mock implementation)."""
        return {
            'control_flow': {
                'nodes': 87,
                'edges': 142,
                'cycles': 3,
                'critical_paths': 5
            },
            'data_dependencies': {
                'strong_dependencies': 23,
                'weak_dependencies': 67,
                'isolated_components': 2
            },
            'graph_metrics': {
                'density': 0.23,
                'clustering_coefficient': 0.67,
                'average_path_length': 3.4
            }
        }
    
    def _run_program_analysis(self) -> Dict[str, Any]:
        """Run program analysis (mock implementation)."""
        return {
            'program_structure': {
                'main_programs': 1,
                'subroutines': 8,
                'call_depth': 3,
                'recursive_calls': 0
            },
            'execution_flow': {
                'sequential_blocks': 15,
                'parallel_blocks': 3,
                'conditional_blocks': 12
            },
            'call_analysis': {
                'jsr_instructions': 8,
                'sbr_instructions': 8,
                'ret_instructions': 8,
                'parameter_passing': 12
            }
        }
    
    def _run_timer_counter_analysis(self) -> Dict[str, Any]:
        """Run timer and counter analysis (mock implementation)."""
        return {
            'timers': {
                'total_timers': 12,
                'ton_timers': 8,
                'tof_timers': 3,
                'rto_timers': 1,
                'average_preset': 5000,
                'timing_chains': 3
            },
            'counters': {
                'total_counters': 4,
                'ctu_counters': 3,
                'ctd_counters': 1,
                'counting_chains': 2
            },
            'critical_timing': {
                'safety_timers': 2,
                'process_timers': 6,
                'diagnostic_timers': 4
            }
        }
    
    def _run_udt_analysis(self) -> Dict[str, Any]:
        """Run UDT analysis (mock implementation)."""
        return {
            'udt_definitions': [
                {
                    'name': 'MotorControl_UDT',
                    'members': 8,
                    'instances': 3,
                    'complexity_score': 4.5
                },
                {
                    'name': 'SafetySystem_UDT',
                    'members': 12,
                    'instances': 1,
                    'complexity_score': 8.2
                }
            ],
            'udt_usage': {
                'total_instances': 4,
                'member_accesses': 67,
                'unused_members': 3
            }
        }
    
    def _run_array_analysis(self) -> Dict[str, Any]:
        """Run array analysis (mock implementation)."""
        return {
            'arrays': [
                {
                    'name': 'MBIT',
                    'dimensions': 1,
                    'size': 64,
                    'type': 'BOOL',
                    'accesses': 23,
                    'usage_pattern': 'static'
                }
            ],
            'array_metrics': {
                'total_arrays': 1,
                'total_elements': 64,
                'access_patterns': {
                    'static': 23,
                    'dynamic': 0
                }
            }
        }
    
    def _run_logic_flow_analysis(self) -> Dict[str, Any]:
        """Run logic flow analysis (mock implementation)."""
        return {
            'logic_patterns': [
                {'pattern': 'START_STOP', 'count': 3, 'critical': True},
                {'pattern': 'SAFETY_CHAIN', 'count': 2, 'critical': True},
                {'pattern': 'TIMER_CHAIN', 'count': 4, 'critical': False},
                {'pattern': 'ALARM_LOGIC', 'count': 2, 'critical': True}
            ],
            'execution_paths': {
                'normal_paths': 8,
                'exception_paths': 4,
                'critical_paths': 3
            },
            'flow_complexity': {
                'cyclomatic_complexity': 8.5,
                'nesting_depth': 4,
                'branch_factor': 2.3
            }
        }
    
    def _generate_integrated_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated insights from all analysis results."""
        
        # Safety analysis integration
        safety_concerns = []
        critical_tags = [tag for tag in analysis_results.get('tags', []) if tag.get('critical', False)]
        
        if critical_tags:
            safety_concerns.append(f"Found {len(critical_tags)} safety-critical tags requiring special handling")
        
        if analysis_results.get('logic_flow_analysis', {}).get('logic_patterns'):
            safety_patterns = [p for p in analysis_results['logic_flow_analysis']['logic_patterns'] 
                             if p.get('critical', False)]
            if safety_patterns:
                safety_concerns.append(f"Found {len(safety_patterns)} critical logic patterns")
        
        # Optimization analysis integration
        optimization_opportunities = []
        
        # Check for instruction complexity
        if analysis_results.get('instructions'):
            complex_ratio = sum(1 for i in analysis_results['instructions'] if i.get('count', 0) > 20) / len(analysis_results['instructions'])
            if complex_ratio > 0.3:
                optimization_opportunities.append("High instruction complexity detected - consider refactoring")
        
        # Check for timer/counter optimization
        if analysis_results.get('timer_counter_analysis', {}).get('timers', {}).get('timing_chains', 0) > 5:
            optimization_opportunities.append("Multiple timing chains detected - consolidation opportunity")
        
        # Check for UDT optimization
        udt_analysis = analysis_results.get('udt_analysis', {})
        if udt_analysis.get('udt_usage', {}).get('unused_members', 0) > 0:
            optimization_opportunities.append("Unused UDT members detected - structure optimization possible")
        
        # Integration summary
        total_components = (
            len(analysis_results.get('tags', [])) +
            len(analysis_results.get('routines', [])) +
            len(analysis_results.get('instructions', []))
        )
        
        complexity_metrics = {
            'routine_complexity': analysis_results.get('routines', [{}])[0].get('complexity_score', 0) if analysis_results.get('routines') else 0,
            'instruction_complexity': analysis_results.get('instruction_complexity', {}).get('average_complexity', 0),
            'flow_complexity': analysis_results.get('logic_flow_analysis', {}).get('flow_complexity', {}).get('cyclomatic_complexity', 0)
        }
        
        overall_complexity = sum(complexity_metrics.values()) / len([v for v in complexity_metrics.values() if v > 0])
        
        return {
            'safety_analysis': {
                'concerns': safety_concerns,
                'critical_tags_count': len(critical_tags),
                'safety_patterns_count': len([p for p in analysis_results.get('logic_flow_analysis', {}).get('logic_patterns', []) 
                                             if p.get('critical', False)]),
                'risk_level': 'HIGH' if len(safety_concerns) > 2 else 'MEDIUM' if len(safety_concerns) > 0 else 'LOW'
            },
            'optimization_analysis': {
                'opportunities': optimization_opportunities,
                'complexity_score': overall_complexity,
                'optimization_potential': 'HIGH' if len(optimization_opportunities) > 3 else 'MEDIUM' if len(optimization_opportunities) > 1 else 'LOW'
            },
            'integration_summary': {
                'total_components_analyzed': total_components,
                'analysis_systems_used': len(analysis_results['metadata']['systems_used']),
                'overall_complexity': overall_complexity,
                'analysis_completeness': len(analysis_results['metadata']['systems_used']) / 8.0,  # 8 total systems
                'recommendations': self._generate_recommendations(analysis_results, safety_concerns, optimization_opportunities)
            }
        }
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any], 
                                safety_concerns: List[str], 
                                optimization_opportunities: List[str]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Safety recommendations
        if safety_concerns:
            recommendations.append({
                'category': 'Safety',
                'priority': 'Critical',
                'recommendation': 'Implement comprehensive safety validation and monitoring',
                'details': safety_concerns
            })
        
        # Performance recommendations
        if optimization_opportunities:
            recommendations.append({
                'category': 'Performance',
                'priority': 'High',
                'recommendation': 'Optimize system performance and resource usage',
                'details': optimization_opportunities
            })
        
        # Code quality recommendations
        avg_complexity = analysis_results.get('integration_summary', {}).get('overall_complexity', 0)
        if avg_complexity > 6:
            recommendations.append({
                'category': 'Code Quality',
                'priority': 'Medium',
                'recommendation': 'Reduce system complexity through refactoring',
                'details': [f'Overall complexity score: {avg_complexity:.1f}/10']
            })
        
        return recommendations
    
    def _cache_results(self, analysis_results: Dict[str, Any]):
        """Cache analysis results."""
        cache_file = self.output_dir / f"analysis_cache_{int(time.time())}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"Analysis results cached: {cache_file}")
    
    def _generate_integration_reports(self, analysis_results: Dict[str, Any]):
        """Generate comprehensive integration reports."""
        # JSON report
        json_report_path = self.output_dir / 'comprehensive_analysis_report.json'
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        # HTML report
        html_report_path = self.output_dir / 'comprehensive_analysis_report.html'
        html_content = self._generate_html_report(analysis_results)
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Integration reports generated: {json_report_path}, {html_report_path}")
    
    def _generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate HTML report for analysis results."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive PLC Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background: #e7f3ff; padding: 10px; border-radius: 3px; margin: 5px 0; }}
        .critical {{ background: #ffe7e7; }}
        .warning {{ background: #fff7e7; }}
        .success {{ background: #e7ffe7; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Comprehensive PLC Analysis Report</h1>
        <p><strong>Controller:</strong> {analysis_results.get('controller', {}).get('name', 'Unknown')}</p>
        <p><strong>Generated:</strong> {analysis_results.get('metadata', {}).get('timestamp', 'Unknown')}</p>
        <p><strong>Analysis Duration:</strong> {analysis_results.get('metadata', {}).get('analysis_duration', 0):.2f} seconds</p>
    </div>
    
    <div class="section">
        <h2>System Overview</h2>
        <div class="metric">Tags: {len(analysis_results.get('tags', []))}</div>
        <div class="metric">Routines: {len(analysis_results.get('routines', []))}</div>
        <div class="metric">Instructions: {len(analysis_results.get('instructions', []))}</div>
        <div class="metric">I/O Modules: {len(analysis_results.get('io_modules', []))}</div>
    </div>
    
    <div class="section">
        <h2>Safety Analysis</h2>
        <div class="metric critical">Risk Level: {analysis_results.get('safety_analysis', {}).get('risk_level', 'Unknown')}</div>
        <div class="metric">Critical Tags: {analysis_results.get('safety_analysis', {}).get('critical_tags_count', 0)}</div>
        <div class="metric">Safety Patterns: {analysis_results.get('safety_analysis', {}).get('safety_patterns_count', 0)}</div>
        <h3>Concerns:</h3>
        <ul>
            {''.join([f'<li>{concern}</li>' for concern in analysis_results.get('safety_analysis', {}).get('concerns', [])])}
        </ul>
    </div>
    
    <div class="section">
        <h2>Optimization Analysis</h2>
        <div class="metric">Optimization Potential: {analysis_results.get('optimization_analysis', {}).get('optimization_potential', 'Unknown')}</div>
        <div class="metric">Complexity Score: {analysis_results.get('optimization_analysis', {}).get('complexity_score', 0):.1f}/10</div>
        <h3>Opportunities:</h3>
        <ul>
            {''.join([f'<li>{opp}</li>' for opp in analysis_results.get('optimization_analysis', {}).get('opportunities', [])])}
        </ul>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <table>
            <tr><th>Category</th><th>Priority</th><th>Recommendation</th></tr>
            {''.join([f'<tr><td>{rec["category"]}</td><td>{rec["priority"]}</td><td>{rec["recommendation"]}</td></tr>' 
                     for rec in analysis_results.get('integration_summary', {}).get('recommendations', [])])}
        </table>
    </div>
    
    <div class="section">
        <h2>Analysis Systems Used</h2>
        <ul>
            {''.join([f'<li>{system}</li>' for system in analysis_results.get('metadata', {}).get('systems_used', [])])}
        </ul>
    </div>
</body>
</html>
"""


class AIIntegratedCodeGenerator:
    """AI-powered code generator integrated with comprehensive PLC analysis."""
    
    def __init__(self, ai_config_path: str = "gemini_config.json"):
        self.ai_manager = AIInterfaceManager(config_path=ai_config_path)
        self.plc_ai_service = PLCAIService(ai_manager=self.ai_manager)
        self.prompt_engineer = create_prompt_engineer()
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_comprehensive_code(self, analysis_results: Dict[str, Any],
                                        requirements: List[str],
                                        constraints: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive Python code from analysis results."""
        start_time = time.time()
        
        try:
            # Create PLC context from analysis
            context = self.prompt_engineer.create_context_from_analysis(analysis_results)
            context.user_requirements = requirements
            context.constraints = constraints or []
            
            # Set PLC context in AI service
            self.plc_ai_service.set_plc_context(
                controller_name=analysis_results.get('controller', {}).get('name', 'Unknown'),
                tags=analysis_results.get('tags', []),
                routines=analysis_results.get('routines', []),
                logic_patterns=analysis_results.get('logic_flow_analysis', {}).get('logic_patterns', []),
                safety_concerns=analysis_results.get('safety_analysis', {}).get('concerns', []),
                optimization_opportunities=analysis_results.get('optimization_analysis', {}).get('opportunities', [])
            )
            
            # Get template recommendations
            template_recommendations = self.prompt_engineer.get_template_recommendations(analysis_results)
            
            generation_results = {
                'metadata': {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'generation_duration': 0,
                    'templates_used': [],
                    'ai_model': self.ai_manager.active_provider
                },
                'template_recommendations': template_recommendations,
                'generated_code': {},
                'analysis_summary': {},
                'validation_results': {}
            }
            
            # Generate code using recommended templates
            for recommendation in template_recommendations[:3]:  # Limit to top 3
                template_name = recommendation['template']
                
                try:
                    self.logger.info(f"Generating code using template: {template_name}")
                    
                    # Build prompt
                    prompt_data = self.prompt_engineer.generate_contextual_prompt(
                        template_name=template_name,
                        analysis_results=analysis_results,
                        user_requirements=requirements,
                        constraints=constraints,
                        model_name="gemini-1.5-flash"
                    )
                    
                    # Generate code using AI
                    code_request = CodeGenerationRequest(
                        task_description=f"Generate code using {template_name} template",
                        target_language="python",
                        include_comments=True,
                        include_error_handling=True,
                        requirements=requirements,
                        constraints=constraints or []
                    )
                    
                    # Use the AI service to generate code
                    messages = [
                        AIMessage(role="system", content=prompt_data['system_prompt']),
                        AIMessage(role="user", content=prompt_data['user_prompt'])
                    ]
                    
                    ai_response = await self.ai_manager.generate_response(messages)
                    
                    generation_results['generated_code'][template_name] = {
                        'code': ai_response.content,
                        'template_info': prompt_data['template_info'],
                        'token_usage': {
                            'total_tokens': ai_response.tokens_used,
                            'prompt_tokens': ai_response.tokens_prompt,
                            'completion_tokens': ai_response.tokens_completion
                        },
                        'generation_time': ai_response.response_time,
                        'model_used': ai_response.model
                    }
                    
                    generation_results['metadata']['templates_used'].append(template_name)
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate code for template {template_name}: {e}")
                    generation_results['generated_code'][template_name] = {
                        'error': str(e),
                        'code': None
                    }
            
            # Generate analysis summary
            generation_results['analysis_summary'] = {
                'total_components': len(analysis_results.get('tags', [])) + len(analysis_results.get('routines', [])),
                'safety_concerns': len(analysis_results.get('safety_analysis', {}).get('concerns', [])),
                'optimization_opportunities': len(analysis_results.get('optimization_analysis', {}).get('opportunities', [])),
                'complexity_score': analysis_results.get('optimization_analysis', {}).get('complexity_score', 0),
                'recommendations_count': len(template_recommendations)
            }
            
            # Basic validation
            generation_results['validation_results'] = self._validate_generated_code(generation_results['generated_code'])
            
            generation_results['metadata']['generation_duration'] = time.time() - start_time
            
            return generation_results
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return {
                'error': str(e),
                'metadata': {
                    'generation_duration': time.time() - start_time,
                    'templates_used': [],
                    'ai_model': self.ai_manager.active_provider
                }
            }
    
    def _validate_generated_code(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation of generated code."""
        validation_results = {}
        
        for template_name, code_data in generated_code.items():
            if code_data.get('error'):
                validation_results[template_name] = {
                    'valid': False,
                    'error': code_data['error'],
                    'score': 0
                }
                continue
            
            code = code_data.get('code', '')
            
            # Basic syntax validation
            try:
                compile(code, '<string>', 'exec')
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False
            
            # Check for required imports
            has_pycomm3 = 'pycomm3' in code or 'LogixDriver' in code
            has_logging = 'logging' in code
            has_error_handling = 'try:' in code and 'except' in code
            
            # Calculate score
            score = 0
            if syntax_valid:
                score += 40
            if has_pycomm3:
                score += 30
            if has_logging:
                score += 15
            if has_error_handling:
                score += 15
            
            validation_results[template_name] = {
                'valid': syntax_valid,
                'score': score,
                'checks': {
                    'syntax_valid': syntax_valid,
                    'has_pycomm3': has_pycomm3,
                    'has_logging': has_logging,
                    'has_error_handling': has_error_handling
                },
                'code_length': len(code),
                'line_count': len(code.split('\n'))
            }
        
        return validation_results


# Main integration function
async def run_integrated_analysis_and_generation(l5x_file_path: str,
                                               requirements: List[str],
                                               constraints: List[str] = None,
                                               output_dir: str = "step18_output") -> Dict[str, Any]:
    """Run complete integrated analysis and AI-powered code generation."""
    
    # Configure analysis systems
    config = AnalysisSystemsConfig(
        l5x_file_path=l5x_file_path,
        output_directory=output_dir,
        generate_reports=True,
        cache_results=True
    )
    
    # Run comprehensive analysis
    integrator = AnalysisSystemsIntegrator(config)
    analysis_results = integrator.run_comprehensive_analysis()
    
    # Generate AI-powered code
    code_generator = AIIntegratedCodeGenerator()
    generation_results = await code_generator.generate_comprehensive_code(
        analysis_results=analysis_results,
        requirements=requirements,
        constraints=constraints
    )
    
    # Combine results
    integrated_results = {
        'analysis_results': analysis_results,
        'generation_results': generation_results,
        'integration_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'l5x_file': l5x_file_path,
            'requirements': requirements,
            'constraints': constraints or [],
            'output_directory': output_dir
        }
    }
    
    # Save integrated results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / 'integrated_analysis_and_generation.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(integrated_results, f, indent=2, ensure_ascii=False, default=str)
    
    return integrated_results


if __name__ == "__main__":
    print("🚀 Step 18: AI Integration System")
    print("=" * 60)
    
    # Example usage
    async def main():
        requirements = [
            "Generate a complete PLC interface for reading motor status",
            "Include comprehensive error handling and logging",
            "Implement safety monitoring for critical tags",
            "Provide real-time data acquisition capabilities"
        ]
        
        constraints = [
            "Use only pycomm3 library for PLC communication",
            "Follow PEP8 coding standards",
            "Maximum 500 lines per generated file",
            "Include unit tests for all methods"
        ]
        
        results = await run_integrated_analysis_and_generation(
            l5x_file_path="Assembly_Controls_Robot.L5X",
            requirements=requirements,
            constraints=constraints
        )
        
        print(f"✅ Integration completed successfully!")
        print(f"📊 Analysis systems used: {len(results['analysis_results']['metadata']['systems_used'])}")
        print(f"🤖 AI templates used: {len(results['generation_results']['metadata']['templates_used'])}")
        print(f"📁 Results saved to: step18_output/")
    
    # Run the example
    asyncio.run(main())
