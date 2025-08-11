#!/usr/bin/env python3
"""
Enhanced PLC Processing Service
Comprehensive integration of all analysis modules with proper file output and ChromaDB integration

This service addresses all the issues:
1. Complete variable and logic parsing from L5X files
2. ChromaDB integration for semantic search
3. Code generation with AI integration
4. File output for reports and generated code
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPLCProcessingService:
    """Enhanced service that integrates all PLC analysis capabilities"""

    def __init__(self, output_dir: str = "static/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all analysis components"""
        logger.info("Initializing enhanced PLC processing components...")

        # Core parser
        try:
            from src.core.l5x_parser import L5XParser
            self.l5x_parser = L5XParser()
            logger.info("✓ L5X Parser initialized")
        except ImportError as e:
            logger.error(f"✗ L5X Parser failed: {e}")
            self.l5x_parser = None

        # Ladder logic parser
        try:
            from src.analysis.ladder_logic_parser import LadderLogicParser
            self.ladder_parser = LadderLogicParser()
            logger.info("✓ Ladder Logic Parser initialized")
        except ImportError as e:
            logger.error(f"✗ Ladder Logic Parser failed: {e}")
            self.ladder_parser = None

        # Instruction analyzer
        try:
            from src.analysis.instruction_analysis import InstructionAnalyzer
            self.instruction_analyzer = InstructionAnalyzer()
            logger.info("✓ Instruction Analyzer initialized")
        except ImportError as e:
            logger.error(f"✗ Instruction Analyzer failed: {e}")
            self.instruction_analyzer = None

        # Advanced analysis modules
        try:
            from src.analysis.graph_builder import AdvancedGraphBuilder
            from src.analysis.graph_query_engine import GraphQueryEngine
            from src.analysis.routine_analyzer import RoutineAnalyzer
            from src.analysis.timer_counter_analyzer import TimerCounterAnalyzer

            self.graph_builder = AdvancedGraphBuilder()
            self.graph_query_engine = GraphQueryEngine(self.graph_builder)
            self.routine_analyzer = RoutineAnalyzer()
            self.timer_counter_analyzer = TimerCounterAnalyzer()
            logger.info("✓ Advanced analysis modules initialized")
        except ImportError as e:
            logger.warning(f"⚠ Advanced analysis modules not available: {e}")
            self.graph_builder = None
            self.graph_query_engine = None
            self.routine_analyzer = None
            self.timer_counter_analyzer = None

        # ChromaDB integration with PyTorch compatibility handling
        try:
            # Import with enhanced error handling
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from src.semantic.chromadb_integration import PLCSemanticSearchEngine
            self.semantic_engine = PLCSemanticSearchEngine()
            logger.info("✓ ChromaDB integration initialized")
        except (ImportError, AttributeError) as e:
            logger.warning(f"⚠ ChromaDB not available (PyTorch compatibility issue): {e}")
            logger.info("  Continuing without semantic search capabilities")
            self.semantic_engine = None
        except Exception as e:
            logger.warning(f"⚠ ChromaDB initialization failed: {e}")
            logger.info("  System will continue without semantic search")
            self.semantic_engine = None

        # Code generation
        try:
            from src.ai.code_generation import CodeGenerator
            from src.ai.ai_interface import AIInterfaceManager

            self.code_generator = CodeGenerator()
            self.ai_interface = AIInterfaceManager()
            logger.info("✓ Code generation initialized")
        except ImportError as e:
            logger.warning(f"⚠ Code generation not available: {e}")
            self.code_generator = None
            self.ai_interface = None

        # Visualization
        try:
            from src.visualization.graph_visualizer import GraphVisualizer
            from src.visualization.advanced_visualization import AdvancedVisualizationEngine

            self.graph_visualizer = GraphVisualizer(self.graph_builder, self.graph_query_engine)
            self.advanced_visualizer = AdvancedVisualizationEngine()
            logger.info("✓ Visualization components initialized")
        except ImportError as e:
            logger.warning(f"⚠ Visualization not available: {e}")
            self.graph_visualizer = None
            self.advanced_visualizer = None

    def process_l5x_file_comprehensive(self, file_path: str) -> Dict[str, Any]:
        """
        Comprehensive L5X file processing with all advanced features

        Returns complete analysis results with file outputs
        """
        start_time = time.time()
        timestamp = datetime.now()

        logger.info(f"🚀 Starting comprehensive L5X processing: {file_path}")

        results = {
            'success': False,
            'timestamp': timestamp.isoformat(),
            'file_path': file_path,
            'processing_time': 0.0,
            'components_used': [],
            'outputs_generated': [],
            'analysis_results': {},
            'errors': []
        }

        try:
            # Step 1: Parse L5X file
            logger.info("📄 Step 1: Parsing L5X file...")
            parse_result = self._parse_l5x_comprehensive(file_path)
            if not parse_result['success']:
                results['errors'].append(f"L5X parsing failed: {parse_result.get('error', 'Unknown error')}")
                return results

            results['analysis_results']['l5x_parsing'] = parse_result
            results['components_used'].append('l5x_parser')
            logger.info(f"✓ L5X parsed: {len(parse_result.get('controller_tags', []))} controller tags, {len(parse_result.get('programs', []))} programs")

            # Step 2: Advanced Ladder Logic Analysis
            logger.info("🔍 Step 2: Advanced ladder logic analysis...")
            ladder_analysis = self._analyze_ladder_logic_comprehensive(parse_result)
            results['analysis_results']['ladder_logic'] = ladder_analysis
            if ladder_analysis.get('success'):
                results['components_used'].append('ladder_logic_analyzer')
                logger.info(f"✓ Ladder logic analyzed: {ladder_analysis.get('total_instructions', 0)} instructions")

            # Step 3: Instruction Analysis
            logger.info("⚙️ Step 3: Instruction analysis...")
            instruction_analysis = self._analyze_instructions_comprehensive(parse_result, ladder_analysis)
            results['analysis_results']['instruction_analysis'] = instruction_analysis
            if instruction_analysis.get('success'):
                results['components_used'].append('instruction_analyzer')
                logger.info(f"✓ Instructions analyzed: {len(instruction_analysis.get('analyzed_instructions', []))} instructions")

            # Step 4: Advanced Graph Analysis
            logger.info("📊 Step 4: Advanced graph analysis...")
            graph_analysis = self._build_advanced_graphs(parse_result, ladder_analysis, instruction_analysis)
            results['analysis_results']['graph_analysis'] = graph_analysis
            if graph_analysis.get('success'):
                results['components_used'].append('graph_analysis')
                logger.info(f"✓ Graphs built: {len(graph_analysis.get('graphs', []))} graph types")

            # Step 5: Routine and Timer/Counter Analysis
            logger.info("🔄 Step 5: Routine and timer/counter analysis...")
            advanced_analysis = self._run_advanced_analysis(parse_result, ladder_analysis)
            results['analysis_results']['advanced_analysis'] = advanced_analysis
            if advanced_analysis.get('success'):
                results['components_used'].append('advanced_analysis')
                logger.info(f"✓ Advanced analysis complete")

            # Step 6: ChromaDB Integration
            logger.info("🔍 Step 6: ChromaDB semantic indexing...")
            chromadb_result = self._integrate_chromadb(results['analysis_results'])
            results['analysis_results']['chromadb'] = chromadb_result
            if chromadb_result.get('success'):
                results['components_used'].append('chromadb')
                logger.info(f"✓ ChromaDB indexed: {chromadb_result.get('documents_indexed', 0)} documents")

            # Step 7: Code Generation
            logger.info("🤖 Step 7: AI-powered code generation...")
            code_generation_result = self._generate_code_comprehensive(results['analysis_results'])
            results['analysis_results']['code_generation'] = code_generation_result
            if code_generation_result.get('success'):
                results['components_used'].append('code_generator')
                logger.info(f"✓ Code generated: {len(code_generation_result.get('generated_code', ''))} characters")

            # Step 8: Generate Output Files
            logger.info("📁 Step 8: Generating output files...")
            output_files = self._generate_output_files(results)
            results['outputs_generated'] = output_files
            logger.info(f"✓ Generated {len(output_files)} output files")

            # Step 9: Create Visualizations
            logger.info("📈 Step 9: Creating visualizations...")
            visualizations = self._create_visualizations(results['analysis_results'])
            results['analysis_results']['visualizations'] = visualizations
            if visualizations.get('success'):
                results['components_used'].append('visualizations')
                logger.info(f"✓ Visualizations created: {len(visualizations.get('files', []))} files")

            results['success'] = True

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            results['errors'].append(error_msg)

        finally:
            results['processing_time'] = time.time() - start_time
            logger.info(f"🏁 Processing complete in {results['processing_time']:.2f}s")

        return results

    def _parse_l5x_comprehensive(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive L5X parsing with enhanced variable extraction"""
        try:
            if not self.l5x_parser:
                return {'success': False, 'error': 'L5X parser not available'}

            # Parse the file
            result = self.l5x_parser.parse_file(file_path)

            # Enhanced processing for better variable extraction
            enhanced_result = {
                'success': True,
                'controller_info': result.get('controller_info', {}),
                'controller_tags': result.get('controller_tags', []),
                'programs': result.get('programs', []),
                'routines': result.get('routines', []),
                'io_modules': result.get('io_modules', []),
                'parsing_time': result.get('parsing_time', 0),
                'enhanced_variables': self._extract_enhanced_variables(result)
            }

            return enhanced_result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _extract_enhanced_variables(self, parse_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive variable information"""
        variables = {
            'controller_scope': [],
            'program_scope': {},
            'io_variables': [],
            'udt_variables': [],
            'array_variables': [],
            'total_count': 0
        }

        try:
            # Controller tags
            controller_tags = parse_result.get('controller_tags', [])
            for tag in controller_tags:
                if isinstance(tag, dict):
                    var_info = {
                        'name': tag.get('name', ''),
                        'data_type': tag.get('data_type', ''),
                        'scope': 'controller',
                        'description': tag.get('description', ''),
                        'array_dimensions': tag.get('array_dimensions', [])
                    }
                    variables['controller_scope'].append(var_info)

                    # Categorize special types
                    if var_info['array_dimensions']:
                        variables['array_variables'].append(var_info)
                    if '_' in var_info['data_type']:  # Likely UDT
                        variables['udt_variables'].append(var_info)

            # Program tags
            programs = parse_result.get('programs', [])
            for program in programs:
                if isinstance(program, dict):
                    program_name = program.get('name', '')
                    program_tags = program.get('tags', [])
                    variables['program_scope'][program_name] = []

                    for tag in program_tags:
                        if isinstance(tag, dict):
                            var_info = {
                                'name': tag.get('name', ''),
                                'data_type': tag.get('data_type', ''),
                                'scope': f'program:{program_name}',
                                'description': tag.get('description', ''),
                                'array_dimensions': tag.get('array_dimensions', [])
                            }
                            variables['program_scope'][program_name].append(var_info)

            # I/O variables
            io_modules = parse_result.get('io_modules', [])
            for module in io_modules:
                if isinstance(module, dict):
                    # Extract I/O point information
                    points = module.get('connection_tags', [])
                    for point in points:
                        if isinstance(point, dict):
                            var_info = {
                                'name': point.get('name', ''),
                                'data_type': 'BOOL',  # Most I/O is boolean
                                'scope': 'io',
                                'module': module.get('name', ''),
                                'description': point.get('description', '')
                            }
                            variables['io_variables'].append(var_info)

            # Calculate totals
            variables['total_count'] = (
                len(variables['controller_scope']) +
                sum(len(tags) for tags in variables['program_scope'].values()) +
                len(variables['io_variables'])
            )

            logger.info(f"Enhanced variables extracted: {variables['total_count']} total variables")

        except Exception as e:
            logger.error(f"Error extracting enhanced variables: {e}")

        return variables

    def _analyze_ladder_logic_comprehensive(self, parse_result: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive ladder logic analysis"""
        if not self.ladder_parser:
            return {'success': False, 'error': 'Ladder logic parser not available'}

        try:
            analysis = {
                'success': True,
                'routines_analyzed': [],
                'total_rungs': 0,
                'total_instructions': 0,
                'instruction_distribution': {},
                'tag_references': set()
            }

            routines = parse_result.get('routines', [])
            for routine in routines:
                if isinstance(routine, dict) and routine.get('ladder_logic'):
                    ladder_logic = routine['ladder_logic']
                    if hasattr(ladder_logic, 'rungs'):
                        routine_analysis = {
                            'name': routine.get('name', ''),
                            'program': routine.get('program_name', ''),
                            'rungs_count': len(ladder_logic.rungs),
                            'instructions': []
                        }

                        for rung in ladder_logic.rungs:
                            if hasattr(rung, 'instructions'):
                                for instruction in rung.instructions:
                                    inst_info = {
                                        'type': instruction.type.value if hasattr(instruction.type, 'value') else str(instruction.type),
                                        'operand': instruction.operand,
                                        'rung': rung.number
                                    }
                                    routine_analysis['instructions'].append(inst_info)
                                    analysis['tag_references'].add(instruction.operand)

                                    # Update distribution
                                    inst_type = inst_info['type']
                                    analysis['instruction_distribution'][inst_type] = analysis['instruction_distribution'].get(inst_type, 0) + 1

                        analysis['routines_analyzed'].append(routine_analysis)
                        analysis['total_rungs'] += routine_analysis['rungs_count']
                        analysis['total_instructions'] += len(routine_analysis['instructions'])

            # Convert set to list for JSON serialization
            analysis['tag_references'] = list(analysis['tag_references'])

            return analysis

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _analyze_instructions_comprehensive(self, parse_result: Dict[str, Any], ladder_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive instruction analysis"""
        if not self.instruction_analyzer:
            return {'success': False, 'error': 'Instruction analyzer not available'}

        try:
            analysis = {
                'success': True,
                'analyzed_instructions': [],
                'instruction_categories': {},
                'tag_dependencies': [],
                'complexity_metrics': {}
            }

            # Analyze instructions from ladder logic
            routines_analyzed = ladder_analysis.get('routines_analyzed', [])
            for routine in routines_analyzed:
                for instruction in routine.get('instructions', []):
                    analyzed_inst = {
                        'type': instruction['type'],
                        'operand': instruction['operand'],
                        'routine': routine['name'],
                        'rung': instruction['rung'],
                        'category': self._categorize_instruction(instruction['type']),
                        'complexity_score': self._calculate_instruction_complexity(instruction)
                    }
                    analysis['analyzed_instructions'].append(analyzed_inst)

                    # Update categories
                    category = analyzed_inst['category']
                    analysis['instruction_categories'][category] = analysis['instruction_categories'].get(category, 0) + 1

            # Calculate overall complexity metrics
            if analysis['analyzed_instructions']:
                total_complexity = sum(inst['complexity_score'] for inst in analysis['analyzed_instructions'])
                analysis['complexity_metrics'] = {
                    'total_complexity': total_complexity,
                    'average_complexity': total_complexity / len(analysis['analyzed_instructions']),
                    'max_complexity': max(inst['complexity_score'] for inst in analysis['analyzed_instructions'])
                }

            return analysis

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _categorize_instruction(self, instruction_type: str) -> str:
        """Categorize instruction type"""
        basic_logic = ['XIC', 'XIO', 'OTE', 'ONS']
        timers = ['TON', 'TOF', 'RTO']
        counters = ['CTU', 'CTD', 'RES']
        math = ['ADD', 'SUB', 'MUL', 'DIV', 'MOV']

        if instruction_type in basic_logic:
            return 'BASIC_LOGIC'
        elif instruction_type in timers:
            return 'TIMING'
        elif instruction_type in counters:
            return 'COUNTING'
        elif instruction_type in math:
            return 'MATH'
        else:
            return 'ADVANCED'

    def _calculate_instruction_complexity(self, instruction: Dict[str, Any]) -> float:
        """Calculate complexity score for instruction"""
        base_scores = {
            'XIC': 1.0, 'XIO': 1.0, 'OTE': 1.0,
            'TON': 3.0, 'TOF': 3.0, 'RTO': 3.5,
            'CTU': 2.5, 'CTD': 2.5,
            'ADD': 2.0, 'SUB': 2.0, 'MUL': 2.5, 'DIV': 3.0,
            'MOV': 1.5
        }
        return base_scores.get(instruction['type'], 2.0)

    def _build_advanced_graphs(self, parse_result: Dict[str, Any], ladder_analysis: Dict[str, Any], instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build advanced graph representations"""
        if not self.graph_builder:
            return {'success': False, 'error': 'Graph builder not available'}

        try:
            # Create a comprehensive data structure for graph building
            plc_data = {
                'controller_info': parse_result.get('controller_info', {}),
                'tags': parse_result.get('controller_tags', []),
                'programs': parse_result.get('programs', []),
                'routines': ladder_analysis.get('routines_analyzed', []),
                'instructions': instruction_analysis.get('analyzed_instructions', [])
            }

            # Build different types of graphs
            graphs = {}
            graph_types = ['CONTROL_FLOW', 'DATA_DEPENDENCY', 'INSTRUCTION_NETWORK', 'EXECUTION_FLOW']

            for graph_type in graph_types:
                try:
                    graph = self.graph_builder.build_comprehensive_graph(plc_data, graph_type)
                    graphs[graph_type] = {
                        'nodes': len(graph.nodes()) if hasattr(graph, 'nodes') else 0,
                        'edges': len(graph.edges()) if hasattr(graph, 'edges') else 0,
                        'created': True
                    }
                except Exception as e:
                    graphs[graph_type] = {'created': False, 'error': str(e)}

            return {
                'success': True,
                'graphs': graphs,
                'graph_types_built': [gt for gt, info in graphs.items() if info.get('created')]
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _run_advanced_analysis(self, parse_result: Dict[str, Any], ladder_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run routine and timer/counter analysis"""
        analysis_results = {'success': True, 'components': {}}

        # Routine analysis
        if self.routine_analyzer:
            try:
                routine_data = {
                    'programs': parse_result.get('programs', []),
                    'routines': ladder_analysis.get('routines_analyzed', [])
                }
                routine_analysis = self.routine_analyzer.analyze_program_structure(routine_data)
                analysis_results['components']['routine_analysis'] = routine_analysis
            except Exception as e:
                analysis_results['components']['routine_analysis'] = {'error': str(e)}

        # Timer/Counter analysis
        if self.timer_counter_analyzer:
            try:
                timer_counter_data = {
                    'instructions': ladder_analysis.get('routines_analyzed', [])
                }
                timer_counter_analysis = self.timer_counter_analyzer.analyze_timers_and_counters(timer_counter_data)
                analysis_results['components']['timer_counter_analysis'] = timer_counter_analysis
            except Exception as e:
                analysis_results['components']['timer_counter_analysis'] = {'error': str(e)}

        return analysis_results

    def _integrate_chromadb(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with ChromaDB for semantic search"""
        if not self.semantic_engine:
            return {'success': False, 'error': 'ChromaDB not available'}

        try:
            # Prepare documents for indexing
            documents = []

            # Index L5X parsing results
            l5x_data = analysis_results.get('l5x_parsing', {})
            if l5x_data.get('success'):
                # Controller tags
                for tag in l5x_data.get('controller_tags', []):
                    if isinstance(tag, dict):
                        doc = {
                            'content': f"Controller tag: {tag.get('name')} ({tag.get('data_type')}) - {tag.get('description', '')}",
                            'metadata': {
                                'type': 'controller_tag',
                                'name': tag.get('name'),
                                'data_type': tag.get('data_type'),
                                'scope': 'controller'
                            }
                        }
                        documents.append(doc)

                # Program information
                for program in l5x_data.get('programs', []):
                    if isinstance(program, dict):
                        doc = {
                            'content': f"Program: {program.get('name')} ({program.get('type')}) - {program.get('description', '')}",
                            'metadata': {
                                'type': 'program',
                                'name': program.get('name'),
                                'program_type': program.get('type')
                            }
                        }
                        documents.append(doc)

            # Index ladder logic
            ladder_data = analysis_results.get('ladder_logic', {})
            if ladder_data.get('success'):
                for routine in ladder_data.get('routines_analyzed', []):
                    doc = {
                        'content': f"Routine: {routine.get('name')} - {len(routine.get('instructions', []))} instructions",
                        'metadata': {
                            'type': 'routine',
                            'name': routine.get('name'),
                            'program': routine.get('program'),
                            'instruction_count': len(routine.get('instructions', []))
                        }
                    }
                    documents.append(doc)

            # Index in ChromaDB
            if documents:
                collection_name = f"plc_analysis_{int(time.time())}"
                indexed_count = self.semantic_engine.index_documents(documents, collection_name)

                return {
                    'success': True,
                    'documents_indexed': indexed_count,
                    'collection': collection_name,
                    'document_types': list(set(doc['metadata']['type'] for doc in documents))
                }
            else:
                return {'success': False, 'error': 'No documents to index'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _generate_code_comprehensive(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Python code using AI"""
        if not self.code_generator or not self.ai_interface:
            return {'success': False, 'error': 'Code generation not available'}

        try:
            # Prepare context for code generation
            context = {
                'controller_info': analysis_results.get('l5x_parsing', {}).get('controller_info', {}),
                'tags': analysis_results.get('l5x_parsing', {}).get('enhanced_variables', {}),
                'ladder_logic': analysis_results.get('ladder_logic', {}),
                'instruction_analysis': analysis_results.get('instruction_analysis', {})
            }

            # Generate code
            generation_result = self.code_generator.generate_plc_interface_code(
                context=context,
                generation_type='FULL_INTERFACE',
                quality_level='PRODUCTION',
                framework='PYCOMM3'
            )

            if generation_result.get('success'):
                return {
                    'success': True,
                    'generated_code': generation_result.get('generated_code', ''),
                    'validation_score': generation_result.get('validation_score', 0),
                    'features': generation_result.get('features', []),
                    'generation_time': generation_result.get('generation_time', 0)
                }
            else:
                return {'success': False, 'error': generation_result.get('error', 'Code generation failed')}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _generate_output_files(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive output files"""
        output_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        try:
            logger.info("Generating JSON report...")
            # 1. Analysis Report (JSON)
            analysis_file = self.output_dir / f"analysis_report_{timestamp}.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            output_files.append(str(analysis_file))
            logger.info(f"✓ Generated {analysis_file}")

            # 2. Generated Python Code
            if results['analysis_results'].get('code_generation', {}).get('success'):
                logger.info("Generating Python code file...")
                code_file = self.output_dir / f"generated_plc_interface_{timestamp}.py"
                generated_code = results['analysis_results']['code_generation'].get('generated_code', '')

                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(f'# Generated PLC Interface Code\\n')
                    f.write(f'# Generated: {datetime.now().isoformat()}\\n')
                    f.write(f'# Source: {results.get("file_path", "Unknown")}\\n\\n')
                    f.write(generated_code)
                output_files.append(str(code_file))
                logger.info(f"✓ Generated {code_file}")

            # 3. Variable Summary (XLSX)
            try:
                logger.info("Generating XLSX variable summary...")
                import pandas as pd
                variables_file = self.output_dir / f"variables_summary_{timestamp}.xlsx"
                enhanced_vars = results['analysis_results'].get('l5x_parsing', {}).get('enhanced_variables', {})

                all_vars = []
                # Controller variables
                for var in enhanced_vars.get('controller_scope', []):
                    all_vars.append([var.get('name'), var.get('data_type'), var.get('scope'), var.get('description')])

                # Program variables
                for program, vars_list in enhanced_vars.get('program_scope', {}).items():
                    for var in vars_list:
                        all_vars.append([var.get('name'), var.get('data_type'), var.get('scope'), var.get('description')])

                df = pd.DataFrame(all_vars, columns=['Name', 'Data Type', 'Scope', 'Description'])
                df.to_excel(variables_file, index=False)

                output_files.append(str(variables_file))
                logger.info(f"✓ Generated {variables_file}")
            except ImportError:
                logger.warning("pandas module not available for variables summary")

            # 4. Instruction Analysis Report (Text)
            logger.info("Generating TXT instruction analysis...")
            instruction_file = self.output_dir / f"instruction_analysis_{timestamp}.txt"
            instruction_data = results['analysis_results'].get('instruction_analysis', {})

            with open(instruction_file, 'w', encoding='utf-8') as f:
                f.write(f"PLC Instruction Analysis Report\\n")
                f.write(f"Generated: {datetime.now().isoformat()}\\n")
                f.write(f"=" * 50 + "\\n\\n")

                if instruction_data.get('success'):
                    f.write(f"Total Instructions Analyzed: {len(instruction_data.get('analyzed_instructions', []))}\\n")
                    f.write(f"Instruction Categories:\\n")
                    for category, count in instruction_data.get('instruction_categories', {}).items():
                        f.write(f"  {category}: {count}\\n")

                    complexity = instruction_data.get('complexity_metrics', {})
                    if complexity:
                        f.write(f"\\nComplexity Metrics:\\n")
                        f.write(f"  Total Complexity: {complexity.get('total_complexity', 0):.2f}\\n")
                        f.write(f"  Average Complexity: {complexity.get('average_complexity', 0):.2f}\\n")
                        f.write(f"  Maximum Complexity: {complexity.get('max_complexity', 0):.2f}\\n")
                else:
                    f.write("Instruction analysis not available\\n")

            output_files.append(str(instruction_file))
            logger.info(f"✓ Generated {instruction_file}")

            # 5. HTML Analysis Report
            logger.info("Generating HTML analysis report...")
            html_file = self.output_dir / f"analysis_report_{timestamp}.html"
            html_content = self._create_analysis_html(results)
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            output_files.append(str(html_file))
            logger.info(f"✓ Generated {html_file}")

            logger.info(f"Generated {len(output_files)} output files")

        except Exception as e:
            logger.error(f"Error generating output files: {e}")

        return output_files

    def _create_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization files"""
        if not self.graph_visualizer and not self.advanced_visualizer:
            return {'success': False, 'error': 'Visualization not available'}

        try:
            visualization_files = []

            # Create basic graph visualizations
            if self.graph_visualizer:
                graph_data = analysis_results.get('graph_analysis', {})
                if graph_data.get('success'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    viz_file = self.output_dir / f"graph_visualization_{timestamp}.html"

                    # Create a simple visualization
                    html_content = self._create_basic_graph_html(graph_data)
                    with open(viz_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    visualization_files.append(str(viz_file))

            # Create advanced visualizations
            if self.advanced_visualizer:
                # Create analytics dashboard
                dashboard_data = self._prepare_dashboard_data(analysis_results)
                dashboard_result = self.advanced_visualizer.create_analytics_dashboard(dashboard_data)

                if dashboard_result.get('success'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dashboard_file = self.output_dir / f"analytics_dashboard_{timestamp}.html"

                    with open(dashboard_file, 'w', encoding='utf-8') as f:
                        f.write(dashboard_result.get('html_content', ''))

                    visualization_files.append(str(dashboard_file))

            return {
                'success': True,
                'files': visualization_files,
                'count': len(visualization_files)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _create_basic_graph_html(self, graph_data: Dict[str, Any]) -> str:
        """Create basic HTML visualization"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PLC Graph Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .graph-info {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>PLC Graph Analysis Results</h1>
            <p>Generated: {datetime.now().isoformat()}</p>

            <h2>Graph Types Built:</h2>
        """

        for graph_type, info in graph_data.get('graphs', {}).items():
            status = "✓" if info.get('created') else "✗"
            css_class = "success" if info.get('created') else "error"

            html += f"""
            <div class="graph-info">
                <h3 class="{css_class}">{status} {graph_type}</h3>
                <p>Nodes: {info.get('nodes', 0)}</p>
                <p>Edges: {info.get('edges', 0)}</p>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def _prepare_dashboard_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for advanced dashboard"""
        return {
            'title': 'PLC Analysis Dashboard',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tags': len(analysis_results.get('l5x_parsing', {}).get('controller_tags', [])),
                'total_programs': len(analysis_results.get('l5x_parsing', {}).get('programs', [])),
                'total_instructions': analysis_results.get('instruction_analysis', {}).get('analyzed_instructions', []),
                'complexity_score': analysis_results.get('instruction_analysis', {}).get('complexity_metrics', {}).get('average_complexity', 0)
            },
            'components_used': analysis_results.get('components_used', []),
            'processing_successful': analysis_results.get('success', False)
        }

    def _create_analysis_html(self, results: Dict[str, Any]) -> str:
        """Creates an HTML report from the analysis results."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PLC Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 800px; margin: auto; background: #f9f9f9; padding: 20px; border-radius: 8px; }}
                .info-box {{ background: #eee; padding: 10px; margin-bottom: 10px; border-left: 4px solid #007bff; }}
                pre {{ background: #ddd; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>PLC Analysis Report</h1>
                <p><strong>File:</strong> {results.get('file_path', 'N/A')}</p>
                <p><strong>Timestamp:</strong> {results.get('timestamp', 'N/A')}</p>
                <p><strong>Processing Time:</strong> {results.get('processing_time', 0):.2f}s</p>

                <h2>Controller Info</h2>
                <div class="info-box">
                    <pre>{json.dumps(results.get('analysis_results', {{}}).get('l5x_parsing', {{}}).get('controller_info', {{}}), indent=2)}</pre>
                </div>

                <h2>Processing Summary</h2>
                <div class="info-box">
                    <p><strong>Components Used:</strong> {', '.join(results.get('components_used', []))}</p>
                    <p><strong>Outputs Generated:</strong> {len(results.get('outputs_generated', []))}</p>
                </div>

                <h2>Errors</h2>
                <div class="info-box">
                    <pre>{json.dumps(results.get('errors', ['No errors']), indent=2)}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        return html

# Test the enhanced service
if __name__ == "__main__":
    service = EnhancedPLCProcessingService()

    # Test with the L5X file if it exists
    l5x_file = "Assembly_Controls_Robot.L5X"
    if os.path.exists(l5x_file):
        print(f"🧪 Testing enhanced processing with {l5x_file}")
        result = service.process_l5x_file_comprehensive(l5x_file)

        print(f"\\n📊 Processing Results:")
        print(f"Success: {result['success']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Components Used: {', '.join(result['components_used'])}")
        print(f"Output Files: {len(result['outputs_generated'])}")

        if result['outputs_generated']:
            print("\\n📁 Generated Files:")
            for file_path in result['outputs_generated']:
                print(f"  - {file_path}")

        if result['errors']:
            print("\\n⚠️ Errors:")
            for error in result['errors']:
                print(f"  - {error}")
    else:
        print(f"❌ {l5x_file} not found in current directory")
