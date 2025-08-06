"""
Processing Pipeline Module
Orchestrates the complete PLC analysis workflow with advanced analysis
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .l5x_parser import L5XParser

# Import advanced analysis modules
try:
    from src.analysis.ladder_logic_parser import LadderLogicParser
    from src.analysis.instruction_analysis import InstructionAnalyzer
    from src.models.knowledge_graph import PLCKnowledgeGraph
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced analysis modules not available: {e}")
    ADVANCED_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStep:
    """Individual processing step configuration"""
    name: str
    description: str
    function: Callable
    enabled: bool = True
    timeout: int = 300  # 5 minutes default
    retry_count: int = 1

@dataclass
class ProcessingResult:
    """Result from a processing step"""
    step_name: str
    success: bool
    data: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    success: bool
    total_execution_time: float
    timestamp: datetime
    steps_results: List[ProcessingResult]
    final_data: Dict[str, Any]
    error_summary: List[str]

class ProcessingPipeline:
    """
    Main processing pipeline that orchestrates L5X file analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = L5XParser()
        self.steps = []
        self._setup_default_pipeline()
    
    def _setup_default_pipeline(self):
        """Setup the default processing pipeline with advanced analysis"""
        self.steps = [
            ProcessingStep(
                name="file_validation",
                description="Validate L5X file format and structure",
                function=self._validate_file_step
            ),
            ProcessingStep(
                name="xml_parsing",
                description="Parse L5X XML content",
                function=self._parse_xml_step
            ),
            ProcessingStep(
                name="data_extraction",
                description="Extract structured data from parsed content",
                function=self._extract_data_step
            ),
            ProcessingStep(
                name="ladder_logic_analysis",
                description="Analyze ladder logic patterns and instructions",
                function=self._analyze_ladder_logic_step,
                enabled=ADVANCED_ANALYSIS_AVAILABLE
            ),
            ProcessingStep(
                name="knowledge_graph_construction",
                description="Build knowledge graph from PLC components",
                function=self._build_knowledge_graph_step,
                enabled=ADVANCED_ANALYSIS_AVAILABLE
            ),
            ProcessingStep(
                name="logic_analysis",
                description="Analyze PLC logic and dependencies",
                function=self._analyze_logic_step
            ),
            ProcessingStep(
                name="documentation_generation",
                description="Generate documentation from analysis",
                function=self._generate_documentation_step
            )
        ]
    
    def process_file(self, file_path: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process L5X file through complete pipeline
        
        Args:
            file_path: Path to the L5X file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete processing results
        """
        start_time = time.time()
        self.logger.info(f"Starting pipeline processing for: {file_path}")
        
        # Initialize results tracking
        step_results = []
        pipeline_data = {"file_path": file_path}
        error_summary = []
        
        try:
            # Execute each pipeline step
            for i, step in enumerate(self.steps):
                if not step.enabled:
                    self.logger.debug(f"Skipping disabled step: {step.name}")
                    continue
                
                # Update progress
                if progress_callback:
                    progress = (i / len(self.steps)) * 100
                    progress_callback(f"Executing {step.description}", progress)
                
                # Execute step with timeout and retry logic
                step_result = self._execute_step(step, pipeline_data)
                step_results.append(step_result)
                
                if step_result.success:
                    # Update pipeline data with step results
                    if step_result.data:
                        pipeline_data.update(step_result.data)
                    self.logger.debug(f"Step '{step.name}' completed successfully in {step_result.execution_time:.2f}s")
                else:
                    error_summary.append(f"{step.name}: {step_result.error_message}")
                    self.logger.error(f"Step '{step.name}' failed: {step_result.error_message}")
                    
                    # Decide whether to continue or stop pipeline
                    if step.name in ["file_validation", "xml_parsing"]:
                        # Critical steps - stop pipeline
                        break
            
            # Final progress update
            if progress_callback:
                progress_callback("Processing complete", 100)
            
            total_time = time.time() - start_time
            success = len(error_summary) == 0
            
            result = {
                'success': success,
                'total_execution_time': total_time,
                'timestamp': datetime.now(),
                'steps_results': [asdict(result) for result in step_results],
                'final_data': pipeline_data,
                'error_summary': error_summary,
                'file_path': file_path,
                'statistics': self._generate_statistics(pipeline_data)
            }
            
            self.logger.info(f"Pipeline processing completed in {total_time:.2f}s (Success: {success})")
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Pipeline processing failed: {e}")
            
            return {
                'success': False,
                'total_execution_time': total_time,
                'timestamp': datetime.now(),
                'steps_results': [asdict(result) for result in step_results],
                'final_data': pipeline_data,
                'error_summary': [f"Pipeline error: {str(e)}"],
                'file_path': file_path,
                'statistics': {}
            }
    
    def _execute_step(self, step: ProcessingStep, data: Dict[str, Any]) -> ProcessingResult:
        """Execute a single pipeline step with retry logic"""
        start_time = time.time()
        
        for attempt in range(step.retry_count):
            try:
                self.logger.debug(f"Executing step: {step.name} (attempt {attempt + 1})")
                
                # Execute the step function
                result_data = step.function(data)
                
                execution_time = time.time() - start_time
                return ProcessingResult(
                    step_name=step.name,
                    success=True,
                    data=result_data,
                    execution_time=execution_time
                )
                
            except Exception as e:
                self.logger.warning(f"Step '{step.name}' attempt {attempt + 1} failed: {e}")
                if attempt == step.retry_count - 1:
                    # Final attempt failed
                    execution_time = time.time() - start_time
                    return ProcessingResult(
                        step_name=step.name,
                        success=False,
                        error_message=str(e),
                        execution_time=execution_time
                    )
                time.sleep(1)  # Brief delay before retry
    
    def _validate_file_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Validate L5X file"""
        file_path = data.get("file_path")
        
        if not file_path:
            raise ValueError("No file path provided")
        
        # Use parser's validation method
        is_valid, error_message = self.parser.validate_l5x_file(file_path)
        
        if not is_valid:
            raise ValueError(f"File validation failed: {error_message}")
        
        # Get file statistics
        file_stats = os.stat(file_path)
        
        return {
            "validation_result": {
                "valid": True,
                "file_size": file_stats.st_size,
                "file_modified": datetime.fromtimestamp(file_stats.st_mtime),
                "validation_message": "File validation successful"
            }
        }
    
    def _parse_xml_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Parse L5X XML content"""
        file_path = data.get("file_path")
        
        # Use the L5X parser
        parse_result = self.parser.parse_file(file_path)
        
        if not parse_result.get("success", False):
            raise ValueError(f"XML parsing failed: {parse_result.get('error_message', 'Unknown error')}")
        
        return {
            "parse_result": parse_result
        }
    
    def _extract_data_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Extract and structure data"""
        parse_result = data.get("parse_result", {})
        
        if not parse_result:
            raise ValueError("No parse result available")
        
        # Extract key information for further processing
        controller_info = parse_result.get("controller_info", {})
        controller_tags = parse_result.get("controller_tags", [])
        programs = parse_result.get("programs", [])
        routines = parse_result.get("routines", [])
        
        # Build comprehensive data structure
        extracted_data = {
            "controller": {
                "name": controller_info.get("name", "Unknown"),
                "type": controller_info.get("type", "Unknown"),
                "firmware": controller_info.get("firmware_revision", "Unknown"),
                "tag_count": len(controller_tags),
                "program_count": len(programs)
            },
            "tags_summary": {
                "controller_tags": len(controller_tags),
                "program_tags": sum(len(p.get("tags", [])) for p in programs),
                "total_tags": len(controller_tags) + sum(len(p.get("tags", [])) for p in programs)
            },
            "programs_summary": {
                "total_programs": len(programs),
                "total_routines": len(routines),
                "program_names": [p.get("name", "") for p in programs]
            },
            "detailed_data": {
                "controller_tags": controller_tags,
                "programs": programs,
                "routines": routines,
                "io_modules": parse_result.get("io_modules", [])
            }
        }
        
        return {
            "extracted_data": extracted_data
        }
    
    def _analyze_logic_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Analyze PLC logic and dependencies"""
        extracted_data = data.get("extracted_data", {})
        
        if not extracted_data:
            raise ValueError("No extracted data available")
        
        # Perform logic analysis
        programs = extracted_data.get("detailed_data", {}).get("programs", [])
        controller_tags = extracted_data.get("detailed_data", {}).get("controller_tags", [])
        
        # Analyze tag usage patterns
        tag_analysis = self._analyze_tag_usage(controller_tags, programs)
        
        # Analyze program dependencies
        program_dependencies = self._analyze_program_dependencies(programs)
        
        # Generate logic insights
        logic_insights = self._generate_logic_insights(programs, controller_tags)
        
        analysis_result = {
            "tag_analysis": tag_analysis,
            "program_dependencies": program_dependencies,
            "logic_insights": logic_insights,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return {
            "logic_analysis": analysis_result
        }
    
    def _analyze_ladder_logic_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4a: Analyze ladder logic patterns and instructions"""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            return {"ladder_analysis": {"available": False, "reason": "Advanced analysis modules not available"}}
        
        parse_result = data.get("parse_result", {})
        
        if not parse_result.get("success", False):
            raise ValueError("No parse result available for ladder logic analysis")
        
        try:
            ladder_parser = LadderLogicParser()
            instruction_analyzer = InstructionAnalyzer()
            
            # Parse ladder logic from routines
            all_routine_logic = []
            routines = parse_result.get("routines", [])
            
            # For now, we'll create mock routine logic since we need the actual XML elements
            # In a full implementation, this would parse the actual routine XML
            analysis_result = {
                "ladder_analysis": {
                    "total_routines_analyzed": len(routines),
                    "instruction_summary": {
                        "total_instructions": 0,
                        "instruction_types": {},
                        "complexity_analysis": {}
                    },
                    "patterns_detected": {
                        "timer_usage": [],
                        "counter_usage": [],
                        "safety_interlocks": [],
                        "seal_circuits": []
                    },
                    "recommendations": [
                        "Detailed ladder logic analysis requires XML element access",
                        "Consider implementing routine-level parsing for complete analysis"
                    ]
                }
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Ladder logic analysis failed: {e}")
            return {
                "ladder_analysis": {
                    "available": False,
                    "error": str(e),
                    "fallback_analysis": {
                        "routine_count": len(parse_result.get("routines", [])),
                        "basic_info": "Advanced ladder logic analysis failed"
                    }
                }
            }
    
    def _build_knowledge_graph_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4b: Build knowledge graph from PLC components"""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            return {"knowledge_graph": {"available": False, "reason": "Advanced analysis modules not available"}}
        
        try:
            # Create knowledge graph
            knowledge_graph = PLCKnowledgeGraph()
            
            # Build graph from all available data
            l5x_data = {
                'final_data': {
                    'extracted_data': data.get('extracted_data', {})
                }
            }
            
            knowledge_graph.build_from_l5x_data(l5x_data)
            
            # Analyze graph structure
            graph_analysis = knowledge_graph.analyze_graph_structure()
            
            # Export graph data for use in other steps
            graph_data = knowledge_graph.export_graph_data()
            
            return {
                "knowledge_graph": {
                    "available": True,
                    "graph_analysis": graph_analysis,
                    "graph_data": graph_data,
                    "statistics": {
                        "nodes": len(knowledge_graph.nodes),
                        "edges": len(knowledge_graph.edges),
                        "node_types": list(set(node.type.value for node in knowledge_graph.nodes.values())),
                        "edge_types": list(set(edge.type.value for edge in knowledge_graph.edges))
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Knowledge graph construction failed: {e}")
            return {
                "knowledge_graph": {
                    "available": False,
                    "error": str(e),
                    "fallback_info": "Knowledge graph construction failed"
                }
            }
    
    def _generate_documentation_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Generate documentation"""
        extracted_data = data.get("extracted_data", {})
        logic_analysis = data.get("logic_analysis", {})
        
        # Generate comprehensive documentation
        documentation = {
            "system_overview": self._generate_system_overview(extracted_data),
            "tag_documentation": self._generate_tag_documentation(extracted_data),
            "program_documentation": self._generate_program_documentation(extracted_data),
            "analysis_summary": self._generate_analysis_summary(logic_analysis),
            "generated_timestamp": datetime.now().isoformat()
        }
        
        return {
            "documentation": documentation
        }
    
    def _analyze_tag_usage(self, controller_tags: List[Dict], programs: List[Dict]) -> Dict[str, Any]:
        """Analyze tag usage patterns"""
        analysis = {
            "data_type_distribution": {},
            "scope_distribution": {"controller": len(controller_tags), "program": 0},
            "array_tags": [],
            "constant_tags": []
        }
        
        # Analyze controller tags
        for tag in controller_tags:
            data_type = tag.get("data_type", "UNKNOWN")
            analysis["data_type_distribution"][data_type] = analysis["data_type_distribution"].get(data_type, 0) + 1
            
            if tag.get("array_dimensions"):
                analysis["array_tags"].append(tag.get("name", ""))
            
            if tag.get("constant", False):
                analysis["constant_tags"].append(tag.get("name", ""))
        
        # Analyze program tags
        for program in programs:
            program_tags = program.get("tags", [])
            analysis["scope_distribution"]["program"] += len(program_tags)
            
            for tag in program_tags:
                data_type = tag.get("data_type", "UNKNOWN")
                analysis["data_type_distribution"][data_type] = analysis["data_type_distribution"].get(data_type, 0) + 1
        
        return analysis
    
    def _analyze_program_dependencies(self, programs: List[Dict]) -> Dict[str, Any]:
        """Analyze dependencies between programs"""
        dependencies = {
            "program_relationships": {},
            "shared_tags": [],
            "main_programs": []
        }
        
        for program in programs:
            program_name = program.get("name", "")
            dependencies["program_relationships"][program_name] = {
                "routines": program.get("routines", []),
                "main_routine": program.get("main_routine", ""),
                "tag_count": len(program.get("tags", []))
            }
            
            if program.get("main_routine"):
                dependencies["main_programs"].append(program_name)
        
        return dependencies
    
    def _generate_logic_insights(self, programs: List[Dict], controller_tags: List[Dict]) -> Dict[str, Any]:
        """Generate insights about the PLC logic"""
        insights = {
            "complexity_indicators": {},
            "recommendations": [],
            "potential_issues": []
        }
        
        # Calculate complexity indicators
        total_routines = sum(len(p.get("routines", [])) for p in programs)
        total_tags = len(controller_tags) + sum(len(p.get("tags", [])) for p in programs)
        
        insights["complexity_indicators"] = {
            "total_programs": len(programs),
            "total_routines": total_routines,
            "total_tags": total_tags,
            "avg_routines_per_program": total_routines / len(programs) if programs else 0,
            "complexity_score": min(100, (total_routines * 2 + total_tags) / 10)  # Simple complexity metric
        }
        
        # Generate recommendations
        if total_tags > 1000:
            insights["recommendations"].append("Consider organizing tags into more structured groups")
        
        if total_routines > 50:
            insights["recommendations"].append("High routine count - consider modular programming approach")
        
        return insights
    
    def _generate_system_overview(self, extracted_data: Dict[str, Any]) -> str:
        """Generate system overview documentation"""
        controller = extracted_data.get("controller", {})
        tags_summary = extracted_data.get("tags_summary", {})
        programs_summary = extracted_data.get("programs_summary", {})
        
        overview = f"""
PLC System Overview
==================

Controller Information:
- Name: {controller.get('name', 'Unknown')}
- Type: {controller.get('type', 'Unknown')}
- Firmware: {controller.get('firmware', 'Unknown')}

System Statistics:
- Total Programs: {programs_summary.get('total_programs', 0)}
- Total Routines: {programs_summary.get('total_routines', 0)}
- Total Tags: {tags_summary.get('total_tags', 0)}
  - Controller Tags: {tags_summary.get('controller_tags', 0)}
  - Program Tags: {tags_summary.get('program_tags', 0)}

Programs:
{chr(10).join(f'- {name}' for name in programs_summary.get('program_names', []))}
        """.strip()
        
        return overview
    
    def _generate_tag_documentation(self, extracted_data: Dict[str, Any]) -> str:
        """Generate tag documentation"""
        controller_tags = extracted_data.get("detailed_data", {}).get("controller_tags", [])
        
        doc = "Tag Documentation\n==================\n\n"
        
        if controller_tags:
            doc += "Controller Tags:\n"
            for tag in controller_tags[:20]:  # Limit to first 20 for readability
                name = tag.get("name", "Unknown")
                data_type = tag.get("data_type", "Unknown")
                description = tag.get("description", "No description")
                doc += f"- {name} ({data_type}): {description}\n"
            
            if len(controller_tags) > 20:
                doc += f"... and {len(controller_tags) - 20} more tags\n"
        
        return doc
    
    def _generate_program_documentation(self, extracted_data: Dict[str, Any]) -> str:
        """Generate program documentation"""
        programs = extracted_data.get("detailed_data", {}).get("programs", [])
        
        doc = "Program Documentation\n====================\n\n"
        
        for program in programs:
            name = program.get("name", "Unknown")
            description = program.get("description", "No description")
            main_routine = program.get("main_routine", "None")
            routines = program.get("routines", [])
            
            doc += f"Program: {name}\n"
            doc += f"Description: {description}\n"
            doc += f"Main Routine: {main_routine}\n"
            doc += f"Routines ({len(routines)}): {', '.join(routines)}\n\n"
        
        return doc
    
    def _generate_analysis_summary(self, logic_analysis: Dict[str, Any]) -> str:
        """Generate analysis summary"""
        if not logic_analysis:
            return "No analysis data available"
        
        insights = logic_analysis.get("logic_insights", {})
        complexity = insights.get("complexity_indicators", {})
        recommendations = insights.get("recommendations", [])
        
        summary = "Analysis Summary\n===============\n\n"
        
        if complexity:
            summary += f"Complexity Score: {complexity.get('complexity_score', 0):.1f}/100\n"
            summary += f"Programs: {complexity.get('total_programs', 0)}\n"
            summary += f"Routines: {complexity.get('total_routines', 0)}\n"
            summary += f"Tags: {complexity.get('total_tags', 0)}\n\n"
        
        if recommendations:
            summary += "Recommendations:\n"
            for rec in recommendations:
                summary += f"- {rec}\n"
        
        return summary
    
    def _generate_statistics(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final statistics"""
        extracted_data = pipeline_data.get("extracted_data", {})
        
        return {
            "processing_date": datetime.now().isoformat(),
            "file_processed": pipeline_data.get("file_path", "Unknown"),
            "controller_name": extracted_data.get("controller", {}).get("name", "Unknown"),
            "total_elements": {
                "programs": extracted_data.get("programs_summary", {}).get("total_programs", 0),
                "routines": extracted_data.get("programs_summary", {}).get("total_routines", 0),
                "tags": extracted_data.get("tags_summary", {}).get("total_tags", 0)
            }
        }

class PLCProcessingService:
    """
    High-level service for PLC file processing
    """
    
    def __init__(self):
        self.pipeline = ProcessingPipeline()
        self.logger = logging.getLogger(__name__)
    
    def process_l5x_file(self, file_path: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process L5X file and return comprehensive results
        
        Args:
            file_path: Path to the L5X file
            progress_callback: Optional progress callback function
            
        Returns:
            Processing results dictionary
        """
        return self.pipeline.process_file(file_path, progress_callback)
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            "service_active": True,
            "pipeline_steps": len(self.pipeline.steps),
            "timestamp": datetime.now().isoformat()
        }
