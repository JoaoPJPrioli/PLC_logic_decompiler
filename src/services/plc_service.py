"""
PLC Service Module
High-level service for PLC operations
"""

import os
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

from src.core.processing_pipeline import PLCProcessingService
from src.models.tags import Tag, TagCollection, TagScope

logger = logging.getLogger(__name__)

@dataclass
class PLCAnalysisSession:
    """Represents a PLC analysis session"""
    session_id: str
    file_path: str
    created_at: datetime
    status: str  # 'pending', 'processing', 'completed', 'failed'
    results: Dict[str, Any] = None
    error_message: str = ""
    progress: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            'session_id': self.session_id,
            'file_path': self.file_path,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'results': self.results,
            'error_message': self.error_message,
            'progress': self.progress
        }

class PLCService:
    """
    High-level PLC service that orchestrates all PLC operations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processing_service = PLCProcessingService()
        self.active_sessions = {}
        
    def create_analysis_session(self, file_path: str) -> str:
        """
        Create a new analysis session
        
        Args:
            file_path: Path to the L5X file
            
        Returns:
            Session ID
        """
        import uuid
        
        session_id = str(uuid.uuid4())
        session = PLCAnalysisSession(
            session_id=session_id,
            file_path=file_path,
            created_at=datetime.now(),
            status='pending'
        )
        
        self.active_sessions[session_id] = session
        self.logger.info(f"Created analysis session {session_id} for file: {file_path}")
        
        return session_id
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session status
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status dictionary or None if not found
        """
        session = self.active_sessions.get(session_id)
        if session:
            return session.to_dict()
        return None
    
    def analyze_l5x_file(self, file_path: str, session_id: str = None, 
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Analyze L5X file and return comprehensive results
        
        Args:
            file_path: Path to the L5X file
            session_id: Optional session ID
            progress_callback: Optional progress callback
            
        Returns:
            Analysis results
        """
        # Update session if provided
        if session_id and session_id in self.active_sessions:
            self.active_sessions[session_id].status = 'processing'
            self.active_sessions[session_id].progress = 0.0
        
        def wrapped_progress_callback(message: str, progress: float):
            """Wrapper for progress callback that updates session"""
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id].progress = progress
            
            if progress_callback:
                progress_callback(message, progress)
        
        try:
            # Process the file
            results = self.processing_service.process_l5x_file(
                file_path, 
                wrapped_progress_callback
            )
            
            # Update session with results
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id].status = 'completed' if results['success'] else 'failed'
                self.active_sessions[session_id].results = results
                self.active_sessions[session_id].progress = 100.0
                if not results['success']:
                    self.active_sessions[session_id].error_message = '; '.join(results.get('error_summary', []))
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error analyzing L5X file: {error_msg}")
            
            # Update session with error
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id].status = 'failed'
                self.active_sessions[session_id].error_message = error_msg
                self.active_sessions[session_id].progress = 0.0
            
            return {
                'success': False,
                'error_summary': [error_msg],
                'timestamp': datetime.now(),
                'file_path': file_path
            }
    
    def extract_tags(self, analysis_results: Dict[str, Any]) -> TagCollection:
        """
        Extract tags from analysis results into a TagCollection
        
        Args:
            analysis_results: Results from analyze_l5x_file
            
        Returns:
            TagCollection with all extracted tags
        """
        collection = TagCollection(name="L5X File Tags")
        
        if not analysis_results.get('success', False):
            return collection
        
        final_data = analysis_results.get('final_data', {})
        extracted_data = final_data.get('extracted_data', {})
        detailed_data = extracted_data.get('detailed_data', {})
        
        # Extract controller tags
        controller_tags = detailed_data.get('controller_tags', [])
        for tag_data in controller_tags:
            try:
                tag = Tag(
                    name=tag_data.get('name', ''),
                    data_type=tag_data.get('data_type', 'UNKNOWN'),
                    scope=TagScope.CONTROLLER,
                    description=tag_data.get('description', ''),
                    value=tag_data.get('value'),
                    external_access=tag_data.get('external_access', 'Read/Write'),
                    constant=tag_data.get('constant', False),
                    array_dimensions=tag_data.get('array_dimensions', [])
                )
                collection.add_tag(tag)
            except Exception as e:
                self.logger.warning(f"Error creating controller tag: {e}")
        
        # Extract program tags
        programs = detailed_data.get('programs', [])
        for program in programs:
            program_name = program.get('name', '')
            program_tags = program.get('tags', [])
            
            for tag_data in program_tags:
                try:
                    tag = Tag(
                        name=tag_data.get('name', ''),
                        data_type=tag_data.get('data_type', 'UNKNOWN'),
                        scope=TagScope.PROGRAM,
                        description=tag_data.get('description', ''),
                        value=tag_data.get('value'),
                        program_name=program_name,
                        external_access=tag_data.get('external_access', 'Read/Write'),
                        constant=tag_data.get('constant', False),
                        array_dimensions=tag_data.get('array_dimensions', [])
                    )
                    collection.add_tag(tag)
                except Exception as e:
                    self.logger.warning(f"Error creating program tag: {e}")
        
        self.logger.info(f"Extracted {len(collection.tags)} tags from analysis results")
        return collection
    
    def generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a human-readable analysis summary
        
        Args:
            analysis_results: Results from analyze_l5x_file
            
        Returns:
            Summary dictionary
        """
        if not analysis_results.get('success', False):
            return {
                'success': False,
                'error': 'Analysis failed',
                'details': analysis_results.get('error_summary', [])
            }
        
        final_data = analysis_results.get('final_data', {})
        extracted_data = final_data.get('extracted_data', {})
        
        # Controller information
        controller = extracted_data.get('controller', {})
        
        # Statistics
        tags_summary = extracted_data.get('tags_summary', {})
        programs_summary = extracted_data.get('programs_summary', {})
        
        # Logic analysis
        logic_analysis = final_data.get('logic_analysis', {})
        
        summary = {
            'success': True,
            'file_info': {
                'path': analysis_results.get('file_path', 'Unknown'),
                'processing_time': analysis_results.get('total_execution_time', 0),
                'processed_at': analysis_results.get('timestamp', datetime.now()).isoformat() if hasattr(analysis_results.get('timestamp', ''), 'isoformat') else str(analysis_results.get('timestamp', ''))
            },
            'controller_info': {
                'name': controller.get('name', 'Unknown'),
                'type': controller.get('type', 'Unknown'),
                'firmware': controller.get('firmware', 'Unknown')
            },
            'system_statistics': {
                'total_programs': programs_summary.get('total_programs', 0),
                'total_routines': programs_summary.get('total_routines', 0),
                'total_tags': tags_summary.get('total_tags', 0),
                'controller_tags': tags_summary.get('controller_tags', 0),
                'program_tags': tags_summary.get('program_tags', 0)
            },
            'analysis_insights': logic_analysis.get('logic_insights', {})
        }
        
        return summary
    
    def validate_l5x_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate L5X file without full processing
        
        Args:
            file_path: Path to the L5X file
            
        Returns:
            Validation results
        """
        try:
            # Use the parser's validation method
            is_valid, message = self.processing_service.pipeline.parser.validate_l5x_file(file_path)
            
            file_info = {}
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                file_info = {
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            
            return {
                'valid': is_valid,
                'message': message,
                'file_path': file_path,
                'file_info': file_info,
                'validation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'valid': False,
                'message': f"Validation error: {str(e)}",
                'file_path': file_path,
                'file_info': {},
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Get information about processing capabilities
        
        Returns:
            Capabilities dictionary
        """
        return {
            'supported_formats': ['L5X'],
            'analysis_features': [
                'Controller information extraction',
                'Tag analysis and categorization',
                'Program and routine mapping',
                'I/O module detection',
                'Logic complexity analysis',
                'Documentation generation'
            ],
            'pipeline_steps': [
                'File validation',
                'XML parsing',
                'Data extraction',
                'Logic analysis',
                'Documentation generation'
            ],
            'service_status': 'active',
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up a completed session
        
        Args:
            session_id: Session ID to clean up
            
        Returns:
            True if session was cleaned up, False if not found
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Cleaned up session {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of all active sessions
        
        Returns:
            List of session dictionaries
        """
        return [session.to_dict() for session in self.active_sessions.values()]
