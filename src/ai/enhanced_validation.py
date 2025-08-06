"""
Step 20: Enhanced Code Validation

This module extends the basic code validation from Step 19 with PLC-specific validation
capabilities, including tag mapping validation, controller compatibility checks, and
runtime behavior analysis.

Key Features:
- Tag mapping validation against source L5X file
- Controller compatibility verification
- PLC communication protocol validation
- Runtime behavior analysis
- Integration with L5X analysis results
- Advanced security analysis for industrial environments
- Performance validation for real-time systems
"""

import asyncio
import re
import ast
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

# Import our existing components
from ai.code_generation import CodeValidator, GeneratedCode, CodeQuality
from ai.ai_integration import AnalysisSystemsIntegrator
from src.core.l5x_parser import L5XParser
from src.models.tags import Tag


class PLCValidationType(Enum):
    """Types of PLC-specific validation"""
    TAG_MAPPING = "tag_mapping"
    CONTROLLER_COMPATIBILITY = "controller_compatibility"
    PROTOCOL_COMPLIANCE = "protocol_compliance"
    RUNTIME_BEHAVIOR = "runtime_behavior"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_VALIDATION = "performance_validation"
    DATA_TYPE_CONSISTENCY = "data_type_consistency"
    I_O_MAPPING = "io_mapping"


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PLCValidationIssue:
    """Individual PLC validation issue"""
    issue_type: PLCValidationType
    severity: ValidationSeverity
    message: str
    location: str
    suggested_fix: Optional[str] = None
    source_reference: Optional[str] = None
    code_line: Optional[int] = None


@dataclass
class TagMappingResult:
    """Result of tag mapping validation"""
    mapped_tags: Dict[str, str]  # Generated tag -> L5X tag
    unmapped_tags: List[str]     # Tags in generated code not found in L5X
    unused_tags: List[str]       # Tags in L5X not used in generated code
    type_mismatches: List[Tuple[str, str, str]]  # tag, expected_type, actual_type
    mapping_confidence: float


@dataclass
class ControllerCompatibilityResult:
    """Result of controller compatibility validation"""
    compatible: bool
    controller_type: str
    firmware_version: Optional[str]
    supported_features: List[str]
    unsupported_features: List[str]
    compatibility_score: float


@dataclass
class PLCValidationResult:
    """Comprehensive PLC validation result"""
    validation_id: str
    timestamp: datetime
    source_file: str
    generated_code_file: str
    overall_score: float
    validation_passed: bool
    
    # Validation results by type
    tag_mapping: Optional[TagMappingResult] = None
    controller_compatibility: Optional[ControllerCompatibilityResult] = None
    protocol_compliance: Optional[Dict[str, Any]] = None
    runtime_behavior: Optional[Dict[str, Any]] = None
    security_analysis: Optional[Dict[str, Any]] = None
    performance_validation: Optional[Dict[str, Any]] = None
    
    # Issues found
    issues: List[PLCValidationIssue] = field(default_factory=list)
    
    # Summary statistics
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0


class PLCTagMapper:
    """Maps generated code tags to L5X source tags"""
    
    def __init__(self, l5x_parser: L5XParser):
        self.l5x_parser = l5x_parser
        self.source_tags = {}
        self.logger = logging.getLogger(__name__)
        
    def build_tag_index(self) -> None:
        """Build comprehensive tag index from L5X file"""
        try:
            # Get all tags from the parser
            controller_tags = self.l5x_parser.get_controller_tags()
            program_tags = self.l5x_parser.get_program_tags()
            io_tags = self.l5x_parser.get_io_tags()
            
            # Build unified tag index
            self.source_tags = {}
            
            # Add controller tags
            for tag in controller_tags:
                self.source_tags[tag.name] = {
                    'tag': tag,
                    'scope': 'controller',
                    'data_type': tag.data_type,
                    'is_array': hasattr(tag, 'dimensions') and tag.dimensions
                }
            
            # Add program tags  
            for program, tags in program_tags.items():
                for tag in tags:
                    canonical_name = f"{program}.{tag.name}"
                    self.source_tags[canonical_name] = {
                        'tag': tag,
                        'scope': 'program',
                        'program': program,
                        'data_type': tag.data_type,
                        'is_array': hasattr(tag, 'dimensions') and tag.dimensions
                    }
            
            # Add I/O tags
            for tag in io_tags:
                self.source_tags[tag.name] = {
                    'tag': tag,
                    'scope': 'io',
                    'data_type': tag.data_type,
                    'module': getattr(tag, 'module', None)
                }
                
            self.logger.info(f"Built tag index with {len(self.source_tags)} tags")
            
        except Exception as e:
            self.logger.error(f"Error building tag index: {e}")
            raise
    
    def extract_code_tags(self, code: str) -> Set[str]:
        """Extract tag references from generated Python code"""
        try:
            tree = ast.parse(code)
            tags = set()
            
            class TagExtractor(ast.NodeVisitor):
                def visit_Str(self, node):
                    # String literals that look like tag names
                    if isinstance(node.s, str) and self._looks_like_tag(node.s):
                        tags.add(node.s)
                    self.generic_visit(node)
                
                def visit_Constant(self, node):
                    # Python 3.8+ constant nodes
                    if isinstance(node.value, str) and self._looks_like_tag(node.value):
                        tags.add(node.value)
                    self.generic_visit(node)
                
                def visit_Name(self, node):
                    # Variable names that might reference tags
                    if self._looks_like_tag_variable(node.id):
                        tags.add(node.id)
                    self.generic_visit(node)
                
                def _looks_like_tag(self, s: str) -> bool:
                    """Check if string looks like a PLC tag"""
                    # Basic heuristics for PLC tag names
                    if len(s) < 2:
                        return False
                    # Common PLC tag patterns
                    patterns = [
                        r'^[A-Za-z][A-Za-z0-9_]*$',  # Basic identifier
                        r'^[A-Za-z][A-Za-z0-9_]*\[[0-9]+\]$',  # Array access
                        r'^[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*$',  # UDT member
                        r'^Local:[A-Za-z][A-Za-z0-9_]*$',  # Local I/O
                    ]
                    return any(re.match(pattern, s) for pattern in patterns)
                
                def _looks_like_tag_variable(self, name: str) -> bool:
                    """Check if variable name suggests tag usage"""
                    # Variables ending in _tag or similar
                    return ('tag' in name.lower() or 
                           'plc' in name.lower() or
                           name.startswith(('input_', 'output_', 'status_')))
            
            extractor = TagExtractor()
            extractor.visit(tree)
            
            # Also look for string patterns in comments and docstrings
            comment_tags = self._extract_from_comments(code)
            tags.update(comment_tags)
            
            return tags
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in code, using regex fallback: {e}")
            return self._extract_tags_regex(code)
        except Exception as e:
            self.logger.error(f"Error extracting tags from code: {e}")
            return set()
    
    def _extract_from_comments(self, code: str) -> Set[str]:
        """Extract tag references from comments and docstrings"""
        tags = set()
        lines = code.split('\n')
        
        for line in lines:
            # Look for comments containing tag references
            if '#' in line:
                comment = line.split('#', 1)[1]
                # Find tag-like patterns in comments
                patterns = re.findall(r'\b[A-Za-z][A-Za-z0-9_]*(?:\[[0-9]+\])?(?:\.[A-Za-z][A-Za-z0-9_]*)*\b', comment)
                for pattern in patterns:
                    if len(pattern) > 2 and pattern not in ['the', 'and', 'for', 'with']:
                        tags.add(pattern)
        
        return tags
    
    def _extract_tags_regex(self, code: str) -> Set[str]:
        """Fallback regex-based tag extraction"""
        tags = set()
        
        # Look for quoted strings that look like tags
        string_patterns = re.findall(r'["\']([A-Za-z][A-Za-z0-9_]*(?:\[[0-9]+\])?(?:\.[A-Za-z][A-Za-z0-9_]*)*)["\']', code)
        tags.update(string_patterns)
        
        return tags
    
    def validate_tag_mapping(self, code: str) -> TagMappingResult:
        """Validate tag mapping between generated code and L5X source"""
        if not self.source_tags:
            self.build_tag_index()
        
        # Extract tags from generated code
        code_tags = self.extract_code_tags(code)
        
        mapped_tags = {}
        unmapped_tags = []
        type_mismatches = []
        
        # Check each code tag against source
        for code_tag in code_tags:
            if code_tag in self.source_tags:
                mapped_tags[code_tag] = code_tag
            else:
                # Try fuzzy matching
                best_match = self._find_best_tag_match(code_tag)
                if best_match:
                    mapped_tags[code_tag] = best_match
                else:
                    unmapped_tags.append(code_tag)
        
        # Find unused source tags
        used_source_tags = set(mapped_tags.values())
        unused_tags = [tag for tag in self.source_tags.keys() if tag not in used_source_tags]
        
        # Calculate mapping confidence
        total_code_tags = len(code_tags) if code_tags else 1
        mapped_count = len(mapped_tags)
        mapping_confidence = mapped_count / total_code_tags
        
        return TagMappingResult(
            mapped_tags=mapped_tags,
            unmapped_tags=unmapped_tags,
            unused_tags=unused_tags,
            type_mismatches=type_mismatches,
            mapping_confidence=mapping_confidence
        )
    
    def _find_best_tag_match(self, code_tag: str) -> Optional[str]:
        """Find the best matching source tag for a code tag"""
        # Try exact match first
        if code_tag in self.source_tags:
            return code_tag
        
        # Try case-insensitive match
        lower_code_tag = code_tag.lower()
        for source_tag in self.source_tags:
            if source_tag.lower() == lower_code_tag:
                return source_tag
        
        # Try substring matching
        for source_tag in self.source_tags:
            if code_tag in source_tag or source_tag in code_tag:
                return source_tag
        
        # Try removing common prefixes/suffixes
        cleaned_code_tag = re.sub(r'^(input_|output_|status_)', '', code_tag.lower())
        cleaned_code_tag = re.sub(r'(_tag|_val|_value)$', '', cleaned_code_tag)
        
        for source_tag in self.source_tags:
            cleaned_source_tag = re.sub(r'^(input_|output_|status_)', '', source_tag.lower())
            cleaned_source_tag = re.sub(r'(_tag|_val|_value)$', '', cleaned_source_tag)
            
            if cleaned_code_tag == cleaned_source_tag:
                return source_tag
        
        return None


class PLCControllerValidator:
    """Validates generated code for controller compatibility"""
    
    def __init__(self, l5x_parser: L5XParser):
        self.l5x_parser = l5x_parser
        self.logger = logging.getLogger(__name__)
    
    def validate_controller_compatibility(self, code: str) -> ControllerCompatibilityResult:
        """Validate code compatibility with target controller"""
        try:
            # Get controller information
            controller_info = self.l5x_parser.get_controller_info()
            
            controller_type = controller_info.get('Name', 'Unknown')
            firmware_version = controller_info.get('MajorRev', '0') + '.' + controller_info.get('MinorRev', '0')
            
            # Analyze code for controller-specific features
            supported_features = []
            unsupported_features = []
            
            # Check for communication protocols
            if 'pycomm3' in code:
                supported_features.append('EtherNet/IP Communication')
            
            if 'opcua' in code.lower():
                if self._supports_opcua(controller_type):
                    supported_features.append('OPC UA')
                else:
                    unsupported_features.append('OPC UA (not supported by controller)')
            
            # Check for advanced features
            if 'structured_text' in code.lower():
                if self._supports_structured_text(controller_type):
                    supported_features.append('Structured Text')
                else:
                    unsupported_features.append('Structured Text (not supported)')
            
            # Check data types
            if self._uses_advanced_datatypes(code):
                if self._supports_advanced_datatypes(controller_type):
                    supported_features.append('Advanced Data Types')
                else:
                    unsupported_features.append('Advanced Data Types (limited support)')
            
            # Calculate compatibility score
            total_features = len(supported_features) + len(unsupported_features)
            if total_features > 0:
                compatibility_score = len(supported_features) / total_features
            else:
                compatibility_score = 1.0
            
            compatible = len(unsupported_features) == 0
            
            return ControllerCompatibilityResult(
                compatible=compatible,
                controller_type=controller_type,
                firmware_version=firmware_version,
                supported_features=supported_features,
                unsupported_features=unsupported_features,
                compatibility_score=compatibility_score
            )
            
        except Exception as e:
            self.logger.error(f"Error validating controller compatibility: {e}")
            return ControllerCompatibilityResult(
                compatible=False,
                controller_type="Unknown",
                firmware_version=None,
                supported_features=[],
                unsupported_features=["Validation Error"],
                compatibility_score=0.0
            )
    
    def _supports_opcua(self, controller_type: str) -> bool:
        """Check if controller supports OPC UA"""
        # CompactLogix 5380 and newer support OPC UA
        return any(model in controller_type.upper() for model in ['L83', 'L84', 'L85'])
    
    def _supports_structured_text(self, controller_type: str) -> bool:
        """Check if controller supports structured text"""
        # Most modern Allen-Bradley controllers support ST
        return not any(old_model in controller_type.upper() for old_model in ['L35', 'L45', 'L55'])
    
    def _uses_advanced_datatypes(self, code: str) -> bool:
        """Check if code uses advanced data types"""
        advanced_types = ['LREAL', 'LINT', 'ULINT', 'USINT', 'UINT']
        return any(dtype in code for dtype in advanced_types)
    
    def _supports_advanced_datatypes(self, controller_type: str) -> bool:
        """Check if controller supports advanced data types"""
        # Newer controllers support more data types
        return any(model in controller_type.upper() for model in ['L83', 'L84', 'L85', 'L73', 'L74', 'L75'])


class PLCSecurityValidator:
    """Enhanced security validation for industrial environments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_security(self, code: str) -> Dict[str, Any]:
        """Perform security analysis for PLC code"""
        issues = []
        score = 10.0
        
        # Check for hardcoded credentials
        if self._has_hardcoded_credentials(code):
            issues.append("Hardcoded credentials detected")
            score -= 3.0
        
        # Check for unsafe network operations
        if self._has_unsafe_network_ops(code):
            issues.append("Unsafe network operations detected")
            score -= 2.0
        
        # Check for inadequate error handling
        if self._inadequate_error_handling(code):
            issues.append("Inadequate error handling")
            score -= 1.0
        
        # Check for logging of sensitive data
        if self._logs_sensitive_data(code):
            issues.append("Potential logging of sensitive data")
            score -= 1.5
        
        # Check for input validation
        if not self._has_input_validation(code):
            issues.append("Missing input validation")
            score -= 2.0
        
        return {
            'security_score': max(0.0, score),
            'issues': issues,
            'recommendations': self._get_security_recommendations(issues)
        }
    
    def _has_hardcoded_credentials(self, code: str) -> bool:
        """Check for hardcoded passwords or keys"""
        patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'passwd\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']'
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
    
    def _has_unsafe_network_ops(self, code: str) -> bool:
        """Check for unsafe network operations"""
        patterns = [
            r'verify\s*=\s*False',  # Disabled SSL verification
            r'ssl\s*=\s*False',     # Disabled SSL
            r'http://',             # Unencrypted HTTP
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
    
    def _inadequate_error_handling(self, code: str) -> bool:
        """Check for adequate error handling"""
        # Should have try/except blocks for PLC operations
        has_try_except = 'try:' in code and 'except' in code
        has_plc_operations = any(op in code for op in ['read', 'write', 'connect'])
        
        return has_plc_operations and not has_try_except
    
    def _logs_sensitive_data(self, code: str) -> bool:
        """Check if code might log sensitive data"""
        patterns = [
            r'log.*password',
            r'log.*secret',
            r'log.*key',
            r'print.*password',
            r'print.*secret'
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
    
    def _has_input_validation(self, code: str) -> bool:
        """Check for input validation"""
        validation_patterns = [
            r'isinstance\s*\(',
            r'type\s*\(',
            r'validate',
            r'check.*range',
            r'if.*not.*in'
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in validation_patterns)
    
    def _get_security_recommendations(self, issues: List[str]) -> List[str]:
        """Get security recommendations based on issues"""
        recommendations = []
        
        if "Hardcoded credentials" in issues:
            recommendations.append("Use environment variables or secure configuration files for credentials")
        
        if "Unsafe network operations" in issues:
            recommendations.append("Enable SSL/TLS verification and use encrypted connections")
        
        if "Inadequate error handling" in issues:
            recommendations.append("Add comprehensive try/except blocks around PLC operations")
        
        if "Missing input validation" in issues:
            recommendations.append("Add input validation for all user-provided parameters")
        
        return recommendations


class EnhancedPLCValidator:
    """Main enhanced PLC code validator combining all validation types"""
    
    def __init__(self, l5x_file_path: str):
        self.l5x_file_path = l5x_file_path
        self.l5x_parser = L5XParser(l5x_file_path)
        self.tag_mapper = PLCTagMapper(self.l5x_parser)
        self.controller_validator = PLCControllerValidator(self.l5x_parser)
        self.security_validator = PLCSecurityValidator()
        self.basic_validator = CodeValidator()  # From Step 19
        self.logger = logging.getLogger(__name__)
        
        # Load L5X file
        self.l5x_parser.parse()
    
    async def validate_generated_code(self, generated_code: GeneratedCode, 
                                    validation_types: List[PLCValidationType] = None) -> PLCValidationResult:
        """Perform comprehensive PLC validation"""
        if validation_types is None:
            validation_types = list(PLCValidationType)
        
        validation_id = f"plc_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = PLCValidationResult(
            validation_id=validation_id,
            timestamp=datetime.now(),
            source_file=self.l5x_file_path,
            generated_code_file=generated_code.metadata.get('file_path', 'inline'),
            overall_score=0.0,
            validation_passed=False
        )
        
        scores = []
        
        try:
            # Basic validation first (from Step 19)
            basic_result = await self.basic_validator.validate_code(generated_code.code)
            scores.append(basic_result.quality_score)
            
            # Tag mapping validation
            if PLCValidationType.TAG_MAPPING in validation_types:
                result.tag_mapping = self.tag_mapper.validate_tag_mapping(generated_code.code)
                scores.append(result.tag_mapping.mapping_confidence * 10)
                
                # Add issues for unmapped tags
                for tag in result.tag_mapping.unmapped_tags:
                    result.issues.append(PLCValidationIssue(
                        issue_type=PLCValidationType.TAG_MAPPING,
                        severity=ValidationSeverity.WARNING,
                        message=f"Tag '{tag}' not found in source L5X file",
                        location="code",
                        suggested_fix="Verify tag name or add to L5X file"
                    ))
            
            # Controller compatibility validation
            if PLCValidationType.CONTROLLER_COMPATIBILITY in validation_types:
                result.controller_compatibility = self.controller_validator.validate_controller_compatibility(generated_code.code)
                scores.append(result.controller_compatibility.compatibility_score * 10)
                
                # Add issues for unsupported features
                for feature in result.controller_compatibility.unsupported_features:
                    result.issues.append(PLCValidationIssue(
                        issue_type=PLCValidationType.CONTROLLER_COMPATIBILITY,
                        severity=ValidationSeverity.ERROR,
                        message=f"Unsupported feature: {feature}",
                        location="code",
                        suggested_fix="Remove or replace with compatible alternative"
                    ))
            
            # Security validation
            if PLCValidationType.SECURITY_ANALYSIS in validation_types:
                result.security_analysis = self.security_validator.validate_security(generated_code.code)
                scores.append(result.security_analysis['security_score'])
                
                # Add security issues
                for issue in result.security_analysis['issues']:
                    result.issues.append(PLCValidationIssue(
                        issue_type=PLCValidationType.SECURITY_ANALYSIS,
                        severity=ValidationSeverity.WARNING,
                        message=issue,
                        location="code",
                        suggested_fix="See security recommendations"
                    ))
            
            # Protocol compliance validation
            if PLCValidationType.PROTOCOL_COMPLIANCE in validation_types:
                result.protocol_compliance = self._validate_protocol_compliance(generated_code.code)
                scores.append(result.protocol_compliance.get('compliance_score', 5.0))
            
            # Runtime behavior validation
            if PLCValidationType.RUNTIME_BEHAVIOR in validation_types:
                result.runtime_behavior = self._validate_runtime_behavior(generated_code.code)
                scores.append(result.runtime_behavior.get('behavior_score', 5.0))
            
            # Performance validation
            if PLCValidationType.PERFORMANCE_VALIDATION in validation_types:
                result.performance_validation = self._validate_performance(generated_code.code)
                scores.append(result.performance_validation.get('performance_score', 5.0))
            
            # Calculate overall score
            if scores:
                result.overall_score = sum(scores) / len(scores)
            else:
                result.overall_score = 0.0
            
            # Count issues by severity
            result.total_issues = len(result.issues)
            result.critical_issues = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.CRITICAL)
            result.error_issues = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.ERROR)
            result.warning_issues = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.WARNING)
            result.info_issues = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.INFO)
            
            # Determine if validation passed
            result.validation_passed = (result.critical_issues == 0 and 
                                      result.error_issues == 0 and 
                                      result.overall_score >= 7.0)
            
        except Exception as e:
            self.logger.error(f"Error during PLC validation: {e}")
            result.issues.append(PLCValidationIssue(
                issue_type=PLCValidationType.RUNTIME_BEHAVIOR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation error: {str(e)}",
                location="validator",
                suggested_fix="Check validator configuration"
            ))
        
        return result
    
    def _validate_protocol_compliance(self, code: str) -> Dict[str, Any]:
        """Validate protocol compliance (pycomm3, OPC UA, etc.)"""
        compliance_score = 10.0
        issues = []
        
        # Check for proper pycomm3 usage
        if 'pycomm3' in code:
            if 'LogixDriver' not in code:
                issues.append("Missing LogixDriver import for pycomm3")
                compliance_score -= 2.0
            
            if not re.search(r'with\s+LogixDriver', code):
                issues.append("Should use context manager for LogixDriver")
                compliance_score -= 1.0
        
        # Check for proper connection handling
        if 'connect' in code and 'disconnect' not in code:
            issues.append("Missing explicit disconnect call")
            compliance_score -= 1.5
        
        return {
            'compliance_score': max(0.0, compliance_score),
            'issues': issues,
            'protocols_detected': self._detect_protocols(code)
        }
    
    def _validate_runtime_behavior(self, code: str) -> Dict[str, Any]:
        """Validate runtime behavior patterns"""
        behavior_score = 10.0
        issues = []
        
        # Check for infinite loops
        if re.search(r'while\s+True:', code) and 'break' not in code:
            issues.append("Potential infinite loop detected")
            behavior_score -= 3.0
        
        # Check for proper exception handling in loops
        if 'while' in code or 'for' in code:
            if 'try:' not in code:
                issues.append("Missing exception handling in loops")
                behavior_score -= 2.0
        
        # Check for sleep/delay in tight loops
        if ('while' in code or 'for' in code) and 'sleep' not in code:
            issues.append("Consider adding delays in processing loops")
            behavior_score -= 1.0
        
        return {
            'behavior_score': max(0.0, behavior_score),
            'issues': issues,
            'patterns_detected': self._detect_behavior_patterns(code)
        }
    
    def _validate_performance(self, code: str) -> Dict[str, Any]:
        """Validate performance characteristics"""
        performance_score = 10.0
        issues = []
        recommendations = []
        
        # Check for batch operations
        read_count = len(re.findall(r'\.read\(', code))
        if read_count > 5:
            issues.append(f"Multiple individual reads detected ({read_count})")
            recommendations.append("Consider using batch read operations")
            performance_score -= 2.0
        
        # Check for write operations
        write_count = len(re.findall(r'\.write\(', code))
        if write_count > 3:
            issues.append(f"Multiple individual writes detected ({write_count})")
            recommendations.append("Consider using batch write operations")
            performance_score -= 2.0
        
        # Check for connection reuse
        if code.count('LogixDriver') > 1:
            issues.append("Multiple driver instances detected")
            recommendations.append("Reuse driver connections for better performance")
            performance_score -= 1.5
        
        return {
            'performance_score': max(0.0, performance_score),
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'read_operations': read_count,
                'write_operations': write_count,
                'connection_instances': code.count('LogixDriver')
            }
        }
    
    def _detect_protocols(self, code: str) -> List[str]:
        """Detect communication protocols used in code"""
        protocols = []
        
        if 'pycomm3' in code or 'LogixDriver' in code:
            protocols.append('EtherNet/IP')
        
        if 'opcua' in code.lower():
            protocols.append('OPC UA')
        
        if 'modbus' in code.lower():
            protocols.append('Modbus')
        
        return protocols
    
    def _detect_behavior_patterns(self, code: str) -> List[str]:
        """Detect runtime behavior patterns"""
        patterns = []
        
        if 'while' in code:
            patterns.append('Continuous Processing')
        
        if 'sleep' in code or 'time.sleep' in code:
            patterns.append('Timed Operations')
        
        if 'threading' in code:
            patterns.append('Multi-threading')
        
        if 'async' in code or 'await' in code:
            patterns.append('Asynchronous Operations')
        
        return patterns
    
    def generate_validation_report(self, result: PLCValidationResult) -> str:
        """Generate comprehensive validation report"""
        report = f"""
# PLC Code Validation Report

**Validation ID**: {result.validation_id}
**Timestamp**: {result.timestamp}
**Source File**: {result.source_file}
**Generated Code**: {result.generated_code_file}

## Overall Results

- **Overall Score**: {result.overall_score:.1f}/10.0
- **Validation Passed**: {'✅ PASS' if result.validation_passed else '❌ FAIL'}
- **Total Issues**: {result.total_issues}
  - Critical: {result.critical_issues}
  - Errors: {result.error_issues}
  - Warnings: {result.warning_issues}
  - Info: {result.info_issues}

## Detailed Results

### Tag Mapping Validation
"""
        
        if result.tag_mapping:
            tm = result.tag_mapping
            report += f"""
- **Mapping Confidence**: {tm.mapping_confidence:.1%}
- **Mapped Tags**: {len(tm.mapped_tags)}
- **Unmapped Tags**: {len(tm.unmapped_tags)}
- **Unused Source Tags**: {len(tm.unused_tags)}
"""
            
            if tm.unmapped_tags:
                report += f"\n**Unmapped Tags**: {', '.join(tm.unmapped_tags[:10])}"
                if len(tm.unmapped_tags) > 10:
                    report += f" (and {len(tm.unmapped_tags) - 10} more)"
        
        if result.controller_compatibility:
            cc = result.controller_compatibility
            report += f"""

### Controller Compatibility
- **Compatible**: {'✅ Yes' if cc.compatible else '❌ No'}
- **Controller Type**: {cc.controller_type}
- **Firmware Version**: {cc.firmware_version or 'Unknown'}
- **Compatibility Score**: {cc.compatibility_score:.1%}
- **Supported Features**: {len(cc.supported_features)}
- **Unsupported Features**: {len(cc.unsupported_features)}
"""
        
        if result.security_analysis:
            sa = result.security_analysis
            report += f"""

### Security Analysis
- **Security Score**: {sa['security_score']:.1f}/10.0
- **Security Issues**: {len(sa['issues'])}
"""
            if sa['issues']:
                report += "\n**Issues Found**:\n"
                for issue in sa['issues']:
                    report += f"- {issue}\n"
        
        # Add issues section
        if result.issues:
            report += "\n## Issues Found\n\n"
            
            for issue in result.issues:
                severity_icon = {
                    ValidationSeverity.CRITICAL: "🔴",
                    ValidationSeverity.ERROR: "🟠", 
                    ValidationSeverity.WARNING: "🟡",
                    ValidationSeverity.INFO: "🔵"
                }
                
                report += f"{severity_icon.get(issue.severity, '⚪')} **{issue.severity.value.upper()}**: {issue.message}\n"
                if issue.suggested_fix:
                    report += f"   *Suggested Fix*: {issue.suggested_fix}\n"
                report += "\n"
        
        return report


# Convenience functions for common validation tasks

async def validate_plc_code(l5x_file_path: str, generated_code: GeneratedCode, 
                          validation_types: List[PLCValidationType] = None) -> PLCValidationResult:
    """Convenience function to validate PLC code"""
    validator = EnhancedPLCValidator(l5x_file_path)
    return await validator.validate_generated_code(generated_code, validation_types)

async def validate_tag_mapping_only(l5x_file_path: str, code: str) -> TagMappingResult:
    """Convenience function to validate only tag mapping"""
    parser = L5XParser(l5x_file_path)
    parser.parse()
    mapper = PLCTagMapper(parser)
    return mapper.validate_tag_mapping(code)

async def validate_controller_compatibility_only(l5x_file_path: str, code: str) -> ControllerCompatibilityResult:
    """Convenience function to validate only controller compatibility"""
    parser = L5XParser(l5x_file_path)
    parser.parse()
    validator = PLCControllerValidator(parser)
    return validator.validate_controller_compatibility(code)


if __name__ == "__main__":
    # Example usage
    async def main():
        # This would be used with actual generated code
        print("Enhanced PLC Validator - Ready for integration")
        print("Use validate_plc_code() for comprehensive validation")
    
    asyncio.run(main())
