"""
Step 32: Reporting and Analytics
Advanced reporting system for comprehensive PLC project analysis and documentation
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from collections import defaultdict, Counter
import statistics

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Visualization libraries not available - using text-based reports")
    VISUALIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of reports that can be generated"""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_ANALYSIS = "technical_analysis"
    PATTERN_ANALYSIS = "pattern_analysis"
    SECURITY_ASSESSMENT = "security_assessment"
    MAINTENANCE_REPORT = "maintenance_report"
    OPTIMIZATION_ROADMAP = "optimization_roadmap"
    COMPLIANCE_AUDIT = "compliance_audit"
    PERFORMANCE_METRICS = "performance_metrics"

class ReportFormat(Enum):
    """Report output formats"""
    HTML = "html"
    PDF = "pdf"
    JSON = "json" 
    TEXT = "text"
    MARKDOWN = "markdown"
    EXCEL = "excel"

class MetricType(Enum):
    """Types of metrics that can be tracked"""
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    SAFETY = "safety"
    COMPLIANCE = "compliance"

@dataclass
class ReportMetric:
    """Individual metric within a report"""
    name: str
    value: Union[int, float, str]
    unit: str
    category: MetricType
    description: str
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    trend: Optional[str] = None  # up, down, stable
    historical_data: List[float] = field(default_factory=list)

@dataclass 
class ReportSection:
    """Section within a report"""
    title: str
    content: str
    metrics: List[ReportMetric] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    subsections: List['ReportSection'] = field(default_factory=list)

@dataclass
class AnalyticsReport:
    """Complete analytics report"""
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    project_info: Dict[str, Any]
    executive_summary: str
    sections: List[ReportSection]
    metrics: List[ReportMetric]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    format: ReportFormat = ReportFormat.HTML

class ReportingEngine:
    """Advanced reporting and analytics engine"""
    
    def __init__(self):
        """Initialize reporting engine"""
        self.report_templates = {}
        self.metric_calculators = {}
        self.visualization_cache = {}
        self.report_history = []
        
        # Analysis data storage
        self.analysis_data = defaultdict(list)
        self.trend_data = defaultdict(list)
        self.benchmark_data = {}
        
        # Configure reporting options
        self.auto_refresh = True
        self.cache_reports = True
        self.include_historical = True
        
        # Initialize report templates
        self._initialize_report_templates()
        self._initialize_metric_calculators()
        
        logger.info("Reporting engine initialized")
    
    def _initialize_report_templates(self):
        """Initialize standard report templates"""
        
        # Executive Summary Template
        self.report_templates[ReportType.EXECUTIVE_SUMMARY] = {
            'title': 'Executive Summary Report',
            'sections': [
                'project_overview',
                'key_metrics',
                'major_findings',
                'recommendations',
                'next_steps'
            ],
            'metrics': [
                'total_instructions',
                'code_quality_score',
                'security_rating',
                'optimization_potential'
            ]
        }
        
        # Technical Analysis Template
        self.report_templates[ReportType.TECHNICAL_ANALYSIS] = {
            'title': 'Technical Analysis Report',
            'sections': [
                'system_architecture',
                'instruction_analysis',
                'program_structure',
                'tag_utilization',
                'routine_complexity',
                'performance_analysis'
            ],
            'metrics': [
                'instruction_count',
                'routine_count',
                'tag_count',
                'complexity_metrics',
                'performance_metrics'
            ]
        }
        
        # Pattern Analysis Template
        self.report_templates[ReportType.PATTERN_ANALYSIS] = {
            'title': 'Pattern Analysis Report',
            'sections': [
                'pattern_summary',
                'best_practices',
                'anti_patterns',
                'optimization_opportunities',
                'pattern_distribution'
            ],
            'metrics': [
                'patterns_detected',
                'anti_patterns_found',
                'best_practice_ratio',
                'optimization_score'
            ]
        }
        
        # Security Assessment Template
        self.report_templates[ReportType.SECURITY_ASSESSMENT] = {
            'title': 'Security Assessment Report',
            'sections': [
                'security_overview',
                'vulnerability_analysis',
                'access_control',
                'security_recommendations',
                'compliance_status'
            ],
            'metrics': [
                'security_score',  
                'vulnerabilities_found',
                'hardcoded_values',
                'security_compliance'
            ]
        }
        
        logger.info(f"Initialized {len(self.report_templates)} report templates")
    
    def _initialize_metric_calculators(self):
        """Initialize metric calculation functions"""
        
        # Code Quality Metrics
        self.metric_calculators['code_quality_score'] = self._calculate_code_quality_score
        self.metric_calculators['complexity_score'] = self._calculate_complexity_score
        self.metric_calculators['maintainability_index'] = self._calculate_maintainability_index
        
        # Security Metrics
        self.metric_calculators['security_score'] = self._calculate_security_score
        self.metric_calculators['vulnerability_count'] = self._calculate_vulnerability_count
        
        # Performance Metrics
        self.metric_calculators['performance_score'] = self._calculate_performance_score
        self.metric_calculators['optimization_potential'] = self._calculate_optimization_potential
        
        # Pattern Metrics
        self.metric_calculators['pattern_density'] = self._calculate_pattern_density
        self.metric_calculators['best_practice_ratio'] = self._calculate_best_practice_ratio
        
        logger.info(f"Initialized {len(self.metric_calculators)} metric calculators")
    
    async def generate_report(self, 
                            report_type: ReportType,
                            project_data: Dict[str, Any],
                            format_type: ReportFormat = ReportFormat.HTML,
                            include_visualizations: bool = True) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        
        logger.info(f"Generating {report_type.value} report")
        
        report_id = str(uuid.uuid4())[:8]
        
        # Get report template
        template = self.report_templates.get(report_type, {})
        
        # Calculate metrics
        metrics = await self._calculate_metrics(project_data, template.get('metrics', []))
        
        # Generate sections
        sections = await self._generate_sections(project_data, template.get('sections', []))
        
        # Create executive summary
        executive_summary = await self._generate_executive_summary(project_data, metrics)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(project_data, metrics)
        
        # Create report
        report = AnalyticsReport(
            report_id=report_id,
            report_type=report_type,
            title=template.get('title', f'{report_type.value.replace("_", " ").title()} Report'),
            generated_at=datetime.now(),
            project_info=self._extract_project_info(project_data),
            executive_summary=executive_summary,
            sections=sections,
            metrics=metrics,
            recommendations=recommendations,
            format=format_type
        )
        
        # Add visualizations if requested
        if include_visualizations and VISUALIZATION_AVAILABLE:
            await self._add_visualizations(report, project_data)
        
        # Cache report
        if self.cache_reports:
            self.report_history.append(report)
        
        logger.info(f"Report {report_id} generated successfully")
        return report
    
    async def _calculate_metrics(self, project_data: Dict[str, Any], metric_names: List[str]) -> List[ReportMetric]:
        """Calculate metrics for the report"""
        
        metrics = []
        
        for metric_name in metric_names:
            try:
                calculator = self.metric_calculators.get(metric_name)
                if calculator:
                    metric = await calculator(project_data)
                    metrics.append(metric)
                else:
                    # Create placeholder metric
                    metric = ReportMetric(
                        name=metric_name,
                        value="N/A",
                        unit="",
                        category=MetricType.CODE_QUALITY,
                        description=f"Metric {metric_name} not available"
                    )
                    metrics.append(metric)
            
            except Exception as e:
                logger.warning(f"Failed to calculate metric {metric_name}: {e}")
                continue
        
        return metrics
    
    async def _generate_sections(self, project_data: Dict[str, Any], section_names: List[str]) -> List[ReportSection]:
        """Generate sections for the report"""
        
        sections = []
        
        for section_name in section_names:
            try:
                section = await self._generate_section(project_data, section_name)
                sections.append(section)
            except Exception as e:
                logger.warning(f"Failed to generate section {section_name}: {e}")
                continue
        
        return sections
    
    async def _generate_section(self, project_data: Dict[str, Any], section_name: str) -> ReportSection:
        """Generate individual report section"""
        
        if section_name == 'project_overview':
            return await self._generate_project_overview_section(project_data)
        elif section_name == 'key_metrics':
            return await self._generate_key_metrics_section(project_data)
        elif section_name == 'major_findings':
            return await self._generate_major_findings_section(project_data)
        elif section_name == 'system_architecture':
            return await self._generate_system_architecture_section(project_data)
        elif section_name == 'instruction_analysis':
            return await self._generate_instruction_analysis_section(project_data)
        elif section_name == 'pattern_summary':
            return await self._generate_pattern_summary_section(project_data)
        elif section_name == 'security_overview':
            return await self._generate_security_overview_section(project_data)
        else:
            # Default section
            return ReportSection(
                title=section_name.replace('_', ' ').title(),
                content=f"Analysis for {section_name} is not yet implemented.",
                recommendations=[f"Implement {section_name} analysis"]
            )
    
    async def _generate_project_overview_section(self, project_data: Dict[str, Any]) -> ReportSection:
        """Generate project overview section"""
        
        instructions = project_data.get('instructions', [])
        tags = project_data.get('tags', [])
        routines = project_data.get('routines', [])
        programs = project_data.get('programs', [])
        
        content = f"""
        ## Project Overview
        
        This PLC project contains {len(instructions)} instructions across {len(routines)} routines
        in {len(programs)} programs, utilizing {len(tags)} tags for control and monitoring.
        
        ### System Statistics:
        - **Instructions**: {len(instructions)}
        - **Routines**: {len(routines)}
        - **Programs**: {len(programs)}
        - **Tags**: {len(tags)}
        
        ### Project Structure:
        The project is organized into logical programs with dedicated routines for specific
        control functions. The tag structure supports efficient data organization and
        communication between system components.
        """
        
        metrics = [
            ReportMetric(
                name="total_instructions",
                value=len(instructions),
                unit="count",
                category=MetricType.CODE_QUALITY,
                description="Total number of instructions in project"
            ),
            ReportMetric(
                name="total_routines", 
                value=len(routines),
                unit="count",
                category=MetricType.CODE_QUALITY,
                description="Total number of routines in project"
            )
        ]
        
        recommendations = []
        if len(instructions) > 10000:
            recommendations.append("Consider modularizing large instruction sets")
        if len(routines) > 100:
            recommendations.append("Review routine organization for maintainability")
        
        return ReportSection(
            title="Project Overview",
            content=content.strip(),
            metrics=metrics,
            recommendations=recommendations
        )
    
    async def _generate_key_metrics_section(self, project_data: Dict[str, Any]) -> ReportSection:
        """Generate key metrics section"""
        
        # Calculate key performance indicators
        instructions = project_data.get('instructions', [])
        patterns = project_data.get('patterns_detected', [])
        anti_patterns = project_data.get('anti_patterns', [])
        
        # Calculate metrics
        instruction_complexity = self._calculate_average_complexity(instructions)
        pattern_coverage = len(patterns) / max(1, len(instructions)) * 100
        code_quality = max(0, 100 - len(anti_patterns) * 10)
        
        content = f"""
        ## Key Performance Indicators
        
        ### Code Quality Metrics:
        - **Instruction Complexity**: {instruction_complexity:.2f} (avg)
        - **Pattern Coverage**: {pattern_coverage:.1f}%
        - **Code Quality Score**: {code_quality:.0f}/100
        
        ### Pattern Analysis:
        - **Patterns Detected**: {len(patterns)}
        - **Anti-patterns Found**: {len(anti_patterns)}
        - **Best Practice Ratio**: {(len(patterns) / max(1, len(patterns) + len(anti_patterns)) * 100):.1f}%
        
        ### System Health:
        The system demonstrates {'good' if code_quality > 80 else 'fair' if code_quality > 60 else 'poor'} 
        overall code quality with {'high' if pattern_coverage > 50 else 'moderate' if pattern_coverage > 25 else 'low'} 
        pattern coverage.
        """
        
        metrics = [
            ReportMetric(
                name="code_quality_score",
                value=code_quality,
                unit="score",
                category=MetricType.CODE_QUALITY,
                description="Overall code quality assessment",
                threshold=80.0,
                status="good" if code_quality > 80 else "warning" if code_quality > 60 else "critical"
            ),
            ReportMetric(
                name="pattern_coverage",
                value=pattern_coverage,
                unit="percent", 
                category=MetricType.CODE_QUALITY,
                description="Percentage of instructions covered by patterns"
            )
        ]
        
        return ReportSection(
            title="Key Metrics",
            content=content.strip(),
            metrics=metrics
        )
    
    async def _generate_major_findings_section(self, project_data: Dict[str, Any]) -> ReportSection:
        """Generate major findings section"""
        
        patterns = project_data.get('patterns_detected', [])
        anti_patterns = project_data.get('anti_patterns', [])
        optimization_opportunities = project_data.get('optimization_opportunities', [])
        
        findings = []
        recommendations = []
        
        # Analyze patterns
        if patterns:
            pattern_types = Counter(p.get('pattern_type', 'unknown') for p in patterns)
            most_common = pattern_types.most_common(1)[0] if pattern_types else None
            if most_common:
                findings.append(f"Most common pattern type: {most_common[0]} ({most_common[1]} instances)")
        
        # Analyze anti-patterns
        if anti_patterns:
            findings.append(f"Found {len(anti_patterns)} anti-patterns requiring attention")
            recommendations.append("Prioritize refactoring anti-patterns to improve code quality")
        
        # Analyze optimizations
        if optimization_opportunities:
            high_priority = [op for op in optimization_opportunities if op.get('priority') == 'high']
            if high_priority:
                findings.append(f"Identified {len(high_priority)} high-priority optimization opportunities")
                recommendations.append("Focus on high-priority optimizations for maximum impact")
        
        content = f"""
        ## Major Findings
        
        ### Pattern Analysis Results:
        {chr(10).join(f"- {finding}" for finding in findings) if findings else "- No significant patterns identified"}
        
        ### Key Observations:
        - Pattern diversity indicates {'good' if len(set(p.get('pattern_type', '') for p in patterns)) > 3 else 'limited'} 
          architectural variety
        - Anti-pattern presence suggests {'moderate' if len(anti_patterns) < 5 else 'significant'} 
          technical debt
        - Optimization potential is {'high' if len(optimization_opportunities) > 10 else 'moderate'}
        
        ### Impact Assessment:
        The analysis reveals opportunities for improvement in code organization, 
        pattern consistency, and technical debt reduction.
        """
        
        return ReportSection(
            title="Major Findings",
            content=content.strip(),
            recommendations=recommendations
        )
    
    async def _generate_system_architecture_section(self, project_data: Dict[str, Any]) -> ReportSection:
        """Generate system architecture analysis section"""
        
        programs = project_data.get('programs', [])
        routines = project_data.get('routines', [])
        tags = project_data.get('tags', [])
        
        # Analyze program structure
        program_routine_map = defaultdict(list)
        for routine in routines:
            program_name = routine.get('program', 'Unknown')
            program_routine_map[program_name].append(routine)
        
        # Analyze tag scopes
        tag_scopes = Counter(tag.get('scope', 'unknown') for tag in tags)
        
        content = f"""
        ## System Architecture Analysis
        
        ### Program Organization:
        - **Total Programs**: {len(programs)}
        - **Average Routines per Program**: {len(routines) / max(1, len(programs)):.1f}
        
        ### Program Breakdown:
        {chr(10).join(f"- **{prog}**: {len(routines)} routines" for prog, routines in list(program_routine_map.items())[:5])}
        
        ### Tag Architecture:
        - **Controller Tags**: {tag_scopes.get('controller', 0)}
        - **Program Tags**: {tag_scopes.get('program', 0)}
        - **Local Tags**: {tag_scopes.get('local', 0)}
        
        ### Architecture Assessment:
        The system shows {'good' if len(programs) > 1 else 'basic'} modular organization with 
        {'appropriate' if tag_scopes.get('controller', 0) < len(tags) * 0.8 else 'excessive'} 
        use of global controller tags.
        """
        
        metrics = [
            ReportMetric(
                name="architectural_complexity",
                value=len(programs) * len(routines) / max(1, len(tags)),
                unit="ratio",
                category=MetricType.CODE_QUALITY,
                description="System architectural complexity indicator"
            )
        ]
        
        return ReportSection(
            title="System Architecture",
            content=content.strip(),
            metrics=metrics
        )
    
    async def _generate_instruction_analysis_section(self, project_data: Dict[str, Any]) -> ReportSection:
        """Generate instruction analysis section"""
        
        instructions = project_data.get('instructions', [])
        
        # Analyze instruction types
        instruction_types = Counter(inst.get('type', 'unknown') for inst in instructions)
        total_instructions = len(instructions)
        
        # Calculate complexity
        complex_instructions = ['JSR', 'JSS', 'SBR', 'RET', 'FOR', 'WHILE', 'REPEAT']
        complexity_count = sum(instruction_types.get(inst_type, 0) for inst_type in complex_instructions)
        complexity_ratio = complexity_count / max(1, total_instructions) * 100
        
        content = f"""
        ## Instruction Analysis
        
        ### Instruction Distribution:
        {chr(10).join(f"- **{inst_type}**: {count} ({count/total_instructions*100:.1f}%)" 
                     for inst_type, count in instruction_types.most_common(10))}
        
        ### Complexity Analysis:
        - **Total Instructions**: {total_instructions}
        - **Complex Instructions**: {complexity_count}
        - **Complexity Ratio**: {complexity_ratio:.1f}%
        
        ### Instruction Patterns:
        - Most used instruction: **{instruction_types.most_common(1)[0][0] if instruction_types else 'None'}**
        - Instruction diversity: {len(instruction_types)} unique types
        - Average instructions per routine: {total_instructions / max(1, len(project_data.get('routines', []))):.1f}
        """
        
        metrics = [
            ReportMetric(
                name="instruction_complexity_ratio",
                value=complexity_ratio,
                unit="percent",
                category=MetricType.CODE_QUALITY,
                description="Percentage of complex instructions",
                threshold=20.0,
                status="normal" if complexity_ratio < 20 else "warning"
            )
        ]
        
        return ReportSection(
            title="Instruction Analysis",
            content=content.strip(),
            metrics=metrics
        )
    
    async def _generate_pattern_summary_section(self, project_data: Dict[str, Any]) -> ReportSection:
        """Generate pattern analysis summary section"""
        
        patterns = project_data.get('patterns_detected', [])
        anti_patterns = project_data.get('anti_patterns', [])
        
        # Analyze pattern categories
        pattern_categories = Counter(p.get('category', 'unknown') for p in patterns)
        pattern_confidence = [p.get('confidence', 0) for p in patterns if 'confidence' in p]
        avg_confidence = statistics.mean(pattern_confidence) if pattern_confidence else 0
        
        content = f"""
        ## Pattern Analysis Summary
        
        ### Pattern Detection Results:
        - **Patterns Detected**: {len(patterns)}
        - **Anti-patterns Found**: {len(anti_patterns)}
        - **Average Confidence**: {avg_confidence:.1%}
        
        ### Pattern Categories:
        {chr(10).join(f"- **{category.replace('_', ' ').title()}**: {count} patterns" 
                     for category, count in pattern_categories.most_common())}
        
        ### Pattern Quality Assessment:
        - **Best Practice Ratio**: {len(patterns) / max(1, len(patterns) + len(anti_patterns)) * 100:.1f}%
        - **Pattern Coverage**: {len(patterns) / max(1, len(project_data.get('instructions', []))) * 100:.1f}%
        - **Detection Quality**: {'High' if avg_confidence > 0.8 else 'Medium' if avg_confidence > 0.6 else 'Low'}
        
        ### Recommendations:
        Based on pattern analysis, the system shows {'strong' if len(patterns) > len(anti_patterns) * 2 else 'moderate'} 
        adherence to best practices with opportunities for improvement in anti-pattern remediation.
        """
        
        recommendations = []
        if len(anti_patterns) > 0:
            recommendations.append(f"Address {len(anti_patterns)} anti-patterns to improve code quality")
        if avg_confidence < 0.7:
            recommendations.append("Review pattern matching accuracy and context")
        
        return ReportSection(
            title="Pattern Analysis Summary",
            content=content.strip(),
            recommendations=recommendations
        )
    
    async def _generate_security_overview_section(self, project_data: Dict[str, Any]) -> ReportSection:
        """Generate security assessment overview section"""
        
        instructions = project_data.get('instructions', [])
        tags = project_data.get('tags', [])
        
        # Security analysis
        hardcoded_values = self._count_hardcoded_values(instructions)
        exposed_tags = len([tag for tag in tags if tag.get('scope') == 'controller'])
        security_score = max(0, 100 - hardcoded_values * 5 - (exposed_tags / max(1, len(tags)) * 20))
        
        content = f"""
        ## Security Assessment Overview
        
        ### Security Metrics:
        - **Security Score**: {security_score:.0f}/100
        - **Hardcoded Values**: {hardcoded_values}
        - **Exposed Controller Tags**: {exposed_tags}
        
        ### Security Analysis:
        - **Data Exposure**: {'Low' if exposed_tags < len(tags) * 0.5 else 'High'} risk
        - **Configuration Security**: {'Good' if hardcoded_values < 10 else 'Poor'} practices
        - **Overall Assessment**: {'Secure' if security_score > 80 else 'Needs Improvement'}
        
        ### Security Recommendations:
        The system security posture is {'strong' if security_score > 80 else 'adequate' if security_score > 60 else 'weak'} 
        with {'minimal' if hardcoded_values < 5 else 'significant'} configuration security concerns.
        """
        
        recommendations = []
        if hardcoded_values > 5:
            recommendations.append("Replace hardcoded values with configurable parameters")
        if exposed_tags > len(tags) * 0.7:
            recommendations.append("Review tag scope assignments for security")
        
        metrics = [
            ReportMetric(
                name="security_score",
                value=security_score,
                unit="score",
                category=MetricType.SECURITY,
                description="Overall security assessment score",
                threshold=80.0,
                status="good" if security_score > 80 else "warning" if security_score > 60 else "critical"
            )
        ]
        
        return ReportSection(
            title="Security Overview",
            content=content.strip(),
            recommendations=recommendations,
            metrics=metrics
        )
    
    def _count_hardcoded_values(self, instructions: List[Dict[str, Any]]) -> int:
        """Count hardcoded values in instructions"""
        hardcoded_count = 0
        for instruction in instructions:
            operands = instruction.get('operands', [])
            for operand in operands:
                if isinstance(operand, str) and operand.isdigit():
                    hardcoded_count += 1
        return hardcoded_count
    
    def _calculate_average_complexity(self, instructions: List[Dict[str, Any]]) -> float:
        """Calculate average instruction complexity"""
        if not instructions:
            return 0.0
        
        complexity_weights = {
            'XIC': 1, 'XIO': 1, 'OTE': 1, 'OTL': 1, 'OTU': 1,
            'TON': 2, 'TOF': 2, 'RTO': 2,
            'ADD': 2, 'SUB': 2, 'MUL': 2, 'DIV': 2,
            'EQU': 2, 'GEQ': 2, 'LEQ': 2, 'NEQ': 2,
            'JSR': 4, 'JSS': 4, 'SBR': 3, 'RET': 2,
            'FOR': 5, 'WHILE': 5, 'REPEAT': 5
        }
        
        total_complexity = sum(complexity_weights.get(inst.get('type', ''), 3) for inst in instructions)
        return total_complexity / len(instructions)
    
    async def _generate_executive_summary(self, project_data: Dict[str, Any], metrics: List[ReportMetric]) -> str:
        """Generate executive summary"""
        
        instructions_count = len(project_data.get('instructions', []))
        patterns_count = len(project_data.get('patterns_detected', []))
        anti_patterns_count = len(project_data.get('anti_patterns', []))
        
        # Find key metrics
        code_quality_metric = next((m for m in metrics if m.name == 'code_quality_score'), None)
        security_metric = next((m for m in metrics if m.name == 'security_score'), None)
        
        code_quality = code_quality_metric.value if code_quality_metric else 'N/A'
        security_score = security_metric.value if security_metric else 'N/A'
        
        summary = f"""
        This comprehensive analysis of the PLC project reveals a system with {instructions_count} 
        instructions organized across multiple programs and routines. The analysis identified 
        {patterns_count} standard patterns and {anti_patterns_count} anti-patterns.
        
        Key findings include a code quality score of {code_quality} and a security rating of 
        {security_score}. The system demonstrates {'strong' if isinstance(code_quality, (int, float)) and code_quality > 80 else 'moderate'} 
        adherence to best practices with opportunities for optimization and improvement.
        
        Immediate attention should be focused on addressing anti-patterns and implementing 
        recommended security enhancements to improve overall system reliability and maintainability.
        """
        
        return summary.strip()
    
    async def _generate_recommendations(self, project_data: Dict[str, Any], metrics: List[ReportMetric]) -> List[str]:
        """Generate system-wide recommendations"""
        
        recommendations = []
        
        # Analyze metrics for recommendations
        for metric in metrics:
            if metric.status == 'critical':
                recommendations.append(f"Critical: Address {metric.name} - current value {metric.value}")
            elif metric.status == 'warning':
                recommendations.append(f"Warning: Monitor {metric.name} - current value {metric.value}")
        
        # Pattern-based recommendations
        patterns = project_data.get('patterns_detected', [])
        anti_patterns = project_data.get('anti_patterns', [])
        
        if len(anti_patterns) > 0:
            recommendations.append(f"Refactor {len(anti_patterns)} anti-patterns to improve code quality")
        
        if len(patterns) < len(project_data.get('instructions', [])) * 0.1:
            recommendations.append("Consider implementing more standardized patterns")
        
        # System-specific recommendations
        instructions = project_data.get('instructions', [])
        if len(instructions) > 5000:
            recommendations.append("Consider modularizing large instruction sets for better maintainability")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _extract_project_info(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key project information"""
        
        return {
            'instruction_count': len(project_data.get('instructions', [])),
            'routine_count': len(project_data.get('routines', [])),
            'program_count': len(project_data.get('programs', [])),
            'tag_count': len(project_data.get('tags', [])),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _add_visualizations(self, report: AnalyticsReport, project_data: Dict[str, Any]):
        """Add visualizations to report sections"""
        
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            # Create instruction type distribution chart
            instructions = project_data.get('instructions', [])
            if instructions:
                instruction_types = Counter(inst.get('type', 'unknown') for inst in instructions)
                
                # Add chart data to appropriate section
                for section in report.sections:
                    if section.title == "Instruction Analysis":
                        section.charts.append({
                            'type': 'bar',
                            'title': 'Instruction Type Distribution',
                            'data': dict(instruction_types.most_common(10))
                        })
            
            # Create pattern distribution chart
            patterns = project_data.get('patterns_detected', [])
            if patterns:
                pattern_types = Counter(p.get('pattern_type', 'unknown') for p in patterns)
                
                for section in report.sections:
                    if section.title == "Pattern Analysis Summary":
                        section.charts.append({
                            'type': 'pie',
                            'title': 'Pattern Type Distribution',
                            'data': dict(pattern_types)
                        })
        
        except Exception as e:
            logger.warning(f"Failed to add visualizations: {e}")
    
    # Metric calculation methods
    async def _calculate_code_quality_score(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate overall code quality score"""
        
        instructions = project_data.get('instructions', [])
        patterns = project_data.get('patterns_detected', [])
        anti_patterns = project_data.get('anti_patterns', [])
        
        # Base score calculation
        base_score = 70
        pattern_bonus = min(30, len(patterns) * 2)
        anti_pattern_penalty = len(anti_patterns) * 5
        
        score = max(0, min(100, base_score + pattern_bonus - anti_pattern_penalty))
        
        return ReportMetric(
            name="code_quality_score",
            value=score,
            unit="score",
            category=MetricType.CODE_QUALITY,
            description="Overall code quality assessment based on patterns and anti-patterns",
            threshold=80.0,
            status="good" if score > 80 else "warning" if score > 60 else "critical"
        )
    
    async def _calculate_security_score(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate security assessment score"""
        
        instructions = project_data.get('instructions', [])
        tags = project_data.get('tags', [])
        
        hardcoded_values = self._count_hardcoded_values(instructions)
        exposed_tags = len([tag for tag in tags if tag.get('scope') == 'controller'])
        
        base_score = 85
        hardcoded_penalty = hardcoded_values * 3
        exposure_penalty = (exposed_tags / max(1, len(tags))) * 15
        
        score = max(0, min(100, base_score - hardcoded_penalty - exposure_penalty))
        
        return ReportMetric(
            name="security_score",
            value=score,
            unit="score",
            category=MetricType.SECURITY,
            description="Security assessment based on hardcoded values and tag exposure",
            threshold=75.0,
            status="good" if score > 75 else "warning" if score > 50 else "critical"
        )
    
    async def _calculate_complexity_score(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate system complexity score"""
        
        instructions = project_data.get('instructions', [])
        routines = project_data.get('routines', [])
        
        avg_complexity = self._calculate_average_complexity(instructions)
        routine_complexity = len(instructions) / max(1, len(routines))
        
        complexity_score = (avg_complexity + routine_complexity) / 2
        
        return ReportMetric(
            name="complexity_score",
            value=complexity_score,
            unit="score",
            category=MetricType.CODE_QUALITY,
            description="System complexity based on instruction and routine analysis"
        )
    
    async def _calculate_maintainability_index(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate maintainability index"""
        
        patterns = project_data.get('patterns_detected', [])
        anti_patterns = project_data.get('anti_patterns', [])
        instructions = project_data.get('instructions', [])
        
        pattern_ratio = len(patterns) / max(1, len(instructions)) * 100
        anti_pattern_impact = len(anti_patterns) * 10
        
        maintainability = max(0, min(100, pattern_ratio * 2 - anti_pattern_impact + 50))
        
        return ReportMetric(
            name="maintainability_index",
            value=maintainability,
            unit="index",
            category=MetricType.MAINTAINABILITY,
            description="Code maintainability index based on patterns and structure"
        )
    
    async def _calculate_vulnerability_count(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate security vulnerability count"""
        
        instructions = project_data.get('instructions', [])
        hardcoded_count = self._count_hardcoded_values(instructions)
        
        return ReportMetric(
            name="vulnerability_count",
            value=hardcoded_count,
            unit="count",
            category=MetricType.SECURITY,
            description="Number of potential security vulnerabilities identified"
        )
    
    async def _calculate_performance_score(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate performance score"""
        
        instructions = project_data.get('instructions', [])
        routines = project_data.get('routines', [])
        
        instruction_efficiency = 100 - (len(instructions) / 1000)  # Penalize large instruction counts
        routine_organization = max(0, 100 - len(routines))  # Penalize too many routines
        
        performance_score = max(0, min(100, (instruction_efficiency + routine_organization) / 2))
        
        return ReportMetric(
            name="performance_score",
            value=performance_score,
            unit="score",
            category=MetricType.PERFORMANCE,
            description="System performance assessment based on organization and efficiency"
        )
    
    async def _calculate_optimization_potential(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate optimization potential"""
        
        optimization_opportunities = project_data.get('optimization_opportunities', [])
        anti_patterns = project_data.get('anti_patterns', [])
        
        optimization_score = len(optimization_opportunities) * 10 + len(anti_patterns) * 15
        capped_score = min(100, optimization_score)
        
        return ReportMetric(
            name="optimization_potential",
            value=capped_score,
            unit="score",
            category=MetricType.OPTIMIZATION,
            description="Potential for system optimization based on identified opportunities"
        )
    
    async def _calculate_pattern_density(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate pattern density"""
        
        patterns = project_data.get('patterns_detected', [])
        instructions = project_data.get('instructions', [])
        
        density = len(patterns) / max(1, len(instructions)) * 100
        
        return ReportMetric(
            name="pattern_density",
            value=density,
            unit="percent",
            category=MetricType.CODE_QUALITY,
            description="Percentage of instructions covered by recognized patterns"
        )
    
    async def _calculate_best_practice_ratio(self, project_data: Dict[str, Any]) -> ReportMetric:
        """Calculate best practice ratio"""
        
        patterns = project_data.get('patterns_detected', [])
        anti_patterns = project_data.get('anti_patterns', [])
        
        total_patterns = len(patterns) + len(anti_patterns)
        ratio = len(patterns) / max(1, total_patterns) * 100
        
        return ReportMetric(
            name="best_practice_ratio",
            value=ratio,
            unit="percent",
            category=MetricType.CODE_QUALITY,
            description="Ratio of best practices to anti-patterns"
        )
    
    async def export_report(self, report: AnalyticsReport, output_path: str) -> str:
        """Export report to specified format and path"""
        
        output_file = None
        
        try:
            if report.format == ReportFormat.JSON:
                output_file = f"{output_path}.json"
                with open(output_file, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
            
            elif report.format == ReportFormat.HTML:
                output_file = f"{output_path}.html"
                html_content = await self._generate_html_report(report)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            elif report.format == ReportFormat.MARKDOWN:
                output_file = f"{output_path}.md"
                md_content = await self._generate_markdown_report(report)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
            
            elif report.format == ReportFormat.TEXT:
                output_file = f"{output_path}.txt"
                text_content = await self._generate_text_report(report)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)
            
            logger.info(f"Report exported to {output_file}")
            return output_file
        
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            raise
    
    async def _generate_html_report(self, report: AnalyticsReport) -> str:
        """Generate HTML report content"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e9f4ff; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .critical {{ background-color: #ffe9e9; }}
                .warning {{ background-color: #fff4e9; }}
                .good {{ background-color: #e9ffe9; }}
                .recommendations {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007cba; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Report ID:</strong> {report.report_id}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{report.executive_summary}</p>
            </div>
        """
        
        # Add sections
        for section in report.sections:
            html_content += f"""
            <div class="section">
                <h2>{section.title}</h2>
                <div>{section.content}</div>
            """
            
            # Add metrics
            for metric in section.metrics:
                status_class = metric.status if metric.status in ['critical', 'warning', 'good'] else ''
                html_content += f"""
                <div class="metric {status_class}">
                    <strong>{metric.name.replace('_', ' ').title()}:</strong> {metric.value} {metric.unit}
                    <br><small>{metric.description}</small>
                </div>
                """
            
            # Add recommendations
            if section.recommendations:
                html_content += """
                <div class="recommendations">
                    <h3>Recommendations:</h3>
                    <ul>
                """
                for rec in section.recommendations:
                    html_content += f"<li>{rec}</li>"
                html_content += "</ul></div>"
            
            html_content += "</div>"
        
        # Add overall recommendations
        if report.recommendations:
            html_content += """
            <div class="section">
                <h2>System Recommendations</h2>
                <div class="recommendations">
                    <ul>
            """
            for rec in report.recommendations:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul></div></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    async def _generate_markdown_report(self, report: AnalyticsReport) -> str:
        """Generate Markdown report content"""
        
        md_content = f"""# {report.title}

**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}  
**Report ID:** {report.report_id}

## Executive Summary

{report.executive_summary}

"""
        
        # Add sections
        for section in report.sections:
            md_content += f"## {section.title}\n\n{section.content}\n\n"
            
            # Add metrics
            if section.metrics:
                md_content += "### Metrics\n\n"
                for metric in section.metrics:
                    status_indicator = "🔴" if metric.status == "critical" else "🟡" if metric.status == "warning" else "🟢"
                    md_content += f"- **{metric.name.replace('_', ' ').title()}:** {metric.value} {metric.unit} {status_indicator}\n"
                    md_content += f"  - {metric.description}\n"
                md_content += "\n"
            
            # Add recommendations
            if section.recommendations:
                md_content += "### Recommendations\n\n"
                for rec in section.recommendations:
                    md_content += f"- {rec}\n"
                md_content += "\n"
        
        # Add overall recommendations
        if report.recommendations:
            md_content += "## System Recommendations\n\n"
            for rec in report.recommendations:
                md_content += f"- {rec}\n"
        
        return md_content
    
    async def _generate_text_report(self, report: AnalyticsReport) -> str:
        """Generate plain text report content"""
        
        text_content = f"""{report.title}
{'=' * len(report.title)}

Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {report.report_id}

EXECUTIVE SUMMARY
-----------------
{report.executive_summary}

"""
        
        # Add sections
        for section in report.sections:
            text_content += f"{section.title.upper()}\n{'-' * len(section.title)}\n\n"
            text_content += f"{section.content}\n\n"
            
            # Add metrics
            if section.metrics:
                text_content += "Metrics:\n"
                for metric in section.metrics:
                    status_text = f" [{metric.status.upper()}]" if metric.status != "normal" else ""
                    text_content += f"  - {metric.name.replace('_', ' ').title()}: {metric.value} {metric.unit}{status_text}\n"
                    text_content += f"    {metric.description}\n"
                text_content += "\n"
            
            # Add recommendations
            if section.recommendations:
                text_content += "Recommendations:\n"
                for rec in section.recommendations:
                    text_content += f"  - {rec}\n"
                text_content += "\n"
        
        # Add overall recommendations
        if report.recommendations:
            text_content += "SYSTEM RECOMMENDATIONS\n----------------------\n"
            for rec in report.recommendations:
                text_content += f"- {rec}\n"
        
        return text_content
    
    def get_report_history(self) -> List[AnalyticsReport]:
        """Get historical reports"""
        return self.report_history.copy()
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics engine summary"""
        
        return {
            'total_reports_generated': len(self.report_history),
            'report_types': list(self.report_templates.keys()),
            'available_metrics': list(self.metric_calculators.keys()),
            'visualization_support': VISUALIZATION_AVAILABLE,
            'last_report': self.report_history[-1].generated_at.isoformat() if self.report_history else None
        }


# Convenience functions
async def create_reporting_engine() -> ReportingEngine:
    """Create and initialize reporting engine"""
    engine = ReportingEngine()
    logger.info("Reporting engine created successfully")
    return engine

async def generate_executive_summary_report(project_data: Dict[str, Any]) -> AnalyticsReport:
    """Generate executive summary report"""
    engine = await create_reporting_engine()
    return await engine.generate_report(ReportType.EXECUTIVE_SUMMARY, project_data)

async def generate_technical_analysis_report(project_data: Dict[str, Any]) -> AnalyticsReport:
    """Generate technical analysis report"""
    engine = await create_reporting_engine()
    return await engine.generate_report(ReportType.TECHNICAL_ANALYSIS, project_data)

async def generate_pattern_analysis_report(project_data: Dict[str, Any]) -> AnalyticsReport:
    """Generate pattern analysis report"""
    engine = await create_reporting_engine()
    return await engine.generate_report(ReportType.PATTERN_ANALYSIS, project_data)

async def generate_security_assessment_report(project_data: Dict[str, Any]) -> AnalyticsReport:
    """Generate security assessment report"""
    engine = await create_reporting_engine()
    return await engine.generate_report(ReportType.SECURITY_ASSESSMENT, project_data)

async def export_reports_to_directory(reports: List[AnalyticsReport], output_dir: str) -> List[str]:
    """Export multiple reports to directory"""
    output_files = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    engine = await create_reporting_engine()
    
    for report in reports:
        try:
            output_path = os.path.join(output_dir, f"{report.report_type.value}_{report.report_id}")
            output_file = await engine.export_report(report, output_path)
            output_files.append(output_file)
        except Exception as e:
            logger.error(f"Failed to export report {report.report_id}: {e}")
    
    return output_files


if __name__ == "__main__":
    # Example usage
    async def main():
        print("Step 32: Reporting and Analytics")
        print("=" * 40)
        
        # Create sample project data
        sample_data = {
            'instructions': [
                {'type': 'XIC', 'operands': ['Input1']},
                {'type': 'OTE', 'operands': ['Output1']},
                {'type': 'TON', 'operands': ['Timer1']}
            ],
            'tags': [
                {'name': 'Input1', 'scope': 'controller'},
                {'name': 'Output1', 'scope': 'controller'},
                {'name': 'Timer1', 'scope': 'program'}
            ],
            'routines': [
                {'name': 'MainRoutine', 'program': 'MainProgram'}
            ],
            'programs': [
                {'name': 'MainProgram'}
            ],
            'patterns_detected': [
                {'pattern_type': 'control', 'category': 'best_practice', 'confidence': 0.95}
            ],
            'anti_patterns': [],
            'optimization_opportunities': [
                {'type': 'consolidation', 'priority': 'medium'}
            ]
        }
        
        # Generate reports
        engine = await create_reporting_engine()
        
        executive_report = await engine.generate_report(ReportType.EXECUTIVE_SUMMARY, sample_data)
        print(f"Generated executive summary: {executive_report.report_id}")
        
        technical_report = await engine.generate_report(ReportType.TECHNICAL_ANALYSIS, sample_data)
        print(f"Generated technical analysis: {technical_report.report_id}")
        
        # Export example
        await engine.export_report(executive_report, "executive_summary_example")
        print("Report exported successfully")
    
    asyncio.run(main())
