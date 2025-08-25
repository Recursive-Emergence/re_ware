"""
Frames Module for RE_ware
=========================

Concrete frame implementations that provide structured views of project state
without requiring LLM calls. Built directly from ontology data each tick.

Frames available:
- Quality: Test coverage, build status, code quality metrics
- Delivery: Release readiness, deployment status, milestone progress
- Architecture: Component relationships, dependency health, technical debt
- Risk: Security issues, performance problems, compliance gaps
- Security: Vulnerabilities, access control, audit trail
- Performance: Benchmarks, profiling data, resource usage
- Team: Contributor activity, code review metrics, knowledge distribution
- Decision: Recent decisions, pending approvals, decision rationale
- Dependency: External dependencies, version conflicts, security updates
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

from .ontology import NodeType, RelationType, OntologyPhenotype


@dataclass
class FrameData:
    """Base structure for all frames"""
    frame_type: str
    generated_at: float
    schema_version: str = "1.0"
    project_id: str = "project:root"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class QualityFrame(FrameData):
    """Quality assurance frame"""
    
    # Test metrics
    total_tests: int = 0
    passing_tests: int = 0
    failing_tests: int = 0
    skipped_tests: int = 0
    test_coverage: float = 0.0
    
    # Build metrics
    total_builds: int = 0
    successful_builds: int = 0
    failed_builds: int = 0
    latest_build_status: str = "unknown"
    
    # Code quality
    open_bugs: int = 0
    critical_issues: int = 0
    technical_debt_count: int = 0
    code_review_pending: int = 0
    
    # Quality gates
    quality_gate_status: str = "unknown"  # pass, fail, pending
    quality_score: float = 0.0  # 0.0 to 1.0
    blockers: List[str] = None
    
    def __post_init__(self):
        if self.blockers is None:
            self.blockers = []


@dataclass 
class DeliveryFrame(FrameData):
    """Delivery and release readiness frame"""
    
    # Release readiness
    release_readiness_score: float = 0.0  # 0.0 to 1.0
    blockers_count: int = 0
    critical_bugs: int = 0
    
    # Requirements and features
    total_requirements: int = 0
    implemented_requirements: int = 0
    verified_requirements: int = 0
    coverage_ratio: float = 0.0
    
    # Milestones and progress
    current_milestone: str = "unknown"
    milestone_progress: float = 0.0
    
    # Deployment status
    deployment_status: str = "unknown"
    last_deployment: Optional[str] = None
    
    # Quality gates for delivery
    all_tests_passing: bool = False
    security_scan_clean: bool = False
    performance_acceptable: bool = False
    documentation_complete: bool = False


@dataclass
class ArchitectureFrame(FrameData):
    """Architecture and component health frame"""
    
    # Component overview
    total_components: int = 0
    healthy_components: int = 0
    components_with_issues: int = 0
    
    # Dependencies
    total_dependencies: int = 0
    outdated_dependencies: int = 0
    vulnerable_dependencies: int = 0
    
    # Architecture health
    coupling_score: float = 0.0  # Lower is better
    cohesion_score: float = 0.0  # Higher is better
    complexity_score: float = 0.0
    
    # Technical debt
    debt_ratio: float = 0.0
    refactoring_candidates: List[str] = None
    
    def __post_init__(self):
        if self.refactoring_candidates is None:
            self.refactoring_candidates = []


@dataclass
class RiskFrame(FrameData):
    """Risk assessment frame"""
    
    # Overall risk assessment
    risk_level: str = "unknown"  # low, medium, high, critical
    risk_score: float = 0.0  # 0.0 to 1.0
    
    # Security risks
    security_vulnerabilities: int = 0
    critical_vulnerabilities: int = 0
    
    # Operational risks
    single_points_of_failure: int = 0
    undocumented_components: int = 0
    key_person_dependencies: int = 0
    
    # Quality risks
    untested_code_ratio: float = 0.0
    build_fragility_score: float = 0.0
    
    # Risk mitigation
    risk_items: List[Dict[str, Any]] = None
    mitigation_plans: List[str] = None
    
    def __post_init__(self):
        if self.risk_items is None:
            self.risk_items = []
        if self.mitigation_plans is None:
            self.mitigation_plans = []


class FrameBuilder:
    """Builds concrete frames from ontology state"""
    
    def __init__(self, ontology: OntologyPhenotype):
        self.ontology = ontology
        self.generated_at = time.time()
    
    def build_quality_frame(self) -> QualityFrame:
        """Build quality assurance frame from ontology"""
        frame = QualityFrame(frame_type="quality", generated_at=self.generated_at)
        
        # Count test nodes and their status
        test_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.TEST]
        frame.total_tests = len(test_nodes)
        
        for test in test_nodes:
            status = test.content.get("status", "unknown").lower()
            if status in ["passed", "pass", "ok", "success"]:
                frame.passing_tests += 1
            elif status in ["failed", "fail", "error", "failure"]:
                frame.failing_tests += 1
            elif status in ["skipped", "skip", "ignored"]:
                frame.skipped_tests += 1
        
        # Count build nodes and their status
        build_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.BUILD]
        frame.total_builds = len(build_nodes)
        
        latest_build = None
        if build_nodes:
            # Sort by timestamp to get latest
            build_nodes.sort(key=lambda n: n.content.get("updated_at", ""), reverse=True)
            latest_build = build_nodes[0]
            frame.latest_build_status = latest_build.content.get("conclusion", "unknown")
            
            for build in build_nodes:
                conclusion = build.content.get("conclusion", "unknown").lower()
                if conclusion == "success":
                    frame.successful_builds += 1
                elif conclusion in ["failure", "cancelled", "timeout"]:
                    frame.failed_builds += 1
        
        # Get coverage from coverage nodes
        coverage_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.COVERAGE]
        if coverage_nodes:
            # Use the most recent coverage report
            coverage_nodes.sort(key=lambda n: n.content.get("timestamp", ""), reverse=True)
            latest_coverage = coverage_nodes[0]
            frame.test_coverage = latest_coverage.content.get("line_coverage", 0.0)
        
        # Count bugs and issues
        bug_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.BUG]
        frame.open_bugs = len(bug_nodes)
        
        # Count critical issues (P0 criticality)
        frame.critical_issues = len([
            n for n in self.ontology.nodes.values()
            if getattr(n.state, 'criticality', None) == 'P0'
        ])
        
        # Count technical debt
        debt_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.TECHNICALDEBT]
        frame.technical_debt_count = len(debt_nodes)
        
        # Calculate quality score (0.0 to 1.0)
        quality_factors = []
        
        # Test pass rate
        if frame.total_tests > 0:
            test_pass_rate = frame.passing_tests / frame.total_tests
            quality_factors.append(test_pass_rate)
        
        # Coverage score
        coverage_score = min(frame.test_coverage / 80.0, 1.0)  # 80% target
        quality_factors.append(coverage_score)
        
        # Build success rate
        if frame.total_builds > 0:
            build_success_rate = frame.successful_builds / frame.total_builds
            quality_factors.append(build_success_rate)
        
        # Issue penalty
        issue_penalty = max(0.0, 1.0 - (frame.critical_issues * 0.2))
        quality_factors.append(issue_penalty)
        
        # Average quality factors
        frame.quality_score = sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
        
        # Determine quality gate status
        if frame.quality_score >= 0.8 and frame.critical_issues == 0:
            frame.quality_gate_status = "pass"
        elif frame.quality_score >= 0.6:
            frame.quality_gate_status = "warning"
        else:
            frame.quality_gate_status = "fail"
        
        # Identify blockers
        if frame.failing_tests > 0:
            frame.blockers.append(f"{frame.failing_tests} failing tests")
        if frame.critical_issues > 0:
            frame.blockers.append(f"{frame.critical_issues} critical issues")
        if frame.latest_build_status in ["failure", "cancelled"]:
            frame.blockers.append("Latest build failed")
        
        return frame
    
    def build_delivery_frame(self) -> DeliveryFrame:
        """Build delivery readiness frame from ontology"""
        frame = DeliveryFrame(frame_type="delivery", generated_at=self.generated_at)
        
        # Get requirement coverage from phi signals
        phi_signals = self.ontology.phi_signals()
        frame.coverage_ratio = phi_signals.get("coverage_ratio", 0.0)
        frame.total_requirements = len([
            n for n in self.ontology.nodes.values()
            if n.type == NodeType.REQUIREMENT
        ])
        
        # Count implemented and verified requirements
        requirement_edges = [
            e for e in self.ontology.edges.values()
            if e.relation in [RelationType.IMPLEMENTS, RelationType.VERIFIES]
        ]
        
        implemented_reqs = set()
        verified_reqs = set()
        
        for edge in requirement_edges:
            if edge.relation == RelationType.IMPLEMENTS:
                if edge.to_node in self.ontology.nodes:
                    to_node = self.ontology.nodes[edge.to_node]
                    if to_node.type == NodeType.REQUIREMENT:
                        implemented_reqs.add(edge.to_node)
            elif edge.relation == RelationType.VERIFIES:
                if edge.to_node in self.ontology.nodes:
                    to_node = self.ontology.nodes[edge.to_node]
                    if to_node.type == NodeType.REQUIREMENT:
                        verified_reqs.add(edge.to_node)
        
        frame.implemented_requirements = len(implemented_reqs)
        frame.verified_requirements = len(verified_reqs)
        
        # Get project status for milestone tracking
        project_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.PROJECT]
        if project_nodes:
            project = project_nodes[0]
            frame.current_milestone = project.state.status.name if hasattr(project.state.status, 'name') else str(project.state.status)
        
        # Calculate milestone progress
        if frame.total_requirements > 0:
            frame.milestone_progress = frame.verified_requirements / frame.total_requirements
        
        # Count critical bugs as blockers
        frame.critical_bugs = len([
            n for n in self.ontology.nodes.values()
            if n.type == NodeType.BUG and getattr(n.state, 'criticality', None) == 'P0'
        ])
        frame.blockers_count = frame.critical_bugs
        
        # Quality gates for delivery
        build_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.BUILD]
        if build_nodes:
            latest_build = max(build_nodes, key=lambda n: n.content.get("updated_at", ""))
            frame.all_tests_passing = latest_build.content.get("conclusion") == "success"
        
        # Security scan status (placeholder - would need security sensor)
        frame.security_scan_clean = frame.critical_bugs == 0  # Simplified
        
        # Documentation completeness (simplified heuristic)
        doc_nodes = [
            n for n in self.ontology.nodes.values()
            if n.type in [NodeType.TECHNICALDOC, NodeType.USERDOC, NodeType.APIDOC]
        ]
        code_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.CODEMODULE]
        
        if code_nodes:
            doc_ratio = len(doc_nodes) / len(code_nodes)
            frame.documentation_complete = doc_ratio >= 0.5  # 50% threshold
        
        # Calculate release readiness score
        readiness_factors = []
        readiness_factors.append(frame.coverage_ratio)
        readiness_factors.append(1.0 if frame.all_tests_passing else 0.0)
        readiness_factors.append(1.0 if frame.security_scan_clean else 0.0)
        readiness_factors.append(1.0 if frame.documentation_complete else 0.0)
        readiness_factors.append(max(0.0, 1.0 - (frame.critical_bugs * 0.25)))
        
        frame.release_readiness_score = sum(readiness_factors) / len(readiness_factors)
        
        return frame
    
    def build_architecture_frame(self) -> ArchitectureFrame:
        """Build architecture health frame from ontology"""
        frame = ArchitectureFrame(frame_type="architecture", generated_at=self.generated_at)
        
        # Count components
        component_types = [NodeType.COMPONENT, NodeType.SERVICE, NodeType.CODEMODULE]
        components = [
            n for n in self.ontology.nodes.values()
            if n.type in component_types
        ]
        frame.total_components = len(components)
        
        # Count healthy components (no critical issues)
        healthy_count = 0
        for comp in components:
            if getattr(comp.state, 'criticality', 'P2') not in ['P0', 'P1']:
                healthy_count += 1
        
        frame.healthy_components = healthy_count
        frame.components_with_issues = frame.total_components - healthy_count
        
        # Count dependencies
        dep_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.DEPENDENCY_SPEC]
        frame.total_dependencies = len(dep_nodes)
        
        # Count technical debt for debt ratio
        debt_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.TECHNICALDEBT]
        if frame.total_components > 0:
            frame.debt_ratio = len(debt_nodes) / frame.total_components
        
        # Calculate coupling score (simplified - based on edges per node)
        total_edges = len(self.ontology.edges)
        total_nodes = len(self.ontology.nodes)
        if total_nodes > 0:
            frame.coupling_score = total_edges / total_nodes  # Higher = more coupled
        
        # Identify refactoring candidates (high debt, high complexity)
        for comp in components:
            if getattr(comp.state, 'criticality', None) == 'P1':
                frame.refactoring_candidates.append(comp.title)
        
        return frame
    
    def build_risk_frame(self) -> RiskFrame:
        """Build risk assessment frame from ontology"""
        frame = RiskFrame(frame_type="risk", generated_at=self.generated_at)
        
        # Count security vulnerabilities
        vuln_nodes = [
            n for n in self.ontology.nodes.values()
            if n.type == NodeType.BUG and "security" in n.content.get("tags", [])
        ]
        frame.security_vulnerabilities = len(vuln_nodes)
        
        # Count critical vulnerabilities
        frame.critical_vulnerabilities = len([
            n for n in vuln_nodes
            if getattr(n.state, 'criticality', None) == 'P0'
        ])
        
        # Calculate untested code ratio
        test_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.TEST]
        code_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.CODEMODULE]
        
        if code_nodes:
            # Simplified: assume each test covers some code
            tested_ratio = min(len(test_nodes) / len(code_nodes), 1.0)
            frame.untested_code_ratio = 1.0 - tested_ratio
        
        # Count undocumented components
        doc_nodes = [
            n for n in self.ontology.nodes.values()
            if n.type in [NodeType.TECHNICALDOC, NodeType.USERDOC, NodeType.APIDOC]
        ]
        
        documented_components = set()
        for edge in self.ontology.edges.values():
            if edge.relation == RelationType.DOCUMENTS:
                documented_components.add(edge.to_node)
        
        total_components = len([
            n for n in self.ontology.nodes.values()
            if n.type in [NodeType.CODEMODULE, NodeType.COMPONENT, NodeType.SERVICE]
        ])
        
        frame.undocumented_components = total_components - len(documented_components)
        
        # Calculate overall risk score
        risk_factors = []
        
        # Security risk
        security_risk = min(frame.critical_vulnerabilities * 0.3, 1.0)
        risk_factors.append(security_risk)
        
        # Quality risk
        quality_risk = frame.untested_code_ratio
        risk_factors.append(quality_risk)
        
        # Documentation risk
        if total_components > 0:
            doc_risk = frame.undocumented_components / total_components
            risk_factors.append(doc_risk)
        
        # Calculate average risk
        frame.risk_score = sum(risk_factors) / len(risk_factors) if risk_factors else 0.0
        
        # Determine risk level
        if frame.risk_score >= 0.8:
            frame.risk_level = "critical"
        elif frame.risk_score >= 0.6:
            frame.risk_level = "high"
        elif frame.risk_score >= 0.4:
            frame.risk_level = "medium"
        else:
            frame.risk_level = "low"
        
        # Add specific risk items
        if frame.critical_vulnerabilities > 0:
            frame.risk_items.append({
                "type": "security",
                "severity": "critical",
                "description": f"{frame.critical_vulnerabilities} critical security vulnerabilities"
            })
        
        if frame.untested_code_ratio > 0.5:
            frame.risk_items.append({
                "type": "quality",
                "severity": "medium",
                "description": f"High untested code ratio: {frame.untested_code_ratio:.1%}"
            })
        
        return frame
    
    def build_all_frames(self) -> Dict[str, FrameData]:
        """Build all available frames"""
        frames = {}
        
        try:
            frames["quality"] = self.build_quality_frame()
        except Exception as e:
            print(f"⚠️  Failed to build quality frame: {e}")
        
        try:
            frames["delivery"] = self.build_delivery_frame()
        except Exception as e:
            print(f"⚠️  Failed to build delivery frame: {e}")
        
        try:
            frames["architecture"] = self.build_architecture_frame()
        except Exception as e:
            print(f"⚠️  Failed to build architecture frame: {e}")
        
        try:
            frames["risk"] = self.build_risk_frame()
        except Exception as e:
            print(f"⚠️  Failed to build risk frame: {e}")
        
        return frames