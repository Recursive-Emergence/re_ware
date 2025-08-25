"""
Quality Gates (Ω) for RE_ware
=============================

Implements executable quality predicates that act as Ω guardrails in the RE pattern.
Gates enforce Definition of Done (DoD), coverage thresholds, and PR policies.
Failed gates block evolution cycles and emit corrective actions.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .ontology import NodeType, RelationType, OntologyPhenotype
from .frames import FrameBuilder


class GateResult(Enum):
    """Quality gate execution results"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"  # Not applicable


@dataclass
class GateEvaluation:
    """Result of evaluating a quality gate"""
    gate_name: str
    result: GateResult
    score: float  # 0.0 to 1.0
    message: str
    evidence: Dict[str, Any]
    blocking: bool = False  # Does this gate block deployment?
    corrective_actions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.corrective_actions is None:
            self.corrective_actions = []


class QualityGate:
    """Base class for quality gate implementations"""
    
    def __init__(self, name: str, description: str, blocking: bool = False):
        self.name = name
        self.description = description
        self.blocking = blocking
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Evaluate this quality gate against current project state"""
        raise NotImplementedError


class CoverageGate(QualityGate):
    """Enforces minimum test coverage thresholds"""
    
    def __init__(self, min_coverage: float = 0.8, blocking: bool = True):
        super().__init__(
            name="coverage_gate",
            description=f"Test coverage must be >= {min_coverage:.0%}",
            blocking=blocking
        )
        self.min_coverage = min_coverage
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Check test coverage against threshold"""
        
        # Get coverage from frames if available
        coverage_ratio = 0.0
        if frames and "quality" in frames:
            coverage_ratio = frames["quality"].test_coverage / 100.0
        else:
            # Fallback: calculate from ontology
            phi_signals = ontology.phi_signals()
            coverage_ratio = phi_signals.get("coverage_ratio", 0.0)
        
        # Evaluate against threshold
        passes = coverage_ratio >= self.min_coverage
        score = min(coverage_ratio / self.min_coverage, 1.0)
        
        evidence = {
            "current_coverage": coverage_ratio,
            "required_coverage": self.min_coverage,
            "coverage_gap": max(0.0, self.min_coverage - coverage_ratio)
        }
        
        corrective_actions = []
        if not passes:
            gap_pct = (self.min_coverage - coverage_ratio) * 100
            corrective_actions.extend([
                {
                    "kind": "fs.write",
                    "title": f"Add tests to increase coverage by {gap_pct:.1f}%",
                    "body": f"Current coverage: {coverage_ratio:.1%}\nRequired: {self.min_coverage:.1%}",
                    "priority": "high",
                    "idempotency_key": f"coverage_gap_{gap_pct:.0f}pct"
                },
                {
                    "kind": "notify", 
                    "title": "Coverage gate failing",
                    "body": f"Test coverage {coverage_ratio:.1%} below required {self.min_coverage:.1%}",
                    "priority": "high",
                    "idempotency_key": "coverage_gate_fail"
                }
            ])
        
        message = f"Coverage {coverage_ratio:.1%} (required: {self.min_coverage:.1%})"
        result = GateResult.PASS if passes else GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class TestPassGate(QualityGate):
    """Enforces that all tests must pass"""
    
    def __init__(self, allow_skipped: bool = True, blocking: bool = True):
        super().__init__(
            name="test_pass_gate", 
            description="All tests must pass",
            blocking=blocking
        )
        self.allow_skipped = allow_skipped
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Check test pass rate"""
        
        if frames and "quality" in frames:
            quality_frame = frames["quality"]
            total_tests = quality_frame.total_tests
            passing_tests = quality_frame.passing_tests
            failing_tests = quality_frame.failing_tests
        else:
            # Fallback: count from ontology
            test_nodes = [n for n in ontology.nodes.values() if n.type == NodeType.TEST]
            total_tests = len(test_nodes)
            passing_tests = 0
            failing_tests = 0
            
            for test in test_nodes:
                status = test.content.get("status", "unknown").lower()
                if status in ["passed", "pass", "ok", "success"]:
                    passing_tests += 1
                elif status in ["failed", "fail", "error", "failure"]:
                    failing_tests += 1
        
        # Evaluate pass rate
        passes = failing_tests == 0 and total_tests > 0
        score = passing_tests / max(total_tests, 1)
        
        evidence = {
            "total_tests": total_tests,
            "passing_tests": passing_tests,
            "failing_tests": failing_tests,
            "pass_rate": score
        }
        
        corrective_actions = []
        if failing_tests > 0:
            corrective_actions.extend([
                {
                    "kind": "fs.write",
                    "title": f"Fix {failing_tests} failing tests",
                    "body": f"Fix failing tests before proceeding with deployment",
                    "priority": "critical",
                    "idempotency_key": f"fix_failing_tests_{failing_tests}"
                },
                {
                    "kind": "github.issue",
                    "title": "Failing tests blocking deployment",
                    "body": f"There are {failing_tests} failing tests that must be fixed.\n\nPass rate: {score:.1%}",
                    "labels": ["bug", "tests", "blocking"],
                    "priority": "critical",
                    "idempotency_key": "failing_tests_issue"
                }
            ])
        elif total_tests == 0:
            corrective_actions.append({
                "kind": "fs.write",
                "title": "Add initial tests",
                "body": "Project has no tests. Add basic test coverage.",
                "priority": "high",
                "idempotency_key": "add_initial_tests"
            })
        
        if total_tests == 0:
            message = "No tests found"
            result = GateResult.WARNING
        elif passes:
            message = f"All {total_tests} tests passing"
            result = GateResult.PASS
        else:
            message = f"{failing_tests} of {total_tests} tests failing"
            result = GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class BuildGate(QualityGate):
    """Enforces that latest build must pass"""
    
    def __init__(self, blocking: bool = True):
        super().__init__(
            name="build_gate",
            description="Latest build must pass",
            blocking=blocking
        )
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Check latest build status"""
        
        latest_build_status = "unknown"
        build_count = 0
        
        if frames and "quality" in frames:
            quality_frame = frames["quality"]
            latest_build_status = quality_frame.latest_build_status
            build_count = quality_frame.total_builds
        else:
            # Fallback: check build nodes directly
            build_nodes = [n for n in ontology.nodes.values() if n.type == NodeType.BUILD]
            build_count = len(build_nodes)
            if build_nodes:
                # Sort by timestamp to get latest
                build_nodes.sort(key=lambda n: n.content.get("updated_at", ""), reverse=True)
                latest_build = build_nodes[0]
                latest_build_status = latest_build.content.get("conclusion", "unknown")
        
        # Evaluate build status
        passes = latest_build_status.lower() == "success"
        score = 1.0 if passes else 0.0
        
        evidence = {
            "latest_build_status": latest_build_status,
            "total_builds": build_count
        }
        
        corrective_actions = []
        if not passes and build_count > 0:
            corrective_actions.extend([
                {
                    "kind": "ci.trigger",
                    "title": "Retry failed build",
                    "body": f"Latest build status: {latest_build_status}",
                    "workflow": "ci.yml",
                    "priority": "high",
                    "idempotency_key": f"retry_build_{latest_build_status}"
                },
                {
                    "kind": "github.issue",
                    "title": "Build failure blocking deployment", 
                    "body": f"Latest build status: {latest_build_status}\n\nFix build issues before deployment.",
                    "labels": ["build", "ci", "blocking"],
                    "priority": "high",
                    "idempotency_key": "build_failure_issue"
                }
            ])
        elif build_count == 0:
            corrective_actions.append({
                "kind": "ci.trigger",
                "title": "Trigger initial build",
                "body": "No builds found. Trigger CI pipeline.",
                "workflow": "ci.yml",
                "priority": "medium",
                "idempotency_key": "initial_build"
            })
        
        if build_count == 0:
            message = "No builds found"
            result = GateResult.WARNING
        elif passes:
            message = f"Latest build: {latest_build_status}"
            result = GateResult.PASS
        else:
            message = f"Latest build failed: {latest_build_status}"
            result = GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class RequirementGate(QualityGate):
    """Enforces that requirements must be implemented and verified"""
    
    def __init__(self, min_implementation_rate: float = 0.8, min_verification_rate: float = 0.8, blocking: bool = True):
        super().__init__(
            name="requirement_gate",
            description=f"Requirements must be {min_implementation_rate:.0%} implemented and {min_verification_rate:.0%} verified",
            blocking=blocking
        )
        self.min_implementation_rate = min_implementation_rate
        self.min_verification_rate = min_verification_rate
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Check requirement implementation and verification rates"""
        
        if frames and "delivery" in frames:
            delivery_frame = frames["delivery"]
            total_requirements = delivery_frame.total_requirements
            implemented_requirements = delivery_frame.implemented_requirements
            verified_requirements = delivery_frame.verified_requirements
        else:
            # Fallback: calculate from ontology
            requirement_nodes = [n for n in ontology.nodes.values() if n.type == NodeType.REQUIREMENT]
            total_requirements = len(requirement_nodes)
            
            # Count implemented requirements (have IMPLEMENTS edges)
            implemented_reqs = set()
            verified_reqs = set()
            
            for edge in ontology.edges.values():
                if edge.relation == RelationType.IMPLEMENTS:
                    if edge.to_node in ontology.nodes:
                        to_node = ontology.nodes[edge.to_node]
                        if to_node.type == NodeType.REQUIREMENT:
                            implemented_reqs.add(edge.to_node)
                elif edge.relation == RelationType.VERIFIES:
                    if edge.to_node in ontology.nodes:
                        to_node = ontology.nodes[edge.to_node]
                        if to_node.type == NodeType.REQUIREMENT:
                            verified_reqs.add(edge.to_node)
            
            implemented_requirements = len(implemented_reqs)
            verified_requirements = len(verified_reqs)
        
        # Calculate rates
        implementation_rate = implemented_requirements / max(total_requirements, 1)
        verification_rate = verified_requirements / max(total_requirements, 1)
        
        # Evaluate against thresholds
        implementation_pass = implementation_rate >= self.min_implementation_rate
        verification_pass = verification_rate >= self.min_verification_rate
        passes = implementation_pass and verification_pass
        
        score = (implementation_rate * 0.5) + (verification_rate * 0.5)
        
        evidence = {
            "total_requirements": total_requirements,
            "implemented_requirements": implemented_requirements,
            "verified_requirements": verified_requirements,
            "implementation_rate": implementation_rate,
            "verification_rate": verification_rate,
            "required_implementation_rate": self.min_implementation_rate,
            "required_verification_rate": self.min_verification_rate
        }
        
        corrective_actions = []
        if not implementation_pass:
            missing_impl = int((self.min_implementation_rate - implementation_rate) * total_requirements)
            corrective_actions.append({
                "kind": "fs.write",
                "title": f"Implement {missing_impl} more requirements",
                "body": f"Implementation rate {implementation_rate:.1%} below required {self.min_implementation_rate:.1%}",
                "priority": "high",
                "idempotency_key": f"implement_requirements_{missing_impl}"
            })
        
        if not verification_pass:
            missing_verify = int((self.min_verification_rate - verification_rate) * total_requirements)
            corrective_actions.append({
                "kind": "fs.write",
                "title": f"Add tests for {missing_verify} more requirements",
                "body": f"Verification rate {verification_rate:.1%} below required {self.min_verification_rate:.1%}",
                "priority": "high",
                "idempotency_key": f"verify_requirements_{missing_verify}"
            })
        
        if total_requirements == 0:
            message = "No requirements found"
            result = GateResult.WARNING
        elif passes:
            message = f"Requirements: {implementation_rate:.1%} implemented, {verification_rate:.1%} verified"
            result = GateResult.PASS
        else:
            message = f"Requirements: {implementation_rate:.1%} implemented (need {self.min_implementation_rate:.1%}), {verification_rate:.1%} verified (need {self.min_verification_rate:.1%})"
            result = GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class SecurityGate(QualityGate):
    """Enforces security standards - no critical vulnerabilities"""
    
    def __init__(self, blocking: bool = True):
        super().__init__(
            name="security_gate",
            description="No critical security vulnerabilities",
            blocking=blocking
        )
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Check for security vulnerabilities"""
        
        critical_vulnerabilities = 0
        total_vulnerabilities = 0
        
        if frames and "risk" in frames:
            risk_frame = frames["risk"]
            critical_vulnerabilities = risk_frame.critical_vulnerabilities
            total_vulnerabilities = risk_frame.security_vulnerabilities
        else:
            # Fallback: check bug nodes with security tags
            bug_nodes = [n for n in ontology.nodes.values() if n.type == NodeType.BUG]
            security_bugs = [n for n in bug_nodes if "security" in n.content.get("tags", [])]
            total_vulnerabilities = len(security_bugs)
            critical_vulnerabilities = len([
                n for n in security_bugs
                if getattr(n.state, 'criticality', None) == 'P0'
            ])
        
        # Evaluate security posture
        passes = critical_vulnerabilities == 0
        score = 1.0 if passes else max(0.0, 1.0 - (critical_vulnerabilities * 0.5))
        
        evidence = {
            "critical_vulnerabilities": critical_vulnerabilities,
            "total_vulnerabilities": total_vulnerabilities
        }
        
        corrective_actions = []
        if critical_vulnerabilities > 0:
            corrective_actions.extend([
                {
                    "kind": "github.issue",
                    "title": f"Fix {critical_vulnerabilities} critical security vulnerabilities",
                    "body": f"Critical security issues must be resolved before deployment.",
                    "labels": ["security", "critical", "vulnerability"],
                    "priority": "critical",
                    "idempotency_key": f"critical_security_issues_{critical_vulnerabilities}"
                },
                {
                    "kind": "notify",
                    "title": "Critical security vulnerabilities found",
                    "body": f"{critical_vulnerabilities} critical security issues block deployment",
                    "priority": "critical",
                    "idempotency_key": "security_gate_critical"
                }
            ])
        
        if passes:
            message = f"Security: {total_vulnerabilities} vulnerabilities, 0 critical"
            result = GateResult.PASS
        else:
            message = f"Security: {critical_vulnerabilities} critical vulnerabilities (blocking)"
            result = GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class SASTGate(QualityGate):
    """Static Application Security Testing (SAST) gate"""
    
    def __init__(self, scan_results_file: Optional[str] = None, max_critical: int = 0, max_high: int = 5, blocking: bool = True):
        super().__init__(
            name="sast_gate", 
            description=f"SAST: ≤{max_critical} critical, ≤{max_high} high severity issues",
            blocking=blocking
        )
        self.scan_results_file = scan_results_file or ".reware/sast_results.json"
        self.max_critical = max_critical
        self.max_high = max_high
    
    def _load_sast_results(self) -> Dict[str, Any]:
        """Load SAST results from scan output file"""
        results_path = Path(self.scan_results_file)
        
        if not results_path.exists():
            return {"critical": 0, "high": 0, "medium": 0, "low": 0, "findings": []}
        
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
                
            # Normalize different SAST tool outputs
            # Support formats: Bandit, Semgrep, CodeQL, etc.
            normalized = {"critical": 0, "high": 0, "medium": 0, "low": 0, "findings": []}
            
            if "results" in data:  # Bandit format
                for finding in data.get("results", []):
                    severity = finding.get("issue_severity", "").lower()
                    confidence = finding.get("issue_confidence", "").lower()
                    
                    # Map Bandit severity to our levels
                    if severity == "high" and confidence in ["high", "medium"]:
                        normalized["high"] += 1
                    elif severity == "medium":
                        normalized["medium"] += 1
                    elif severity == "low":
                        normalized["low"] += 1
                    
                    normalized["findings"].append({
                        "severity": severity,
                        "confidence": confidence,
                        "test_name": finding.get("test_name", ""),
                        "filename": finding.get("filename", ""),
                        "line_number": finding.get("line_number", 0)
                    })
            
            elif "findings" in data:  # Generic format
                for finding in data.get("findings", []):
                    severity = finding.get("severity", "").lower()
                    if severity in normalized:
                        normalized[severity] += 1
                    normalized["findings"].append(finding)
            
            elif "vulnerabilities" in data:  # Semgrep/other format
                for vuln in data.get("vulnerabilities", []):
                    severity = vuln.get("severity", "").lower()
                    # Map common severity levels
                    if severity in ["critical", "error"]:
                        normalized["critical"] += 1
                    elif severity in ["high", "warning"]:
                        normalized["high"] += 1
                    elif severity == "medium":
                        normalized["medium"] += 1
                    elif severity in ["low", "info"]:
                        normalized["low"] += 1
                    
                    normalized["findings"].append(vuln)
            
            return normalized
            
        except (json.JSONDecodeError, FileNotFoundError):
            return {"critical": 0, "high": 0, "medium": 0, "low": 0, "findings": []}
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Evaluate SAST scan results"""
        
        results = self._load_sast_results()
        
        critical_count = results.get("critical", 0)
        high_count = results.get("high", 0)
        medium_count = results.get("medium", 0)
        low_count = results.get("low", 0)
        total_findings = critical_count + high_count + medium_count + low_count
        
        # Check thresholds
        critical_pass = critical_count <= self.max_critical
        high_pass = high_count <= self.max_high
        passes = critical_pass and high_pass
        
        # Calculate score based on thresholds
        critical_score = 1.0 if critical_pass else max(0.0, 1.0 - (critical_count - self.max_critical) * 0.2)
        high_score = 1.0 if high_pass else max(0.0, 1.0 - (high_count - self.max_high) * 0.1)
        score = (critical_score + high_score) / 2.0
        
        evidence = {
            "critical_findings": critical_count,
            "high_findings": high_count,
            "medium_findings": medium_count,
            "low_findings": low_count,
            "total_findings": total_findings,
            "max_critical_allowed": self.max_critical,
            "max_high_allowed": self.max_high,
            "scan_results_file": str(self.scan_results_file),
            "results_available": Path(self.scan_results_file).exists()
        }
        
        corrective_actions = []
        if not passes:
            issues = []
            if not critical_pass:
                excess = critical_count - self.max_critical
                issues.append(f"{excess} excess critical issues")
            if not high_pass:
                excess = high_count - self.max_high
                issues.append(f"{excess} excess high-severity issues")
            
            corrective_actions.extend([
                {
                    "kind": "github.issue",
                    "title": f"Fix SAST security issues: {', '.join(issues)}",
                    "body": f"**SAST Scan Results**\n\n" +
                           f"- Critical: {critical_count} (max: {self.max_critical})\n" +
                           f"- High: {high_count} (max: {self.max_high})\n" +
                           f"- Medium: {medium_count}\n" +
                           f"- Low: {low_count}\n\n" +
                           f"**Action Required**: Fix {', '.join(issues)} before deployment.\n\n" +
                           f"Scan results: `{self.scan_results_file}`",
                    "labels": ["security", "sast", "vulnerability", "blocking"],
                    "priority": "high" if critical_count > 0 else "medium",
                    "idempotency_key": f"sast_issues_{critical_count}c_{high_count}h"
                },
                {
                    "kind": "ci.trigger",
                    "title": "Re-run SAST scan after fixes",
                    "body": "Trigger SAST scan workflow to verify fixes",
                    "workflow": "security.yml",
                    "priority": "medium",
                    "idempotency_key": "sast_rescan"
                }
            ])
        elif total_findings == 0 and not Path(self.scan_results_file).exists():
            # No scan results available
            corrective_actions.append({
                "kind": "ci.trigger",
                "title": "Run initial SAST scan",
                "body": f"No SAST results found at {self.scan_results_file}. Run security scan.",
                "workflow": "security.yml",
                "priority": "medium",
                "idempotency_key": "initial_sast_scan"
            })
        
        if not Path(self.scan_results_file).exists():
            message = "No SAST scan results available"
            result = GateResult.WARNING
            score = 0.5  # Partial score for missing scan
        elif passes:
            message = f"SAST: {total_findings} findings ({critical_count} critical, {high_count} high)"
            result = GateResult.PASS
        else:
            message = f"SAST: {critical_count} critical (max: {self.max_critical}), {high_count} high (max: {self.max_high})"
            result = GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking and result == GateResult.FAIL,
            corrective_actions=corrective_actions
        )


class DependencyVulnerabilityGate(QualityGate):
    """Dependency vulnerability scanning gate (npm audit, pip-audit, etc.)"""
    
    def __init__(self, scan_results_file: Optional[str] = None, max_critical: int = 0, max_high: int = 3, blocking: bool = True):
        super().__init__(
            name="dependency_vulnerability_gate",
            description=f"Dependencies: ≤{max_critical} critical, ≤{max_high} high severity vulnerabilities", 
            blocking=blocking
        )
        self.scan_results_file = scan_results_file or ".reware/dependency_vulns.json"
        self.max_critical = max_critical
        self.max_high = max_high
    
    def _load_dependency_scan_results(self) -> Dict[str, Any]:
        """Load dependency vulnerability scan results"""
        results_path = Path(self.scan_results_file)
        
        if not results_path.exists():
            return {"critical": 0, "high": 0, "moderate": 0, "low": 0, "vulnerabilities": []}
        
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            # Normalize npm audit, pip-audit, or other formats
            normalized = {"critical": 0, "high": 0, "moderate": 0, "low": 0, "vulnerabilities": []}
            
            if "vulnerabilities" in data and isinstance(data["vulnerabilities"], dict):
                # npm audit format
                for vuln_name, vuln_data in data["vulnerabilities"].items():
                    severity = vuln_data.get("severity", "").lower()
                    if severity in normalized:
                        normalized[severity] += vuln_data.get("range", "").count(",") + 1
                    
                    normalized["vulnerabilities"].append({
                        "name": vuln_name,
                        "severity": severity,
                        "via": vuln_data.get("via", []),
                        "fixAvailable": vuln_data.get("fixAvailable", False)
                    })
            
            elif "vulnerabilities" in data and isinstance(data["vulnerabilities"], list):
                # pip-audit or generic list format
                for vuln in data["vulnerabilities"]:
                    severity = vuln.get("severity", "").lower()
                    if severity == "critical":
                        normalized["critical"] += 1
                    elif severity == "high":
                        normalized["high"] += 1
                    elif severity in ["moderate", "medium"]:
                        normalized["moderate"] += 1
                    elif severity == "low":
                        normalized["low"] += 1
                    
                    normalized["vulnerabilities"].append(vuln)
            
            return normalized
            
        except (json.JSONDecodeError, FileNotFoundError):
            return {"critical": 0, "high": 0, "moderate": 0, "low": 0, "vulnerabilities": []}
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Evaluate dependency vulnerability scan results"""
        
        results = self._load_dependency_scan_results()
        
        critical_count = results.get("critical", 0)
        high_count = results.get("high", 0)
        moderate_count = results.get("moderate", 0)
        low_count = results.get("low", 0)
        total_vulns = critical_count + high_count + moderate_count + low_count
        
        # Check thresholds
        critical_pass = critical_count <= self.max_critical
        high_pass = high_count <= self.max_high
        passes = critical_pass and high_pass
        
        # Calculate score
        if total_vulns == 0:
            score = 1.0
        else:
            critical_penalty = max(0, critical_count - self.max_critical) * 0.3
            high_penalty = max(0, high_count - self.max_high) * 0.1
            score = max(0.0, 1.0 - critical_penalty - high_penalty)
        
        evidence = {
            "critical_vulnerabilities": critical_count,
            "high_vulnerabilities": high_count,
            "moderate_vulnerabilities": moderate_count,
            "low_vulnerabilities": low_count,
            "total_vulnerabilities": total_vulns,
            "max_critical_allowed": self.max_critical,
            "max_high_allowed": self.max_high,
            "scan_results_file": str(self.scan_results_file),
            "scan_available": Path(self.scan_results_file).exists()
        }
        
        corrective_actions = []
        if not passes:
            corrective_actions.extend([
                {
                    "kind": "github.issue",
                    "title": f"Fix {critical_count + high_count} critical/high dependency vulnerabilities",
                    "body": f"**Dependency Vulnerability Scan Results**\n\n" +
                           f"- Critical: {critical_count} (max allowed: {self.max_critical})\n" +
                           f"- High: {high_count} (max allowed: {self.max_high})\n" +
                           f"- Moderate: {moderate_count}\n" +
                           f"- Low: {low_count}\n\n" +
                           f"**Action Required**: Update vulnerable dependencies before deployment.\n\n" +
                           f"Run `npm audit fix` or equivalent to resolve automatically fixable issues.\n" +
                           f"Scan results: `{self.scan_results_file}`",
                    "labels": ["security", "dependencies", "vulnerability", "blocking"],
                    "priority": "critical" if critical_count > 0 else "high",
                    "idempotency_key": f"dependency_vulns_{critical_count}c_{high_count}h"
                },
                {
                    "kind": "fs.write",
                    "title": "Create dependency update script",
                    "body": f"# Dependency Vulnerability Fixes\n\n## Issues Found\n- Critical: {critical_count}\n- High: {high_count}\n\n## Commands to run:\n- npm audit fix --force\n- pip-audit --fix\n- Or manually update vulnerable packages",
                    "priority": "high", 
                    "idempotency_key": "dependency_fix_script"
                }
            ])
        elif total_vulns == 0 and not Path(self.scan_results_file).exists():
            corrective_actions.append({
                "kind": "ci.trigger",
                "title": "Run dependency vulnerability scan",
                "body": f"No dependency scan results at {self.scan_results_file}. Run security scan.",
                "workflow": "security.yml",
                "priority": "medium",
                "idempotency_key": "initial_dependency_scan"
            })
        
        if not Path(self.scan_results_file).exists():
            message = "No dependency scan results available"
            result = GateResult.WARNING
            score = 0.5
        elif passes:
            message = f"Dependencies: {total_vulns} vulnerabilities ({critical_count} critical, {high_count} high)"
            result = GateResult.PASS
        else:
            message = f"Dependencies: {critical_count} critical, {high_count} high vulnerabilities exceed limits"
            result = GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking and result == GateResult.FAIL,
            corrective_actions=corrective_actions
        )


class ReleaseReadinessGate(QualityGate):
    """Enforces release readiness criteria for deployment"""
    
    def __init__(self, threshold: float = 0.85, blocking: bool = True):
        super().__init__(
            name="release_readiness_gate",
            description=f"Delivery readiness >= {threshold:.0%}",
            blocking=blocking
        )
        self.threshold = threshold
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Evaluate release readiness based on delivery frame"""
        
        if frames is None:
            frames = FrameBuilder(ontology).build_all_frames()
        
        delivery_frame = frames.get("delivery")
        if not delivery_frame:
            return GateEvaluation(
                gate_name=self.name,
                result=GateResult.FAIL,
                score=0.0,
                message="No delivery frame available",
                evidence={},
                blocking=self.blocking
            )
        
        # Check all release readiness criteria
        readiness_ok = (
            delivery_frame.release_readiness_score >= self.threshold and
            delivery_frame.all_tests_passing and
            delivery_frame.security_scan_clean and
            delivery_frame.performance_acceptable and
            delivery_frame.documentation_complete and
            delivery_frame.critical_bugs == 0 and
            delivery_frame.blockers_count == 0
        )
        
        evidence = {
            "release_readiness_score": delivery_frame.release_readiness_score,
            "required_threshold": self.threshold,
            "all_tests_passing": delivery_frame.all_tests_passing,
            "security_scan_clean": delivery_frame.security_scan_clean,
            "performance_acceptable": delivery_frame.performance_acceptable,
            "documentation_complete": delivery_frame.documentation_complete,
            "critical_bugs": delivery_frame.critical_bugs,
            "blockers_count": delivery_frame.blockers_count
        }
        
        score = min(delivery_frame.release_readiness_score / self.threshold, 1.0)
        
        corrective_actions = []
        if not readiness_ok:
            issues = []
            if delivery_frame.release_readiness_score < self.threshold:
                issues.append(f"Readiness score {delivery_frame.release_readiness_score:.1%} below {self.threshold:.1%}")
            if not delivery_frame.all_tests_passing:
                issues.append("Tests not all passing")
            if not delivery_frame.security_scan_clean:
                issues.append("Security scan not clean")
            if not delivery_frame.performance_acceptable:
                issues.append("Performance not acceptable")
            if not delivery_frame.documentation_complete:
                issues.append("Documentation incomplete")
            if delivery_frame.critical_bugs > 0:
                issues.append(f"{delivery_frame.critical_bugs} critical bugs")
            
            corrective_actions.append({
                "kind": "github.issue",
                "title": "Close release gaps",
                "body": "Release readiness issues:\n" + "\n".join(f"- {issue}" for issue in issues),
                "labels": ["release", "blocking"],
                "priority": "high",
                "idempotency_key": "release_readiness_gaps"
            })
        
        message = "Ready to release" if readiness_ok else f"Release readiness {delivery_frame.release_readiness_score:.1%} (need {self.threshold:.1%})"
        result = GateResult.PASS if readiness_ok else GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class StabilityWindowGate(QualityGate):
    """Enforces CI/test stability window for safe releases"""
    
    def __init__(self, min_green_days: int = 3, blocking: bool = True):
        super().__init__(
            name="stability_window_gate",
            description=f"CI/test stability >= {min_green_days} days",
            blocking=blocking
        )
        self.min_green_days = min_green_days
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Evaluate CI stability window"""
        
        # Get CI stability from phi signals
        phi_signals = ontology.phi_signals()
        green_days = phi_signals.get("ci_green_days", 0)
        
        stability_ok = green_days >= self.min_green_days
        score = min(green_days / self.min_green_days, 1.0) if self.min_green_days > 0 else 1.0
        
        evidence = {
            "ci_green_days": green_days,
            "required_green_days": self.min_green_days,
            "stability_window_met": stability_ok
        }
        
        corrective_actions = []
        if not stability_ok:
            days_needed = self.min_green_days - green_days
            corrective_actions.append({
                "kind": "notify",
                "title": "Stability window not met",
                "body": f"Need {days_needed} more day(s) of green CI. Current: {green_days} days.",
                "priority": "medium",
                "idempotency_key": f"stability_window_{days_needed}days"
            })
        
        message = f"Green for {green_days} days" if stability_ok else f"Need {self.min_green_days - green_days} more days"
        result = GateResult.PASS if stability_ok else GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class TwoKeyReleaseGate(QualityGate):
    """Enforces two-key approval for critical deployments (merge_main, release_prod)"""
    
    def __init__(self, approval_file: Optional[str] = None, required_operations: List[str] = None, blocking: bool = True):
        super().__init__(
            name="two_key_release_gate",
            description="Requires two distinct approvers for critical operations",
            blocking=blocking
        )
        self.approval_file = approval_file or ".reware/approvals.json"
        self.required_operations = required_operations or ["merge_main", "release_prod"]
        
    def _load_approvals(self) -> Dict[str, Any]:
        """Load approval state from persistent storage"""
        approval_path = Path(self.approval_file)
        if not approval_path.exists():
            # Ensure directory exists
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            return {}
        
        try:
            with open(approval_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_approvals(self, approvals: Dict[str, Any]):
        """Save approval state to persistent storage"""
        approval_path = Path(self.approval_file)
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(approval_path, 'w') as f:
            json.dump(approvals, f, indent=2)
    
    def add_approval(self, operation: str, approver: str, git_commit: str = None) -> bool:
        """Add an approval for a specific operation"""
        approvals = self._load_approvals()
        
        if operation not in approvals:
            approvals[operation] = {
                "approvers": [],
                "created_at": datetime.now().isoformat(),
                "git_commit": git_commit
            }
        
        # Check if this approver already approved this operation
        existing_approvers = [a["name"] for a in approvals[operation]["approvers"]]
        if approver in existing_approvers:
            return False  # Already approved by this person
        
        # Add new approval
        approvals[operation]["approvers"].append({
            "name": approver,
            "approved_at": datetime.now().isoformat(),
            "git_commit": git_commit
        })
        
        self._save_approvals(approvals)
        return True
    
    def clear_approvals(self, operation: str = None):
        """Clear approvals for specific operation or all operations"""
        approvals = self._load_approvals()
        
        if operation:
            if operation in approvals:
                del approvals[operation]
        else:
            approvals.clear()
            
        self._save_approvals(approvals)
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Check if required operations have two-key approval"""
        approvals = self._load_approvals()
        
        # Check each required operation
        operation_status = {}
        overall_approved = True
        total_approvers = 0
        
        for operation in self.required_operations:
            operation_approvals = approvals.get(operation, {})
            approvers = operation_approvals.get("approvers", [])
            num_approvers = len(approvers)
            
            operation_status[operation] = {
                "approvers": num_approvers,
                "required": 2,
                "approved": num_approvers >= 2,
                "approver_names": [a["name"] for a in approvers]
            }
            
            total_approvers += num_approvers
            if num_approvers < 2:
                overall_approved = False
        
        # Calculate score
        expected_total = len(self.required_operations) * 2
        score = min(total_approvers / max(expected_total, 1), 1.0)
        
        evidence = {
            "operations": operation_status,
            "total_approvers": total_approvers,
            "expected_total": expected_total,
            "all_operations_approved": overall_approved
        }
        
        corrective_actions = []
        if not overall_approved:
            pending_operations = [
                op for op, status in operation_status.items() 
                if not status["approved"]
            ]
            
            corrective_actions.extend([
                {
                    "kind": "notify",
                    "title": f"Two-key approval needed for: {', '.join(pending_operations)}",
                    "body": f"Operations requiring 2 approvals: {', '.join(pending_operations)}\n\nApprovals handled automatically by conscious project entity.",
                    "priority": "high",
                    "idempotency_key": f"two_key_approval_needed_{len(pending_operations)}"
                },
                {
                    "kind": "github.issue",
                    "title": "Two-key approval required for deployment",
                    "body": f"The following operations require two distinct approvals:\n\n" + 
                           "\n".join(f"- **{op}**: {status['approvers']}/2 approvals ({', '.join(status['approver_names']) if status['approver_names'] else 'none'})" 
                                   for op, status in operation_status.items() if not status['approved']),
                    "labels": ["approval", "deployment", "blocking"],
                    "priority": "high",
                    "idempotency_key": "two_key_approval_issue"
                }
            ])
        
        if overall_approved:
            message = "All operations have two-key approval"
            result = GateResult.PASS
        else:
            pending_count = len([s for s in operation_status.values() if not s["approved"]])
            message = f"{pending_count} operations need two-key approval"
            result = GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class RequirementsCoverageGate(QualityGate):
    """Enforces that every REQUIREMENT has at least one TEST that verifies it"""
    
    def __init__(self, blocking: bool = True):
        super().__init__(
            name="requirements_coverage_gate",
            description="Every requirement must be verified by at least one test",
            blocking=blocking
        )
    
    def evaluate(self, ontology: OntologyPhenotype, frames: Dict[str, Any] = None) -> GateEvaluation:
        """Check that all requirements have test coverage"""
        
        # Get all requirements
        requirement_nodes = [n for n in ontology.nodes.values() if n.type == NodeType.REQUIREMENT]
        total_requirements = len(requirement_nodes)
        
        if total_requirements == 0:
            return GateEvaluation(
                gate_name=self.name,
                result=GateResult.WARNING,
                score=1.0,
                message="No requirements found",
                evidence={"total_requirements": 0, "covered_requirements": 0},
                blocking=False
            )
        
        # Find requirements that have test coverage
        covered_requirements = set()
        uncovered_requirements = []
        
        for req_node in requirement_nodes:
            has_test_coverage = False
            
            # Check for direct TEST -> VERIFIES -> REQUIREMENT edges
            for edge in ontology.edges.values():
                if (edge.relation == RelationType.VERIFIES and 
                    edge.to_node == req_node.id):
                    # Found a test that verifies this requirement
                    from_node = ontology.nodes.get(edge.from_node)
                    if from_node and from_node.type == NodeType.TEST:
                        has_test_coverage = True
                        break
            
            # Also check for indirect coverage: CODE -> IMPLEMENTS -> REQ and TEST -> VERIFIES -> CODE
            if not has_test_coverage:
                # Find code that implements this requirement
                implementing_code = []
                for edge in ontology.edges.values():
                    if (edge.relation == RelationType.IMPLEMENTS and 
                        edge.to_node == req_node.id):
                        implementing_code.append(edge.from_node)
                
                # Check if any of the implementing code has tests
                for code_id in implementing_code:
                    for edge in ontology.edges.values():
                        if (edge.relation == RelationType.VERIFIES and 
                            edge.to_node == code_id):
                            from_node = ontology.nodes.get(edge.from_node)
                            if from_node and from_node.type == NodeType.TEST:
                                has_test_coverage = True
                                break
                    if has_test_coverage:
                        break
            
            if has_test_coverage:
                covered_requirements.add(req_node.id)
            else:
                uncovered_requirements.append({
                    "id": req_node.id,
                    "title": req_node.content.get("title", "Untitled requirement"),
                    "description": req_node.content.get("description", "")
                })
        
        # Calculate coverage
        covered_count = len(covered_requirements)
        uncovered_count = len(uncovered_requirements)
        coverage_ratio = covered_count / total_requirements
        
        passes = uncovered_count == 0
        score = coverage_ratio
        
        evidence = {
            "total_requirements": total_requirements,
            "covered_requirements": covered_count,
            "uncovered_requirements": uncovered_count,
            "coverage_ratio": coverage_ratio,
            "uncovered_details": uncovered_requirements[:5]  # Limit to first 5 for brevity
        }
        
        corrective_actions = []
        if uncovered_count > 0:
            corrective_actions.extend([
                {
                    "kind": "fs.write",
                    "title": f"Add tests for {uncovered_count} uncovered requirements",
                    "body": f"Requirements without test coverage:\n" + 
                           "\n".join(f"- {req['title']} ({req['id']})" for req in uncovered_requirements[:10]),
                    "priority": "high",
                    "idempotency_key": f"cover_requirements_{uncovered_count}"
                },
                {
                    "kind": "github.issue",
                    "title": f"Add test coverage for {uncovered_count} requirements",
                    "body": f"**Requirements Coverage Gap**\n\n" +
                           f"Found {uncovered_count} requirements without test coverage ({coverage_ratio:.1%} covered):\n\n" +
                           "\n".join(f"- [ ] {req['title']} (`{req['id']}`)" for req in uncovered_requirements[:10]) +
                           (f"\n\n...and {uncovered_count - 10} more" if uncovered_count > 10 else "") +
                           f"\n\n**Required**: Every requirement must have at least one test that verifies it.\n" +
                           f"**Current**: {covered_count}/{total_requirements} requirements have test coverage.",
                    "labels": ["testing", "requirements", "coverage", "blocking"],
                    "priority": "high", 
                    "idempotency_key": "requirements_coverage_gap"
                }
            ])
        
        if passes:
            message = f"All {total_requirements} requirements have test coverage"
            result = GateResult.PASS
        else:
            message = f"{uncovered_count} requirements lack test coverage ({coverage_ratio:.1%} covered)"
            result = GateResult.FAIL
        
        return GateEvaluation(
            gate_name=self.name,
            result=result,
            score=score,
            message=message,
            evidence=evidence,
            blocking=self.blocking,
            corrective_actions=corrective_actions
        )


class QualityGateRunner:
    """Orchestrates quality gate evaluation"""
    
    def __init__(self, ontology: OntologyPhenotype):
        self.ontology = ontology
        self.gates: List[QualityGate] = []
        self._setup_default_gates()
    
    def _setup_default_gates(self):
        """Setup default quality gates for production readiness"""
        self.gates = [
            TestPassGate(blocking=True),
            CoverageGate(min_coverage=0.8, blocking=True),
            BuildGate(blocking=True), 
            RequirementGate(min_implementation_rate=0.8, min_verification_rate=0.7, blocking=True),
            RequirementsCoverageGate(blocking=True),  # Every requirement must have a test
            SecurityGate(blocking=True),  # General security gate
            SASTGate(max_critical=0, max_high=3, blocking=True),  # Static code analysis
            DependencyVulnerabilityGate(max_critical=0, max_high=2, blocking=True),  # Dependency scanning
            TwoKeyReleaseGate(blocking=True),  # Two-key approval for critical operations
            ReleaseReadinessGate(threshold=0.85, blocking=True),
            StabilityWindowGate(min_green_days=3, blocking=True)
        ]
    
    def add_gate(self, gate: QualityGate):
        """Add a custom quality gate"""
        self.gates.append(gate)
    
    def evaluate_all_gates(self, frames: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate all quality gates and return results"""
        
        if frames is None:
            # Generate frames if not provided
            frame_builder = FrameBuilder(self.ontology)
            frames = frame_builder.build_all_frames()
        
        evaluations = []
        all_corrective_actions = []
        blocking_failures = []
        warnings = []
        
        # Evaluate each gate
        for gate in self.gates:
            try:
                evaluation = gate.evaluate(self.ontology, frames)
                evaluations.append(evaluation)
                
                # Collect corrective actions
                all_corrective_actions.extend(evaluation.corrective_actions)
                
                # Track blocking failures and warnings
                if evaluation.result == GateResult.FAIL:
                    if evaluation.blocking:
                        blocking_failures.append(evaluation.gate_name)
                elif evaluation.result == GateResult.WARNING:
                    warnings.append(evaluation.gate_name)
                    
            except Exception as e:
                # Handle gate evaluation errors gracefully
                error_eval = GateEvaluation(
                    gate_name=gate.name,
                    result=GateResult.FAIL,
                    score=0.0,
                    message=f"Gate evaluation error: {e}",
                    evidence={"error": str(e)},
                    blocking=gate.blocking
                )
                evaluations.append(error_eval)
                if gate.blocking:
                    blocking_failures.append(gate.name)
        
        # Calculate overall gate status
        total_gates = len(evaluations)
        passing_gates = len([e for e in evaluations if e.result == GateResult.PASS])
        failing_gates = len([e for e in evaluations if e.result == GateResult.FAIL])
        warning_gates = len([e for e in evaluations if e.result == GateResult.WARNING])
        
        overall_score = sum(e.score for e in evaluations) / max(total_gates, 1)
        
        # Determine overall status
        if blocking_failures:
            overall_status = "blocked"
        elif failing_gates > 0:
            overall_status = "failing"
        elif warning_gates > 0:
            overall_status = "warning"
        else:
            overall_status = "passing"
        
        return {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "summary": {
                "total_gates": total_gates,
                "passing": passing_gates,
                "failing": failing_gates,
                "warnings": warning_gates,
                "blocking_failures": len(blocking_failures)
            },
            "blocking_failures": blocking_failures,
            "warnings": warnings,
            "evaluations": [
                {
                    "gate_name": e.gate_name,
                    "result": e.result.value,
                    "score": e.score,
                    "message": e.message,
                    "blocking": e.blocking,
                    "evidence": e.evidence
                } for e in evaluations
            ],
            "corrective_actions": all_corrective_actions,
            "evaluated_at": time.time()
        }
    
    def is_deployment_blocked(self, frames: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """Check if deployment is blocked by quality gates"""
        results = self.evaluate_all_gates(frames)
        is_blocked = results["overall_status"] == "blocked"
        blocking_gates = results["blocking_failures"]
        return is_blocked, blocking_gates