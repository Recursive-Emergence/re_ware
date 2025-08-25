"""
Quality Gates (Ω) for RE_ware
=============================

Implements executable quality predicates that act as Ω guardrails in the RE pattern.
Gates enforce Definition of Done (DoD), coverage thresholds, and PR policies.
Failed gates block evolution cycles and emit corrective actions.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

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
            SecurityGate(blocking=True),
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