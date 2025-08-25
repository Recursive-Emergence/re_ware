"""
Advice Schema Validation for RE_ware
====================================

Validates LLM-generated advice frames against expected schema before dispatch.
Ensures structured advice conforms to AdviceInput/AdviceOutput format.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationResult(Enum):
    """Validation result status"""
    VALID = "valid"
    INVALID = "invalid" 
    REPAIRED = "repaired"


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    field: str
    issue: str
    severity: str  # "error", "warning", "info"
    suggested_fix: Optional[str] = None


@dataclass
class AdviceValidationResult:
    """Result of advice schema validation"""
    result: ValidationResult
    issues: List[ValidationIssue]
    repaired_advice: Optional[Dict[str, Any]] = None
    message: str = ""


class AdviceSchemaValidator:
    """Validates and repairs LLM advice against expected schema"""
    
    def __init__(self):
        self.required_top_level = ["actions", "tensions", "phi_updates"]
        self.action_required_fields = ["kind", "idempotency_key", "title"]
        self.valid_action_kinds = {
            "github.issue", "github.pr", "ci.trigger", 
            "fs.write", "notify", "graph.update"
        }
        self.valid_priorities = {"critical", "high", "medium", "low"}
    
    def validate_advice_frame(self, advice: Dict[str, Any], repair: bool = True) -> AdviceValidationResult:
        """Validate advice frame against schema"""
        issues = []
        repaired_advice = advice.copy() if repair else None
        
        # Validate top-level structure
        issues.extend(self._validate_top_level(advice, repaired_advice))
        
        # Validate actions array
        if "actions" in advice:
            action_issues, repaired_actions = self._validate_actions(advice["actions"], repair)
            issues.extend(action_issues)
            if repair and repaired_actions is not None:
                repaired_advice["actions"] = repaired_actions
        
        # Validate tensions array
        if "tensions" in advice:
            tension_issues = self._validate_tensions(advice["tensions"])
            issues.extend(tension_issues)
        
        # Validate phi_updates
        if "phi_updates" in advice:
            phi_issues = self._validate_phi_updates(advice["phi_updates"])
            issues.extend(phi_issues)
        
        # Determine result
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])
        
        if error_count == 0:
            if warning_count == 0:
                result = ValidationResult.VALID
                message = "Advice schema is valid"
            else:
                result = ValidationResult.VALID
                message = f"Advice schema valid with {warning_count} warnings"
        else:
            if repair and repaired_advice and error_count <= 3:  # Attempt repair for minor issues
                result = ValidationResult.REPAIRED
                message = f"Advice schema repaired ({error_count} errors, {warning_count} warnings fixed)"
            else:
                result = ValidationResult.INVALID
                message = f"Advice schema invalid ({error_count} errors, {warning_count} warnings)"
                repaired_advice = None
        
        return AdviceValidationResult(
            result=result,
            issues=issues,
            repaired_advice=repaired_advice,
            message=message
        )
    
    def _validate_top_level(self, advice: Dict[str, Any], repaired: Optional[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate top-level advice structure"""
        issues = []
        
        for required_field in self.required_top_level:
            if required_field not in advice:
                issues.append(ValidationIssue(
                    field=required_field,
                    issue=f"Missing required top-level field: {required_field}",
                    severity="error",
                    suggested_fix=f"Add empty {required_field} array"
                ))
                
                # Repair: add empty field
                if repaired is not None:
                    if required_field == "phi_updates":
                        repaired[required_field] = {}
                    else:
                        repaired[required_field] = []
        
        # Check for unexpected top-level fields
        expected_fields = set(self.required_top_level + ["metadata", "pull_more", "advice_id"])
        for field in advice.keys():
            if field not in expected_fields:
                issues.append(ValidationIssue(
                    field=field,
                    issue=f"Unexpected top-level field: {field}",
                    severity="warning",
                    suggested_fix="Remove or move to metadata"
                ))
        
        return issues
    
    def _validate_actions(self, actions: Any, repair: bool) -> Tuple[List[ValidationIssue], Optional[List[Dict[str, Any]]]]:
        """Validate actions array"""
        issues = []
        repaired_actions = [] if repair else None
        
        if not isinstance(actions, list):
            issues.append(ValidationIssue(
                field="actions",
                issue="Actions must be an array",
                severity="error",
                suggested_fix="Convert to array format"
            ))
            return issues, repaired_actions
        
        for i, action in enumerate(actions):
            action_issues, repaired_action = self._validate_single_action(action, f"actions[{i}]", repair)
            issues.extend(action_issues)
            
            if repair:
                if repaired_action is not None:
                    repaired_actions.append(repaired_action)
                elif len(action_issues) == 0 or not any(issue.severity == "error" for issue in action_issues):
                    # Keep action if no errors or only warnings
                    repaired_actions.append(action)
                # Otherwise skip broken action
        
        return issues, repaired_actions
    
    def _validate_single_action(self, action: Any, field_path: str, repair: bool) -> Tuple[List[ValidationIssue], Optional[Dict[str, Any]]]:
        """Validate individual action"""
        issues = []
        repaired_action = action.copy() if repair and isinstance(action, dict) else None
        
        if not isinstance(action, dict):
            issues.append(ValidationIssue(
                field=field_path,
                issue="Action must be an object",
                severity="error"
            ))
            return issues, None
        
        # Check required fields
        for required_field in self.action_required_fields:
            if required_field not in action:
                issues.append(ValidationIssue(
                    field=f"{field_path}.{required_field}",
                    issue=f"Missing required field: {required_field}",
                    severity="error",
                    suggested_fix=f"Add {required_field} field"
                ))
                
                # Repair: add default values
                if repaired_action is not None:
                    if required_field == "idempotency_key":
                        # Generate from kind + title
                        kind = action.get("kind", "unknown")
                        title = action.get("title", "untitled")
                        repaired_action["idempotency_key"] = f"{kind}_{hash(title) % 10000}"
                    elif required_field == "title":
                        repaired_action["title"] = f"Generated {action.get('kind', 'action')}"
                    elif required_field == "kind":
                        repaired_action["kind"] = "notify"  # Safe default
        
        # Validate action kind
        kind = action.get("kind", "")
        if kind and kind not in self.valid_action_kinds:
            issues.append(ValidationIssue(
                field=f"{field_path}.kind",
                issue=f"Unknown action kind: {kind}",
                severity="error",
                suggested_fix=f"Use one of: {', '.join(self.valid_action_kinds)}"
            ))
            
            # Repair: map to closest valid kind
            if repaired_action is not None:
                if "github" in kind.lower():
                    repaired_action["kind"] = "github.issue"
                elif "file" in kind.lower() or "write" in kind.lower():
                    repaired_action["kind"] = "fs.write" 
                elif "ci" in kind.lower() or "build" in kind.lower():
                    repaired_action["kind"] = "ci.trigger"
                else:
                    repaired_action["kind"] = "notify"
        
        # Validate priority if present
        priority = action.get("priority")
        if priority and priority not in self.valid_priorities:
            issues.append(ValidationIssue(
                field=f"{field_path}.priority",
                issue=f"Invalid priority: {priority}",
                severity="warning",
                suggested_fix=f"Use one of: {', '.join(self.valid_priorities)}"
            ))
            
            # Repair: normalize priority
            if repaired_action is not None:
                priority_lower = priority.lower()
                if priority_lower in self.valid_priorities:
                    repaired_action["priority"] = priority_lower
                elif priority_lower in ["urgent", "p0", "blocker"]:
                    repaired_action["priority"] = "critical"
                elif priority_lower in ["important", "p1"]:
                    repaired_action["priority"] = "high"
                else:
                    repaired_action["priority"] = "medium"
        
        # Validate idempotency_key format
        idem_key = action.get("idempotency_key", "")
        if idem_key and (len(idem_key) < 3 or len(idem_key) > 100):
            issues.append(ValidationIssue(
                field=f"{field_path}.idempotency_key",
                issue="Idempotency key should be 3-100 characters",
                severity="warning",
                suggested_fix="Use descriptive key with reasonable length"
            ))
            
            # Repair: truncate or regenerate
            if repaired_action is not None:
                if len(idem_key) > 100:
                    repaired_action["idempotency_key"] = idem_key[:97] + "..."
                elif len(idem_key) < 3:
                    title = action.get("title", "action")
                    repaired_action["idempotency_key"] = f"{kind}_{hash(title) % 10000}"
        
        # Validate GitHub actions have labels
        if kind in ["github.issue", "github.pr"] and "labels" not in action.get("params", {}):
            issues.append(ValidationIssue(
                field=f"{field_path}.params.labels",
                issue="GitHub actions should include labels",
                severity="warning",
                suggested_fix="Add relevant labels array"
            ))
            
            # Repair: add default labels
            if repaired_action is not None:
                if "params" not in repaired_action:
                    repaired_action["params"] = {}
                repaired_action["params"]["labels"] = ["automated"]
        
        return issues, repaired_action
    
    def _validate_tensions(self, tensions: Any) -> List[ValidationIssue]:
        """Validate tensions array"""
        issues = []
        
        if not isinstance(tensions, list):
            issues.append(ValidationIssue(
                field="tensions",
                issue="Tensions must be an array",
                severity="error"
            ))
            return issues
        
        for i, tension in enumerate(tensions):
            if not isinstance(tension, dict):
                issues.append(ValidationIssue(
                    field=f"tensions[{i}]",
                    issue="Tension must be an object",
                    severity="warning"
                ))
                continue
            
            if "description" not in tension:
                issues.append(ValidationIssue(
                    field=f"tensions[{i}].description",
                    issue="Tension missing description",
                    severity="warning"
                ))
        
        return issues
    
    def _validate_phi_updates(self, phi_updates: Any) -> List[ValidationIssue]:
        """Validate phi_updates object"""
        issues = []
        
        if not isinstance(phi_updates, dict):
            issues.append(ValidationIssue(
                field="phi_updates",
                issue="Phi updates must be an object",
                severity="error"
            ))
            return issues
        
        # Could add specific phi signal validation here
        # For now, accept any dict structure
        
        return issues
    
    def format_validation_report(self, validation_result: AdviceValidationResult) -> str:
        """Format validation result as human-readable report"""
        lines = []
        
        # Status
        status_icon = {
            ValidationResult.VALID: "✅",
            ValidationResult.REPAIRED: "🔧", 
            ValidationResult.INVALID: "❌"
        }[validation_result.result]
        
        lines.append(f"{status_icon} {validation_result.message}")
        
        if not validation_result.issues:
            return lines[0]
        
        # Group issues by severity
        errors = [i for i in validation_result.issues if i.severity == "error"]
        warnings = [i for i in validation_result.issues if i.severity == "warning"]
        infos = [i for i in validation_result.issues if i.severity == "info"]
        
        if errors:
            lines.append(f"\n❌ Errors ({len(errors)}):")
            for error in errors:
                lines.append(f"  • {error.field}: {error.issue}")
                if error.suggested_fix:
                    lines.append(f"    Fix: {error.suggested_fix}")
        
        if warnings:
            lines.append(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                lines.append(f"  • {warning.field}: {warning.issue}")
                if warning.suggested_fix:
                    lines.append(f"    Suggestion: {warning.suggested_fix}")
        
        if infos:
            lines.append(f"\nℹ️  Info ({len(infos)}):")
            for info in infos:
                lines.append(f"  • {info.field}: {info.issue}")
        
        return "\n".join(lines)