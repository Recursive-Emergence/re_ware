"""
Action Dispatcher for RE_ware
=============================

Executes actions from LLM advice frames by routing to appropriate executors.
Maps action kinds to tool registry capabilities with graceful degradation.
"""

import os
import json
import subprocess
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActionResult:
    """Result of an executed action"""
    success: bool
    kind: str
    idempotency_key: str
    message: str
    details: Optional[Dict[str, Any]] = None
    external_refs: Optional[List[str]] = None  # GitHub URLs, etc.


class GitHubExecutor:
    """GitHub-specific action executor using gh CLI"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.gh_available = self._check_gh_cli()
    
    def _check_gh_cli(self) -> bool:
        """Check if gh CLI is available and authenticated"""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    async def create_issue(self, action: Dict[str, Any]) -> ActionResult:
        """Create GitHub issue"""
        if not self.gh_available:
            return ActionResult(
                success=False,
                kind="github.issue",
                idempotency_key=action.get("idempotency_key", ""),
                message="GitHub CLI not available or not authenticated"
            )
        
        try:
            title = action.get("title", "Untitled Issue")
            body = action.get("body", "")
            params = action.get("params", {})
            
            # Build gh issue create command
            cmd = ["gh", "issue", "create", "--title", title, "--body", body]
            
            # Add labels if specified
            if "labels" in params:
                labels = params["labels"]
                if isinstance(labels, list):
                    labels = ",".join(labels)
                cmd.extend(["--label", labels])
            
            # Add assignees if specified  
            if "assignee" in params:
                cmd.extend(["--assignee", params["assignee"]])
            
            # Execute command
            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                issue_url = stdout.decode().strip()
                return ActionResult(
                    success=True,
                    kind="github.issue",
                    idempotency_key=action.get("idempotency_key", ""),
                    message=f"Created GitHub issue: {issue_url}",
                    external_refs=[issue_url]
                )
            else:
                return ActionResult(
                    success=False,
                    kind="github.issue", 
                    idempotency_key=action.get("idempotency_key", ""),
                    message=f"Failed to create issue: {stderr.decode()}"
                )
                
        except Exception as e:
            return ActionResult(
                success=False,
                kind="github.issue",
                idempotency_key=action.get("idempotency_key", ""),
                message=f"GitHub issue creation error: {e}"
            )
    
    async def create_pr(self, action: Dict[str, Any]) -> ActionResult:
        """Create GitHub pull request"""
        if not self.gh_available:
            return ActionResult(
                success=False,
                kind="github.pr",
                idempotency_key=action.get("idempotency_key", ""),
                message="GitHub CLI not available or not authenticated"
            )
        
        try:
            title = action.get("title", "Untitled PR")
            body = action.get("body", "")
            params = action.get("params", {})
            
            # Build gh pr create command
            cmd = ["gh", "pr", "create", "--title", title, "--body", body]
            
            # Add base branch if specified
            if "base" in params:
                cmd.extend(["--base", params["base"]])
            
            # Add head branch if specified
            if "head" in params:
                cmd.extend(["--head", params["head"]])
            
            # Add draft flag if specified
            if params.get("draft", False):
                cmd.append("--draft")
            
            # Execute command
            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                pr_url = stdout.decode().strip()
                return ActionResult(
                    success=True,
                    kind="github.pr",
                    idempotency_key=action.get("idempotency_key", ""),
                    message=f"Created GitHub PR: {pr_url}",
                    external_refs=[pr_url]
                )
            else:
                return ActionResult(
                    success=False,
                    kind="github.pr",
                    idempotency_key=action.get("idempotency_key", ""),
                    message=f"Failed to create PR: {stderr.decode()}"
                )
                
        except Exception as e:
            return ActionResult(
                success=False,
                kind="github.pr",
                idempotency_key=action.get("idempotency_key", ""),
                message=f"GitHub PR creation error: {e}"
            )


class CIExecutor:
    """CI/CD action executor for GitHub Actions"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.gh_available = self._check_gh_cli()
    
    def _check_gh_cli(self) -> bool:
        """Check if gh CLI is available"""
        try:
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    async def trigger_workflow(self, action: Dict[str, Any]) -> ActionResult:
        """Trigger GitHub Actions workflow"""
        if not self.gh_available:
            return ActionResult(
                success=False,
                kind="ci.trigger",
                idempotency_key=action.get("idempotency_key", ""),
                message="GitHub CLI not available for workflow dispatch"
            )
        
        try:
            params = action.get("params", {})
            workflow = params.get("workflow", "ci.yml")
            ref = params.get("ref", "main")
            
            # Build gh workflow run command
            cmd = ["gh", "workflow", "run", workflow, "--ref", ref]
            
            # Add inputs if specified
            if "inputs" in params:
                for key, value in params["inputs"].items():
                    cmd.extend(["--field", f"{key}={value}"])
            
            # Execute command
            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return ActionResult(
                    success=True,
                    kind="ci.trigger",
                    idempotency_key=action.get("idempotency_key", ""),
                    message=f"Triggered workflow: {workflow}",
                    details={"workflow": workflow, "ref": ref}
                )
            else:
                return ActionResult(
                    success=False,
                    kind="ci.trigger",
                    idempotency_key=action.get("idempotency_key", ""),
                    message=f"Failed to trigger workflow: {stderr.decode()}"
                )
                
        except Exception as e:
            return ActionResult(
                success=False,
                kind="ci.trigger",
                idempotency_key=action.get("idempotency_key", ""),
                message=f"CI trigger error: {e}"
            )


class LocalExecutor:
    """Local filesystem and notification executor"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    async def write_file(self, action: Dict[str, Any]) -> ActionResult:
        """Write content to filesystem"""
        try:
            params = action.get("params", {})
            targets = action.get("targets", [])
            
            if not targets:
                return ActionResult(
                    success=False,
                    kind="fs.write",
                    idempotency_key=action.get("idempotency_key", ""),
                    message="No file targets specified"
                )
            
            files_written = []
            for target in targets:
                if target.get("type") != "file":
                    continue
                
                file_path = self.project_root / target.get("id", "")
                content = params.get("content", action.get("body", ""))
                
                # Ensure parent directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file
                file_path.write_text(content, encoding="utf-8")
                files_written.append(str(file_path))
            
            if files_written:
                return ActionResult(
                    success=True,
                    kind="fs.write",
                    idempotency_key=action.get("idempotency_key", ""),
                    message=f"Wrote {len(files_written)} files",
                    details={"files": files_written}
                )
            else:
                return ActionResult(
                    success=False,
                    kind="fs.write",
                    idempotency_key=action.get("idempotency_key", ""),
                    message="No valid file targets found"
                )
                
        except Exception as e:
            return ActionResult(
                success=False,
                kind="fs.write",
                idempotency_key=action.get("idempotency_key", ""),
                message=f"File write error: {e}"
            )
    
    async def send_notification(self, action: Dict[str, Any]) -> ActionResult:
        """Send notification (console output for now)"""
        try:
            title = action.get("title", "RE_ware Notification")
            body = action.get("body", "")
            
            print(f"🔔 {title}")
            if body:
                print(f"   {body}")
            
            return ActionResult(
                success=True,
                kind="notify",
                idempotency_key=action.get("idempotency_key", ""),
                message=f"Notification sent: {title}"
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                kind="notify",
                idempotency_key=action.get("idempotency_key", ""),
                message=f"Notification error: {e}"
            )


class ActionDispatcher:
    """Main action dispatcher that routes actions to appropriate executors"""
    
    def __init__(self, project_root: Path, ontology=None):
        self.project_root = Path(project_root)
        self.ontology = ontology
        
        # Initialize executors
        self.github_executor = GitHubExecutor(self.project_root)
        self.ci_executor = CIExecutor(self.project_root)
        self.local_executor = LocalExecutor(self.project_root)
        
        # Action kind routing
        self.executors = {
            "github.issue": self.github_executor.create_issue,
            "github.pr": self.github_executor.create_pr,
            "ci.trigger": self.ci_executor.trigger_workflow,
            "fs.write": self.local_executor.write_file,
            "notify": self.local_executor.send_notification,
            "graph.update": self._update_graph,  # Special handler
        }
    
    async def dispatch_action(self, action: Dict[str, Any]) -> ActionResult:
        """Dispatch a single action to appropriate executor"""
        kind = action.get("kind", "unknown")
        idempotency_key = action.get("idempotency_key", "")
        
        # Check if we have an executor for this kind
        if kind not in self.executors:
            return ActionResult(
                success=False,
                kind=kind,
                idempotency_key=idempotency_key,
                message=f"No executor available for action kind: {kind}"
            )
        
        # Check for idempotency (simple in-memory cache for now)
        if hasattr(self, '_executed_actions'):
            if idempotency_key in self._executed_actions:
                prev_result = self._executed_actions[idempotency_key]
                return ActionResult(
                    success=prev_result.success,
                    kind=kind,
                    idempotency_key=idempotency_key,
                    message=f"Action already executed: {prev_result.message}",
                    details={"cached": True}
                )
        else:
            self._executed_actions = {}
        
        try:
            # Execute the action
            executor_func = self.executors[kind]
            result = await executor_func(action)
            
            # Cache result for idempotency
            if idempotency_key:
                self._executed_actions[idempotency_key] = result
            
            return result
            
        except Exception as e:
            return ActionResult(
                success=False,
                kind=kind,
                idempotency_key=idempotency_key,
                message=f"Executor error: {e}"
            )
    
    async def dispatch_actions(self, actions: List[Dict[str, Any]]) -> List[ActionResult]:
        """Dispatch multiple actions and return results"""
        results = []
        
        for action in actions:
            result = await self.dispatch_action(action)
            results.append(result)
            
            # Log result
            status = "✅" if result.success else "❌"
            print(f"{status} {result.kind}: {result.message}")
        
        return results
    
    async def _update_graph(self, action: Dict[str, Any]) -> ActionResult:
        """Special handler for graph updates"""
        try:
            # This would update the ontology directly
            # For now, just log the request
            title = action.get("title", "Graph update")
            
            # In a full implementation, this would:
            # 1. Parse the update parameters
            # 2. Apply changes to self.ontology
            # 3. Trigger a sensor rescan if needed
            
            return ActionResult(
                success=True,
                kind="graph.update",
                idempotency_key=action.get("idempotency_key", ""),
                message=f"Graph update logged: {title}",
                details={"deferred": True}
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                kind="graph.update",
                idempotency_key=action.get("idempotency_key", ""),
                message=f"Graph update error: {e}"
            )
    
    def get_available_executors(self) -> Dict[str, Dict[str, Any]]:
        """Return available executors and their capabilities"""
        return {
            "github.issue": {
                "available": self.github_executor.gh_available,
                "description": "Create GitHub issues"
            },
            "github.pr": {
                "available": self.github_executor.gh_available,
                "description": "Create GitHub pull requests"
            },
            "ci.trigger": {
                "available": self.ci_executor.gh_available,
                "description": "Trigger GitHub Actions workflows"
            },
            "fs.write": {
                "available": True,
                "description": "Write files to local filesystem"
            },
            "notify": {
                "available": True,
                "description": "Send notifications (console)"
            },
            "graph.update": {
                "available": True,
                "description": "Update ontology graph (deferred)"
            }
        }