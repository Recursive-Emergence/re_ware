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
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta

from .advice_validator import AdviceSchemaValidator, ValidationResult


@dataclass
class ActionResult:
    """Result of an executed action"""
    success: bool
    kind: str
    idempotency_key: str
    message: str
    details: Optional[Dict[str, Any]] = None
    external_refs: Optional[List[str]] = None  # GitHub URLs, etc.


@dataclass
class ActionLogEntry:
    """Audit log entry for action execution"""
    timestamp: str
    idempotency_key: str
    kind: str
    action: Dict[str, Any]
    result: ActionResult
    git_commit: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass 
class QueuedAction:
    """Action queued for retry or deferred execution"""
    action: Dict[str, Any]
    retry_count: int
    max_retries: int
    last_attempt: str
    next_retry: str
    reason: str
    priority: str = "medium"


class ActionAuditLogger:
    """Persistent audit logger for action execution and idempotency"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file or ".reware/actions.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._idempotency_cache = None
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None
    
    def _load_idempotency_cache(self) -> Dict[str, ActionLogEntry]:
        """Load idempotency cache from log file"""
        if self._idempotency_cache is not None:
            return self._idempotency_cache
        
        self._idempotency_cache = {}
        
        if not self.log_file.exists():
            return self._idempotency_cache
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        log_data = json.loads(line)
                        entry = ActionLogEntry(
                            timestamp=log_data["timestamp"],
                            idempotency_key=log_data["idempotency_key"],
                            kind=log_data["kind"],
                            action=log_data["action"],
                            result=ActionResult(**log_data["result"]),
                            git_commit=log_data.get("git_commit"),
                            execution_time_ms=log_data.get("execution_time_ms")
                        )
                        
                        if entry.idempotency_key:
                            self._idempotency_cache[entry.idempotency_key] = entry
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Skip malformed entries
                        continue
        except FileNotFoundError:
            pass
        
        return self._idempotency_cache
    
    def is_action_executed(self, idempotency_key: str) -> Optional[ActionResult]:
        """Check if action has already been executed"""
        if not idempotency_key:
            return None
        
        cache = self._load_idempotency_cache()
        entry = cache.get(idempotency_key)
        
        if entry:
            # Return cached result with additional metadata
            cached_result = ActionResult(
                success=entry.result.success,
                kind=entry.result.kind,
                idempotency_key=entry.result.idempotency_key,
                message=f"[CACHED] {entry.result.message}",
                details={
                    **(entry.result.details or {}),
                    "cached": True,
                    "original_timestamp": entry.timestamp,
                    "git_commit": entry.git_commit
                },
                external_refs=entry.result.external_refs
            )
            return cached_result
        
        return None
    
    def log_action(self, action: Dict[str, Any], result: ActionResult, execution_time_ms: Optional[float] = None):
        """Log action execution for audit trail"""
        try:
            entry = ActionLogEntry(
                timestamp=datetime.now().isoformat(),
                idempotency_key=result.idempotency_key,
                kind=result.kind,
                action=action,
                result=result,
                git_commit=self._get_git_commit(),
                execution_time_ms=execution_time_ms
            )
            
            # Convert to JSON and append to log file
            log_data = {
                "timestamp": entry.timestamp,
                "idempotency_key": entry.idempotency_key,
                "kind": entry.kind,
                "action": entry.action,
                "result": asdict(entry.result),
                "git_commit": entry.git_commit,
                "execution_time_ms": entry.execution_time_ms
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_data, separators=(',', ':')) + '\n')
            
            # Update cache
            if self._idempotency_cache is not None and entry.idempotency_key:
                self._idempotency_cache[entry.idempotency_key] = entry
                
        except Exception as e:
            # Don't fail action execution due to logging errors
            print(f"⚠️  Action audit logging error: {e}")
    
    def get_action_history(self, limit: Optional[int] = None, kind_filter: Optional[str] = None) -> List[ActionLogEntry]:
        """Get action execution history"""
        history = []
        
        if not self.log_file.exists():
            return history
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                
            # Process most recent entries first
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    log_data = json.loads(line)
                    entry = ActionLogEntry(
                        timestamp=log_data["timestamp"],
                        idempotency_key=log_data["idempotency_key"],
                        kind=log_data["kind"],
                        action=log_data["action"],
                        result=ActionResult(**log_data["result"]),
                        git_commit=log_data.get("git_commit"),
                        execution_time_ms=log_data.get("execution_time_ms")
                    )
                    
                    # Apply kind filter if specified
                    if kind_filter and entry.kind != kind_filter:
                        continue
                    
                    history.append(entry)
                    
                    # Apply limit if specified
                    if limit and len(history) >= limit:
                        break
                        
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        except FileNotFoundError:
            pass
        
        return history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        history = self.get_action_history()
        
        if not history:
            return {
                "total_actions": 0,
                "success_rate": 0.0,
                "action_kinds": {},
                "recent_activity": None
            }
        
        total_actions = len(history)
        successful_actions = sum(1 for entry in history if entry.result.success)
        success_rate = successful_actions / total_actions
        
        # Count by action kind
        action_kinds = {}
        for entry in history:
            kind = entry.kind
            if kind not in action_kinds:
                action_kinds[kind] = {"count": 0, "success_count": 0}
            action_kinds[kind]["count"] += 1
            if entry.result.success:
                action_kinds[kind]["success_count"] += 1
        
        # Calculate success rates by kind
        for kind_stats in action_kinds.values():
            kind_stats["success_rate"] = kind_stats["success_count"] / kind_stats["count"]
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": success_rate,
            "action_kinds": action_kinds,
            "recent_activity": history[0].timestamp if history else None,
            "log_file_size_kb": self.log_file.stat().st_size / 1024 if self.log_file.exists() else 0
        }


class ActionQueue:
    """Persistent queue for actions requiring retry or deferred execution"""
    
    def __init__(self, queue_file: Optional[str] = None):
        self.queue_file = Path(queue_file or ".reware/action_queue.json")
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self._queue_cache = None
    
    def _load_queue(self) -> List[QueuedAction]:
        """Load queued actions from persistent storage"""
        if self._queue_cache is not None:
            return self._queue_cache
        
        if not self.queue_file.exists():
            self._queue_cache = []
            return self._queue_cache
        
        try:
            with open(self.queue_file, 'r') as f:
                data = json.load(f)
            
            self._queue_cache = []
            for item in data:
                queued_action = QueuedAction(
                    action=item["action"],
                    retry_count=item["retry_count"],
                    max_retries=item["max_retries"],
                    last_attempt=item["last_attempt"],
                    next_retry=item["next_retry"],
                    reason=item["reason"],
                    priority=item.get("priority", "medium")
                )
                self._queue_cache.append(queued_action)
                
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            self._queue_cache = []
        
        return self._queue_cache
    
    def _save_queue(self):
        """Save queue to persistent storage"""
        if self._queue_cache is None:
            return
        
        try:
            data = []
            for queued_action in self._queue_cache:
                data.append({
                    "action": queued_action.action,
                    "retry_count": queued_action.retry_count,
                    "max_retries": queued_action.max_retries,
                    "last_attempt": queued_action.last_attempt,
                    "next_retry": queued_action.next_retry,
                    "reason": queued_action.reason,
                    "priority": queued_action.priority
                })
            
            with open(self.queue_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Action queue save error: {e}")
    
    def enqueue_action(self, action: Dict[str, Any], reason: str, max_retries: int = 3, priority: str = "medium"):
        """Add action to retry queue"""
        now = datetime.now()
        next_retry_time = now + timedelta(minutes=5)  # Retry in 5 minutes
        
        queued_action = QueuedAction(
            action=action,
            retry_count=0,
            max_retries=max_retries,
            last_attempt=now.isoformat(),
            next_retry=next_retry_time.isoformat(),
            reason=reason,
            priority=priority
        )
        
        queue = self._load_queue()
        
        # Check if this action is already queued (by idempotency key)
        idempotency_key = action.get("idempotency_key", "")
        if idempotency_key:
            for existing in queue:
                if existing.action.get("idempotency_key") == idempotency_key:
                    # Update existing entry
                    existing.last_attempt = queued_action.last_attempt
                    existing.retry_count += 1
                    existing.next_retry = queued_action.next_retry
                    existing.reason = reason
                    self._save_queue()
                    return
        
        queue.append(queued_action)
        self._queue_cache = queue
        self._save_queue()
    
    def get_ready_actions(self) -> List[QueuedAction]:
        """Get actions ready for retry (past next_retry time)"""
        queue = self._load_queue()
        now = datetime.now()
        ready_actions = []
        
        for queued_action in queue:
            try:
                next_retry_time = datetime.fromisoformat(queued_action.next_retry)
                if now >= next_retry_time and queued_action.retry_count < queued_action.max_retries:
                    ready_actions.append(queued_action)
            except ValueError:
                # Invalid datetime format, consider it ready
                ready_actions.append(queued_action)
        
        return ready_actions
    
    def remove_action(self, idempotency_key: str):
        """Remove action from queue (after successful execution or max retries)"""
        queue = self._load_queue()
        original_len = len(queue)
        
        self._queue_cache = [qa for qa in queue if qa.action.get("idempotency_key") != idempotency_key]
        
        if len(self._queue_cache) != original_len:
            self._save_queue()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue statistics and status"""
        queue = self._load_queue()
        now = datetime.now()
        
        ready_count = 0
        pending_count = 0
        expired_count = 0
        
        by_kind = {}
        by_priority = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for qa in queue:
            # Count by readiness
            try:
                next_retry_time = datetime.fromisoformat(qa.next_retry)
                if qa.retry_count >= qa.max_retries:
                    expired_count += 1
                elif now >= next_retry_time:
                    ready_count += 1
                else:
                    pending_count += 1
            except ValueError:
                ready_count += 1
            
            # Count by kind
            kind = qa.action.get("kind", "unknown")
            by_kind[kind] = by_kind.get(kind, 0) + 1
            
            # Count by priority
            if qa.priority in by_priority:
                by_priority[qa.priority] += 1
        
        return {
            "total_queued": len(queue),
            "ready_for_retry": ready_count,
            "pending_retry": pending_count,
            "max_retries_exceeded": expired_count,
            "by_action_kind": by_kind,
            "by_priority": by_priority,
            "queue_file": str(self.queue_file)
        }
    
    def get_human_facing_ledger(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get ledger of all human-facing work (queued GitHub actions)"""
        queue = self._load_queue()
        
        # Filter for human-facing actions (GitHub issues/PRs)
        human_facing = []
        for qa in queue:
            if qa.action.get("kind", "").startswith("github."):
                human_facing.append({
                    "kind": qa.action.get("kind"),
                    "title": qa.action.get("title", "Untitled"),
                    "priority": qa.priority,
                    "retry_count": qa.retry_count,
                    "max_retries": qa.max_retries,
                    "last_attempt": qa.last_attempt,
                    "next_retry": qa.next_retry,
                    "reason": qa.reason,
                    "idempotency_key": qa.action.get("idempotency_key", ""),
                    "status": "expired" if qa.retry_count >= qa.max_retries else "pending"
                })
        
        # Sort by priority and retry time
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        human_facing.sort(key=lambda x: (priority_order.get(x["priority"], 4), x["last_attempt"]))
        
        if limit:
            human_facing = human_facing[:limit]
        
        return human_facing


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
    
    def __init__(self, project_root: Path, ontology=None, audit_log_file: Optional[str] = None, queue_file: Optional[str] = None):
        self.project_root = Path(project_root)
        self.ontology = ontology
        
        # Initialize audit logger, action queue, and validator
        self.audit_logger = ActionAuditLogger(audit_log_file)
        self.action_queue = ActionQueue(queue_file)
        self.advice_validator = AdviceSchemaValidator()
        
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
    
    async def dispatch_action(self, action: Dict[str, Any], allow_queue: bool = True) -> ActionResult:
        """Dispatch a single action to appropriate executor"""
        kind = action.get("kind", "unknown")
        idempotency_key = action.get("idempotency_key", "")
        
        # Check if we have an executor for this kind
        if kind not in self.executors:
            result = ActionResult(
                success=False,
                kind=kind,
                idempotency_key=idempotency_key,
                message=f"No executor available for action kind: {kind}"
            )
            self.audit_logger.log_action(action, result)
            return result
        
        # Check for persistent idempotency
        cached_result = self.audit_logger.is_action_executed(idempotency_key)
        if cached_result:
            return cached_result
        
        # Record execution start time
        start_time = time.time()
        
        try:
            # Execute the action
            executor_func = self.executors[kind]
            result = await executor_func(action)
            
            # Record execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Handle failed actions that should be queued for retry
            if not result.success and allow_queue and self._should_queue_for_retry(result, kind):
                # Queue for retry instead of immediate failure
                priority = action.get("priority", "medium")
                self.action_queue.enqueue_action(action, result.message, max_retries=3, priority=priority)
                
                # Return a queued result
                result = ActionResult(
                    success=False,
                    kind=kind,
                    idempotency_key=idempotency_key,
                    message=f"Queued for retry: {result.message}",
                    details={"queued": True, "original_message": result.message}
                )
            elif result.success:
                # Remove from queue if it was queued and now succeeded
                if idempotency_key:
                    self.action_queue.remove_action(idempotency_key)
            
            # Log action for audit trail and idempotency
            self.audit_logger.log_action(action, result, execution_time_ms)
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            result = ActionResult(
                success=False,
                kind=kind,
                idempotency_key=idempotency_key,
                message=f"Executor error: {e}"
            )
            
            # Queue for retry if appropriate
            if allow_queue and self._should_queue_for_retry(result, kind):
                priority = action.get("priority", "medium")
                self.action_queue.enqueue_action(action, str(e), max_retries=3, priority=priority)
                result.message = f"Queued for retry: {e}"
                result.details = {"queued": True, "original_error": str(e)}
            
            self.audit_logger.log_action(action, result, execution_time_ms)
            return result
    
    def _should_queue_for_retry(self, result: ActionResult, kind: str) -> bool:
        """Determine if a failed action should be queued for retry"""
        # Queue GitHub actions when gh CLI is not available
        if kind.startswith("github.") and ("not available" in result.message or "not authenticated" in result.message):
            return True
        
        # Queue CI actions when gh CLI is not available
        if kind == "ci.trigger" and "not available" in result.message:
            return True
        
        # Don't queue local operations or successful actions
        if kind in ["fs.write", "notify", "graph.update"]:
            return False
        
        # Queue network-related failures
        if any(keyword in result.message.lower() for keyword in ["timeout", "connection", "network", "unavailable"]):
            return True
        
        return False
    
    def validate_advice_frame(self, advice: Dict[str, Any], repair: bool = True) -> Dict[str, Any]:
        """Validate advice frame schema and optionally repair issues"""
        validation_result = self.advice_validator.validate_advice_frame(advice, repair=repair)
        
        # Log validation results
        print(f"📋 Advice validation: {validation_result.message}")
        
        if validation_result.issues:
            report = self.advice_validator.format_validation_report(validation_result)
            print(report)
        
        return {
            "validation_result": validation_result.result.value,
            "message": validation_result.message,
            "issues_count": len(validation_result.issues),
            "error_count": len([i for i in validation_result.issues if i.severity == "error"]),
            "warning_count": len([i for i in validation_result.issues if i.severity == "warning"]),
            "repaired_advice": validation_result.repaired_advice,
            "original_advice": advice
        }
    
    async def dispatch_advice_frame(self, advice: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        """Validate and dispatch an entire advice frame"""
        results = {
            "validation": None,
            "actions_dispatched": 0,
            "actions_succeeded": 0,
            "actions_failed": 0,
            "actions_queued": 0,
            "action_results": []
        }
        
        # Validate advice frame if requested
        if validate:
            validation_info = self.validate_advice_frame(advice, repair=True)
            results["validation"] = validation_info
            
            # Use repaired advice if validation was successful
            if validation_info["validation_result"] in ["valid", "repaired"] and validation_info.get("repaired_advice"):
                advice = validation_info["repaired_advice"]
            elif validation_info["validation_result"] == "invalid":
                results["error"] = "Advice frame validation failed - cannot dispatch actions"
                return results
        
        # Extract and dispatch actions
        actions = advice.get("actions", [])
        if not actions:
            results["warning"] = "No actions found in advice frame"
            return results
        
        # Dispatch each action
        action_results = await self.dispatch_actions(actions)
        results["actions_dispatched"] = len(action_results)
        results["action_results"] = action_results
        
        # Count results by type
        for result in action_results:
            if result.success:
                results["actions_succeeded"] += 1
            else:
                results["actions_failed"] += 1
                if result.details and result.details.get("queued"):
                    results["actions_queued"] += 1
        
        return results
    
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
    
    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        return self.audit_logger.get_stats()
    
    def get_action_history(self, limit: Optional[int] = None, kind_filter: Optional[str] = None) -> List[ActionLogEntry]:
        """Get action execution history"""
        return self.audit_logger.get_action_history(limit, kind_filter)
    
    async def process_retry_queue(self) -> Dict[str, Any]:
        """Process actions ready for retry"""
        ready_actions = self.action_queue.get_ready_actions()
        
        results = {
            "total_ready": len(ready_actions),
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "requeued": 0
        }
        
        for queued_action in ready_actions:
            try:
                # Retry the action
                result = await self.dispatch_action(queued_action.action, allow_queue=False)
                results["processed"] += 1
                
                if result.success:
                    results["succeeded"] += 1
                    # Remove from queue
                    idempotency_key = queued_action.action.get("idempotency_key", "")
                    if idempotency_key:
                        self.action_queue.remove_action(idempotency_key)
                else:
                    results["failed"] += 1
                    # Update retry count or remove if max retries exceeded
                    if queued_action.retry_count >= queued_action.max_retries - 1:
                        # Max retries exceeded, remove from queue
                        idempotency_key = queued_action.action.get("idempotency_key", "")
                        if idempotency_key:
                            self.action_queue.remove_action(idempotency_key)
                    else:
                        # Re-queue with incremented retry count
                        priority = queued_action.action.get("priority", "medium")
                        self.action_queue.enqueue_action(
                            queued_action.action, 
                            f"Retry {queued_action.retry_count + 1}: {result.message}",
                            max_retries=queued_action.max_retries,
                            priority=priority
                        )
                        results["requeued"] += 1
                        
            except Exception as e:
                print(f"⚠️  Error processing queued action: {e}")
                results["failed"] += 1
        
        return results
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get action queue status"""
        return self.action_queue.get_queue_status()
    
    def get_human_facing_ledger(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get unified ledger of all human-facing work (issues, PRs, etc.)"""
        # Get queued human-facing work
        queued_work = self.action_queue.get_human_facing_ledger(limit)
        
        # Get recent successful human-facing actions from audit log
        recent_actions = self.audit_logger.get_action_history(limit=limit or 20, kind_filter="github.")
        successful_work = []
        
        for entry in recent_actions:
            if entry.kind.startswith("github.") and entry.result.success:
                successful_work.append({
                    "kind": entry.kind,
                    "title": entry.action.get("title", "Untitled"),
                    "priority": entry.action.get("priority", "medium"),
                    "completed_at": entry.timestamp,
                    "idempotency_key": entry.idempotency_key,
                    "status": "completed",
                    "external_refs": entry.result.external_refs or []
                })
        
        return {
            "queued_work": queued_work,
            "completed_work": successful_work[:limit] if limit else successful_work,
            "queue_stats": self.action_queue.get_queue_status(),
            "summary": {
                "total_queued": len(queued_work),
                "total_completed": len(successful_work),
                "high_priority_queued": len([w for w in queued_work if w["priority"] == "high"]),
                "critical_priority_queued": len([w for w in queued_work if w["priority"] == "critical"])
            }
        }