"""
Sensor Implementations - Individual sensors that feed the SensorHub
==================================================================

Each sensor implements SensorInterface and provides events from different sources:
- GitSensor: Tracks commits, branches, file changes via git log
- FsSensor: Watches filesystem changes via inotify/polling  
- GhSensor: Polls GitHub API for issues/PRs/comments
- CliSensor: Accepts manual events from CLI/human input
"""

import os
import time
import subprocess
from typing import List, Optional, Dict, Any
from pathlib import Path

from .sensor_hub import SensorInterface, DomainEvent

class GitSensor(SensorInterface):
    """Tracks Git repository changes via git log and working tree diff"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.last_tracked_commit: Optional[str] = None
        self.is_git_repo = self._check_git_repo()
        
    def _check_git_repo(self) -> bool:
        """Check if directory is a git repository"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @property
    def sensor_type(self) -> str:
        return "git"
    
    def reset_watermark(self, watermark: Optional[str] = None):
        """Reset to specific commit SHA"""
        self.last_tracked_commit = watermark
        if watermark:
            print(f"   📍 Git sensor watermark: {watermark[:8]}...")
        
    def poll(self) -> List[DomainEvent]:
        """Poll for new git changes since last watermark"""
        if not self.is_git_repo:
            return []
        
        events = []
        
        try:
            # Get current HEAD
            current_head = self._get_current_head()
            
            if self.last_tracked_commit is None:
                # First run - get recent commits (last 10)
                events.extend(self._get_recent_commits(limit=10))
            elif current_head != self.last_tracked_commit:
                # Get commits since last watermark
                events.extend(self._get_commits_since(self.last_tracked_commit))
            
            # Always check working tree changes
            events.extend(self._get_working_tree_changes())
            
            # Update watermark to current HEAD
            if current_head:
                self.last_tracked_commit = current_head
                
        except Exception as e:
            print(f"⚠️  Git polling error: {e}")
        
        return events
    
    def _get_current_head(self) -> Optional[str]:
        """Get current HEAD commit SHA"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_recent_commits(self, limit: int = 10) -> List[DomainEvent]:
        """Get recent commits for bootstrap"""
        events = []
        
        try:
            result = subprocess.run([
                "git", "log", f"-{limit}", "--pretty=format:%H|%an|%ct|%s", "--name-status"
            ], cwd=self.repo_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                events.extend(self._parse_git_log_output(result.stdout))
                
        except Exception as e:
            print(f"⚠️  Error getting recent commits: {e}")
        
        return events
    
    def _get_commits_since(self, since_commit: str) -> List[DomainEvent]:
        """Get commits since specific SHA"""
        events = []
        
        try:
            result = subprocess.run([
                "git", "log", f"{since_commit}..HEAD", "--pretty=format:%H|%an|%ct|%s", "--name-status"
            ], cwd=self.repo_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                events.extend(self._parse_git_log_output(result.stdout))
                
        except Exception as e:
            print(f"⚠️  Error getting commits since {since_commit[:8]}: {e}")
        
        return events
    
    def _parse_git_log_output(self, output: str) -> List[DomainEvent]:
        """Parse git log output into DomainEvents"""
        events = []
        lines = output.strip().split('\n')
        
        current_commit = None
        current_author = None
        current_timestamp = None
        current_message = None
        
        for line in lines:
            if not line.strip():
                continue
                
            if '|' in line and '\t' not in line:
                # Commit header line
                parts = line.split('|')
                if len(parts) >= 4:
                    current_commit = parts[0]
                    current_author = parts[1] 
                    current_timestamp = float(parts[2])
                    current_message = '|'.join(parts[3:])
            elif line.startswith(('A\t', 'M\t', 'D\t', 'R\t')):
                # File change line
                if current_commit:
                    change_type = line[0]
                    file_path = line[2:]  # Skip "A\t" prefix
                    
                    # Map git status to our event kinds
                    kind_map = {'A': 'create', 'M': 'modify', 'D': 'delete', 'R': 'rename'}
                    kind = kind_map.get(change_type, 'modify')
                    
                    event = DomainEvent(
                        source="git",
                        kind=kind,
                        path=file_path,
                        sha=current_commit,
                        ref="refs/heads/main",  # Simplified
                        ts=current_timestamp,
                        actor=current_author,
                        meta={
                            "commit_message": current_message,
                            "change_type": change_type
                        }
                    )
                    events.append(event)
        
        return events
    
    def _get_working_tree_changes(self) -> List[DomainEvent]:
        """Get uncommitted changes in working tree"""
        events = []
        
        try:
            # Get modified files
            result = subprocess.run([
                "git", "diff", "--name-status"
            ], cwd=self.repo_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and '\t' in line:
                        change_type, file_path = line.split('\t', 1)
                        
                        event = DomainEvent(
                            source="git",
                            kind="modify",
                            path=file_path,
                            sha=None,  # Working tree change
                            ts=time.time(),
                            actor="local",
                            meta={"working_tree_change": True, "change_type": change_type}
                        )
                        events.append(event)
            
            # Get untracked files
            result = subprocess.run([
                "git", "ls-files", "--others", "--exclude-standard"
            ], cwd=self.repo_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                for file_path in result.stdout.strip().split('\n'):
                    if file_path:
                        event = DomainEvent(
                            source="git",
                            kind="create",
                            path=file_path,
                            ts=time.time(),
                            actor="local",
                            meta={"untracked": True}
                        )
                        events.append(event)
                        
        except Exception as e:
            print(f"⚠️  Error getting working tree changes: {e}")
        
        return events

class FsSensor(SensorInterface):
    """Watches filesystem changes (simplified polling implementation)"""
    
    def __init__(self, watch_root: Path, poll_interval: float = 2.0):
        self.watch_root = watch_root
        self.poll_interval = poll_interval
        self.last_scan_time = time.time()
        self.known_files: Dict[str, float] = {}
        
    @property
    def sensor_type(self) -> str:
        return "fs"
    
    def reset_watermark(self, watermark: Optional[float] = None):
        """Reset to specific timestamp"""
        if watermark is None:
            # None watermark means scan everything from beginning
            self.last_scan_time = 0.0
        else:
            self.last_scan_time = watermark
        self._scan_initial_state()
    
    def _scan_initial_state(self):
        """Scan current filesystem state to establish baseline"""
        self.known_files.clear()
        
        # Immediately scan to establish baseline without generating change events
        try:
            for file_path in self.watch_root.rglob("*"):
                if file_path.is_file() and not self._should_ignore(file_path):
                    rel_path = str(file_path.relative_to(self.watch_root))
                    mtime = file_path.stat().st_mtime
                    self.known_files[rel_path] = mtime
            
            print(f"🔍 FsSensor: Established baseline with {len(self.known_files)} known files")
        except Exception as e:
            print(f"⚠️ FsSensor baseline scan failed: {e}")
            self.known_files.clear()
    
    def poll(self) -> List[DomainEvent]:
        """Poll for filesystem changes"""
        events = []
        current_time = time.time()
        
        # Only scan if enough time has passed
        if current_time - self.last_scan_time < self.poll_interval:
            return events
        
        try:
            current_files: Dict[str, float] = {}
            
            # Scan current state
            for file_path in self.watch_root.rglob("*"):
                if file_path.is_file() and not self._should_ignore(file_path):
                    rel_path = str(file_path.relative_to(self.watch_root))
                    mtime = file_path.stat().st_mtime
                    current_files[rel_path] = mtime
                    
                    # Check for new or modified files
                    if rel_path not in self.known_files:
                        # New file
                        events.append(DomainEvent(
                            source="fs",
                            kind="create",
                            path=rel_path,
                            ts=mtime,
                            meta={"size": file_path.stat().st_size}
                        ))
                    elif self.known_files[rel_path] != mtime:
                        # Modified file
                        events.append(DomainEvent(
                            source="fs", 
                            kind="modify",
                            path=rel_path,
                            ts=mtime,
                            meta={"size": file_path.stat().st_size}
                        ))
            
            # Check for deleted files
            for rel_path in self.known_files:
                if rel_path not in current_files:
                    events.append(DomainEvent(
                        source="fs",
                        kind="delete", 
                        path=rel_path,
                        ts=current_time
                    ))
            
            # Update known state
            self.known_files = current_files
            self.last_scan_time = current_time
            
        except Exception as e:
            print(f"⚠️  FS polling error: {e}")
        
        return events
    
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = [
            ".git", "__pycache__", ".pytest_cache", "node_modules",
            ".venv", "venv", ".DS_Store", "*.pyc", "*.pyo"
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)

class GhSensor(SensorInterface):
    """Polls GitHub API for issues, PRs, comments (requires gh CLI)"""
    
    def __init__(self, repo: Optional[str] = None, poll_interval: int = 300):
        self.repo = repo  # "owner/repo" format
        self.poll_interval = poll_interval
        self.last_poll_time = time.time()
        self.has_gh_cli = self._check_gh_cli()
        
    def _check_gh_cli(self) -> bool:
        """Check if gh CLI is available"""
        try:
            result = subprocess.run(["gh", "--version"], capture_output=True)
            return result.returncode == 0
        except Exception:
            return False
    
    @property 
    def sensor_type(self) -> str:
        return "gh"
    
    def reset_watermark(self, watermark: Optional[float] = None):
        """Reset to specific poll time"""
        self.last_poll_time = watermark or time.time()
    
    def poll(self) -> List[DomainEvent]:
        """Poll GitHub for new issues/PRs/comments"""
        events = []
        current_time = time.time()
        
        # Check poll interval
        if current_time - self.last_poll_time < self.poll_interval:
            return events
        
        if not self.has_gh_cli or not self.repo:
            return events
        
        try:
            # Get recent issues
            events.extend(self._get_recent_issues())
            
            # Get recent PRs  
            events.extend(self._get_recent_prs())
            
            self.last_poll_time = current_time
            
        except Exception as e:
            print(f"⚠️  GitHub polling error: {e}")
        
        return events
    
    def _get_recent_issues(self) -> List[DomainEvent]:
        """Get recent issues from GitHub"""
        events = []
        
        try:
            result = subprocess.run([
                "gh", "issue", "list", "--repo", self.repo, "--limit", "10", "--json", 
                "number,title,state,createdAt,updatedAt,author"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                issues = json.loads(result.stdout)
                
                for issue in issues:
                    events.append(DomainEvent(
                        source="gh",
                        kind="issue",
                        path=f"issues/{issue['number']}",
                        ts=time.time(),  # Simplified
                        actor=issue['author']['login'],
                        meta={
                            "title": issue['title'],
                            "state": issue['state'],
                            "issue_number": issue['number']
                        }
                    ))
                    
        except Exception as e:
            print(f"⚠️  Error fetching issues: {e}")
        
        return events
    
    def _get_recent_prs(self) -> List[DomainEvent]:
        """Get recent pull requests"""
        events = []
        
        try:
            result = subprocess.run([
                "gh", "pr", "list", "--repo", self.repo, "--limit", "10", "--json",
                "number,title,state,createdAt,updatedAt,author"  
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                prs = json.loads(result.stdout)
                
                for pr in prs:
                    events.append(DomainEvent(
                        source="gh", 
                        kind="pr",
                        path=f"pull/{pr['number']}",
                        ts=time.time(),  # Simplified
                        actor=pr['author']['login'],
                        meta={
                            "title": pr['title'],
                            "state": pr['state'], 
                            "pr_number": pr['number']
                        }
                    ))
                    
        except Exception as e:
            print(f"⚠️  Error fetching PRs: {e}")
        
        return events

class CliSensor(SensorInterface):
    """Accepts manual events from CLI/human input"""
    
    def __init__(self):
        self.event_queue: List[DomainEvent] = []
        
    @property
    def sensor_type(self) -> str:
        return "cli"
    
    def reset_watermark(self, watermark: Any = None):
        """CLI sensor doesn't use watermarks"""
        pass
    
    def poll(self) -> List[DomainEvent]:
        """Return queued events and clear"""
        events = self.event_queue.copy()
        self.event_queue.clear()
        return events
    
    def add_event(self, kind: str, path: str, text: str, actor: str = "human"):
        """Add manual event to queue"""
        event = DomainEvent(
            source="cli",
            kind=kind,
            path=path,
            actor=actor,
            meta={"text": text, "manual": True}
        )
        self.event_queue.append(event)
        print(f"📝 Added CLI event: {kind} {path}")