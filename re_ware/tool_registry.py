"""
Dynamic Tool Discovery and Integration
======================================

Discovers available development tools and integrates them with the ontological graph.
Enables organic workflow adaptation based on what tools are actually available.
"""

import subprocess
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .ontology import OntologyGraph, NodeType, RelationType, create_node, create_edge

@dataclass
class ToolCapability:
    """Represents a capability provided by a tool"""
    name: str
    tool: str
    command_pattern: str
    input_types: List[str]
    output_types: List[str]
    ontology_mapping: Dict[str, str]  # Maps tool output to ontology node types

@dataclass
class ToolInfo:
    """Information about a discovered tool"""
    name: str
    version: str
    path: str
    capabilities: List[ToolCapability]
    available: bool
    last_checked: float

class ToolAdapter(ABC):
    """Base class for tool adapters"""
    
    @abstractmethod
    def check_available(self) -> bool:
        """Check if tool is available"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[ToolCapability]:
        """Get tool capabilities"""
        pass
    
    @abstractmethod
    async def execute_capability(self, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool capability"""
        pass
    
    @abstractmethod
    def map_to_ontology(self, capability: str, output: Dict[str, Any], graph: OntologyGraph) -> List[str]:
        """Map tool output to ontological graph nodes"""
        pass

class GitHubCLIAdapter(ToolAdapter):
    """GitHub CLI (gh) tool adapter"""
    
    def __init__(self):
        self.name = "gh"
        self.tool_path = shutil.which("gh")
    
    def check_available(self) -> bool:
        """Check if GitHub CLI is available and authenticated"""
        if not self.tool_path:
            return False
        
        try:
            # Check if authenticated
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return False
    
    def get_capabilities(self) -> List[ToolCapability]:
        """Get GitHub CLI capabilities"""
        return [
            ToolCapability(
                name="list_issues",
                tool="gh",
                command_pattern="gh issue list --json number,title,state,labels,assignees,createdAt",
                input_types=["repository"],
                output_types=["issues"],
                ontology_mapping={"issues": "ISSUE"}
            ),
            ToolCapability(
                name="list_prs",
                tool="gh",
                command_pattern="gh pr list --json number,title,state,labels,author,createdAt",
                input_types=["repository"],
                output_types=["pull_requests"],
                ontology_mapping={"pull_requests": "PULLREQUEST"}
            ),
            ToolCapability(
                name="create_issue",
                tool="gh",
                command_pattern="gh issue create --title '{title}' --body '{body}' --label '{labels}'",
                input_types=["title", "body", "labels"],
                output_types=["issue"],
                ontology_mapping={"issue": "ISSUE"}
            ),
            ToolCapability(
                name="repo_info",
                tool="gh",
                command_pattern="gh repo view --json name,description,language,stargazerCount,forkCount",
                input_types=["repository"],
                output_types=["repository_info"],
                ontology_mapping={"repository_info": "PROJECT"}
            ),
            ToolCapability(
                name="workflow_runs",
                tool="gh",
                command_pattern="gh run list --json status,conclusion,workflowName,createdAt",
                input_types=["repository"],
                output_types=["workflows"],
                ontology_mapping={"workflows": "PIPELINE"}
            )
        ]
    
    async def execute_capability(self, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub CLI capability"""
        capabilities = {cap.name: cap for cap in self.get_capabilities()}
        
        if capability not in capabilities:
            raise ValueError(f"Unknown capability: {capability}")
        
        cap = capabilities[capability]
        
        try:
            if capability == "list_issues":
                result = subprocess.run(
                    ["gh", "issue", "list", "--json", "number,title,state,labels,assignees,createdAt"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    issues = json.loads(result.stdout) if result.stdout.strip() else []
                    return {"issues": issues, "count": len(issues)}
                else:
                    return {"error": result.stderr, "issues": []}
            
            elif capability == "list_prs":
                result = subprocess.run(
                    ["gh", "pr", "list", "--json", "number,title,state,labels,author,createdAt"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    prs = json.loads(result.stdout) if result.stdout.strip() else []
                    return {"pull_requests": prs, "count": len(prs)}
                else:
                    return {"error": result.stderr, "pull_requests": []}
            
            elif capability == "repo_info":
                result = subprocess.run(
                    ["gh", "repo", "view", "--json", "name,description,language,stargazerCount,forkCount"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    repo_info = json.loads(result.stdout) if result.stdout.strip() else {}
                    return {"repository_info": repo_info}
                else:
                    return {"error": result.stderr, "repository_info": {}}
            
            elif capability == "workflow_runs":
                result = subprocess.run(
                    ["gh", "run", "list", "--json", "status,conclusion,workflowName,createdAt", "--limit", "20"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    workflows = json.loads(result.stdout) if result.stdout.strip() else []
                    return {"workflows": workflows, "count": len(workflows)}
                else:
                    return {"error": result.stderr, "workflows": []}
            
            elif capability == "create_issue":
                title = params.get("title", "")
                body = params.get("body", "")
                labels = params.get("labels", "")
                
                cmd = ["gh", "issue", "create", "--title", title, "--body", body]
                if labels:
                    cmd.extend(["--label", labels])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    return {"issue_url": result.stdout.strip(), "success": True}
                else:
                    return {"error": result.stderr, "success": False}
            
            else:
                return {"error": f"Capability {capability} not implemented"}
        
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {e}"}
        except Exception as e:
            return {"error": f"Execution error: {e}"}
    
    def map_to_ontology(self, capability: str, output: Dict[str, Any], graph: OntologyGraph) -> List[str]:
        """Map GitHub CLI output to ontological graph nodes"""
        created_nodes = []
        
        if capability == "list_issues" and "issues" in output:
            for issue in output["issues"]:
                node = create_node(
                    NodeType.ISSUE,
                    title=f"Issue #{issue['number']}: {issue['title']}",
                    content={
                        "github_number": issue["number"],
                        "state": issue["state"],
                        "labels": [label["name"] for label in issue.get("labels", [])],
                        "assignees": [assignee["login"] for assignee in issue.get("assignees", [])],
                        "created_at": issue.get("createdAt"),
                        "source": "github_cli"
                    }
                )
                graph.add_node(node)
                created_nodes.append(node.id)
        
        elif capability == "list_prs" and "pull_requests" in output:
            for pr in output["pull_requests"]:
                node = create_node(
                    NodeType.PULLREQUEST,
                    title=f"PR #{pr['number']}: {pr['title']}",
                    content={
                        "github_number": pr["number"],
                        "state": pr["state"],
                        "labels": [label["name"] for label in pr.get("labels", [])],
                        "author": pr.get("author", {}).get("login", ""),
                        "created_at": pr.get("createdAt"),
                        "source": "github_cli"
                    }
                )
                graph.add_node(node)
                created_nodes.append(node.id)
        
        elif capability == "repo_info" and "repository_info" in output:
            repo_info = output["repository_info"]
            if repo_info:
                node = create_node(
                    NodeType.PROJECT,
                    title=repo_info.get("name", "Unknown Project"),
                    content={
                        "description": repo_info.get("description", ""),
                        "language": repo_info.get("language", ""),
                        "stars": repo_info.get("stargazerCount", 0),
                        "forks": repo_info.get("forkCount", 0),
                        "source": "github_cli"
                    }
                )
                graph.add_node(node)
                created_nodes.append(node.id)
        
        elif capability == "workflow_runs" and "workflows" in output:
            for workflow in output["workflows"]:
                node = create_node(
                    NodeType.PIPELINE,
                    title=f"Workflow: {workflow.get('workflowName', 'Unknown')}",
                    content={
                        "status": workflow.get("status"),
                        "conclusion": workflow.get("conclusion"),
                        "created_at": workflow.get("createdAt"),
                        "source": "github_cli"
                    }
                )
                graph.add_node(node)
                created_nodes.append(node.id)
        
        return created_nodes

class DockerAdapter(ToolAdapter):
    """Docker tool adapter"""
    
    def __init__(self):
        self.name = "docker"
        self.tool_path = shutil.which("docker")
    
    def check_available(self) -> bool:
        """Check if Docker is available"""
        if not self.tool_path:
            return False
        
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return False
    
    def get_capabilities(self) -> List[ToolCapability]:
        """Get Docker capabilities"""
        return [
            ToolCapability(
                name="list_containers",
                tool="docker",
                command_pattern="docker ps -a --format json",
                input_types=[],
                output_types=["containers"],
                ontology_mapping={"containers": "ENVIRONMENT"}
            ),
            ToolCapability(
                name="list_images",
                tool="docker",
                command_pattern="docker images --format json",
                input_types=[],
                output_types=["images"],
                ontology_mapping={"images": "ARTIFACT"}
            )
        ]
    
    async def execute_capability(self, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Docker capability"""
        try:
            if capability == "list_containers":
                result = subprocess.run(
                    ["docker", "ps", "-a", "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    # Parse JSON lines
                    containers = []
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            containers.append(json.loads(line))
                    return {"containers": containers, "count": len(containers)}
                else:
                    return {"error": result.stderr, "containers": []}
            
            elif capability == "list_images":
                result = subprocess.run(
                    ["docker", "images", "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    images = []
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            images.append(json.loads(line))
                    return {"images": images, "count": len(images)}
                else:
                    return {"error": result.stderr, "images": []}
            
            else:
                return {"error": f"Capability {capability} not implemented"}
        
        except Exception as e:
            return {"error": f"Execution error: {e}"}
    
    def map_to_ontology(self, capability: str, output: Dict[str, Any], graph: OntologyGraph) -> List[str]:
        """Map Docker output to ontological graph nodes"""
        created_nodes = []
        
        if capability == "list_containers" and "containers" in output:
            for container in output["containers"]:
                node = create_node(
                    NodeType.ENVIRONMENT,
                    title=f"Container: {container.get('Names', 'Unknown')}",
                    content={
                        "image": container.get("Image"),
                        "status": container.get("Status"),
                        "ports": container.get("Ports"),
                        "source": "docker"
                    }
                )
                graph.add_node(node)
                created_nodes.append(node.id)
        
        elif capability == "list_images" and "images" in output:
            for image in output["images"]:
                node = create_node(
                    NodeType.ARTIFACT,
                    title=f"Docker Image: {image.get('Repository', 'Unknown')}",
                    content={
                        "tag": image.get("Tag"),
                        "size": image.get("Size"),
                        "created": image.get("CreatedAt"),
                        "source": "docker"
                    }
                )
                graph.add_node(node)
                created_nodes.append(node.id)
        
        return created_nodes

class ToolRegistry:
    """Registry for discovering and managing development tools"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tools: Dict[str, ToolInfo] = {}
        self.adapters: Dict[str, ToolAdapter] = {}
        self.capabilities: Dict[str, List[str]] = {}  # capability -> tools that provide it
        
        # Register built-in adapters
        self._register_builtin_adapters()
    
    def _register_builtin_adapters(self):
        """Register built-in tool adapters"""
        adapters = [
            GitHubCLIAdapter(),
            DockerAdapter(),
        ]
        
        for adapter in adapters:
            self.adapters[adapter.name] = adapter
    
    def discover_tools(self) -> Dict[str, bool]:
        """Discover available development tools"""
        print("🔍 Discovering available development tools...")
        
        discovered = {}
        
        for name, adapter in self.adapters.items():
            try:
                available = adapter.check_available()
                discovered[name] = available
                
                if available:
                    # Get tool info
                    version = self._get_tool_version(name)
                    path = shutil.which(name) or "unknown"
                    capabilities = adapter.get_capabilities()
                    
                    self.tools[name] = ToolInfo(
                        name=name,
                        version=version,
                        path=path,
                        capabilities=capabilities,
                        available=True,
                        last_checked=time.time()
                    )
                    
                    # Register capabilities
                    for cap in capabilities:
                        if cap.name not in self.capabilities:
                            self.capabilities[cap.name] = []
                        self.capabilities[cap.name].append(name)
                    
                    print(f"   ✅ {name} ({version}) - {len(capabilities)} capabilities")
                else:
                    print(f"   ❌ {name} - not available")
            
            except Exception as e:
                print(f"   ⚠️  {name} - discovery failed: {e}")
                discovered[name] = False
        
        return discovered
    
    def _get_tool_version(self, tool_name: str) -> str:
        """Get tool version"""
        version_commands = {
            "gh": ["gh", "--version"],
            "docker": ["docker", "--version"],
            "git": ["git", "--version"],
            "npm": ["npm", "--version"],
            "python": ["python", "--version"],
        }
        
        if tool_name not in version_commands:
            return "unknown"
        
        try:
            result = subprocess.run(
                version_commands[tool_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Extract version from output
                output = result.stdout.strip()
                if tool_name == "gh":
                    # "gh version 2.40.1 (2023-12-13)"
                    parts = output.split()
                    return parts[2] if len(parts) > 2 else "unknown"
                elif tool_name == "docker":
                    # "Docker version 24.0.7, build afdd53b"
                    parts = output.split()
                    return parts[2].rstrip(',') if len(parts) > 2 else "unknown"
                else:
                    return output.split()[-1] if output.split() else "unknown"
            
            return "unknown"
        except:
            return "unknown"
    
    def get_available_capabilities(self) -> List[str]:
        """Get list of available capabilities"""
        return list(self.capabilities.keys())
    
    def can_execute(self, capability: str) -> bool:
        """Check if capability can be executed"""
        return capability in self.capabilities and len(self.capabilities[capability]) > 0
    
    async def execute_capability(self, capability: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a capability using available tools"""
        if not self.can_execute(capability):
            return {"error": f"Capability {capability} not available"}
        
        params = params or {}
        
        # Try tools that provide this capability
        for tool_name in self.capabilities[capability]:
            if tool_name in self.adapters and tool_name in self.tools:
                adapter = self.adapters[tool_name]
                
                try:
                    result = await adapter.execute_capability(capability, params)
                    return {"tool": tool_name, "result": result}
                except Exception as e:
                    print(f"   ⚠️  {tool_name} failed for {capability}: {e}")
                    continue
        
        return {"error": f"All tools failed for capability {capability}"}
    
    def integrate_with_ontology(self, capability: str, output: Dict[str, Any], graph: OntologyGraph) -> List[str]:
        """Integrate tool output with ontological graph"""
        created_nodes = []
        
        if "tool" not in output or "result" not in output:
            return created_nodes
        
        tool_name = output["tool"]
        tool_output = output["result"]
        
        if tool_name in self.adapters:
            adapter = self.adapters[tool_name]
            try:
                nodes = adapter.map_to_ontology(capability, tool_output, graph)
                created_nodes.extend(nodes)
            except Exception as e:
                print(f"   ⚠️  Ontology mapping failed for {tool_name}: {e}")
        
        return created_nodes
    
    def get_tool_summary(self) -> Dict[str, Any]:
        """Get summary of discovered tools"""
        available_tools = [name for name, info in self.tools.items() if info.available]
        total_capabilities = sum(len(info.capabilities) for info in self.tools.values() if info.available)
        
        return {
            "total_tools": len(self.tools),
            "available_tools": len(available_tools),
            "tools": available_tools,
            "total_capabilities": total_capabilities,
            "capabilities": list(self.capabilities.keys())
        }