#!/usr/bin/env python3
"""
RE_ware Interactive Agent - Web Dashboard & CLI Interface
========================================================

Multi-mode RE_ware interface supporting both CLI and web-based interaction.
Web mode provides interactive Psi graph visualization with real-time updates.

Usage:
    python evolve.py                    # Interactive CLI mode (default)
    python evolve.py --web              # Launch with web dashboard
    python evolve.py --web --port 8080  # Web dashboard on custom port
    
    Interactive commands:
    re_ware> status                     # Show consciousness state
    re_ware> advice                     # Get project reasoning
    re_ware> tick                       # Execute evolution cycle
    re_ware> auto 5                     # Enable autonomous mode (5min intervals)
    re_ware> save                       # Save current state
    re_ware> quit                       # Exit system
"""

import sys
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

# Add the re_ware package to path
sys.path.insert(0, str(Path(__file__).parent))

from re_ware.evolution import REWareInteractive, REWareEvolver
from re_ware.ontology import list_available_gene_templates
from re_ware.sensors import GitSensor, FsSensor, CliSensor, GhSensor
# from re_ware.ci_sensors import GitHubActionsSensor, JUnitTestSensor, PytestCoverageSensor, TestResultSensor

# Web server imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
    import uvicorn
    WEB_DEPENDENCIES_AVAILABLE = True
except ImportError:
    WEB_DEPENDENCIES_AVAILABLE = False


class REWareWebServer:
    """Web-based RE_ware dashboard with interactive graph visualization"""
    
    def __init__(self, project_root: Path, schema_name: str = "project_manager", port: int = 8000, enable_web_ui: bool = False):
        self.project_root = project_root
        self.schema_name = schema_name
        self.port = port
        self.enable_web_ui = enable_web_ui
        self.agent = None
        
        # Configure FastAPI based on mode
        if enable_web_ui:
            self.app = FastAPI(title="RE_ware Dashboard", description="Interactive Psi Graph Visualization")
        else:
            self.app = FastAPI(title="RE_ware API", description="REware Core API")
        
        self.active_connections: List[WebSocket] = []
        self.auto_tick_interval = 0  # 0 = manual, >0 = auto interval in minutes
        self.auto_tick_task = None
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes and static files"""
        
        # Root route that dynamically serves dashboard or API info
        @self.app.get("/", response_class=HTMLResponse)
        async def root_handler(request: Request):
            """Dynamic root handler - dashboard if web UI enabled, API info otherwise"""
            if self.enable_web_ui:
                return HTMLResponse(content=self.get_dashboard_html(), status_code=200)
            else:
                # Return API info as JSON but handle HTML response type
                from fastapi.responses import JSONResponse
                return JSONResponse({
                    "name": "RE_ware Core API",
                    "version": "1.0.0",
                    "endpoints": {
                        "/api/status": "System status",
                        "/api/graph": "Ontology graph",
                        "/api/tick": "Evolution tick",
                        "/api/advice": "Get advice",
                        "/api/actions": "List actions"
                    },
                    "note": "Use 'ui' command in CLI to enable web dashboard"
                })
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current system status"""
            return await self.get_status_data()
        
        @self.app.get("/api/graph")
        async def get_graph():
            """Get current graph data for visualization"""
            return await self.get_graph_data()
        
        @self.app.post("/api/tick")
        async def manual_tick():
            """Trigger manual evolution tick"""
            if not self.agent:
                return {"error": "Agent not initialized"}
            
            await self.agent._cmd_tick()
            
            # Broadcast update to connected clients
            await self.broadcast_update()
            
            return {"status": "Tick completed"}
        
        @self.app.post("/api/advice")
        async def get_advice():
            """Generate project advice with caching"""
            if not self.agent:
                return {"error": "Agent not initialized"}
            
            # Use the new caching method
            return await self.agent.get_advice_with_caching()
        
        @self.app.get("/api/actions")
        async def get_actions():
            """Get current idempotent actions from last advice frame"""
            if not self.agent:
                return {"error": "Agent not initialized"}
            
            try:
                phi_signals = self.agent.ontology.phi_signals()
                advice_frame = await self.agent.re_agent.generate_advice_frame(
                    species_id="project_manager_v1",
                    instance_id=getattr(self.agent.ontology, 'instance_id', 'unknown'),
                    phi_state=phi_signals
                )
                
                actions = advice_frame.get('actions', [])
                
                # Process actions for UI display
                processed_actions = []
                for action in actions:
                    processed_actions.append({
                        "id": action.get('idempotency_key', 'no-key'),
                        "kind": action.get('kind', 'unknown'),
                        "title": action.get('title', 'No title'),
                        "body": action.get('body', ''),
                        "targets": action.get('targets', []),
                        "params": action.get('params', {}),
                        "idempotency_key": action.get('idempotency_key', 'no-key'),
                        "status": "pending"  # Could be "pending", "executed", "failed"
                    })
                
                return {
                    "actions": processed_actions,
                    "total": len(processed_actions),
                    "phi_state": phi_signals
                }
            except Exception as e:
                return {"error": f"Action retrieval failed: {e}"}
        
        @self.app.post("/api/execute")
        async def execute_actions():
            """Execute actions from most recent advice"""
            if not self.agent or not self.agent.action_dispatcher:
                return {"error": "Agent or action dispatcher not initialized"}
            
            try:
                # Find the most recent advice with actions
                from re_ware.ontology import NodeType
                advice_nodes = [n for n in self.agent.ontology.nodes.values() if n.type == NodeType.ADVICE]
                if not advice_nodes:
                    return {"error": "No advice found. Generate advice first."}
                
                # Sort by creation time (most recent first)
                advice_nodes.sort(key=lambda n: n.content.get('generated_at', 0), reverse=True)
                most_recent = advice_nodes[0]
                
                actions = most_recent.content.get('actions', [])
                if not actions:
                    return {"message": "No actions to execute in most recent advice", "executed": 0}
                
                # Execute actions
                results = await self.agent.action_dispatcher.dispatch_actions(actions)
                
                # Gather external references
                external_refs = []
                for result in results:
                    if result.external_refs:
                        external_refs.extend(result.external_refs)
                
                successful = sum(1 for r in results if r.success)
                failed = len(results) - successful
                
                return {
                    "message": f"Executed {len(actions)} actions",
                    "executed": len(actions),
                    "successful": successful,
                    "failed": failed,
                    "external_refs": external_refs,
                    "results": [
                        {
                            "kind": r.kind,
                            "success": r.success,
                            "message": r.message,
                            "idempotency_key": r.idempotency_key
                        } for r in results
                    ]
                }
                
            except Exception as e:
                return {"error": f"Execution failed: {e}"}
        
        @self.app.post("/api/save")
        async def save_snapshot():
            """Save current Ψ snapshot"""
            if not self.agent:
                return {"error": "Agent not initialized"}
            
            try:
                # Get current phi signals for snapshot
                phi_signals = self.agent.ontology.phi_signals()
                
                # Use the existing snapshot path or create default
                snapshot_path = self.agent.ontology.warm_snapshot_path
                if not snapshot_path:
                    from pathlib import Path
                    from re_ware.core import SNAPSHOT_FILENAME
                    snapshot_path = Path(SNAPSHOT_FILENAME)
                
                # Save the snapshot
                success = self.agent.ontology.save_snapshot(snapshot_path, phi_data={
                    "phi0": 0.0,  # Will be computed by system
                    "coverage": phi_signals.get("coverage_ratio", 0.0),
                    "counters": {
                        "nodes": len(self.agent.ontology.nodes),
                        "edges": len(self.agent.ontology.edges),
                        "changed": len(self.agent.ontology.hot_state.changed_nodes)
                    },
                    "computed_at": __import__('time').time()
                })
                
                if success:
                    return {
                        "status": "Snapshot saved successfully",
                        "path": str(snapshot_path),
                        "nodes": len(self.agent.ontology.nodes),
                        "edges": len(self.agent.ontology.edges)
                    }
                else:
                    return {"error": "Failed to save snapshot"}
                    
            except Exception as e:
                return {"error": f"Save failed: {e}"}
        
        @self.app.post("/api/expand-group/{group_id}")
        async def expand_group(group_id: str):
            """Expand a group to show its individual files"""
            if not self.agent:
                return {"error": "Agent not initialized"}
            
            try:
                # Get detailed graph data for the specific group
                detailed_graph = self._get_expanded_group_data(group_id)
                return detailed_graph
            except Exception as e:
                return {"error": f"Group expansion failed: {e}"}
        
        @self.app.post("/api/auto-tick/{interval}")
        async def set_auto_tick(interval: int):
            """Set automatic tick interval in minutes (0 = disabled)"""
            self.auto_tick_interval = max(0, interval)
            
            # Cancel existing auto-tick task
            if self.auto_tick_task and not self.auto_tick_task.done():
                self.auto_tick_task.cancel()
            
            # Start new auto-tick task if interval > 0
            if self.auto_tick_interval > 0:
                self.auto_tick_task = asyncio.create_task(self.auto_tick_loop())
            
            return {"status": f"Auto-tick set to {self.auto_tick_interval}m"}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    await websocket.receive_text()  # Keep connection alive
            except WebSocketDisconnect:
                # Safe removal - only remove if present
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
            except Exception as e:
                # Handle any other websocket errors
                print(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    async def auto_tick_loop(self):
        """Background task for automatic ticking"""
        try:
            while self.auto_tick_interval > 0:
                # Convert minutes to seconds
                await asyncio.sleep(self.auto_tick_interval * 60)
                if self.agent:
                    await self.agent._cmd_tick()
                    await self.broadcast_update()
        except asyncio.CancelledError:
            pass
    
    async def broadcast_update(self):
        """Broadcast graph updates to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        # Get current graph and status data directly
        graph_data = await self.get_graph_data()
        status_data = await self.get_status_data()
        
        update_data = {
            "type": "graph_update",
            "graph": graph_data,
            "status": status_data
        }
        
        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(update_data))
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)
    
    async def get_status_data(self):
        """Get current system status data"""
        if not self.agent:
            return {
                "phi_metrics": {
                    "phi0": 0.0,
                    "coherence": 0.0,
                    "stability": False,
                    "cycles": 0
                },
                "ontology_state": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "changed_nodes": 0,
                    "coverage_ratio": 0.0,
                    "entropy_hint": 0.0
                },
                "auto_tick_interval": self.auto_tick_interval,
                "status": "initializing"
            }
        
        try:
            phi_signals = self.agent.ontology.phi_signals()
            return {
                "phi_metrics": {
                    "phi0": self.agent.state.phi0,
                    "coherence": self.agent.state.phi_coherence,
                    "stability": self.agent.state.stability_check,
                    "cycles": self.agent.state.cycles_completed
                },
                "ontology_state": {
                    "total_nodes": len(self.agent.ontology.nodes),
                    "total_edges": len(self.agent.ontology.edges),
                    "changed_nodes": len(self.agent.ontology.hot_state.changed_nodes),
                    "coverage_ratio": phi_signals.get('coverage_ratio', 0.0),
                    "entropy_hint": phi_signals.get('entropy_hint', 0.0)
                },
                "auto_tick_interval": self.auto_tick_interval,
                "status": "active"
            }
        except Exception as e:
            return {
                "phi_metrics": {
                    "phi0": 0.0,
                    "coherence": 0.0,
                    "stability": False,
                    "cycles": 0
                },
                "ontology_state": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "changed_nodes": 0,
                    "coverage_ratio": 0.0,
                    "entropy_hint": 0.0
                },
                "auto_tick_interval": self.auto_tick_interval,
                "status": "error",
                "error": str(e)
            }
    
    async def get_graph_data(self):
        """Get hierarchical graph data for visualization"""
        if not self.agent:
            # Return a placeholder while agent is initializing
            return {
                "nodes": [{
                    "id": "initializing",
                    "label": "🧠 RE_ware Initializing...",
                    "type": "SYSTEM",
                    "status": "initializing",
                    "criticality": "P1",
                    "version": "1.0",
                    "changed": True,
                    "full_title": "RE_ware System Initializing",
                    "level": "system"
                }],
                "edges": []
            }
        
        try:
            return self._create_hierarchical_graph()
        except Exception as e:
            # Return error state if graph creation fails
            return {
                "nodes": [{
                    "id": "error", 
                    "label": f"❌ Graph Error: {str(e)[:50]}...",
                    "type": "ERROR",
                    "status": "error",
                    "criticality": "P0",
                    "version": "1.0",
                    "changed": True,
                    "full_title": f"Graph generation error: {e}",
                    "level": "system"
                }],
                "edges": []
            }
    
    def _create_hierarchical_graph(self):
        """Create hierarchical graph with major components and expandable groups"""
        nodes = []
        edges = []
        changed_nodes = set(self.agent.ontology.hot_state.changed_nodes)
        
        # Get all ontology nodes
        ontology_nodes = self.agent.ontology.nodes
        
        # Group nodes by type for hierarchical display
        node_groups = {
            "CODE": [],
            "TESTS": [],
            "DOCS": [],
            "CONFIG": [],
            "OTHER": []
        }
        
        # Categorize nodes
        for node_id, node in ontology_nodes.items():
            node_type = node.type.name
            
            if node_type == "PROJECT":
                # PROJECT node is always shown at top level
                nodes.append({
                    "id": node_id,
                    "label": node.title,
                    "type": node_type,
                    "status": node.state.status.name if hasattr(node.state.status, 'name') else str(node.state.status),
                    "criticality": node.state.criticality,
                    "version": node.state.version,
                    "changed": node_id in changed_nodes,
                    "full_title": node.title,
                    "level": "project",
                    "expanded": False
                })
            elif node_type in ["CODEMODULE"]:
                node_groups["CODE"].append((node_id, node))
            elif node_type in ["TEST", "TESTSUITE"]:
                node_groups["TESTS"].append((node_id, node))
            elif node_type in ["TECHNICALDOC", "USERDOC", "APIDOC"]:
                node_groups["DOCS"].append((node_id, node))
            elif node_type in ["DEPENDENCY_SPEC", "CONFIGURATION"]:
                node_groups["CONFIG"].append((node_id, node))
            else:
                node_groups["OTHER"].append((node_id, node))
        
        # Create group nodes for major categories
        for group_name, group_nodes in node_groups.items():
            if not group_nodes:
                continue
                
            # Count changes in this group
            group_changed = sum(1 for node_id, _ in group_nodes if node_id in changed_nodes)
            total_in_group = len(group_nodes)
            
            # Create group node
            group_id = f"GROUP_{group_name}"
            group_label = f"{group_name} ({total_in_group})"
            if group_changed > 0:
                group_label += f" • {group_changed} changed"
            
            nodes.append({
                "id": group_id,
                "label": group_label,
                "type": f"GROUP_{group_name}",
                "status": "active",
                "criticality": "P1",
                "version": "1.0",
                "changed": group_changed > 0,
                "full_title": f"{group_name} Component Group",
                "level": "group",
                "expanded": False,
                "child_count": total_in_group,
                "changed_count": group_changed,
                "children": [node_id for node_id, _ in group_nodes]
            })
            
            # Create edge from PROJECT to group
            project_nodes = [n for n in nodes if n.get("level") == "project"]
            if project_nodes:
                edges.append({
                    "id": f"project_to_{group_name}",
                    "from": project_nodes[0]["id"],
                    "to": group_id,
                    "type": "CONTAINS",
                    "label": "contains"
                })
        
        return {"nodes": nodes, "edges": edges}
    
    def _get_expanded_group_data(self, group_id: str):
        """Get detailed graph data when a group is expanded"""
        # First get the hierarchical graph
        base_graph = self._create_hierarchical_graph()
        
        if not group_id.startswith("GROUP_"):
            return base_graph
        
        group_type = group_id.replace("GROUP_", "")
        
        # Find the group node
        group_node = None
        for node in base_graph["nodes"]:
            if node["id"] == group_id:
                group_node = node
                break
        
        if not group_node or not group_node.get("children"):
            return base_graph
        
        # Get the ontology nodes for this group
        ontology_nodes = self.agent.ontology.nodes
        changed_nodes = set(self.agent.ontology.hot_state.changed_nodes)
        
        # Add individual file nodes for this group
        expanded_nodes = list(base_graph["nodes"])  # Start with existing nodes
        expanded_edges = list(base_graph["edges"])
        
        for child_id in group_node["children"]:
            if child_id in ontology_nodes:
                node = ontology_nodes[child_id]
                expanded_nodes.append({
                    "id": child_id,
                    "label": node.title[:30] + ("..." if len(node.title) > 30 else ""),
                    "type": node.type.name,
                    "status": node.state.status.name if hasattr(node.state.status, 'name') else str(node.state.status),
                    "criticality": node.state.criticality,
                    "version": node.state.version,
                    "changed": child_id in changed_nodes,
                    "full_title": node.title,
                    "level": "file",
                    "parent_group": group_id
                })
                
                # Create edge from group to child
                expanded_edges.append({
                    "id": f"{group_id}_to_{child_id}",
                    "from": group_id,
                    "to": child_id,
                    "type": "CONTAINS",
                    "label": ""
                })
        
        # Mark the group as expanded
        for node in expanded_nodes:
            if node["id"] == group_id:
                node["expanded"] = True
                node["label"] = node["label"].replace(f" ({group_node['child_count']})", f" [EXPANDED]")
                break
        
        return {"nodes": expanded_nodes, "edges": expanded_edges}
    
    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML with embedded JavaScript"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RE_ware Dashboard - Ψ Graph Visualization</title>
    <script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
        }
        
        .header {
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .title {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .btn {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn:hover {
            background: rgba(255,255,255,0.2);
            transform: translateY(-1px);
        }
        
        .btn.active {
            background: #4CAF50;
            border-color: #45a049;
        }
        
        .main-content {
            display: flex;
            height: calc(100vh - 80px);
        }
        
        .graph-panel {
            flex: 1;
            position: relative;
            background: rgba(255,255,255,0.05);
            margin: 1rem;
            border-radius: 8px;
            overflow: hidden;
        }
        
        #psi-graph {
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.1);
        }
        
        .sidebar {
            width: 300px;
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            overflow-y: auto;
        }
        
        .panel {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .panel h3 {
            margin-top: 0;
            color: #4CAF50;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-weight: bold;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 0.5rem;
        }
        
        .status-pass { background: #4CAF50; }
        .status-fail { background: #f44336; }
        .status-warn { background: #ff9800; }
        
        .advice-content {
            max-height: 200px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            padding: 0.5rem;
            border-radius: 4px;
            margin-top: 0.5rem;
        }
        
        .node-info {
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
            padding: 0.5rem;
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }
        
        input[type="range"] {
            width: 100%;
            margin: 0.5rem 0;
        }
        
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin: 1rem;
            font-size: 0.9rem;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">🧠 RE_ware Dashboard - Ψ Graph</div>
        <div class="controls">
            <button class="btn" onclick="manualTick()">Manual Tick ⚡</button>
            <button class="btn" onclick="getAdvice()">Get Advice 🤖</button>
            <button class="btn" onclick="executeActions()">Execute Actions 🚀</button>
            <button class="btn" onclick="resetGraph()">Reset View 🔄</button>
            <span>Auto: <input type="range" min="0" max="10" value="0" id="autoSlider" oninput="setAutoTick(this.value)"> <span id="autoValue">Manual</span></span>
        </div>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(45deg, #FFD700, #FFA500); border-radius: 50%; width: 20px; height: 20px;"></div>
            <span><strong>Project Hub ⭐</strong></span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #4CAF50; width: 20px; height: 20px; border-radius: 3px;"></div>
            <span><strong>CODE Group</strong> (double-click to expand)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #FF9800; width: 20px; height: 20px; border-radius: 3px;"></div>
            <span><strong>TESTS Group</strong> (double-click to expand)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #2196F3; width: 20px; height: 20px; border-radius: 3px;"></div>
            <span><strong>DOCS Group</strong> (double-click to expand)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #9C27B0; width: 20px; height: 20px; border-radius: 3px;"></div>
            <span><strong>CONFIG Group</strong> (double-click to expand)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f44336;"></div>
            <span>Changed/New Files</span>
        </div>
    </div>
    
    <div class="main-content">
        <div class="graph-panel">
            <div id="psi-graph"></div>
        </div>
        
        <div class="sidebar">
            <div class="panel">
                <h3>📊 Phi Metrics</h3>
                <div class="metric">
                    <span>Φ₀ (Stability):</span>
                    <span class="metric-value" id="phi0">0.000</span>
                    <span class="status-indicator status-fail" id="phi0-status"></span>
                </div>
                <div class="metric">
                    <span>Coherence:</span>
                    <span class="metric-value" id="coherence">0.000</span>
                </div>
                <div class="metric">
                    <span>Stability Check:</span>
                    <span class="metric-value" id="stability">FAIL</span>
                    <span class="status-indicator status-fail" id="stability-status"></span>
                </div>
                <div class="metric">
                    <span>Evolution Cycles:</span>
                    <span class="metric-value" id="cycles">0</span>
                </div>
            </div>
            
            <div class="panel">
                <h3>🧬 Ontology State</h3>
                <div class="metric">
                    <span>Total Nodes:</span>
                    <span class="metric-value" id="total-nodes">0</span>
                </div>
                <div class="metric">
                    <span>Total Edges:</span>
                    <span class="metric-value" id="total-edges">0</span>
                </div>
                <div class="metric">
                    <span>Changed Nodes:</span>
                    <span class="metric-value" id="changed-nodes">0</span>
                </div>
                <div class="metric">
                    <span>Coverage Ratio:</span>
                    <span class="metric-value" id="coverage-ratio">0.00</span>
                </div>
            </div>
            
            <div class="panel">
                <h3>🤖 AI Advice</h3>
                <div class="advice-content" id="advice-content">
                    Click "Get Advice" to receive AI recommendations...
                </div>
            </div>
            
            <div class="panel">
                <h3>🔍 Node Details</h3>
                <div class="node-info" id="node-info">
                    Click on a node to see details...
                </div>
            </div>
            
            <div class="panel">
                <h3>⚡ Action Hub</h3>
                <div class="metric">
                    <span>Pending Actions:</span>
                    <span class="metric-value" id="action-count">0</span>
                </div>
                <div class="advice-content" id="action-content">
                    <button class="btn" onclick="loadActions()" style="width: 100%; margin-bottom: 0.5rem;">Load Actions</button>
                    <div id="action-list">
                        Click "Load Actions" to see idempotent actions...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let network;
        let nodes = new vis.DataSet();
        let edges = new vis.DataSet();
        let ws;
        
        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => console.log('WebSocket connected');
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'graph_update') {
                    updateGraph(data.graph);
                    updateStatus(data.status);
                }
            };
            ws.onclose = () => {
                console.log('WebSocket disconnected, attempting reconnect...');
                setTimeout(initWebSocket, 3000);
            };
        }
        
        // Initialize graph visualization
        function initGraph() {
            const container = document.getElementById('psi-graph');
            const data = { nodes: nodes, edges: edges };
            
            const options = {
                nodes: {
                    shape: 'dot',
                    size: 20,
                    font: { color: 'white', size: 12 },
                    borderWidth: 2,
                    shadow: true
                },
                edges: {
                    width: 2,
                    color: { color: 'rgba(255,255,255,0.4)' },
                    arrows: { to: { enabled: true, scaleFactor: 0.8 } },
                    smooth: { type: 'continuous' }
                },
                physics: {
                    stabilization: { iterations: 150 },
                    barnesHut: { 
                        gravitationalConstant: -3000, 
                        springLength: 200,
                        springConstant: 0.04,
                        damping: 0.09
                    },
                    maxVelocity: 50,
                    minVelocity: 0.1,
                    timestep: 0.5
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 300
                }
            };
            
            network = new vis.Network(container, data, options);
            
            // Center PROJECT node after stabilization
            network.on('stabilizationIterationsDone', () => {
                centerProjectNode();
            });
            
            // Node selection handler
            network.on('click', (params) => {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    showNodeDetails(nodeId);
                }
            });
            
            // Double-click handler for group expansion
            network.on('doubleClick', (params) => {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    expandGroup(nodeId);
                }
            });
        }
        
        // Center the PROJECT node in the visualization
        function centerProjectNode() {
            if (!network) return;
            
            // Find PROJECT node
            const projectNodeId = nodes.get().find(node => 
                node.id === 'project:root' || 
                (typeof node.type !== 'undefined' && node.type === 'PROJECT')
            )?.id;
            
            if (projectNodeId) {
                // Position PROJECT node at center (0, 0)
                network.moveNode(projectNodeId, 0, 0);
                
                // Briefly focus on the project node to center the view
                network.focus(projectNodeId, {
                    scale: 0.8,
                    animation: {
                        duration: 1000,
                        easingFunction: 'easeInOutQuad'
                    }
                });
                
                console.log('Centered PROJECT node:', projectNodeId);
            }
        }
        
        // Expand a group to show individual files
        async function expandGroup(nodeId) {
            // Check if this is a group node that can be expanded
            const node = nodes.get(nodeId);
            if (!node || !nodeId.startsWith('GROUP_')) {
                return;
            }
            
            console.log('Expanding group:', nodeId);
            
            try {
                const response = await fetch(`/api/expand-group/${nodeId}`, { method: 'POST' });
                const expandedData = await response.json();
                
                if (expandedData.error) {
                    console.error('Group expansion failed:', expandedData.error);
                    return;
                }
                
                // Update the graph with expanded data
                updateGraph(expandedData);
                
                // Re-center on the project node
                setTimeout(() => centerProjectNode(), 500);
                
            } catch (error) {
                console.error('Group expansion error:', error);
            }
        }
        
        // Update graph with new data
        function updateGraph(graphData) {
            const nodeUpdates = graphData.nodes.map(node => {
                // Special treatment for different node levels
                const isProject = node.type === 'PROJECT' || node.id === 'project:root';
                const isGroup = node.type && node.type.startsWith('GROUP_');
                
                let nodeConfig = {
                    id: node.id,
                    label: node.label,
                    color: getNodeColor(node),
                    physics: true
                };
                
                if (isProject) {
                    // PROJECT node: central star
                    nodeConfig.title = `${node.full_title}\\\\nProject Hub - Click groups to expand`;
                    nodeConfig.borderWidth = 6;
                    nodeConfig.font = { color: 'white', size: 18, face: 'arial black' };
                    nodeConfig.size = 40;
                    nodeConfig.shape = 'star';
                    nodeConfig.physics = false;
                    nodeConfig.fixed = { x: true, y: true };
                } else if (isGroup) {
                    // GROUP nodes: expandable containers
                    const groupType = node.type.replace('GROUP_', '');
                    nodeConfig.title = `${node.full_title}\\\\nFiles: ${node.child_count}\\\\nChanged: ${node.changed_count}\\\\nDouble-click to expand`;
                    nodeConfig.borderWidth = node.changed ? 4 : 3;
                    nodeConfig.font = { 
                        color: 'white', 
                        size: node.changed ? 16 : 14,
                        face: 'arial'
                    };
                    nodeConfig.size = 30 + Math.min(node.child_count * 2, 20); // Size based on content
                    nodeConfig.shape = 'box';
                } else {
                    // Individual file nodes (when expanded)
                    nodeConfig.title = `${node.type}: ${node.full_title}\\\\nStatus: ${node.status}\\\\nCriticality: ${node.criticality}`;
                    nodeConfig.borderWidth = node.changed ? 4 : 2;
                    nodeConfig.font = { 
                        color: 'white', 
                        size: node.changed ? 14 : 12,
                        face: 'arial'
                    };
                    nodeConfig.size = node.changed ? 25 : 20;
                    nodeConfig.shape = 'dot';
                }
                
                return nodeConfig;
            });
            
            const edgeUpdates = graphData.edges.map(edge => ({
                id: edge.id,
                from: edge.from,
                to: edge.to,
                label: edge.label,
                title: edge.type
            }));
            
            nodes.update(nodeUpdates);
            edges.update(edgeUpdates);
        }
        
        // Get node color based on type and status
        function getNodeColor(node) {
            // Special handling for PROJECT node (central hub)
            if (node.type === 'PROJECT' || node.id === 'project:root') {
                return { 
                    background: 'linear-gradient(45deg, #FFD700, #FFA500)', 
                    border: '#FF8C00',
                    highlight: { background: '#FFD700', border: '#FF8C00' }
                };
            }
            
            if (node.changed) {
                return { background: '#f44336', border: '#d32f2f' };  // Red for changed
            }
            
            // Special handling for GROUP nodes
            if (node.type && node.type.startsWith('GROUP_')) {
                const groupType = node.type.replace('GROUP_', '');
                switch (groupType) {
                    case 'CODE':
                        return { background: '#4CAF50', border: '#2E7D32', highlight: { background: '#66BB6A', border: '#2E7D32' } };
                    case 'TESTS':
                        return { background: '#FF9800', border: '#E65100', highlight: { background: '#FFB74D', border: '#E65100' } };
                    case 'DOCS':
                        return { background: '#2196F3', border: '#1565C0', highlight: { background: '#42A5F5', border: '#1565C0' } };
                    case 'CONFIG':
                        return { background: '#9C27B0', border: '#6A1B9A', highlight: { background: '#BA68C8', border: '#6A1B9A' } };
                    default:
                        return { background: '#607D8B', border: '#37474F', highlight: { background: '#78909C', border: '#37474F' } };
                }
            }
            
            switch (node.type) {
                case 'CODEMODULE':
                    return { background: '#4CAF50', border: '#388E3C' };
                case 'TECHNICALDOC':
                    return { background: '#2196F3', border: '#1976D2' };
                case 'TEST':
                case 'TESTSUITE':
                    return { background: '#FF9800', border: '#F57C00' };
                case 'REQUIREMENT':
                    return { background: '#9C27B0', border: '#7B1FA2' };
                default:
                    return { background: '#607D8B', border: '#455A64' };
            }
        }
        
        // Update status panel
        function updateStatus(statusData) {
            if (!statusData || statusData.error) return;
            
            const phi = statusData.phi_metrics;
            const ontology = statusData.ontology_state;
            
            document.getElementById('phi0').textContent = phi.phi0.toFixed(3);
            document.getElementById('coherence').textContent = phi.coherence.toFixed(3);
            document.getElementById('stability').textContent = phi.stability ? 'PASS' : 'FAIL';
            document.getElementById('cycles').textContent = phi.cycles;
            
            document.getElementById('total-nodes').textContent = ontology.total_nodes;
            document.getElementById('total-edges').textContent = ontology.total_edges;
            document.getElementById('changed-nodes').textContent = ontology.changed_nodes;
            document.getElementById('coverage-ratio').textContent = ontology.coverage_ratio.toFixed(2);
            
            // Update status indicators
            document.getElementById('phi0-status').className = 
                `status-indicator ${phi.phi0 > 0.7 ? 'status-pass' : 'status-fail'}`;
            document.getElementById('stability-status').className = 
                `status-indicator ${phi.stability ? 'status-pass' : 'status-fail'}`;
        }
        
        // Show node details
        function showNodeDetails(nodeId) {
            fetch('/api/graph')
                .then(response => response.json())
                .then(data => {
                    const node = data.nodes.find(n => n.id === nodeId);
                    if (node) {
                        document.getElementById('node-info').innerHTML = `
                            <strong>${node.full_title}</strong><br>
                            <em>Type:</em> ${node.type}<br>
                            <em>Status:</em> ${node.status}<br>
                            <em>Criticality:</em> ${node.criticality}<br>
                            <em>Version:</em> ${node.version}<br>
                            <em>Changed:</em> ${node.changed ? 'Yes' : 'No'}
                        `;
                    }
                });
        }
        
        // Manual tick
        async function manualTick() {
            try {
                const response = await fetch('/api/tick', { method: 'POST' });
                const result = await response.json();
                console.log('Manual tick:', result);
            } catch (error) {
                console.error('Manual tick failed:', error);
            }
        }
        
        // Execute actions
        async function executeActions() {
            try {
                const response = await fetch('/api/execute', { method: 'POST' });
                const result = await response.json();
                
                if (result.error) {
                    console.error('Execute actions failed:', result.error);
                    alert('Execute Actions Error: ' + result.error);
                } else {
                    console.log('Actions executed:', result);
                    let message = `${result.message}\\n✅ Successful: ${result.successful}\\n❌ Failed: ${result.failed}`;
                    
                    if (result.external_refs && result.external_refs.length > 0) {
                        message += '\\n\\n🔗 Created:\\n' + result.external_refs.join('\\n');
                    }
                    
                    alert(message);
                    
                    // Refresh the graph after execution
                    loadInitialData();
                }
            } catch (error) {
                console.error('Execute actions failed:', error);
                alert('Execute Actions Error: ' + error.message);
            }
        }
        
        // Get advice
        async function getAdvice() {
            try {
                const response = await fetch('/api/advice', { method: 'POST' });
                const advice = await response.json();
                
                if (advice.error) {
                    document.getElementById('advice-content').innerHTML = `<em>Error: ${advice.error}</em>`;
                } else {
                    let content = `<strong>Assessment:</strong><br>${advice.judgement}<br><br>`;
                    if (advice.actions && advice.actions.length > 0) {
                        content += '<strong>Recommended Actions:</strong><br>';
                        advice.actions.forEach((action, i) => {
                            content += `${i + 1}. ${action}<br>`;
                        });
                    }
                    if (advice.notes) {
                        content += `<br><strong>Notes:</strong><br>${advice.notes}`;
                    }
                    document.getElementById('advice-content').innerHTML = content;
                }
            } catch (error) {
                console.error('Get advice failed:', error);
                document.getElementById('advice-content').innerHTML = '<em>Failed to get advice</em>';
            }
        }
        
        // Load actions
        async function loadActions() {
            try {
                const response = await fetch('/api/actions');
                const actionData = await response.json();
                
                if (actionData.error) {
                    document.getElementById('action-list').innerHTML = `<em>Error: ${actionData.error}</em>`;
                    document.getElementById('action-count').textContent = '0';
                } else {
                    const actions = actionData.actions;
                    document.getElementById('action-count').textContent = actions.length;
                    
                    if (actions.length === 0) {
                        document.getElementById('action-list').innerHTML = '<em>No actions available</em>';
                    } else {
                        let content = '';
                        actions.forEach((action, i) => {
                            const kindIcon = getActionIcon(action.kind);
                            const statusColor = action.status === 'pending' ? '#ff9800' : 
                                               action.status === 'executed' ? '#4CAF50' : '#f44336';
                            
                            content += `
                                <div class="action-item" style="border-left: 3px solid ${statusColor}; padding: 0.5rem; margin: 0.5rem 0; background: rgba(0,0,0,0.2);">
                                    <div style="display: flex; align-items: center; margin-bottom: 0.3rem;">
                                        <span style="margin-right: 0.5rem;">${kindIcon}</span>
                                        <strong style="font-size: 0.9rem;">${action.title}</strong>
                                    </div>
                                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8);">
                                        <div>Kind: ${action.kind}</div>
                                        <div>Key: ${action.idempotency_key}</div>
                                        ${action.targets.length > 0 ? `<div>Targets: ${action.targets.length} items</div>` : ''}
                                        ${action.body ? `<div style="margin-top: 0.3rem; font-style: italic;">${action.body.substring(0, 100)}${action.body.length > 100 ? '...' : ''}</div>` : ''}
                                    </div>
                                </div>
                            `;
                        });
                        document.getElementById('action-list').innerHTML = content;
                    }
                }
            } catch (error) {
                console.error('Load actions failed:', error);
                document.getElementById('action-list').innerHTML = '<em>Failed to load actions</em>';
                document.getElementById('action-count').textContent = '0';
            }
        }
        
        // Get action icon based on kind
        function getActionIcon(kind) {
            const icons = {
                'github.issue': '🐛',
                'github.pr': '🔄',
                'git.branch': '🌿',
                'fs.write': '📝',
                'graph.update': '🔗',
                'ci.trigger': '⚙️',
                'notify': '📢'
            };
            return icons[kind] || '⚡';
        }
        
        // Set auto tick interval
        async function setAutoTick(interval) {
            const intValue = parseInt(interval);
            try {
                await fetch(`/api/auto-tick/${intValue}`, { method: 'POST' });
                document.getElementById('autoValue').textContent = 
                    intValue === 0 ? 'Manual' : `${intValue}m`;
            } catch (error) {
                console.error('Set auto tick failed:', error);
            }
        }
        
        // Reset graph view
        function resetGraph() {
            if (network) {
                network.fit();
            }
        }
        
        // Load initial data
        async function loadInitialData() {
            try {
                const [graphResponse, statusResponse] = await Promise.all([
                    fetch('/api/graph'),
                    fetch('/api/status')
                ]);
                
                const graphData = await graphResponse.json();
                const statusData = await statusResponse.json();
                
                updateGraph(graphData);
                updateStatus(statusData);
            } catch (error) {
                console.error('Failed to load initial data:', error);
            }
        }
        
        // Initialize everything
        document.addEventListener('DOMContentLoaded', () => {
            initGraph();
            initWebSocket();
            loadInitialData();
            
            // Refresh data every 30 seconds as fallback
            setInterval(loadInitialData, 30000);
        });
    </script>
</body>
</html>
        """
    
    async def run(self):
        """Initialize RE_ware agent and start web server"""
        print("🚀 Starting RE_ware Web Dashboard...")
        
        # Initialize RE_ware agent
        self.agent = REWareInteractive(self.project_root, self.schema_name)
        success = await self.agent.initialize()
        
        if not success:
            print("❌ Failed to initialize RE_ware agent")
            return False
        
        # Register sensors
        sensors = [
            GitSensor(self.project_root),
            FsSensor(self.project_root),
            CliSensor(),
            GhSensor()
            # TODO: Add CI sensors when interface is complete
            # GitHubActionsSensor(self.project_root),
            # JUnitTestSensor(self.project_root),
            # PytestCoverageSensor(self.project_root),
            # TestResultSensor(self.project_root)
        ]
        
        for sensor in sensors:
            self.agent.sensor_hub.register_sensor(sensor)
        
        # Show current ontology state (from loaded snapshot or fresh)
        node_count = len(self.agent.ontology.nodes)
        edge_count = len(self.agent.ontology.edges)
        phi_signals = self.agent.ontology.phi_signals()
        
        print(f"🌐 Dashboard available at: http://localhost:{self.port}")
        print(f"🧠 Current Ψ state: {node_count} nodes, {edge_count} edges, coverage {phi_signals.get('coverage_ratio', 0.0):.2f}")
        print("📊 Features:")
        print("   • Interactive Psi Graph Visualization")
        print("   • Real-time node highlighting (new/changed)")
        print("   • Manual/Auto tick controls") 
        print("   • Live AI advice generation")
        print("   • WebSocket live updates")
        
        # Start the server FIRST so graph is immediately available
        # Start web server with interactive CLI (suppress uvicorn startup messages)
        config = uvicorn.Config(
            self.app, 
            host="0.0.0.0", 
            port=self.port, 
            log_level="critical",  # Suppress logs
            access_log=False       # Suppress access logs
        )
        server = uvicorn.Server(config)
        
        # Run server and interactive CLI concurrently
        server_task = asyncio.create_task(server.serve())
        cli_task = asyncio.create_task(self.run_interactive_cli())
        
        # Bootstrap sensors in background after server starts
        async def background_bootstrap():
            await asyncio.sleep(1)  # Let server start
            try:
                result = self.agent.sensor_hub.bootstrap_from_watermarks()
                if result.get('events_applied', 0) > 0:
                    print(f"📡 Sensor bootstrap completed: {result['events_applied']} new changes detected")
                    # Broadcast update to connected clients
                    await self.broadcast_update()
                else:
                    print("📡 Sensor bootstrap completed: No new changes since last session")
            except Exception as e:
                print(f"⚠️ Background sensor bootstrap failed: {e}")
        
        bootstrap_task = asyncio.create_task(background_bootstrap())
        
        try:
            # Wait for main tasks to complete (ignore bootstrap task completion)
            done, pending = await asyncio.wait(
                [server_task, cli_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Check if bootstrap task is still running and cancel it
            if not bootstrap_task.done():
                bootstrap_task.cancel()
                try:
                    await asyncio.wait_for(bootstrap_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Graceful shutdown: signal server to stop
            server.should_exit = True
            
            # Cancel remaining tasks gracefully with timeout
            for task in pending:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass  # Expected during shutdown
                except Exception:
                    pass  # Ignore other shutdown errors
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n👋 Shutdown requested")
            server.should_exit = True
            for task in [server_task, cli_task, bootstrap_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass  # Ignore shutdown errors
        except Exception as e:
            print(f"❌ Server error: {e}")
            server.should_exit = True
            # Cancel tasks gracefully
            for task in [server_task, cli_task, bootstrap_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass  # Ignore shutdown errors
        
        return True
    
    async def run_interactive_cli(self):
        """Run interactive CLI commands"""
        import sys
        
        try:
            while True:
                try:
                    # Show prompt
                    print("re_ware> ", end="", flush=True)
                    
                    # Read command (using asyncio-friendly approach)
                    command = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    command = command.strip().lower()
                    
                    if not command:
                        continue
                        
                    # Handle commands
                    if command in ['quit', 'exit', 'q']:
                        break
                    elif command == 'status':
                        await self.handle_cli_status()
                    elif command == 'advice':
                        await self.handle_cli_advice()  
                    elif command == 'tick':
                        await self.handle_cli_tick()
                    elif command.startswith('auto'):
                        await self.handle_cli_auto(command)
                    elif command == 'save':
                        await self.handle_cli_save()
                    elif command == 'consolidate':
                        await self.handle_cli_consolidate()
                    elif command == 'ui':
                        await self.handle_cli_ui()
                        # Explicitly continue the loop after ui command
                        continue
                    elif command in ['help', '?']:
                        self.show_cli_help()
                    else:
                        print(f"❓ Unknown command: {command}")
                        print("💡 Type 'help' for available commands")
                        
                except (EOFError, KeyboardInterrupt):
                    break
                except asyncio.CancelledError:
                    # Handle graceful cancellation during shutdown
                    break
                except Exception as e:
                    print(f"❌ CLI error: {e}")
                    
        except asyncio.CancelledError:
            # Handle task cancellation during shutdown
            pass
        except Exception:
            # Ignore other errors during shutdown
            pass
    
    def show_cli_help(self):
        """Show CLI help"""
        print("🎯 Available commands:")
        print("   status      - Show project consciousness state")
        print("   advice      - Get project reasoning and recommendations") 
        print("   tick        - Execute single evolution cycle")
        print("   auto [min]  - Enable autonomous mode (e.g. 'auto 5' for 5min intervals)")
        print("   save        - Save current Ψ state to snapshot")
        print("   consolidate - Merge duplicate file nodes")
        print("   ui          - Enable web UI dashboard and open browser")
        print("   help        - Show this help")
        print("   quit        - Shutdown system gracefully")
    
    async def handle_cli_status(self):
        """Handle status command"""
        try:
            # Call the agent's status command
            await self.agent._cmd_status()
        except Exception as e:
            print(f"❌ Status error: {e}")
    
    async def handle_cli_advice(self):
        """Handle advice command"""  
        try:
            # Call the agent's advice command
            await self.agent._cmd_advice()
        except Exception as e:
            print(f"❌ Advice error: {e}")
    
    async def handle_cli_tick(self):
        """Handle tick command"""
        try:
            # Call the agent's tick command
            await self.agent._cmd_tick()
        except Exception as e:
            print(f"❌ Tick error: {e}")
    
    async def handle_cli_auto(self, command):
        """Handle autonomous mode command"""
        try:
            # Parse interval from command (e.g. "auto 5")
            parts = command.split()
            if len(parts) == 1:
                # Default to 5 minute intervals
                interval = 5
            else:
                try:
                    interval = int(parts[1])
                    if interval <= 0:
                        print("❌ Interval must be positive")
                        return
                except ValueError:
                    print("❌ Invalid interval. Use 'auto <minutes>' (e.g. 'auto 5')")
                    return
            
            print(f"🤖 Starting autonomous mode (interval: {interval}m)")
            print("   • Project will manage itself autonomously")
            print("   • Press Ctrl+C to stop autonomous mode")
            print("")
            
            await self.run_autonomous_mode(interval)
            
        except Exception as e:
            print(f"❌ Autonomous mode error: {e}")
    
    async def run_autonomous_mode(self, interval_minutes):
        """Run project in autonomous mode"""
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                print(f"🔄 Autonomous cycle {cycle_count}")
                
                # Execute evolution cycle
                await self.agent._cmd_tick()
                
                # Wait for next cycle
                print(f"⏸️  Waiting {interval_minutes} minutes until next cycle...")
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Autonomous mode stopped after {cycle_count} cycles")
            print("🧠 Project returned to interactive mode")
        except Exception as e:
            print(f"❌ Autonomous mode error: {e}")

    async def handle_cli_save(self):
        """Handle save command"""
        try:
            # Call the agent's save command
            await self.agent._cmd_save()
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    async def handle_cli_ui(self):
        """Handle UI command - enable web UI and open browser"""
        try:
            if self.enable_web_ui:
                print("💡 Web UI is already enabled!")
                print(f"📊 Dashboard: http://localhost:{self.port}/")
            else:
                print("🌐 Enabling Web UI...")
                self.enable_web_ui = True
                
                print(f"📊 Web Dashboard enabled: http://localhost:{self.port}/")
                print("💡 Refresh your browser to see the dashboard")
                
            # Try to open browser (run in background to avoid blocking CLI)
            async def open_browser():
                try:
                    import webbrowser
                    # Run in executor to avoid blocking the CLI
                    await asyncio.get_event_loop().run_in_executor(
                        None, webbrowser.open, f"http://localhost:{self.port}/"
                    )
                    print("🔗 Browser opened automatically")
                except Exception as e:
                    print(f"💡 Please open browser manually: http://localhost:{self.port}/")
            
            # Start browser opening in background, don't wait for it
            asyncio.create_task(open_browser())
            
            # Ensure CLI remains responsive
            print("🎯 CLI remains active for commands")
                    
        except Exception as e:
            print(f"❌ UI enable error: {e}")

    async def handle_cli_consolidate(self):
        """Handle consolidate command"""
        try:
            # Call the agent's consolidate command
            await self.agent._cmd_consolidate()
        except Exception as e:
            print(f"❌ Consolidate error: {e}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="RE_ware System - Conscious Project Entity")
    
    # Simple arguments for interactive system
    parser.add_argument("--web", action="store_true", help="Enable web UI dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port for server (default: 8000)")
    parser.add_argument("--schema", "-s", type=str, default="project_manager", help="Gene schema to use")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--list-schemas", action="store_true", help="List available gene schemas and exit")
    
    args = parser.parse_args()
    
    # Handle schema listing
    if args.list_schemas:
        schemas = list_available_gene_templates()
        print("🧬 Available Gene Schemas:")
        for schema in schemas:
            print(f"   • {schema}")
        return 0
    
    # Always run the interactive system
    project_root = Path(args.project_root).resolve()
    
    try:
        # Always run the system with API + optional web UI + interactive CLI
        if not WEB_DEPENDENCIES_AVAILABLE:
            print("❌ Web dependencies not installed. Install with:")
            print("   pip install fastapi uvicorn jinja2")
            return 1
        
        # Create and run server with interactive CLI
        server = REWareWebServer(project_root, args.schema, args.port, enable_web_ui=args.web)
        
        if args.web:
            print(f"🚀 Starting RE_ware System + Web UI on port {args.port}")
            print(f"📊 Web Dashboard: http://localhost:{args.port}/")
        else:
            print(f"🚀 Starting RE_ware System on port {args.port}")
            print(f"📡 API available at: http://localhost:{args.port}/api/")
            print("💡 Use 'python evolve.py --web' to enable web dashboard")
        
        print(f"🎯 Interactive commands: status, advice, tick, auto, save, quit")
        print(f"📋 API endpoints available at /api/")
        print("")
        
        # Run the server with interactive CLI
        await server.run()
    
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested")
        return 0
    
    except Exception as e:
        import traceback
        print(f"❌ Unexpected error: {e}")
        print("📋 Full traceback:")
        print(traceback.format_exc())
        return 1
    
    return 0



if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)