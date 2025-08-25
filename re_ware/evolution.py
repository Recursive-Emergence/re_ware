"""
RE_ware Evolution Engine - Core evolution logic and interactive agent
==================================================================

This module contains the core evolution engine that was refactored from evolve.py
to properly encapsulate the RE (Recursive Emergence) pattern implementation.

Classes:
- REWareState: Core state tracking for the system
- REWareInteractive: Interactive command-line agent
- REWareEvolver: Original evolution mode logic  
- GitSafetyManager: Safe git operations for evolution
"""

import sys
import os
import asyncio
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any

from .ontology import (
    OntologyPhenotype, create_ontology_with_gene, 
    NodeType, RelationType, Status, create_node, create_edge
)
from .sensor_hub import SensorHub
from .sensors import GitSensor, FsSensor, CliSensor, GhSensor
# from .ci_sensors import GitHubActionsSensor, JUnitTestSensor, PytestCoverageSensor, TestResultSensor
from .llm_integration import create_project_intelligence
from .action_dispatcher import ActionDispatcher
from .frames import FrameBuilder
from .quality_gates import QualityGateRunner


@dataclass 
class REWareState:
    """Core state tracking for the system"""
    phi0: float = 0.0  # Φ₀ stability metric
    phi_coherence: float = 0.0  # Overall coherence
    stability_check: bool = False  # Ω safety gate
    cycles_completed: int = 0  # Number of evolution cycles
    last_update: float = 0.0  # Timestamp of last update
    
    def update(self, phi0: float, coherence: float, stability: bool):
        """Update state with new metrics"""
        self.phi0 = phi0
        self.phi_coherence = coherence
        self.stability_check = stability
        self.cycles_completed += 1
        self.last_update = time.time()


class REWareInteractive:
    """Interactive RE_ware agent - the main evolution consciousness"""
    
    def __init__(self, project_root: Path, schema_name: str = "project_manager"):
        self.project_root = Path(project_root)
        self.schema_name = schema_name
        self.state = REWareState()
        
        # Core components - will be initialized in bootstrap
        self.ontology: OntologyPhenotype = None
        self.sensor_hub: SensorHub = None
        self.re_agent = None
        self.action_dispatcher: ActionDispatcher = None
        
        # Evolution control
        self.running = True
        
    async def initialize(self) -> bool:
        """Initialize all RE_ware components"""
        try:
            # 1. Initialize ontology (Ψ - externalized memory)
            success = await self._bootstrap_ontology()
            if not success:
                return False
            
            # 2. Initialize sensor hub (must come before consciousness)
            self.sensor_hub = SensorHub(self.ontology)
                
            # 3. Initialize consciousness and agent
            success = await self._bootstrap_consciousness()
            if not success:
                return False
            
            # 4. Initialize action dispatcher
            self.action_dispatcher = ActionDispatcher(self.project_root, self.ontology)
                
            print("🎯 RE_ware consciousness online - interactive mode ready")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def _bootstrap_ontology(self) -> bool:
        """1. Bootstrap ontology with gene schema"""
        try:
            print(f"🧬 Bootstrapping ontology with schema: {self.schema_name}")
            self.ontology = create_ontology_with_gene(self.schema_name)
            if not self.ontology:
                print(f"❌ Failed to create ontology with schema: {self.schema_name}")
                return False
            
            # Try to load existing psi snapshot first
            snapshot_path = self.project_root / "psi_snapshot.json"
            if snapshot_path.exists():
                success = self.ontology.load_snapshot(snapshot_path)
                if success:
                    print(f"🔥 Restored Ψ consciousness: {len(self.ontology.nodes)} nodes, {len(self.ontology.edges)} edges, Φ₀={self.ontology.phi_signals().get('coverage_ratio', 0.0)}")
                    # Skip creating project hub if we loaded from snapshot
                    return True
                else:
                    print("⚠️  Failed to load snapshot, starting fresh")
            
            # Create central PROJECT node as graph hub (only if no snapshot loaded)
            await self._create_project_hub()
            
            print("✅ Ontology initialized with project hub")
            return True
        except Exception as e:
            print(f"❌ Ontology bootstrap failed: {e}")
            return False
    
    async def _bootstrap_consciousness(self) -> bool:
        """3. Bootstrap Ψ consciousness with correct state"""
        try:
            print("🔥 Bootstrapping Ψ consciousness...")
            
            # Removed redundant ontology snapshot loading - only use psi snapshots now
            
            # Bootstrap from watermarks - CRITICAL for proper RE pattern
            result = self.sensor_hub.bootstrap_from_watermarks()
            if result["events_applied"] > 0:
                print(f"📍 Bootstrap applied {result['events_applied']} events")
            
            # Create RE intelligence system (Φ - coherence projection)
            self.re_agent = create_project_intelligence(self.ontology)
            if not self.re_agent:
                print("❌ Failed to create RE intelligence")
                return False
            
            print("✅ Ψ consciousness bootstrapped successfully")
            return True
            
        except Exception as e:
            print(f"❌ Consciousness bootstrap failed: {e}")
            return False
    
    async def _create_project_hub(self) -> bool:
        """Create central PROJECT node as the graph hub"""
        try:
            from .ontology import NodeType, create_node
            import os
            
            # Get project information
            project_name = self.project_root.name
            git_repo = None
            
            # Try to get git remote info
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"], 
                    cwd=self.project_root,
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    git_repo = result.stdout.strip()
            except:
                pass
            
            # Create the PROJECT node
            project_node = create_node(
                NodeType.PROJECT,
                f"Project: {project_name}",
                content={
                    "name": project_name,
                    "root_path": str(self.project_root),
                    "git_repository": git_repo,
                    "schema": self.schema_name,
                    "node_id": "project:root",  # Canonical ID for sensor targeting
                    "created_by": "re_ware_evolution",
                    "description": f"Central hub for {project_name} project graph"
                }
            )
            
            # Set the canonical ID that sensors reference
            project_node.id = "project:root"
            
            # Add to ontology
            success = self.ontology.add_node(project_node)
            if success:
                print(f"🏗️  Created project hub: {project_name} (id: project:root)")
                return True
            else:
                print("❌ Failed to add project hub to ontology")
                return False
                
        except Exception as e:
            print(f"❌ Project hub creation failed: {e}")
            return False
    
    async def run_interactive(self):
        """Main interactive loop"""
        print("🎮 Entering interactive mode...")
        print("Commands: status, advice, tick, execute, frames, gates, nodes, save, help, exit")
        
        try:
            while self.running:
                try:
                    # Get user input
                    cmd = input("RE_ware> ").strip().lower()
                    
                    if not cmd:
                        continue
                    
                    # Handle commands
                    if cmd in ['exit', 'quit', 'q']:
                        await self._cmd_exit()
                    elif cmd in ['help', 'h']:
                        await self._cmd_help()
                    elif cmd == 'status':
                        await self._cmd_status()
                    elif cmd == 'advice':
                        await self._cmd_advice()
                    elif cmd == 'tick':
                        await self._cmd_tick()
                    elif cmd == 'execute':
                        await self._cmd_execute()
                    elif cmd == 'frames':
                        await self._cmd_frames()
                    elif cmd == 'gates':
                        await self._cmd_gates()
                    elif cmd == 'nodes':
                        await self._cmd_nodes()
                    elif cmd == 'save':
                        await self._cmd_save()
                    elif cmd == 'consolidate':
                        await self._cmd_consolidate()
                    else:
                        print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                        
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit gracefully")
                except EOFError:
                    break
                    
        except Exception as e:
            print(f"❌ Interactive loop error: {e}")
        
        print("👋 Interactive session ended")
    
    async def _cmd_help(self):
        """Help command handler"""
        print("""
🎯 RE_ware Interactive Commands:
    
    status      - Show current system status and phi metrics
    advice      - Get LLM advice on current project state  
    tick        - Run one evolution cycle (sensors + phi update)
    execute     - Execute actions from the most recent advice
    frames      - Show concrete project frames (Quality/Delivery/Risk)
    gates       - Evaluate quality gates (Ω guardrails)
    nodes       - Show ontology nodes and hot state
    save        - Save current Ψ snapshot
    consolidate - Consolidate duplicate file nodes
    help        - Show this help message
    exit        - Exit interactive mode
    
🧬 System Status:
    Φ₀ = {:.2f} (stability: {})
    Cycles: {} completed
    Hot nodes: {} changed
        """.format(
            self.state.phi0,
            "✅" if self.state.stability_check else "⚠️",
            self.state.cycles_completed,
            len(self.ontology.hot_state.changed_nodes) if self.ontology else 0
        ))
    
    async def _cmd_status(self):
        """Status command handler"""
        if not self.ontology:
            print("❌ Ontology not initialized")
            return
            
        # Get phi signals
        phi_signals = self.ontology.phi_signals()
        
        print(f"""
🎯 RE_ware System Status:
    
📊 Phi Metrics:
    Φ₀ = {self.state.phi0:.3f} (stability threshold: 0.7)
    Coherence = {self.state.phi_coherence:.3f} 
    Stability Check = {"✅ PASS" if self.state.stability_check else "⚠️  FAIL"}
    
🧬 Ontology State:
    Total nodes: {len(self.ontology.nodes)}
    Total edges: {len(self.ontology.edges)}
    Changed nodes: {phi_signals['changed_nodes']}
    Coverage ratio: {phi_signals['coverage_ratio']:.2f}
    Entropy hint: {phi_signals['entropy_hint']:.3f}
    
⏱️  Evolution:
    Cycles completed: {self.state.cycles_completed}
    Last update: {datetime.fromtimestamp(self.state.last_update).strftime('%Y-%m-%d %H:%M:%S') if self.state.last_update else 'Never'}
        """)
    
    async def _cmd_advice(self):
        """Advice command handler with caching"""
        if not self.re_agent or not self.ontology:
            print("❌ RE intelligence not initialized")
            return
        
        # Check for recent cached advice first
        cached_advice = self._get_recent_advice()
        if cached_advice:
            print("🤖 Using cached advice (no significant changes detected)...")
            self._display_advice(cached_advice)
            return
            
        print("🤖 Generating fresh advice...")
        
        try:
            # Generate advice using the LLM integration
            phi_signals = self.ontology.phi_signals()
            advice_frame = await self.re_agent.generate_advice_frame(
                species_id="project_manager_v1",
                instance_id=getattr(self.ontology, 'instance_id', 'unknown'),
                phi_state=phi_signals
            )
            
            if advice_frame:
                # Store advice as node in graph
                advice_node_id = self._store_advice_node(advice_frame, phi_signals)
                if advice_node_id:
                    print(f"💾 Stored advice as node: {advice_node_id}")
                
                self._display_advice(advice_frame)
            else:
                print("❌ Failed to generate advice frame")
        except Exception as e:
            print(f"❌ Advice generation error: {e}")
    
    def _display_advice(self, advice_frame: Dict[str, Any]):
        """Display advice in consistent format"""
        judgement = advice_frame.get('judgement', 'No judgement available')
        actions = advice_frame.get('actions', [])
        
        print(f"\n🎯 Project Advice:")
        print(f"📋 Assessment: {judgement}")
        
        if actions:
            print(f"\n🚀 Recommended Actions:")
            for i, action in enumerate(actions[:5], 1):
                if isinstance(action, dict):
                    action_text = action.get('title', action.get('description', str(action)))
                else:
                    action_text = str(action)
                print(f"   {i}. {action_text}")
        else:
            print("   No specific actions recommended")
    
    def _get_recent_advice(self) -> Dict[str, Any]:
        """Check for recent advice that's still valid"""
        if not self.ontology:
            return None
        
        # Look for recent ADVICE nodes
        advice_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.ADVICE]
        if not advice_nodes:
            return None
        
        # Sort by creation time (most recent first)
        advice_nodes.sort(key=lambda n: n.content.get('generated_at', 0), reverse=True)
        most_recent = advice_nodes[0]
        
        # Check if advice is still fresh (less than 5 minutes old and no significant changes)
        generated_at = most_recent.content.get('generated_at', 0)
        age_seconds = time.time() - generated_at
        
        if age_seconds < 300:  # 5 minutes
            # Check if there have been significant changes since advice was generated
            phi_signals = self.ontology.phi_signals()
            cached_phi = most_recent.content.get('phi_state', {})
            
            # Compare key metrics - if they haven't changed much, use cached advice
            coverage_changed = abs(phi_signals.get('coverage_ratio', 0) - cached_phi.get('coverage_ratio', 0)) > 0.1
            nodes_changed_significantly = abs(phi_signals.get('changed_nodes', 0) - cached_phi.get('changed_nodes', 0)) > 3
            
            if not (coverage_changed or nodes_changed_significantly):
                return {
                    'judgement': most_recent.content.get('assessment', 'Cached assessment'),
                    'actions': most_recent.content.get('actions', []),
                    'notes': f"Cached advice (generated {age_seconds:.0f}s ago)"
                }
        
        return None
    
    def _store_advice_node(self, advice_frame: Dict[str, Any], phi_signals: Dict[str, Any]) -> str:
        """Store advice as a node in the graph"""
        try:
            # Create advice node
            advice_node = create_node(
                NodeType.ADVICE,
                f"Project Advice {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content={
                    "advice_type": "project_assessment",
                    "assessment": advice_frame.get('judgement', ''),
                    "actions": advice_frame.get('actions', []),
                    "phi_state": phi_signals,
                    "generated_at": time.time(),
                    "need_more": advice_frame.get('need_more', False),
                    "notes": advice_frame.get('notes', '')
                }
            )
            
            # Add to ontology
            success = self.ontology.add_node(advice_node)
            if not success:
                print("⚠️  Failed to add advice node to ontology")
                return None
            
            # Create relationship to project
            project_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.PROJECT]
            if project_nodes:
                project_node = project_nodes[0]  # Use first project node
                advice_edge = create_edge(
                    RelationType.ADDRESSES,
                    advice_node.id,
                    project_node.id
                )
                self.ontology.add_edge(advice_edge)
            
            return advice_node.id
            
        except Exception as e:
            print(f"⚠️  Failed to store advice node: {e}")
            return None
    
    async def _cmd_execute(self):
        """Execute command handler - execute actions from recent advice"""
        if not self.action_dispatcher:
            print("❌ Action dispatcher not initialized")
            return
        
        # Find the most recent advice with actions
        advice_nodes = [n for n in self.ontology.nodes.values() if n.type == NodeType.ADVICE]
        if not advice_nodes:
            print("❌ No advice found. Run 'advice' command first.")
            return
        
        # Sort by creation time (most recent first)
        advice_nodes.sort(key=lambda n: n.content.get('generated_at', 0), reverse=True)
        most_recent = advice_nodes[0]
        
        actions = most_recent.content.get('actions', [])
        if not actions:
            print("📭 No actions to execute in most recent advice")
            return
        
        print(f"🚀 Executing {len(actions)} actions from recent advice...")
        print(f"   Advice from: {datetime.fromtimestamp(most_recent.content.get('generated_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check available executors
        available_executors = self.action_dispatcher.get_available_executors()
        print("\n📋 Available executors:")
        for kind, info in available_executors.items():
            status = "✅" if info["available"] else "❌"
            print(f"   {status} {kind}: {info['description']}")
        
        print(f"\n⚡ Executing actions:")
        
        # Execute actions
        results = await self.action_dispatcher.dispatch_actions(actions)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"\n📊 Execution Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        
        # Show external references (GitHub URLs, etc.)
        external_refs = []
        for result in results:
            if result.external_refs:
                external_refs.extend(result.external_refs)
        
        if external_refs:
            print(f"\n🔗 External references created:")
            for ref in external_refs:
                print(f"   {ref}")
    
    async def _cmd_frames(self):
        """Frames command handler - show concrete project frames"""
        if not self.ontology:
            print("❌ Ontology not initialized")
            return
        
        print("📊 Building concrete project frames...")
        
        try:
            frame_builder = FrameBuilder(self.ontology)
            frames = frame_builder.build_all_frames()
            
            for frame_type, frame_data in frames.items():
                print(f"\n📈 {frame_type.upper()} FRAME")
                print("=" * 40)
                
                if frame_type == "quality":
                    frame = frame_data
                    print(f"Quality Score: {frame.quality_score:.2f}")
                    print(f"Quality Gate: {frame.quality_gate_status.upper()}")
                    print(f"Tests: {frame.passing_tests}/{frame.total_tests} passing")
                    print(f"Coverage: {frame.test_coverage:.1f}%")
                    print(f"Builds: {frame.successful_builds}/{frame.total_builds} successful")
                    print(f"Critical Issues: {frame.critical_issues}")
                    if frame.blockers:
                        print(f"Blockers: {', '.join(frame.blockers)}")
                
                elif frame_type == "delivery":
                    frame = frame_data
                    print(f"Release Readiness: {frame.release_readiness_score:.2f}")
                    print(f"Milestone: {frame.current_milestone}")
                    print(f"Progress: {frame.milestone_progress:.1%}")
                    print(f"Requirements: {frame.verified_requirements}/{frame.total_requirements} verified")
                    print(f"Coverage Ratio: {frame.coverage_ratio:.2f}")
                    print(f"Critical Bugs: {frame.critical_bugs}")
                    print(f"Tests Passing: {'✅' if frame.all_tests_passing else '❌'}")
                    print(f"Security Clean: {'✅' if frame.security_scan_clean else '❌'}")
                    print(f"Docs Complete: {'✅' if frame.documentation_complete else '❌'}")
                
                elif frame_type == "architecture":
                    frame = frame_data
                    print(f"Components: {frame.healthy_components}/{frame.total_components} healthy")
                    print(f"Dependencies: {frame.total_dependencies} total")
                    print(f"Debt Ratio: {frame.debt_ratio:.2f}")
                    print(f"Coupling Score: {frame.coupling_score:.2f}")
                    if frame.refactoring_candidates:
                        print(f"Refactoring Candidates: {', '.join(frame.refactoring_candidates[:3])}")
                
                elif frame_type == "risk":
                    frame = frame_data
                    print(f"Risk Level: {frame.risk_level.upper()}")
                    print(f"Risk Score: {frame.risk_score:.2f}")
                    print(f"Security Vulnerabilities: {frame.security_vulnerabilities}")
                    print(f"Critical Vulnerabilities: {frame.critical_vulnerabilities}")
                    print(f"Untested Code: {frame.untested_code_ratio:.1%}")
                    print(f"Undocumented Components: {frame.undocumented_components}")
                    
                    if frame.risk_items:
                        print("Risk Items:")
                        for item in frame.risk_items[:3]:
                            print(f"  • {item['severity'].upper()}: {item['description']}")
            
            print(f"\n💾 Frames generated at: {datetime.fromtimestamp(frame_builder.generated_at).strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
    
    async def _cmd_gates(self):
        """Quality gates command handler - evaluate Ω guardrails"""
        if not self.ontology:
            print("❌ Ontology not initialized")
            return
        
        print("🛡️  Evaluating quality gates (Ω guardrails)...")
        
        try:
            # Create quality gate runner
            gate_runner = QualityGateRunner(self.ontology)
            
            # Build frames for gate evaluation
            frame_builder = FrameBuilder(self.ontology)
            frames = frame_builder.build_all_frames()
            
            # Evaluate all gates
            results = gate_runner.evaluate_all_gates(frames)
            
            print(f"\n🚨 QUALITY GATES EVALUATION")
            print("=" * 50)
            
            # Overall status
            status_icon = {
                "passing": "✅",
                "warning": "⚠️",
                "failing": "❌", 
                "blocked": "🚫"
            }.get(results["overall_status"], "❓")
            
            print(f"Overall Status: {status_icon} {results['overall_status'].upper()}")
            print(f"Overall Score: {results['overall_score']:.2f}")
            
            # Summary
            summary = results["summary"]
            print(f"\nGate Summary:")
            print(f"  ✅ Passing: {summary['passing']}")
            print(f"  ❌ Failing: {summary['failing']}")
            print(f"  ⚠️  Warnings: {summary['warnings']}")
            print(f"  🚫 Blocking: {summary['blocking_failures']}")
            
            # Individual gate results
            print(f"\n📋 Individual Gate Results:")
            for evaluation in results["evaluations"]:
                result_icon = {
                    "pass": "✅",
                    "fail": "❌",
                    "warning": "⚠️",
                    "skip": "⏭️"
                }.get(evaluation["result"], "❓")
                
                blocking_indicator = " 🚫" if evaluation["blocking"] else ""
                score_indicator = f" ({evaluation['score']:.2f})"
                
                print(f"  {result_icon} {evaluation['gate_name']}: {evaluation['message']}{score_indicator}{blocking_indicator}")
            
            # Blocking failures (critical)
            if results["blocking_failures"]:
                print(f"\n🚫 BLOCKING FAILURES:")
                for gate_name in results["blocking_failures"]:
                    print(f"  • {gate_name}")
                print("   ⚠️  Deployment blocked until these gates pass!")
            
            # Corrective actions
            corrective_actions = results["corrective_actions"]
            if corrective_actions:
                print(f"\n🔧 Corrective Actions ({len(corrective_actions)}):")
                for i, action in enumerate(corrective_actions[:5], 1):  # Show first 5
                    priority_icon = {"critical": "🚨", "high": "⚠️", "medium": "📋", "low": "💡"}.get(action.get("priority", "medium"), "📋")
                    print(f"  {i}. {priority_icon} {action.get('title', 'Unknown action')} ({action.get('kind', 'unknown')})")
                
                if len(corrective_actions) > 5:
                    print(f"   ... and {len(corrective_actions) - 5} more actions")
                
                print(f"\n💡 Tip: Use 'execute' command to run corrective actions")
            
            # Deployment readiness
            is_blocked, blocking_gates = gate_runner.is_deployment_blocked(frames)
            if is_blocked:
                print(f"\n🚨 DEPLOYMENT BLOCKED by {len(blocking_gates)} gates:")
                for gate in blocking_gates:
                    print(f"   • {gate}")
            else:
                print(f"\n🚀 DEPLOYMENT READY - All quality gates passing!")
            
            print(f"\n📊 Evaluated at: {datetime.fromtimestamp(results['evaluated_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Quality gates evaluation error: {e}")
    
    async def get_advice_with_caching(self) -> Dict[str, Any]:
        """Get advice with caching for API endpoints"""
        if not self.re_agent or not self.ontology:
            return {"error": "RE intelligence not initialized"}
        
        # Check for recent cached advice first
        cached_advice = self._get_recent_advice()
        if cached_advice:
            return {
                **cached_advice,
                "cached": True,
                "source": "stored_advice_node"
            }
        
        try:
            # Generate fresh advice
            phi_signals = self.ontology.phi_signals()
            advice_frame = await self.re_agent.generate_advice_frame(
                species_id="project_manager_v1",
                instance_id=getattr(self.ontology, 'instance_id', 'unknown'),
                phi_state=phi_signals
            )
            
            if advice_frame:
                # Store advice as node in graph
                advice_node_id = self._store_advice_node(advice_frame, phi_signals)
                
                return {
                    "judgement": advice_frame.get('judgement', 'No assessment available'),
                    "actions": advice_frame.get('actions', []),
                    "need_more": advice_frame.get('need_more', False),
                    "notes": advice_frame.get('notes', ''),
                    "cached": False,
                    "advice_node_id": advice_node_id
                }
            else:
                return {"error": "Failed to generate advice frame"}
                
        except Exception as e:
            return {"error": f"Advice generation failed: {e}"}
    
    async def _cmd_consolidate(self):
        """Consolidate command handler - merge duplicate file nodes"""
        if not self.sensor_hub or not self.ontology:
            print("❌ System not properly initialized")
            return
            
        print("🔧 Consolidating duplicate nodes...")
        
        try:
            before_count = len(self.ontology.nodes)
            stats = self.sensor_hub.consolidate_duplicate_nodes()
            after_count = len(self.ontology.nodes)
            
            print(f"""
🧹 Node Consolidation Complete:
    Nodes before: {before_count}
    Nodes after: {after_count}
    Files merged: {stats['merged']}
    Duplicates removed: {stats['removed']}
            """)
            
            # Update phi state
            if stats['merged'] > 0:
                self.state.update(
                    phi0=self.state.phi0,  # Keep current phi0
                    coherence=self.state.phi_coherence + 0.05,  # Slight coherence boost
                    stability=True  # Consolidation improves stability
                )
                
        except Exception as e:
            print(f"❌ Consolidation failed: {e}")
    
    async def _cmd_tick(self):
        """Tick command handler - single evolution cycle"""
        if not self.ontology or not self.sensor_hub:
            print("❌ System not properly initialized")
            return
            
        print("⚡ Running evolution tick...")
        cycle_start = time.time()
        
        try:
            # 1. Try sensor polling with error handling
            nodes_updated = 0
            try:
                result = self.sensor_hub.poll_and_apply()
                nodes_updated = len(result.get("updated_nodes", []))
                if nodes_updated > 0:
                    print(f"📡 Sensors updated {nodes_updated} nodes")
                else:
                    print("📡 No sensor updates")
            except Exception as sensor_error:
                print(f"📡 Sensor error (continuing): {sensor_error}")
                # Continue with empty result
            
            # 2. Calculate phi signals and compute real Φ₀
            phi_signals = self.ontology.phi_signals()
            phi0 = self._compute_phi0(phi_signals)
            coherence = float(phi_signals.get('coverage_ratio', 0.0))
            
            # 3. Check stability (0/1 check based on Φ₀)
            stability_check = phi0 > 0.7 and coherence > 0.6  # Ω gates
            
            # 4. Update state
            self.state.update(phi0, coherence, stability_check)
            
            # 5. Evaluate quality gates (Ω guardrails)
            gate_runner = QualityGateRunner(self.ontology)
            frame_builder = FrameBuilder(self.ontology)
            frames = frame_builder.build_all_frames()
            
            gate_results = gate_runner.evaluate_all_gates(frames)
            is_blocked, blocking_gates = gate_runner.is_deployment_blocked(frames)
            
            if is_blocked:
                print(f"🚫 Quality gates blocking: {', '.join(blocking_gates)}")
                # Downgrade stability if gates are blocking
                stability_check = False
                self.state.update(phi0, coherence, stability_check)
            
            # 6. Check and save snapshot if watermark updates triggered persistence
            self.ontology.check_and_save_pending_snapshot()
            
            cycle_result = {
                "success": True,
                "cycle_time": time.time() - cycle_start,
                "phi0": phi0,
                "coherence": coherence,
                "stability": stability_check,
                "nodes_updated": nodes_updated
            }
            
            print(f"✅ Tick completed in {cycle_result['cycle_time']:.2f}s")
            print(f"   Φ₀={phi0:.3f}, coherence={coherence:.3f}, stable={stability_check}")
            
        except Exception as e:
            print(f"❌ Tick failed: {e}")
    
    async def _cmd_nodes(self):
        """Nodes command handler"""
        if not self.ontology:
            print("❌ Ontology not initialized")
            return
            
        print(f"""
🧬 Ontology Nodes ({len(self.ontology.nodes)} total):
        """)
        
        # Show recent changes first
        changed_nodes = list(self.ontology.hot_state.changed_nodes)
        if changed_nodes:
            print("🔥 Recently Changed:")
            for node_id in changed_nodes[:10]:  # Show first 10
                if node_id in self.ontology.nodes:
                    node = self.ontology.nodes[node_id]
                    print(f"   {node.type.name}: {node.title[:60]}")
            
            if len(changed_nodes) > 10:
                print(f"   ... and {len(changed_nodes) - 10} more changed nodes")
        
        # Show node type summary
        type_counts = {}
        for node in self.ontology.nodes.values():
            type_name = node.type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        print("📊 Node Types:")
        for node_type, count in sorted(type_counts.items()):
            print(f"   {node_type}: {count}")
    
    async def _cmd_save(self):
        """Save command handler - saves snapshot with current state"""
        print("💾 Saving Ψ snapshot...")
        snapshot_path = self.project_root / "psi_snapshot.json"
        
        # Create phi data with current state
        phi_data = {
            "phi0": self.state.phi0,
            "coherence": self.state.phi_coherence,
            "stability": self.state.stability_check,
            "saved_at": time.time()
        }
        
        # Use the ontology's snapshot method
        success = self.ontology.save_snapshot(snapshot_path, phi_data)
        if success:
            print(f"✅ Ψ snapshot saved to {snapshot_path}")
        else:
            print("❌ Ψ snapshot save failed")
    
    async def _cmd_exit(self):
        """Exit command handler"""
        print("💾 Saving final snapshot before exit...")
        await self._cmd_save()
        self.running = False
        print("👋 Goodbye!")
    
    def _compute_phi0(self, phi_signals: Dict[str, Any]) -> float:
        """Compute integrated information (Φ₀) from ontology signals"""
        try:
            # Extract key signals
            coverage_ratio = phi_signals.get('coverage_ratio', 0.0)
            changed_nodes = phi_signals.get('changed_nodes', 0)
            uncovered_requirements = phi_signals.get('uncovered_requirements', 0)
            open_p0_issues = phi_signals.get('open_p0_issues', 0)
            entropy_hint = phi_signals.get('entropy_hint', 1.0)
            
            # Total nodes for normalization
            total_nodes = len(self.ontology.nodes) if self.ontology.nodes else 1
            
            # Base integration from coverage (primary signal)
            base_integration = coverage_ratio * 0.4  # Up to 0.4 from coverage
            
            # Connectivity bonus (changed nodes indicate active integration)
            activity_ratio = min(changed_nodes / total_nodes, 1.0) 
            connectivity_bonus = activity_ratio * 0.3  # Up to 0.3 from activity
            
            # Quality gates
            requirement_penalty = min(uncovered_requirements / max(total_nodes * 0.1, 1), 0.2)  # Up to -0.2
            issue_penalty = open_p0_issues * 0.1  # -0.1 per P0 issue
            
            # Information coherence (lower entropy = higher integration)
            coherence_bonus = (1.0 - min(entropy_hint, 1.0)) * 0.2  # Up to 0.2
            
            # Compute Φ₀ (0.0 to 1.0 range)
            phi0 = base_integration + connectivity_bonus + coherence_bonus - requirement_penalty - issue_penalty
            phi0 = max(0.0, min(1.0, phi0))  # Clamp to [0, 1]
            
            return phi0
            
        except Exception as e:
            print(f"⚠️  Φ₀ computation error: {e}")
            return 0.0  # Safe fallback


class GitSafetyManager:
    """Manages git operations for safe evolution with non-interactive policy"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.safety_enabled = True
        
    def is_git_repo(self) -> bool:
        """Check if project is a git repository"""
        git_dir = self.project_root / ".git"
        return git_dir.exists()
    
    def get_git_status(self) -> Dict[str, Any]:
        """Get current git status"""
        if not self.is_git_repo():
            return {"error": "Not a git repository"}
            
        try:
            # Get status
            result = subprocess.run(
                ["git", "status", "--porcelain"], 
                cwd=self.project_root,
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return {"error": f"Git status failed: {result.stderr}"}
            
            # Parse output
            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            modified = []
            untracked = []
            
            for line in lines:
                if line.startswith(' M '):
                    modified.append(line[3:])
                elif line.startswith('??'):
                    untracked.append(line[3:])
            
            return {
                "clean": len(lines) == 0,
                "modified": modified,
                "untracked": untracked,
                "total_changes": len(lines)
            }
            
        except Exception as e:
            return {"error": f"Git status error: {e}"}
    
    def create_safety_commit(self, message: str = "Auto-commit before evolution") -> bool:
        """Create a safety commit with all changes"""
        if not self.safety_enabled or not self.is_git_repo():
            return False
            
        try:
            # Add all changes
            subprocess.run(
                ["git", "add", "."], 
                cwd=self.project_root, 
                check=True,
                timeout=30
            )
            
            # Create commit
            subprocess.run(
                ["git", "commit", "-m", message], 
                cwd=self.project_root,
                check=True,
                timeout=30
            )
            
            return True
            
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False


class REWareEvolver:
    """Original evolution mode - systematic project evolution"""
    
    def __init__(self, project_root: Path, schema_name: str = "project_manager"):
        self.project_root = Path(project_root)
        self.schema_name = schema_name
        
        # Core components
        self.ontology: OntologyPhenotype = None
        self.sensor_hub: SensorHub = None
        self.re_agent = None
        self.git_manager = GitSafetyManager(project_root)
        
        # Evolution settings
        self.max_cycles = 50
        self.stability_threshold = 0.8
        self.auto_commit = True
    
    async def run_evolution(self) -> Dict[str, Any]:
        """Run full evolution cycle"""
        print("🌱 Starting RE_ware Evolution Mode")
        
        try:
            # Initialize system
            success = await self._initialize_system()
            if not success:
                return {"success": False, "error": "System initialization failed"}
            
            # Run evolution cycles
            results = await self._evolution_loop()
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _initialize_system(self) -> bool:
        """Initialize all evolution components"""
        try:
            print("🧬 Initializing evolution system...")
            
            # Create ontology
            self.ontology = create_ontology_with_gene(self.schema_name)
            if not self.ontology:
                return False
            
            # Create sensor hub
            self.sensor_hub = SensorHub(self.ontology)
            
            # Create RE agent
            self.re_agent = create_project_intelligence(self.ontology)
            
            print("✅ Evolution system initialized")
            return True
            
        except Exception as e:
            print(f"❌ System initialization error: {e}")
            return False
    
    async def _evolution_loop(self) -> List[Dict[str, Any]]:
        """Main evolution loop"""
        results = []
        
        for cycle in range(self.max_cycles):
            print(f"\n🔄 Evolution Cycle {cycle + 1}/{self.max_cycles}")
            
            try:
                # Run single evolution cycle
                result = await self._single_evolution_cycle()
                results.append(result)
                
                # Check for stability
                if result.get("stable", False):
                    print(f"🎯 Stability achieved at cycle {cycle + 1}")
                    break
                    
            except Exception as e:
                print(f"❌ Cycle {cycle + 1} failed: {e}")
                break
        
        return results
    
    async def _single_evolution_cycle(self) -> Dict[str, Any]:
        """Single evolution cycle"""
        cycle_start = time.time()
        
        # Sensor pulse
        sensor_result = self.sensor_hub.poll_and_apply()
        
        # Calculate phi metrics from ontology
        phi_signals = self.ontology.phi_signals()
        phi0 = self._compute_phi0(phi_signals)
        coherence = float(phi_signals.get('coverage_ratio', 0.0))
        
        # Check stability
        stable = phi0 > self.stability_threshold and coherence > 0.7
        
        cycle_time = time.time() - cycle_start
        
        return {
            "phi0": phi0,
            "coherence": coherence, 
            "stable": stable,
            "cycle_time": cycle_time,
            "sensor_updates": len(sensor_result.get("updated_nodes", []))
        }
    
    def _compute_phi0(self, phi_signals: Dict[str, Any]) -> float:
        """Compute integrated information (Φ₀) from ontology signals"""
        try:
            # Extract key signals
            coverage_ratio = phi_signals.get('coverage_ratio', 0.0)
            changed_nodes = phi_signals.get('changed_nodes', 0)
            uncovered_requirements = phi_signals.get('uncovered_requirements', 0)
            open_p0_issues = phi_signals.get('open_p0_issues', 0)
            entropy_hint = phi_signals.get('entropy_hint', 1.0)
            
            # Total nodes for normalization
            total_nodes = phi_signals.get('total_nodes', 1)
            
            # Base coherence from coverage (0.0 to 1.0)
            base_coherence = coverage_ratio
            
            # Change activity (normalize by total nodes)
            change_activity = min(changed_nodes / max(total_nodes, 1), 1.0) if total_nodes > 0 else 0.0
            
            # Requirement penalty (normalized)
            req_penalty = min(uncovered_requirements * 0.1, 0.5) if uncovered_requirements > 0 else 0.0
            
            # P0 issue penalty (critical issues heavily penalize)
            p0_penalty = min(open_p0_issues * 0.2, 0.8) if open_p0_issues > 0 else 0.0
            
            # Entropy bonus (lower entropy = more structured = higher phi)
            entropy_bonus = (1.0 - min(entropy_hint, 1.0)) * 0.1
            
            # Compute integrated information
            # Phi combines coherence with information integration
            phi0 = (base_coherence * 0.6 +      # Primary signal
                   change_activity * 0.2 +      # Activity integration
                   entropy_bonus               # Structure bonus
                   - req_penalty              # Requirements penalty
                   - p0_penalty)              # Critical issues penalty
            
            # Ensure bounds
            phi0 = max(0.0, min(1.0, phi0))
            
            return float(phi0)
            
        except Exception as e:
            print(f"⚠️  Error computing Φ₀: {e}")
            return 0.0