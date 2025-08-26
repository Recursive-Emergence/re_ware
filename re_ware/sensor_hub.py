"""
SensorHub - Core sensory system for RE_ware
==========================================

The sensory core that feeds the Ψ (externalized memory) before any planning/decisions.
Follows RE pattern: sensors → Ψ → Φ → Ω → actions

Architecture:
- SensorHub: Single ingestion bus that normalizes all events
- DomainEvent: Standardized event format from all sources  
- Sensors: Git, FS, GitHub, CLI - all feed into the hub
- Watermark persistence: Never lose track of where we left off
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

# Import ontology functions that are used throughout the module
from .ontology import create_edge, RelationType, NodeType

@dataclass
class DomainEvent:
    """Normalized event from any source into Ψ"""
    source: str  # "git|fs|gh|cli"
    kind: str    # "create|modify|delete|rename|issue|pr|comment"
    path: str    # "src/x.py" or "issues/42" 
    sha: Optional[str] = None  # commit sha if applicable
    ref: Optional[str] = None  # "refs/heads/main"
    ts: float = 0.0           # unix timestamp
    actor: str = "system"     # who/what triggered this
    meta: Dict[str, Any] = None  # extra context
    
    def __post_init__(self):
        if self.ts == 0.0:
            self.ts = time.time()
        if self.meta is None:
            self.meta = {}
    
    @property 
    def event_id(self) -> str:
        """Unique ID for idempotency (sha + path)"""
        key_data = f"{self.source}:{self.kind}:{self.path}:{self.sha or 'none'}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SensorInterface(ABC):
    """Interface for all sensors that feed the hub"""
    
    @abstractmethod
    def poll(self) -> List[DomainEvent]:
        """Return new events since last poll"""
        pass
    
    @abstractmethod
    def reset_watermark(self, watermark: Any = None):
        """Reset sensor to specific point (for bootstrap)"""
        pass
    
    @property
    @abstractmethod
    def sensor_type(self) -> str:
        """Type identifier for this sensor"""
        pass

class SensorHub:
    """
    Central sensory system that ingests from multiple sources
    and applies batches to the ontological graph (Ψ)
    """
    
    def __init__(self, ontology_graph, batch_window_ms: int = 300):
        self.graph = ontology_graph
        self.batch_window_ms = batch_window_ms
        self.sensors: Dict[str, SensorInterface] = {}
        self.event_buffer: List[DomainEvent] = []
        self.seen_event_ids: Set[str] = set()
        self.last_batch_ts = time.time()
        self.mapping_rules = self._load_mapping_rules()
        
        # Watermark persistence
        self.watermarks = {
            "last_tracked_commit": None,
            "sensor_versions": {},
            "schema_version": "1.0"
        }
    
    def register_sensor(self, sensor: SensorInterface):
        """Register a sensor to feed events into the hub"""
        self.sensors[sensor.sensor_type] = sensor
        print(f"📡 Registered sensor: {sensor.sensor_type}")
    
    def bootstrap_from_watermarks(self):
        """
        Critical: Load watermarks and sync from last known state
        This ensures Ψ is correct before any planning begins
        """
        print("🔄 Bootstrapping sensory system from watermarks...")
        
        # First, clean up any existing mess
        print("   🧹 Pre-bootstrap cleanup of irrelevant nodes...")
        cleanup_stats = self.nuclear_cleanup()
        if cleanup_stats["removed"] > 0:
            print(f"   ☢️  Cleaned up {cleanup_stats['removed']} irrelevant nodes before bootstrap")
        
        # Load watermarks from Ψ snapshot
        self._load_watermarks_from_snapshot()
        
        # Reset all sensors to their watermarks
        for sensor_type, sensor in self.sensors.items():
            watermark = self.watermarks.get(sensor_type)
            sensor.reset_watermark(watermark)
            print(f"   📡 {sensor_type} reset to watermark: {watermark}")
        
        # Do passive sync first - this is critical for correct Ψ
        print("   🔍 Performing passive sync from watermarks...")
        initial_events = []
        
        for sensor in self.sensors.values():
            events = sensor.poll()  # This should return all changes since watermark
            initial_events.extend(events)
        
        if initial_events:
            print(f"   📥 Ingesting {len(initial_events)} bootstrap events")
            result = self._apply_batch(initial_events, is_bootstrap=True)
            
            # Aggressive post-bootstrap cleanup
            print("   🔧 Post-bootstrap consolidation...")
            consolidate_stats = self.consolidate_duplicate_nodes()
            if consolidate_stats["merged"] > 0:
                print(f"   🔧 Post-bootstrap consolidated {consolidate_stats['merged']} files, removed {consolidate_stats['removed']} duplicates")
            
            print("   🧹 Post-bootstrap aggressive pruning...")
            prune_stats = self.prune_irrelevant_nodes()
            if prune_stats["pruned"] > 0:
                print(f"   🗂️  Post-bootstrap pruned {prune_stats['pruned']} more irrelevant nodes")
            
            # Now save watermarks and trigger snapshot after cleanup
            self._save_watermarks_to_snapshot()
            
            return {
                "events_applied": len(initial_events),
                "nodes_changed": result.get("nodes_changed", []),
                "bootstrap": True
            }
        else:
            print("   ✅ Ψ already up to date")
            return {
                "events_applied": 0,
                "nodes_changed": [],
                "bootstrap": True
            }
    
    def poll_and_apply(self) -> Dict[str, Any]:
        """
        Main sensing cycle: poll sensors → batch events → apply to Ψ
        Returns pulse data for Φ computation
        """
        current_time = time.time()
        
        # Collect events from all sensors
        for sensor in self.sensors.values():
            try:
                events = sensor.poll()
                for event in events:
                    if event.event_id not in self.seen_event_ids:
                        self.event_buffer.append(event)
                        self.seen_event_ids.add(event.event_id)
            except Exception as e:
                print(f"⚠️  Sensor {sensor.sensor_type} error: {e}")
        
        # Check if batch window expired or buffer is full
        batch_ready = (
            len(self.event_buffer) > 0 and 
            (current_time - self.last_batch_ts) * 1000 >= self.batch_window_ms
        ) or len(self.event_buffer) >= 50  # Force batch on size
        
        if batch_ready and self.event_buffer:
            batch = self.event_buffer.copy()
            self.event_buffer.clear()
            self.last_batch_ts = current_time
            
            return self._apply_batch(batch)
        
        return {"events_applied": 0, "nodes_changed": [], "pulse_data": None}
    
    def _apply_batch(self, events: List[DomainEvent], is_bootstrap: bool = False) -> Dict[str, Any]:
        """Apply a batch of events to the ontological graph"""
        if not events:
            return {"events_applied": 0, "nodes_changed": [], "pulse_data": None}
        
        print(f"📥 Applying batch: {len(events)} events {'(bootstrap)' if is_bootstrap else ''}")
        
        changed_nodes = []
        pulse_data = {
            "events_processed": len(events),
            "source_breakdown": {},
            "kind_breakdown": {},
            "batch_timestamp": time.time()
        }
        
        # Process each event according to mapping rules
        for event in events:
            try:
                # Update source/kind breakdowns
                pulse_data["source_breakdown"][event.source] = pulse_data["source_breakdown"].get(event.source, 0) + 1
                pulse_data["kind_breakdown"][event.kind] = pulse_data["kind_breakdown"].get(event.kind, 0) + 1
                
                # Apply mapping rules to update ontology
                nodes_updated = self._apply_mapping_rules(event)
                changed_nodes.extend(nodes_updated)
                
                # Update watermarks
                if event.source == "git" and event.sha:
                    self.watermarks["last_tracked_commit"] = event.sha
                
            except Exception as e:
                print(f"⚠️  Failed to apply event {event.event_id}: {e}")
        
        # Update hot state in graph (skip for bootstrap to avoid marking all files as "changed")
        if changed_nodes and not is_bootstrap:
            self.graph.hot_state.changed_nodes.update(changed_nodes)
            print(f"   🔥 Updated hot state: {len(changed_nodes)} nodes changed")
        elif is_bootstrap and changed_nodes:
            print(f"   🔍 Bootstrap: Created/updated {len(changed_nodes)} nodes (not marked as changed)")
            
            # CRITICAL: Infer relationships after node creation/updates
            if len(changed_nodes) >= 5:  # Only for substantial batches to avoid noise
                relationships_added = self._infer_relationships(changed_nodes)
                if relationships_added > 0:
                    print(f"   🔗 Inferred {relationships_added} relationships from node patterns")
        
        # Save updated watermarks (but don't trigger snapshot save during bootstrap)
        if not is_bootstrap:
            self._save_watermarks_to_snapshot()
        
        return {
            "events_applied": len(events),
            "nodes_changed": list(set(changed_nodes)),
            "pulse_data": pulse_data
        }
    
    def _infer_relationships(self, changed_node_ids: Set[str]) -> int:
        """Infer logical relationships between nodes based on patterns and content"""
        relationships_added = 0
        
        try:
            
            # Get all nodes for relationship inference
            all_nodes = {nid: node for nid, node in self.graph.nodes.items()}
            changed_nodes = {nid: all_nodes[nid] for nid in changed_node_ids if nid in all_nodes}
            
            # 1. Code-Documentation relationships 
            code_nodes = [n for n in changed_nodes.values() if n.type == NodeType.CODEMODULE]
            doc_nodes = [n for n in changed_nodes.values() if n.type == NodeType.TECHNICALDOC]
            
            for code_node in code_nodes:
                for doc_node in doc_nodes:
                    # Link if documentation mentions the code file
                    code_filename = code_node.content.get("path", "").split("/")[-1].replace(".py", "")
                    doc_title = doc_node.title.lower()
                    
                    if code_filename and len(code_filename) > 3 and code_filename.lower() in doc_title:
                        edge = create_edge(RelationType.EXPLAINS, doc_node.id, code_node.id)
                        if edge.id not in self.graph.edges:
                            self.graph.add_edge(edge)
                            relationships_added += 1
            
            # 2. Test-Code relationships
            test_nodes = [n for n in changed_nodes.values() 
                         if n.type == NodeType.CODEMODULE and 
                         any(pattern in n.title.lower() for pattern in ['test_', 'tests/', '_test.py'])]
            
            for test_node in test_nodes:
                test_path = test_node.content.get("path", "").lower()
                # Extract what this test might be testing
                if "test_" in test_path:
                    tested_module = test_path.replace("test_", "").replace("tests/", "").replace(".py", "")
                    
                    # Find corresponding code module
                    for code_node in code_nodes:
                        code_path = code_node.content.get("path", "").lower()
                        if tested_module in code_path and test_node.id != code_node.id:
                            edge = create_edge(RelationType.VERIFIES, test_node.id, code_node.id)
                            if edge.id not in self.graph.edges:
                                self.graph.add_edge(edge)
                                relationships_added += 1
            
            # 3. Project structure relationships
            # Link main files to project
            main_files = ['evolve.py', 'main.py', '__init__.py', 'setup.py']
            main_nodes = [n for n in changed_nodes.values() 
                         if n.type == NodeType.CODEMODULE and
                         any(main_file in n.title.lower() for main_file in main_files)]
            
            # Create a virtual PROJECT node if we have main files but no project node
            project_nodes = [n for n in all_nodes.values() if n.type == NodeType.PROJECT]
            if main_nodes and not project_nodes:
                # Let the main files document the project for now
                for i, main_node in enumerate(main_nodes[:1]):  # Just the first one
                    for doc_node in doc_nodes:
                        if 'readme' in doc_node.title.lower() or 'project' in doc_node.title.lower():
                            edge = create_edge(RelationType.EXPLAINS, doc_node.id, main_node.id)
                            if edge.id not in self.graph.edges:
                                self.graph.add_edge(edge)
                                relationships_added += 1
            
            # 4. Module hierarchy relationships
            # Link modules in the same directory/package
            for code_node1 in code_nodes:
                path1_parts = code_node1.content.get("path", "").split("/")
                if len(path1_parts) > 1:  # Has directory structure
                    package_path = "/".join(path1_parts[:-1])  # Remove filename
                    
                    for code_node2 in code_nodes:
                        if code_node1.id != code_node2.id:
                            path2 = code_node2.content.get("path", "")
                            if path2.startswith(package_path):
                                # Same package - relate them
                                edge = create_edge(RelationType.DEPENDS_ON, code_node1.id, code_node2.id)
                                if edge.id not in self.graph.edges:
                                    self.graph.add_edge(edge)
                                    relationships_added += 1
                                    # Only add one relationship per node to avoid explosion
                                    break
        
        except Exception as e:
            print(f"⚠️  Relationship inference failed: {e}")
            return 0
        
        return relationships_added
    
    def _apply_mapping_rules(self, event: DomainEvent) -> List[str]:
        """Apply mapping rules to translate event into ontology updates"""
        changed_node_ids = []
        
        # Find first matching rule for this event (rules are ordered by specificity)
        matching_rule = None
        for rule in self.mapping_rules.get("rules", []):
            if self._path_matches_pattern(event.path, rule.get("match", "")):
                # Check excludes
                excludes = rule.get("exclude", [])
                excluded = False
                for exclude_pattern in excludes:
                    if self._path_matches_pattern(event.path, exclude_pattern):
                        excluded = True
                        break
                
                if not excluded:
                    matching_rule = rule
                    break
        
        # Apply the first matching rule only
        if matching_rule:
            try:
                node_id = self._apply_node_rule(event, matching_rule)
                if node_id:
                    changed_node_ids.append(node_id)
                    
                # Apply edge rules
                edge_rules = matching_rule.get("edges", [])
                for edge_rule in edge_rules:
                    self._apply_edge_rule(event, edge_rule, node_id)
                    
            except Exception as e:
                print(f"⚠️  Rule application error: {e}")
        
        # Fallback: basic file mapping if no rules matched
        if not matching_rule:
            node_id = self._apply_default_mapping(event)
            if node_id:
                changed_node_ids.append(node_id)
        
        return changed_node_ids
    
    def _apply_node_rule(self, event: DomainEvent, rule: Dict[str, Any]) -> Optional[str]:
        """Apply a node creation/update rule"""
        from .ontology import create_node
        
        node_config = rule.get("node", {})
        node_type_str = node_config.get("type", "CODEMODULE")
        
        # Map string to NodeType enum
        try:
            node_type = getattr(NodeType, node_type_str)
        except AttributeError:
            print(f"⚠️  Unknown node type: {node_type_str}")
            node_type = NodeType.CODEMODULE
        
        # Check if a node already exists for this file path and type
        existing_node_id = self._find_existing_node_for_path(event.path, node_type)
        
        # Prepare content for node
        node_title = f"{node_type_str}: {Path(event.path).name}"
        node_fields = node_config.get("fields", {})
        
        content = {
            **node_fields,
            "path": event.path,
            "last_modified": event.ts,
            "source": event.source,
            "actor": event.actor
        }
        
        # Add specific metadata based on event
        if event.meta:
            content.update(event.meta)
        
        if existing_node_id:
            # Update existing node
            existing_node = self.graph.nodes[existing_node_id]
            existing_node.content.update(content)
            existing_node.touch(f"Updated by {event.source}", by=event.actor)
            return existing_node_id
        else:
            # Create new node
            node = create_node(node_type, node_title, content=content)
            
            # Add to graph
            success = self.graph.add_node(node)
            if success:
                return node.id
            
            return None
    
    def _find_existing_node_for_path(self, file_path: str, node_type: NodeType) -> Optional[str]:
        """Find existing node for the same file path and type"""
        for node_id, node in self.graph.nodes.items():
            if (node.type == node_type and 
                node.content.get("path") == file_path):
                return node_id
        return None
    
    def consolidate_duplicate_nodes(self) -> Dict[str, int]:
        """Consolidate duplicate nodes for the same file paths"""
        print(f"🔧 DEBUG: Starting consolidation with {len(self.graph.nodes)} nodes")
        path_to_nodes = {}
        consolidation_stats = {"merged": 0, "removed": 0}
        
        # Group nodes by path and type
        for node_id, node in list(self.graph.nodes.items()):
            if hasattr(node, 'content') and 'path' in node.content:
                path = node.content['path']
                node_type = node.type
                key = (path, node_type)
                
                if key not in path_to_nodes:
                    path_to_nodes[key] = []
                path_to_nodes[key].append((node_id, node))
        
        # Consolidate duplicates
        for (path, node_type), node_list in path_to_nodes.items():
            if len(node_list) > 1:
                # Keep the most recently modified node
                node_list.sort(key=lambda x: x[1].content.get('last_modified', 0), reverse=True)
                keeper_id, keeper_node = node_list[0]
                
                # Mark keeper as changed since we're consolidating
                self.graph.hot_state.changed_nodes.add(keeper_id)
                
                # Remove duplicates
                for dup_id, dup_node in node_list[1:]:
                    # Transfer any important content
                    if hasattr(dup_node, 'state') and dup_node.state.change_summary:
                        keeper_node.state.change_summary = f"{keeper_node.state.change_summary}; {dup_node.state.change_summary}"
                    
                    # Remove duplicate from graph
                    if dup_id in self.graph.nodes:
                        del self.graph.nodes[dup_id]
                    if dup_id in self.graph.node_edges:
                        del self.graph.node_edges[dup_id]
                    if dup_id in self.graph.hot_state.changed_nodes:
                        self.graph.hot_state.changed_nodes.remove(dup_id)
                    # Clean up corresponding llm_card
                    if dup_id in self.graph.llm_cards:
                        del self.graph.llm_cards[dup_id]
                    
                    consolidation_stats["removed"] += 1
                
                consolidation_stats["merged"] += 1
                print(f"   📝 Consolidated {len(node_list)} nodes for {path}")
        
        print(f"🔧 DEBUG: After consolidation: {len(self.graph.nodes)} nodes remaining")
        
        # Clean up orphaned llm_cards
        self._cleanup_orphaned_llm_cards()
        
        return consolidation_stats
    
    def prune_irrelevant_nodes(self) -> Dict[str, int]:
        """Prune nodes that shouldn't be in the system (venv, build artifacts, etc.)"""
        prune_stats = {"pruned": 0, "kept": 0}
        
        # Aggressive patterns for paths that should be pruned
        prune_patterns = [
            "/venv/", "/env/", "/.venv/", "/site-packages/", 
            "/node_modules/", "/__pycache__/", "/.git/", "/.git_back/",
            "/build/", "/dist/", "/.pytest_cache/",
            "/coverage/", "/.coverage", "/htmlcov/",
            # More aggressive patterns
            "venv/", "site-packages/", "lib/python", "/lib/",
            "/.tox/", "/eggs/", "/.eggs/", "/sdist/",
            "/wheel/", "/develop-eggs/", "/.cache/",
            # Specific to this project's structure
            "anthropic/", "certifi/", "charset_normalizer/", 
            "click/", "distro/", "h11/", "httpcore/", "httpx/",
            "idna/", "jinja2/", "pydantic/", "sniffio/", "typing_extensions/",
            "urllib3/", "uvicorn/", "fastapi/"
        ]
        
        # Additional file extensions to prune
        prune_extensions = [".pyc", ".pyo", ".pyd", "__pycache__", ".whl", ".egg"]
        
        nodes_to_remove = []
        
        # Whitelist: Only keep files from these directories
        keep_patterns = [
            "/re_ware/", "/tests/", 
            "setup.py", "evolve.py", "requirements", 
            ".yml", ".yaml", ".md", ".txt", ".json", ".toml"
        ]
        
        for node_id, node in list(self.graph.nodes.items()):
            should_prune = False
            
            # Skip non-file nodes (PROJECT, ADVICE, etc.)
            if not hasattr(node, 'content') or 'path' not in node.content:
                continue
                
            path = node.content['path']
            
            # Whitelist check: if it's a file node, only keep if it matches whitelist
            if hasattr(node, 'type') and str(node.type) == 'NodeType.CODEMODULE':
                should_keep = False
                for pattern in keep_patterns:
                    if pattern in path:
                        should_keep = True
                        break
                if not should_keep:
                    should_prune = True
            else:
                # For non-CODEMODULE nodes, use blacklist approach
                # Check against prune patterns
                for pattern in prune_patterns:
                    if pattern in path:
                        should_prune = True
                        break
                
                # Check file extensions
                if not should_prune:
                    for ext in prune_extensions:
                        if path.endswith(ext):
                            should_prune = True
                            break
            
            # Check for nodes with generic auto-generated titles or git hashes
            if not should_prune and node.title:
                generic_titles = ["Untitled Node", "Auto-generated", "Unknown"]
                if any(generic in node.title for generic in generic_titles):
                    should_prune = True
                
                # Check for git object hash-like titles
                if "File: " in node.title and len(node.title.split("File: ")[-1]) >= 32:
                    # Looks like "File: <hash>" - probably a git object
                    should_prune = True
            
            if should_prune:
                nodes_to_remove.append(node_id)
            else:
                prune_stats["kept"] += 1
        
        # Remove pruned nodes
        for node_id in nodes_to_remove:
            if node_id in self.graph.nodes:
                del self.graph.nodes[node_id]
            if node_id in self.graph.node_edges:
                del self.graph.node_edges[node_id]
            if node_id in self.graph.hot_state.changed_nodes:
                self.graph.hot_state.changed_nodes.remove(node_id)
            # Clean up corresponding llm_card
            if node_id in self.graph.llm_cards:
                del self.graph.llm_cards[node_id]
            prune_stats["pruned"] += 1
        
        if prune_stats["pruned"] > 0:
            print(f"   🗂️ Pruned {prune_stats['pruned']} irrelevant nodes")
        
        # Clean up orphaned llm_cards
        self._cleanup_orphaned_llm_cards()
        
        return prune_stats
    
    def _cleanup_orphaned_llm_cards(self):
        """Remove llm_cards that don't have corresponding nodes"""
        orphaned_cards = []
        for card_id in list(self.graph.llm_cards.keys()):
            if card_id not in self.graph.nodes:
                orphaned_cards.append(card_id)
        
        for card_id in orphaned_cards:
            del self.graph.llm_cards[card_id]
        
        if orphaned_cards:
            print(f"   🧹 Cleaned up {len(orphaned_cards)} orphaned llm_cards")
    
    def nuclear_cleanup(self) -> Dict[str, int]:
        """Nuclear option: Remove ALL file nodes and rebuild from scratch"""
        cleanup_stats = {"removed": 0, "kept": 0}
        
        # Keep only essential nodes (PROJECT, ADVICE, etc. - not file-based nodes)
        essential_types = ["PROJECT", "ADVICE", "REQUIREMENT", "BUG", "BUILD", "COVERAGE"]
        nodes_to_remove = []
        
        for node_id, node in self.graph.nodes.items():
            if hasattr(node, 'type'):
                node_type_str = str(node.type).split('.')[-1] if hasattr(node.type, 'name') else str(node.type)
                
                if node_type_str in essential_types:
                    cleanup_stats["kept"] += 1
                else:
                    nodes_to_remove.append(node_id)
            else:
                nodes_to_remove.append(node_id)
        
        # Remove all file nodes
        for node_id in nodes_to_remove:
            if node_id in self.graph.nodes:
                del self.graph.nodes[node_id]
            if node_id in self.graph.node_edges:
                del self.graph.node_edges[node_id]
            if node_id in self.graph.hot_state.changed_nodes:
                self.graph.hot_state.changed_nodes.remove(node_id)
            cleanup_stats["removed"] += 1
        
        # Clear edges that reference removed nodes
        edges_to_remove = []
        for edge_id, edge in self.graph.edges.items():
            if edge.from_node not in self.graph.nodes or edge.to_node not in self.graph.nodes:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            if edge_id in self.graph.edges:
                del self.graph.edges[edge_id]
        
        # Clear node_edges mapping for removed nodes
        for node_id in nodes_to_remove:
            if node_id in self.graph.node_edges:
                del self.graph.node_edges[node_id]
        
        # Fix corrupted node_edges mappings for remaining nodes
        for node_id, edge_ids in list(self.graph.node_edges.items()):
            if node_id in self.graph.nodes:  # Only process valid nodes
                # Filter out invalid edge IDs
                valid_edge_ids = set()
                for edge_id in edge_ids:
                    if edge_id in self.graph.edges:
                        valid_edge_ids.add(edge_id)
                    else:
                        print(f"   🔧 Removing corrupted edge reference: {edge_id}")
                self.graph.node_edges[node_id] = valid_edge_ids
            else:
                # Remove node_edges entries for nodes that don't exist
                del self.graph.node_edges[node_id]
        
        # Clear any orphaned references in hot state
        if hasattr(self.graph, 'hot_state'):
            # Remove any changed_nodes that no longer exist
            valid_changed_nodes = set()
            for node_id in self.graph.hot_state.changed_nodes:
                if node_id in self.graph.nodes:
                    valid_changed_nodes.add(node_id)
            self.graph.hot_state.changed_nodes = valid_changed_nodes
        
        # Final validation - ensure graph integrity
        self._validate_graph_integrity()
        
        print(f"   ☢️  Nuclear cleanup: removed {cleanup_stats['removed']} nodes, kept {cleanup_stats['kept']} essential nodes")
        return cleanup_stats
    
    def _validate_graph_integrity(self):
        """Validate and fix any graph integrity issues"""
        try:
            # Check for orphaned edges
            orphaned_edges = []
            for edge_id, edge in self.graph.edges.items():
                if edge.from_node not in self.graph.nodes or edge.to_node not in self.graph.nodes:
                    orphaned_edges.append(edge_id)
            
            # Remove orphaned edges
            for edge_id in orphaned_edges:
                if edge_id in self.graph.edges:
                    del self.graph.edges[edge_id]
            
            if orphaned_edges:
                print(f"   🔧 Cleaned up {len(orphaned_edges)} orphaned edges")
            
            # Fix corrupted node_edges mappings
            corrupted_refs = 0
            for node_id, edge_ids in list(self.graph.node_edges.items()):
                if node_id not in self.graph.nodes:
                    # Remove mapping for non-existent nodes
                    del self.graph.node_edges[node_id]
                    continue
                    
                # Filter out invalid edge references
                valid_edge_ids = set()
                for edge_id in edge_ids:
                    if edge_id in self.graph.edges:
                        valid_edge_ids.add(edge_id)
                    else:
                        corrupted_refs += 1
                        
                self.graph.node_edges[node_id] = valid_edge_ids
            
            if corrupted_refs > 0:
                print(f"   🔧 Fixed {corrupted_refs} corrupted edge references")
                
        except Exception as e:
            print(f"   ⚠️  Graph integrity check failed: {e}")
    
    def _apply_edge_rule(self, event: DomainEvent, edge_rule: Dict[str, Any], from_node_id: str):
        """Apply edge creation rule to build traceability links"""
        try:
            rel_type = edge_rule.get("rel", "RELATES_TO")
            to_pattern = edge_rule.get("to", "")
            
            # Parse target pattern
            target_node_ids = self._resolve_edge_targets(event, to_pattern, from_node_id)
            
            for target_id in target_node_ids:
                if target_id != from_node_id and target_id in self.graph.nodes:
                    # Create the edge using the ontology's create_edge method
                    
                    # Map string relation types to enum values
                    relation_map = {
                        "IMPLEMENTS": RelationType.IMPLEMENTS,
                        "VERIFIES": RelationType.VERIFIES, 
                        "TESTS": RelationType.VERIFIES,
                        "DOCUMENTS": RelationType.DOCUMENTS,
                        "DEPENDS_ON": RelationType.DEPENDS_ON,
                        "BELONGS_TO": RelationType.BELONGS_TO,
                        "RELATES_TO": RelationType.RELATES_TO,
                        "CONFIGURES": RelationType.CONFIGURES
                    }
                    
                    relation = relation_map.get(rel_type.upper(), RelationType.RELATES_TO)
                    
                    # Create edge using standalone function, then add to graph
                    edge = create_edge(relation, from_node_id, target_id)
                    success = self.graph.add_edge(edge)
                    
                    if success:
                        print(f"   🔗 Created edge: {from_node_id} --{rel_type}--> {target_id}")
                    else:
                        print(f"   ⚠️  Edge creation failed: {from_node_id} --{rel_type}--> {target_id}")
                        
        except Exception as e:
            print(f"   ❌ Edge rule application failed: {e}")
    
    def _resolve_edge_targets(self, event: DomainEvent, to_pattern: str, from_node_id: str) -> List[str]:
        """Resolve edge target patterns to actual node IDs"""
        targets = []
        
        if not to_pattern:
            return targets
            
        # Handle different target pattern types
        if to_pattern.startswith("req:"):
            # Find requirement nodes by pattern matching
            pattern = to_pattern[4:]  # Remove "req:" prefix
            targets.extend(self._find_nodes_by_pattern(pattern, "REQUIREMENT"))
            
        elif to_pattern.startswith("test:"):
            # Find test nodes
            pattern = to_pattern[5:]  # Remove "test:" prefix  
            targets.extend(self._find_nodes_by_pattern(pattern, "TEST"))
            
        elif to_pattern.startswith("code:"):
            # Find code module nodes
            pattern = to_pattern[5:]  # Remove "code:" prefix
            targets.extend(self._find_nodes_by_pattern(pattern, "CODEMODULE"))
            
        elif to_pattern.startswith("doc:"):
            # Find documentation nodes
            pattern = to_pattern[4:]  # Remove "doc:" prefix
            targets.extend(self._find_nodes_by_pattern(pattern, "TECHNICALDOC"))
            
        elif "{infer}" in to_pattern:
            # Infer relationships based on file patterns
            targets.extend(self._infer_targets_from_file(event, to_pattern))
            
        elif to_pattern in self.graph.nodes:
            # Direct node ID reference (like "project:root")
            targets.append(to_pattern)
        else:
            # Pattern-based search
            targets.extend(self._find_nodes_by_pattern(to_pattern, None))
            
        return targets
    
    def _find_nodes_by_pattern(self, pattern: str, node_type: str = None) -> List[str]:
        """Find nodes matching a pattern and optional type"""
        matches = []
        
        for node_id, node in self.graph.nodes.items():
            # Type filter
            if node_type and node.type.name != node_type:
                continue
                
            # Pattern matching on title or path
            if pattern == "{infer}" or pattern in node.title.lower() or pattern in node_id:
                matches.append(node_id)
                
        return matches[:5]  # Limit to prevent explosion
    
    def _infer_targets_from_file(self, event: DomainEvent, pattern: str) -> List[str]:
        """Infer edge targets based on file naming conventions"""
        targets = []
        path = event.path
        
        if not path:
            return targets
            
        # Common inference patterns
        base_name = Path(path).stem
        
        # test_foo.py -> foo.py relationship
        if base_name.startswith("test_"):
            target_name = base_name[5:]  # Remove "test_" prefix
            targets.extend(self._find_nodes_by_title_pattern(target_name))
            
        # foo_test.py -> foo.py relationship  
        elif base_name.endswith("_test"):
            target_name = base_name[:-5]  # Remove "_test" suffix
            targets.extend(self._find_nodes_by_title_pattern(target_name))
            
        # README.md in directory -> directory modules
        elif "readme" in base_name.lower():
            dir_path = str(Path(path).parent)
            targets.extend(self._find_nodes_in_directory(dir_path))
            
        return targets
    
    def _find_nodes_by_title_pattern(self, pattern: str) -> List[str]:
        """Find nodes with titles containing the pattern"""
        matches = []
        for node_id, node in self.graph.nodes.items():
            if pattern.lower() in node.title.lower():
                matches.append(node_id)
        return matches
    
    def _find_nodes_in_directory(self, dir_path: str) -> List[str]:
        """Find nodes representing files in a directory"""
        matches = []
        for node_id, node in self.graph.nodes.items():
            if hasattr(node, 'meta') and 'path' in node.meta:
                node_dir = str(Path(node.meta['path']).parent)
                if node_dir == dir_path:
                    matches.append(node_id)
        return matches
    
    def _apply_default_mapping(self, event: DomainEvent) -> Optional[str]:
        """Fallback mapping when no rules match"""
        from .ontology import create_node
        
        # Simple file-based mapping
        file_path = Path(event.path)
        
        if file_path.suffix == ".py":
            node_type = NodeType.CODEMODULE
            title = f"Python Module: {file_path.name}"
        elif file_path.suffix in [".md", ".txt"]:
            node_type = NodeType.TECHNICALDOC
            title = f"Documentation: {file_path.name}"
        elif "test" in file_path.name.lower():
            node_type = NodeType.TEST
            title = f"Test: {file_path.name}"
        else:
            node_type = NodeType.CODEMODULE
            title = f"File: {file_path.name}"
        
        content = {
            "path": event.path,
            "last_modified": event.ts,
            "source": event.source,
            "detected_via": "default_mapping"
        }
        
        node = create_node(node_type, title, content=content)
        success = self.graph.add_node(node)
        
        return node.id if success else None
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Glob-style pattern matching with ** support"""
        import fnmatch
        
        # Handle complex patterns like **/requirements/**/*.md
        if '**' in pattern:
            # Convert ** patterns to simpler matching
            pattern_parts = pattern.split('**')
            if len(pattern_parts) == 3:  # **/requirements/**/*.md
                prefix, middle, suffix = pattern_parts
                # Remove leading/trailing slashes
                middle = middle.strip('/')
                suffix = suffix.lstrip('/')
                
                # Check if path contains the middle part and ends with suffix pattern
                if middle in path and fnmatch.fnmatch(path.split('/')[-1], suffix):
                    return True
            elif len(pattern_parts) == 2:  # **/*.md
                suffix = pattern_parts[1].lstrip('/')
                return fnmatch.fnmatch(path.split('/')[-1], suffix)
        
        return fnmatch.fnmatch(path, pattern)
    
    def _load_mapping_rules(self) -> Dict[str, Any]:
        """Load mapping rules from sensors.yml"""
        rules_file = Path("sensors.yml")
        
        if rules_file.exists():
            import yaml
            try:
                with open(rules_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"⚠️  Error loading sensors.yml: {e}")
        
        # Return default rules
        return {
            "rules": [
                {
                    "match": "src/**/*.py",
                    "node": {"type": "CODEMODULE", "fields": {"language": "python"}},
                    "edges": []
                },
                {
                    "match": "tests/**/*test*.py", 
                    "node": {"type": "TEST", "fields": {"framework": "pytest"}},
                    "edges": []
                },
                {
                    "match": "docs/**/*.md",
                    "node": {"type": "TECHNICALDOC", "fields": {"format": "markdown"}},
                    "edges": []
                }
            ]
        }
    
    def _load_watermarks_from_snapshot(self):
        """Load watermarks from Ψ snapshot"""
        if hasattr(self.graph, 'last_snapshot_data') and self.graph.last_snapshot_data:
            snapshot_watermarks = self.graph.last_snapshot_data.get("watermarks", {})
            self.watermarks.update(snapshot_watermarks)
            print(f"   📍 Loaded watermarks: {list(self.watermarks.keys())}")
    
    def _save_watermarks_to_snapshot(self):
        """Save current watermarks to Ψ snapshot"""
        if not hasattr(self.graph, 'last_snapshot_data'):
            self.graph.last_snapshot_data = {}
        
        self.graph.last_snapshot_data["watermarks"] = self.watermarks.copy()
        
        # Only trigger snapshot save if not explicitly disabled
        if hasattr(self.graph, '_should_save_snapshot') and self.graph._should_save_snapshot is not False:
            self.graph._should_save_snapshot = True