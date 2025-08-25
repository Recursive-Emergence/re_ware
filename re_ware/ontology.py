"""
Ontological Graph Core for Software Engineering
================================================

Implements the core graph schema for managing complete software lifecycle:
- Gene: Immutable schema/types that define agent capabilities and structure
- Phenotype: Mutable runtime substrate that holds actual project data and state

Gene contains:
- Node types: Product, Project, Requirement, Decision, CodeModule, Test, etc.
- Relations: implements, verifies, derives_from, supersedes, depends_on, etc.
- Data structure templates and validation rules

Phenotype contains:
- Actual node/edge instances and their evolving state
- Hot/warm memory management for LLM integration
- Runtime graph operations and tension detection
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum
from collections import deque
import time
import json
import uuid
from pathlib import Path

# Core Enums
class NodeType(Enum):
    # Project Management
    DOMAIN = "Domain"
    PRODUCT = "Product" 
    PROGRAM = "Program"
    PROJECT = "Project"
    EPIC = "Epic"
    STORY = "Story"
    SPRINT = "Sprint"
    ITERATION = "Iteration"
    MILESTONE = "Milestone"
    BACKLOG = "Backlog"
    
    # Requirements & Design
    REQUIREMENT = "Requirement"
    SPECIFICATION = "Specification"
    DESIGN = "Design"
    DECISION = "Decision"
    RISK = "Risk"
    ASSUMPTION = "Assumption"
    CONSTRAINT = "Constraint"
    
    # Architecture
    COMPONENT = "Component"
    SERVICE = "Service"
    API = "API"
    INTERFACE = "Interface"
    DATABASE = "Database"
    SCHEMA = "Schema"
    CONFIGURATION = "Configuration"
    DEPENDENCY_SPEC = "DependencySpec"
    
    # Development
    CODEMODULE = "CodeModule"
    CLASS = "Class"
    FUNCTION = "Function"
    VARIABLE = "Variable"
    COMMIT = "Commit"
    BRANCH = "Branch"
    PULLREQUEST = "PullRequest"
    MERGE = "Merge"
    
    # Quality & Testing
    TEST = "Test"
    TESTSUITE = "TestSuite"
    BUG = "Bug"
    CODEREVIEW = "CodeReview"
    TECHNICALDEBT = "TechnicalDebt"
    COVERAGE = "Coverage"
    PERFORMANCE = "Performance"
    
    # Operations
    BUILD = "Build"
    PIPELINE = "Pipeline"
    DEPLOYMENT = "Deployment"
    RELEASE = "Release"
    ENVIRONMENT = "Environment"
    INFRASTRUCTURE = "Infrastructure"
    MONITORING = "Monitoring"
    
    # Documentation
    TECHNICALDOC = "TechnicalDoc"
    USERDOC = "UserDoc"
    APIDOC = "APIDoc"
    RUNBOOK = "Runbook"
    TUTORIAL = "Tutorial"
    REFERENCE = "Reference"
    
    # Collaboration
    ACTOR = "Actor"
    TEAM = "Team"
    ROLE = "Role"
    ISSUE = "Issue"
    COMMENT = "Comment"
    DISCUSSION = "Discussion"
    MEETING = "Meeting"
    
    # Metrics & Governance
    METRIC = "Metric"
    KPI = "KPI"
    OKR = "OKR"
    POLICY = "Policy"
    COMPLIANCE = "Compliance"
    AUDIT = "Audit"
    
    # AI/LLM Generated Content
    ADVICE = "Advice"
    
    # Legacy compatibility
    SPEC = "Spec"  # Alias for SPECIFICATION
    INCIDENT = "Incident"  # Could be under Operations
    ARTIFACT = "Artifact"  # Generic artifact type

class RelationType(Enum):
    # Core Relations
    IMPLEMENTS = "implements"
    VERIFIES = "verifies"
    DERIVES_FROM = "derives_from"
    SUPERSEDES = "supersedes"
    DEPENDS_ON = "depends_on"
    BELONGS_TO = "belongs_to"
    ADDRESSES = "addresses"
    PRODUCES = "produces"
    ROLLS_UP_TO = "rolls_up_to"
    OWNED_BY = "owned_by"
    OBSERVES = "observes"
    CONSTRAINS = "constrains"
    RELATES_TO = "relates_to"
    
    # Development Relations
    REVIEWS = "reviews"
    MERGES = "merges"
    COMMITS_TO = "commits_to"
    BRANCHES_FROM = "branches_from"
    FIXES = "fixes"
    REFACTORS = "refactors"
    
    # Operations Relations
    DEPLOYS = "deploys"
    MONITORS = "monitors"
    CONFIGURES = "configures"
    SCALES = "scales"
    
    # Documentation Relations
    DOCUMENTS = "documents"
    EXPLAINS = "explains"
    REFERENCES = "references"
    
    # Quality Relations
    TESTS = "tests"
    COVERS = "covers"
    VALIDATES = "validates"
    
    # Collaboration Relations
    ASSIGNS = "assigns"
    PARTICIPATES_IN = "participates_in"
    COMMENTS_ON = "comments_on"
    APPROVES = "approves"

class Status(Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    IN_DEV = "in_dev"
    RELEASED = "released"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    BLOCKED = "blocked"

class Criticality(Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"

@dataclass
class NodeState:
    """State and versioning for every graph node"""
    status: Status = Status.DRAFT
    version: str = "0.1"
    last_changed: float = field(default_factory=time.time)
    change_summary: str = ""
    provenance: Dict[str, str] = field(default_factory=dict)
    criticality: Criticality = Criticality.P2

@dataclass
class GraphNode:
    """Base node in the ontological graph"""
    id: str
    type: NodeType
    title: str
    state: NodeState = field(default_factory=NodeState)
    meta: Dict[str, Any] = field(default_factory=dict)
    content: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self, change_summary: str = "", by: str = "system"):
        """Update node timestamp and provenance"""
        self.state.last_changed = time.time()
        if change_summary:
            self.state.change_summary = change_summary
        self.state.provenance["by"] = by
        self.state.provenance["updated_at"] = time.time()

@dataclass 
class GraphEdge:
    """Relationship between graph nodes"""
    id: str
    relation: RelationType
    from_node: str
    to_node: str
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class LLMCard:
    """Compressed representation of a graph node for LLM context"""
    id: str
    type: str
    title: str
    status: str
    version: str
    last_change: float
    owners: List[str] = field(default_factory=list)
    delta: List[str] = field(default_factory=list)
    impacts: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    key_links: Dict[str, str] = field(default_factory=dict)
    rollup: List[str] = field(default_factory=list)
    digest: str = ""
    
    def to_json(self) -> str:
        """Serialize to deterministic JSON for LLM consumption"""
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))

@dataclass
class GraphPulse:
    """Compressed delta frame for LLM consumption"""
    timestamp: float
    changed_nodes: List[str]
    key_edges: List[str] 
    counters: Dict[str, int]
    last_events: List[str]
    phi_snapshot: Dict[str, Any]
    
    def to_json(self) -> str:
        """Serialize pulse for LLM"""
        return json.dumps({
            "timestamp": self.timestamp,
            "changed_nodes": self.changed_nodes[:5],  # Cap to 5
            "key_edges": self.key_edges[:3],          # Cap to 3
            "counters": self.counters,
            "last_events": self.last_events[:3],      # Cap to 3
            "phi_snapshot": self.phi_snapshot
        }, separators=(',', ':'))

# ============================================================================
# GENE: Immutable Schema & Templates
# ============================================================================

@dataclass
class OntologyGene:
    """Immutable gene defining the ontological schema and capabilities"""
    species_id: str = "software_engineering_v1"
    version: str = "1.0.0"
    
    # Supported node types for this gene
    node_types: Set[NodeType] = field(default_factory=lambda: set(NodeType))
    
    # Supported relation types for this gene  
    relation_types: Set[RelationType] = field(default_factory=lambda: set(RelationType))
    
    # Validation rules
    required_node_fields: Dict[str, List[str]] = field(default_factory=dict)
    node_type_constraints: Dict[NodeType, Dict[str, Any]] = field(default_factory=dict)
    relation_constraints: Dict[RelationType, Dict[str, Any]] = field(default_factory=dict)
    
    # LLM integration parameters
    max_llm_cards: int = 8
    max_pulse_nodes: int = 5
    max_pulse_edges: int = 3
    
    @classmethod
    def from_schema_file(cls, schema_name: str) -> 'OntologyGene':
        """Load gene from JSON schema file"""
        import json
        from pathlib import Path
        
        # Try to find schema file
        current_dir = Path(__file__).parent
        schema_path = current_dir / "schemas" / f"{schema_name}.json"
        
        if not schema_path.exists():
            print(f"⚠️  Schema {schema_name} not found, falling back to default")
            return cls.default_software_gene()
        
        try:
            with open(schema_path, 'r') as f:
                schema_data = json.load(f)
            
            gene = cls()
            gene.species_id = schema_data.get("species_id", "unknown_v1")
            gene.version = schema_data.get("version", "1.0.0")
            
            # Load supported types (convert string names to enum objects)
            if "supported_node_types" in schema_data:
                gene.node_types = set()
                for type_name in schema_data["supported_node_types"]:
                    try:
                        gene.node_types.add(NodeType(type_name))
                    except ValueError:
                        # Silently skip unknown types to reduce noise
                        pass
            
            if "supported_relation_types" in schema_data:
                gene.relation_types = set()
                for rel_name in schema_data["supported_relation_types"]:
                    try:
                        gene.relation_types.add(RelationType(rel_name))
                    except ValueError:
                        # Silently skip unknown relation types to reduce noise
                        pass
            
            # Load validation rules
            gene.required_node_fields = schema_data.get("required_node_fields", {})
            
            # Convert string keys to NodeType enums in constraints
            gene.node_type_constraints = {}
            for type_name, constraints in schema_data.get("node_type_constraints", {}).items():
                try:
                    node_type = NodeType(type_name)
                    gene.node_type_constraints[node_type] = constraints
                except ValueError:
                    # Silently skip unknown types
                    pass
            
            gene.relation_constraints = {}
            for rel_name, constraints in schema_data.get("relation_constraints", {}).items():
                try:
                    rel_type = RelationType(rel_name)
                    # Convert string type names to NodeType enums
                    processed_constraints = {}
                    for key, type_names in constraints.items():
                        if key in ["from_types", "to_types"]:
                            processed_constraints[key] = []
                            for type_name in type_names:
                                try:
                                    # Try exact match first
                                    node_type = NodeType(type_name)
                                    processed_constraints[key].append(node_type)
                                except ValueError:
                                    # Try mapping from schema names to enum values
                                    schema_to_enum = {
                                        "CODEMODULE": "CodeModule",
                                        "TECHNICALDOC": "TechnicalDoc", 
                                        "USERDOC": "UserDoc",
                                        "APIDOC": "APIDoc",
                                        "PULLREQUEST": "PullRequest",
                                        "TECHNICALDEBT": "TechnicalDebt",
                                        "CODEREVIEW": "CodeReview",
                                        "TESTSUITE": "TestSuite",
                                        "FUNCTION": "Function",
                                        "CLASS": "Class", 
                                        "TEST": "Test",
                                        "BUG": "Bug",
                                        "ISSUE": "Issue",
                                        "COMMIT": "Commit",
                                        "ACTOR": "Actor",
                                        "DESIGN": "Design",
                                        "COMPONENT": "Component",
                                        "REQUIREMENT": "Requirement",
                                        "STORY": "Story", 
                                        "SPECIFICATION": "Specification",
                                        "RELEASE": "Release",
                                        "PROJECT": "Project",
                                        "ADVICE": "Advice"
                                    }
                                    enum_value = schema_to_enum.get(type_name, type_name)
                                    try:
                                        node_type = NodeType(enum_value)
                                        processed_constraints[key].append(node_type)
                                    except ValueError:
                                        # Silently skip unknown constraint types
                                        pass
                        else:
                            processed_constraints[key] = type_names
                    gene.relation_constraints[rel_type] = processed_constraints
                except ValueError:
                    # Silently skip unknown relation types
                    pass
            
            # Load LLM integration parameters
            llm_config = schema_data.get("llm_integration", {})
            gene.max_llm_cards = llm_config.get("max_llm_cards", 8)
            gene.max_pulse_nodes = llm_config.get("max_pulse_nodes", 5)
            gene.max_pulse_edges = llm_config.get("max_pulse_edges", 3)
            
            print(f"🧬 Loaded gene schema: {gene.species_id} v{gene.version}")
            return gene
            
        except Exception as e:
            print(f"❌ Failed to load schema {schema_name}: {e}")
            return cls.default_software_gene()
    
    @classmethod
    def default_software_gene(cls) -> 'OntologyGene':
        """Create default software engineering gene - fallback when no schema available"""
        gene = cls()
        
        # Define core validation rules
        gene.required_node_fields = {
            "all": ["id", "type", "title", "state"],
            NodeType.REQUIREMENT.value: ["content.priority"],
            NodeType.TEST.value: ["content.test_type"],
            NodeType.CODEMODULE.value: ["content.path"]
        }
        
        # Define node type constraints
        gene.node_type_constraints = {
            NodeType.REQUIREMENT: {"max_title_length": 200},
            NodeType.TEST: {"required_relations": [RelationType.VERIFIES]},
            NodeType.CODEMODULE: {"file_extensions": [".py", ".js", ".ts", ".java"]}
        }
        
        # Define relation constraints
        gene.relation_constraints = {
            RelationType.IMPLEMENTS: {
                "from_types": [NodeType.CODEMODULE, NodeType.COMPONENT],
                "to_types": [NodeType.REQUIREMENT, NodeType.SPECIFICATION]
            },
            RelationType.VERIFIES: {
                "from_types": [NodeType.TEST],
                "to_types": [NodeType.REQUIREMENT, NodeType.CODEMODULE]
            }
        }
        
        return gene
    
    @classmethod
    def list_available_schemas(cls) -> List[str]:
        """List all available schema files"""
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        schemas_dir = current_dir / "schemas"
        
        if not schemas_dir.exists():
            return []
        
        schema_files = []
        for schema_file in schemas_dir.glob("*.json"):
            schema_files.append(schema_file.stem)  # Remove .json extension
        
        return sorted(schema_files)
    
    def validate_node(self, node: 'GraphNode') -> List[str]:
        """Validate node against gene constraints"""
        errors = []
        
        # Check required fields
        for field_path in self.required_node_fields.get("all", []):
            if not self._has_field(node, field_path):
                errors.append(f"Missing required field: {field_path}")
        
        type_specific = self.required_node_fields.get(node.type.value, [])
        for field_path in type_specific:
            if not self._has_field(node, field_path):
                errors.append(f"Missing type-specific field: {field_path}")
        
        # Check type-specific constraints
        constraints = self.node_type_constraints.get(node.type, {})
        if "max_title_length" in constraints:
            if len(node.title) > constraints["max_title_length"]:
                errors.append(f"Title too long: {len(node.title)} > {constraints['max_title_length']}")
        
        return errors
    
    def validate_edge(self, edge: 'GraphEdge', from_node: 'GraphNode', to_node: 'GraphNode') -> List[str]:
        """Validate edge against gene constraints"""
        errors = []
        
        constraints = self.relation_constraints.get(edge.relation, {})
        
        # Check from_node type constraints
        if "from_types" in constraints:
            if from_node.type not in constraints["from_types"]:
                errors.append(f"Invalid from_node type: {from_node.type} not in {constraints['from_types']}")
        
        # Check to_node type constraints  
        if "to_types" in constraints:
            if to_node.type not in constraints["to_types"]:
                errors.append(f"Invalid to_node type: {to_node.type} not in {constraints['to_types']}")
        
        return errors
    
    def _has_field(self, obj, field_path: str) -> bool:
        """Check if object has field using dot notation"""
        try:
            parts = field_path.split('.')
            current = obj
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            return current is not None
        except:
            return False

@dataclass
class HotState:
    """Hot (RAM) memory for immediate access"""
    recent_pulses: deque = field(default_factory=lambda: deque(maxlen=100))  # Ring buffer
    changed_nodes: Set[str] = field(default_factory=set)
    hot_cards: Dict[str, LLMCard] = field(default_factory=dict)
    last_pulse_time: float = field(default_factory=time.time)
    current_pulse: Optional[GraphPulse] = None
    
    def add_pulse(self, pulse_data: Dict[str, Any]):
        """Add a graph pulse to hot memory"""
        pulse = {
            "timestamp": time.time(),
            "data": pulse_data
        }
        self.recent_pulses.append(pulse)
        self.last_pulse_time = pulse["timestamp"]

# ============================================================================
# PHENOTYPE: Mutable Runtime Substrate
# ============================================================================

class OntologyPhenotype:
    """Mutable runtime substrate holding actual project data and evolving state"""
    
    def __init__(self, gene: OntologyGene = None):
        # Runtime data structures
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.node_edges: Dict[str, Set[str]] = {}  # node_id -> edge_ids
        self.llm_cards: Dict[str, LLMCard] = {}
        
        # Gene reference (immutable template)
        self.gene = gene or OntologyGene.default_software_gene()
        
        # Hot/Warm memory system
        self.hot_state = HotState()
        self.warm_snapshot_path: Optional[Path] = None
        
        # Snapshot persistence trigger (set by watermark updates)
        self._should_save_snapshot = False
        self.last_snapshot_data = {}  # Store watermarks and other metadata
        
        # NetworkX backing graph for fast queries (optional optimization)
        self._nx_graph = None
        self._nx_enabled = self._init_networkx()
    
    def _init_networkx(self) -> bool:
        """Initialize NetworkX backing graph if available"""
        try:
            import networkx as nx
            self._nx_graph = nx.DiGraph()
            return True
        except ImportError:
            print("ℹ️  NetworkX not available - using fallback graph queries (install with: pip install networkx)")
            return False
    
    def _sync_to_nx(self, node_id: str = None, edge_id: str = None):
        """Sync changes to NetworkX backing graph"""
        if not self._nx_enabled:
            return
            
        try:
            import networkx as nx
            
            # Sync specific node
            if node_id and node_id in self.nodes:
                node = self.nodes[node_id]
                self._nx_graph.add_node(node_id, 
                    node_obj=node,
                    type=node.type.value,
                    status=node.state.status.value,
                    last_changed=node.state.last_changed
                )
            
            # Sync specific edge
            if edge_id and edge_id in self.edges:
                edge = self.edges[edge_id]
                self._nx_graph.add_edge(
                    edge.from_node, 
                    edge.to_node,
                    edge_id=edge_id,
                    edge_obj=edge,
                    relation=edge.relation.value,
                    created_at=edge.created_at
                )
                
        except Exception as e:
            print(f"⚠️  NetworkX sync failed: {e}")
        
    def add_node(self, node: GraphNode) -> bool:
        """Add node to graph with gene validation"""
        # Validate against gene constraints
        validation_errors = self.gene.validate_node(node)
        if validation_errors:
            print(f"⚠️  Node validation failed: {validation_errors}")
            return False
        
        self.nodes[node.id] = node
        self.node_edges[node.id] = set()
        self._generate_llm_card(node.id)
        self.mark_changed(node.id)  # Track in hot state
        
        # Sync to NetworkX backing graph
        self._sync_to_nx(node_id=node.id)
        
        return True
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """Add edge to graph with gene validation"""
        if edge.from_node not in self.nodes or edge.to_node not in self.nodes:
            return False
        
        # Validate against gene constraints
        from_node = self.nodes[edge.from_node]
        to_node = self.nodes[edge.to_node]
        validation_errors = self.gene.validate_edge(edge, from_node, to_node)
        if validation_errors:
            print(f"⚠️  Edge validation failed: {validation_errors}")
            return False
            
        self.edges[edge.id] = edge
        self.node_edges[edge.from_node].add(edge.id)
        self.node_edges[edge.to_node].add(edge.id)
        
        # Regenerate LLM cards for impacted nodes
        self._generate_llm_card(edge.from_node)
        self._generate_llm_card(edge.to_node)
        
        # Track changes in hot state
        self.mark_changed(edge.from_node)
        self.mark_changed(edge.to_node)
        
        # Sync to NetworkX backing graph
        self._sync_to_nx(edge_id=edge.id)
        
        return True
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str, relation: Optional[RelationType] = None) -> List[GraphNode]:
        """Get neighboring nodes, optionally filtered by relation type"""
        # Use NetworkX for faster queries when available
        if self._nx_enabled and node_id in self._nx_graph:
            try:
                neighbors = []
                
                # Get both predecessors and successors from NetworkX
                for neighbor_id in list(self._nx_graph.predecessors(node_id)) + list(self._nx_graph.successors(node_id)):
                    if neighbor_id in self.nodes:
                        # Check relation filter if specified
                        if relation:
                            # Get edge between nodes to check relation
                            edge_data = self._nx_graph.get_edge_data(neighbor_id, node_id) or self._nx_graph.get_edge_data(node_id, neighbor_id)
                            if edge_data and edge_data.get('relation') == relation.value:
                                neighbors.append(self.nodes[neighbor_id])
                        else:
                            neighbors.append(self.nodes[neighbor_id])
                
                return neighbors
                
            except Exception as e:
                print(f"⚠️  NetworkX neighbor query failed, falling back to dict lookup: {e}")
        
        # Fallback to dictionary-based lookup
        neighbors = []
        if node_id not in self.node_edges:
            return neighbors
            
        for edge_id in self.node_edges[node_id]:
            edge = self.edges[edge_id]
            if relation and edge.relation != relation:
                continue
                
            neighbor_id = edge.to_node if edge.from_node == node_id else edge.from_node
            if neighbor_id in self.nodes:
                neighbors.append(self.nodes[neighbor_id])
                
        return neighbors
    
    def _generate_llm_card(self, node_id: str) -> LLMCard:
        """Generate compressed LLM card for a node"""
        node = self.nodes.get(node_id)
        if not node:
            return None
            
        # Get impacts (neighboring nodes)
        impacts = [n.id for n in self.get_neighbors(node_id)][:6]  # Limit to 6
        
        # Extract owners from metadata or content
        owners = node.content.get("owners", [])
        if not owners and "owner" in node.content:
            owners = [node.content["owner"]]
            
        # Generate digest
        digest = self._generate_digest(node)
        
        card = LLMCard(
            id=node.id,
            type=node.type.value,
            title=node.title,
            status=node.state.status.value,
            version=node.state.version,
            last_change=node.state.last_changed,
            owners=owners,
            delta=[node.state.change_summary] if node.state.change_summary else [],
            impacts=impacts,
            risks=node.content.get("risks", []),
            open_questions=node.content.get("open_questions", []),
            key_links=node.content.get("links", {}),
            rollup=self._get_rollup_chain(node_id),
            digest=digest
        )
        
        self.llm_cards[node_id] = card
        return card
    
    def _generate_digest(self, node: GraphNode) -> str:
        """Generate 1-3 sentence digest of node"""
        type_name = node.type.value
        title = node.title
        status = node.state.status.value
        version = node.state.version
        
        if node.state.change_summary:
            return f"{type_name} '{title}' v{version} is {status}. {node.state.change_summary}"
        else:
            return f"{type_name} '{title}' v{version} is {status}."
    
    def _get_rollup_chain(self, node_id: str) -> List[str]:
        """Get the rollup hierarchy chain (Story → Epic → Project → Product)"""
        # Use NetworkX for faster rollup traversal when available
        if self._nx_enabled and node_id in self._nx_graph:
            try:
                rollup = []
                current_id = node_id
                
                for _ in range(5):  # Prevent infinite loops
                    # Find outgoing ROLLS_UP_TO edges
                    parents = []
                    for successor_id in self._nx_graph.successors(current_id):
                        edge_data = self._nx_graph.get_edge_data(current_id, successor_id)
                        if edge_data and edge_data.get('relation') == RelationType.ROLLS_UP_TO.value:
                            parents.append(successor_id)
                    
                    if not parents:
                        break
                        
                    parent_id = parents[0]  # Take first parent
                    rollup.append(parent_id)
                    current_id = parent_id
                    
                return rollup
                
            except Exception as e:
                print(f"⚠️  NetworkX rollup query failed, falling back: {e}")
        
        # Fallback to dictionary-based lookup
        rollup = []
        current_id = node_id
        
        # Follow rolls_up_to relationships
        for _ in range(5):  # Prevent infinite loops
            parents = []
            for edge_id in self.node_edges.get(current_id, []):
                edge = self.edges[edge_id]
                if (edge.relation == RelationType.ROLLS_UP_TO and 
                    edge.from_node == current_id):
                    parents.append(edge.to_node)
                    
            if not parents:
                break
                
            parent_id = parents[0]  # Take first parent
            rollup.append(parent_id)
            current_id = parent_id
            
        return rollup
    
    def get_llm_card(self, node_id: str) -> Optional[LLMCard]:
        """Get LLM card for node, generating if needed"""
        if node_id not in self.llm_cards:
            self._generate_llm_card(node_id)
        return self.llm_cards.get(node_id)
    
    def search_nodes(self, 
                    node_type: Optional[NodeType] = None,
                    status: Optional[Status] = None,
                    criticality: Optional[Criticality] = None) -> List[GraphNode]:
        """Search nodes by criteria"""
        results = []
        for node in self.nodes.values():
            if node_type and node.type != node_type:
                continue
            if status and node.state.status != status:
                continue
            if criticality and node.state.criticality != criticality:
                continue
            results.append(node)
        return results
    
    def detect_graph_tensions(self) -> List[Dict[str, Any]]:
        """Detect tensions in the ontological graph that require LLM reasoning"""
        tensions = []
        
        # Detect version misalignments
        for node in self.nodes.values():
            if node.state.change_summary:  # Something changed
                impacted_nodes = self.get_neighbors(node.id)
                for neighbor in impacted_nodes:
                    if neighbor.state.last_changed < node.state.last_changed - 3600:  # 1 hour lag
                        tensions.append({
                            "type": "version_misalignment",
                            "source_node": node.id,
                            "affected_node": neighbor.id,
                            "description": f"Change in {node.title} may impact {neighbor.title}",
                            "severity": "medium",
                            "requires_llm_analysis": True
                        })
        
        # Detect orphaned relationships
        for edge in self.edges.values():
            if edge.from_node not in self.nodes or edge.to_node not in self.nodes:
                tensions.append({
                    "type": "broken_relationship",
                    "edge_id": edge.id,
                    "description": f"Relationship {edge.relation.value} references non-existent nodes",
                    "severity": "high",
                    "requires_llm_analysis": False  # Structural issue
                })
        
        # Detect state inconsistencies requiring reasoning
        for node in self.nodes.values():
            neighbors = self.get_neighbors(node.id)
            if len(neighbors) > 0:
                tensions.append({
                    "type": "state_analysis_needed",
                    "node_id": node.id,
                    "neighbor_count": len(neighbors),
                    "description": f"Node {node.title} state may need validation against {len(neighbors)} relationships",
                    "severity": "low",
                    "requires_llm_analysis": True,
                    "context_nodes": [n.id for n in neighbors[:5]]  # Limit context
                })
        
        return tensions
    
    def detect_technical_debt(self) -> Dict[str, List[str]]:
        """Detect various forms of technical debt and cleanup opportunities"""
        debt_items = {
            "orphaned_files": [],
            "unused_modules": [],
            "obsolete_archives": [],
            "stale_branches": [],
            "redundant_artifacts": [],
            "outdated_dependencies": [],
            "missing_documentation": [],
            "test_gaps": []
        }
        
        # Detect orphaned files (no incoming relationships)
        for node in self.nodes.values():
            if node.type in [NodeType.CODEMODULE, NodeType.ARTIFACT]:
                incoming_edges = [e for e in self.edges.values() if e.to_node == node.id]
                if not incoming_edges:
                    # Check if it's in archive path - likely obsolete
                    if "archive" in node.title.lower() or "/archive/" in node.content.get("path", ""):
                        debt_items["obsolete_archives"].append(node.id)
                    else:
                        debt_items["orphaned_files"].append(node.id)
        
        # Detect unused modules (no implementations or tests)
        for node in self.search_nodes(NodeType.CODEMODULE):
            implementations = self.get_neighbors(node.id, RelationType.IMPLEMENTS)
            tests = self.get_neighbors(node.id, RelationType.TESTS)
            if not implementations and not tests:
                debt_items["unused_modules"].append(node.id)
        
        # Detect stale branches (old commits, no recent activity)
        current_time = time.time()
        stale_threshold = 30 * 24 * 3600  # 30 days
        
        for node in self.search_nodes(NodeType.BRANCH):
            if current_time - node.state.last_changed > stale_threshold:
                # Check if merged or has active work
                merges = self.get_neighbors(node.id, RelationType.MERGES)
                commits = self.get_neighbors(node.id, RelationType.COMMITS_TO)
                if not merges and not commits:
                    debt_items["stale_branches"].append(node.id)
        
        # Detect redundant artifacts (multiple versions, only latest needed)
        artifact_groups = {}
        for node in self.search_nodes(NodeType.ARTIFACT):
            base_name = node.title.split('.')[0]  # Remove version/extension
            if base_name not in artifact_groups:
                artifact_groups[base_name] = []
            artifact_groups[base_name].append(node)
        
        for group_name, artifacts in artifact_groups.items():
            if len(artifacts) > 1:
                # Sort by version, mark older ones as redundant
                artifacts.sort(key=lambda n: n.state.version, reverse=True)
                for old_artifact in artifacts[1:]:  # Keep newest
                    debt_items["redundant_artifacts"].append(old_artifact.id)
        
        # Detect missing documentation for public APIs
        for node in self.search_nodes(NodeType.API):
            docs = self.get_neighbors(node.id, RelationType.DOCUMENTS)
            if not docs:
                debt_items["missing_documentation"].append(node.id)
        
        # Detect test coverage gaps
        for node in self.search_nodes(NodeType.CODEMODULE):
            tests = self.get_neighbors(node.id, RelationType.TESTS)
            if not tests and node.state.status in [Status.IN_DEV, Status.RELEASED]:
                debt_items["test_gaps"].append(node.id)
        
        return debt_items
    
    def suggest_cleanup_actions(self) -> List[Dict[str, Any]]:
        """Generate intelligent cleanup suggestions based on detected patterns"""
        debt_items = self.detect_technical_debt()
        actions = []
        
        # Archive cleanup suggestions
        if debt_items["obsolete_archives"]:
            actions.append({
                "type": "archive_cleanup",
                "priority": "low",
                "description": f"Archive directory contains {len(debt_items['obsolete_archives'])} obsolete files",
                "items": debt_items["obsolete_archives"],
                "suggested_action": "Move to deep archive or remove if confirmed obsolete",
                "risk_level": "low"
            })
        
        # Unused module cleanup
        if debt_items["unused_modules"]:
            actions.append({
                "type": "unused_code_removal",
                "priority": "medium", 
                "description": f"Found {len(debt_items['unused_modules'])} unused code modules",
                "items": debt_items["unused_modules"],
                "suggested_action": "Review for removal or refactor into active code",
                "risk_level": "medium"
            })
        
        # Stale branch cleanup
        if debt_items["stale_branches"]:
            actions.append({
                "type": "branch_cleanup",
                "priority": "low",
                "description": f"Found {len(debt_items['stale_branches'])} stale branches", 
                "items": debt_items["stale_branches"],
                "suggested_action": "Review and delete if no longer needed",
                "risk_level": "low"
            })
        
        # Documentation gaps
        if debt_items["missing_documentation"]:
            actions.append({
                "type": "documentation_debt",
                "priority": "high",
                "description": f"Found {len(debt_items['missing_documentation'])} APIs without documentation",
                "items": debt_items["missing_documentation"],
                "suggested_action": "Create API documentation for public interfaces",
                "risk_level": "low"
            })
        
        # Test coverage gaps
        if debt_items["test_gaps"]:
            actions.append({
                "type": "test_coverage_debt",
                "priority": "high",
                "description": f"Found {len(debt_items['test_gaps'])} modules without tests",
                "items": debt_items["test_gaps"],
                "suggested_action": "Add unit tests for untested modules",
                "risk_level": "medium"
            })
        
        return sorted(actions, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)
    
    # Hot/Warm Memory Management
    def save_snapshot(self, snapshot_path: Path, phi_data: Dict[str, Any] = None) -> bool:
        """Save warm snapshot for fast restart - this IS the warm layer of Ψ"""
        try:
            # Compute current Φ projection if not provided
            if not phi_data:
                coverage = self.coverage_ratio()
                counters = {}
                for node_type in [NodeType.REQUIREMENT, NodeType.TEST, NodeType.CODEMODULE, NodeType.ISSUE]:
                    count = len([n for n in self.nodes.values() if n.type == node_type])
                    counters[node_type.value.lower()] = count
                
                phi_data = {
                    "phi0": 0.0,  # Will be computed by RE agent
                    "coverage": coverage,
                    "counters": counters,
                    "computed_at": time.time()
                }
            
            snapshot_data = {
                "timestamp": time.time(),
                # Ψ core: externalized memory substrate
                "nodes": {nid: self._serialize_node(node) for nid, node in self.nodes.items()},
                "edges": {eid: self._serialize_edge(edge) for eid, edge in self.edges.items()},
                "node_edges": {k: list(v) for k, v in self.node_edges.items()},
                "llm_cards": {k: self._serialize_card(v) for k, v in self.llm_cards.items()},
                "hot_state": {
                    "changed_nodes": list(self.hot_state.changed_nodes),
                    "last_pulse_time": self.hot_state.last_pulse_time
                },
                # Sensor watermarks for bootstrap
                "watermarks": getattr(self, 'last_snapshot_data', {}).get('watermarks', {}),
                # Φ projection: emergent coherence state
                "phi": phi_data,
                # Identity & versioning for self/other recognition
                "identity": {
                    "schema_version": "1.0.0",
                    "species_id": self.gene.species_id if hasattr(self, 'gene') else "software_engineering_v1",
                    "instance_id": getattr(self, 'instance_id', f"instance_{uuid.uuid4().hex[:12]}"),
                    "gene_version": self.gene.version if hasattr(self, 'gene') else "1.0.0"
                }
            }
            
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            
            self.warm_snapshot_path = snapshot_path
            print(f"💾 Saved Ψ snapshot: {len(self.nodes)} nodes, {len(self.edges)} edges, Φ₀={phi_data.get('phi0', 'N/A')}")
            
            # Optional: Create Snapshot node in Ψ for audit trail
            if hasattr(self, '_track_snapshots') and self._track_snapshots:
                self._create_snapshot_audit_node(snapshot_data)
            
            # Reset the persistence trigger since we just saved
            self._should_save_snapshot = False
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save snapshot: {e}")
            return False
    
    def check_and_save_pending_snapshot(self) -> bool:
        """Check if snapshot save is pending due to watermark updates and execute if needed"""
        if not self._should_save_snapshot:
            return False
            
        # Use existing warm snapshot path or default
        snapshot_path = self.warm_snapshot_path or Path("psi_snapshot.json")
        
        try:
            success = self.save_snapshot(snapshot_path)
            if success:
                print("💾 Auto-saved snapshot due to watermark updates")
            return success
        except Exception as e:
            print(f"❌ Auto-save failed: {e}")
            return False
    
    def load_snapshot(self, snapshot_path: Path) -> bool:
        """Load warm snapshot, validate identity, and restore Ψ consciousness"""
        try:
            if not snapshot_path.exists():
                return False
                
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            # Validate identity and schema compatibility
            identity = snapshot_data.get("identity", {})
            schema_version = identity.get("schema_version", "0.0.0")
            species_id = identity.get("species_id", "unknown")
            
            if hasattr(self, 'gene'):
                # Check species compatibility
                if species_id != self.gene.species_id:
                    print(f"⚠️  Species mismatch: snapshot={species_id}, gene={self.gene.species_id}")
                    # Could choose to reject or attempt migration
                
                # Check schema version compatibility  
                if schema_version != "1.0.0":
                    print(f"⚠️  Schema version mismatch: {schema_version} (may need migration)")
            
            # Store identity for self-recognition
            self.instance_id = identity.get("instance_id", f"restored_{uuid.uuid4().hex[:12]}")
            
            # Restore basic Ψ structures
            self.nodes = {}
            self.edges = {}
            self.node_edges = {}
            self.llm_cards = {}
            
            # Rebuild nodes
            for nid, node_data in snapshot_data["nodes"].items():
                node = self._deserialize_node(node_data)
                self.nodes[nid] = node
            
            # Rebuild edges  
            for eid, edge_data in snapshot_data["edges"].items():
                edge = self._deserialize_edge(edge_data)
                self.edges[eid] = edge
            
            # Rebuild node-edge mapping
            for nid, edge_list in snapshot_data["node_edges"].items():
                self.node_edges[nid] = set(edge_list)
            
            # Rebuild LLM cards
            for cid, card_data in snapshot_data["llm_cards"].items():
                card = self._deserialize_card(card_data)
                self.llm_cards[cid] = card
                
            # Restore hot state
            hot_data = snapshot_data.get("hot_state", {})
            self.hot_state.changed_nodes = set(hot_data.get("changed_nodes", []))
            self.hot_state.last_pulse_time = hot_data.get("last_pulse_time", time.time())
            
            # Restore Φ projection if available
            phi_data = snapshot_data.get("phi", {})
            self._last_phi = phi_data
            
            # Restore watermarks for sensor bootstrap
            self.last_snapshot_data = snapshot_data  # SensorHub reads this
            
            # Rebuild hot cache with recent cards
            self._rebuild_hot_cache()
            
            # DELTA-FIRST CONSCIOUSNESS: Create GraphPulse immediately from restored hot state
            restored_pulse = self._create_restoration_pulse(phi_data)
            self.hot_state.current_pulse = restored_pulse
            
            self.warm_snapshot_path = snapshot_path
            phi0_display = phi_data.get("phi0", "N/A")
            print(f"🔥 Restored Ψ consciousness: {len(self.nodes)} nodes, {len(self.edges)} edges, Φ₀={phi0_display}")
            print(f"   Identity: {species_id} instance {self.instance_id}")
            print(f"   Delta-first pulse: {len(restored_pulse.changed_nodes)} changed nodes")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load snapshot: {e}")
            return False
    
    def _rebuild_hot_cache(self):
        """Rebuild hot cache from warm data"""
        # Cache LLM cards for recently changed nodes
        for node_id in self.hot_state.changed_nodes:
            if node_id in self.llm_cards:
                self.hot_state.hot_cards[node_id] = self.llm_cards[node_id]
        
        # Add pulse for reload event
        self.hot_state.add_pulse({
            "event": "hot_cache_rebuilt",
            "node_count": len(self.nodes),
            "hot_cards": len(self.hot_state.hot_cards)
        })
    
    def top_k_changed(self, node_types: Optional[List[NodeType]] = None, k: int = 5) -> List[str]:
        """Get top-k most recently changed nodes"""
        # Filter by type if specified
        candidates = []
        for node_id in self.hot_state.changed_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if not node_types or node.type in node_types:
                    candidates.append((node_id, node.state.last_changed))
        
        # Sort by modification time and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in candidates[:k]]
    
    def coverage_ratio(self) -> Dict[str, float]:
        """Calculate coverage ratios for different relationships"""
        ratios = {}
        
        # Spec -> Requirement coverage
        spec_nodes = [n for n in self.nodes.values() if n.type == NodeType.SPECIFICATION]
        req_nodes = [n for n in self.nodes.values() if n.type == NodeType.REQUIREMENT]
        
        if spec_nodes and req_nodes:
            covered_reqs = set()
            for spec in spec_nodes:
                for neighbor in self.get_neighbors(spec.id, RelationType.IMPLEMENTS):
                    if neighbor.type == NodeType.REQUIREMENT:
                        covered_reqs.add(neighbor.id)
            
            ratios["spec_requirement_coverage"] = len(covered_reqs) / len(req_nodes)
        
        # Test -> Requirement coverage
        test_nodes = [n for n in self.nodes.values() if n.type == NodeType.TEST]
        
        if test_nodes and req_nodes:
            tested_reqs = set()
            for test in test_nodes:
                for neighbor in self.get_neighbors(test.id, RelationType.VERIFIES):
                    if neighbor.type == NodeType.REQUIREMENT:
                        tested_reqs.add(neighbor.id)
            
            ratios["test_requirement_coverage"] = len(tested_reqs) / len(req_nodes)
        
        return ratios
    
    def phi_signals(self) -> Dict[str, Union[int, float]]:
        """Generate Φ signals for LLM consumption (matches system prompt expectations)"""
        # Get coverage ratios
        coverage = self.coverage_ratio()
        
        # Count requirements without tests (uncovered)
        req_nodes = [n for n in self.nodes.values() if n.type == NodeType.REQUIREMENT]
        test_nodes = [n for n in self.nodes.values() if n.type == NodeType.TEST]
        
        tested_req_ids = set()
        for test in test_nodes:
            for neighbor in self.get_neighbors(test.id, RelationType.VERIFIES):
                if neighbor.type == NodeType.REQUIREMENT:
                    tested_req_ids.add(neighbor.id)
        
        uncovered_requirements = len(req_nodes) - len(tested_req_ids)
        
        # Count open P0 issues
        open_p0_issues = len([
            n for n in self.nodes.values() 
            if (n.type in [NodeType.ISSUE, NodeType.BUG] and 
                n.state.criticality == Criticality.P0 and 
                n.state.status not in [Status.RELEASED, Status.DEPRECATED])
        ])
        
        # Primary coverage ratio (test coverage is most important)
        primary_coverage = coverage.get("test_requirement_coverage", 0.0)
        
        # Auto-update project status based on phi signals
        self._auto_update_project_status(primary_coverage, open_p0_issues, uncovered_requirements)
        
        # Calculate CI stability window
        ci_green_days = self._calculate_ci_green_days()
        
        return {
            "coverage_ratio": primary_coverage,
            "changed_nodes": len(self.hot_state.changed_nodes),
            "uncovered_requirements": max(0, uncovered_requirements),  # Ensure non-negative
            "open_p0_issues": open_p0_issues,
            "context_budget_used": 0,  # Will be set by context budget system
            "entropy_hint": self._calculate_entropy_hint(),
            "ci_green_days": ci_green_days
        }
    
    def _calculate_ci_green_days(self) -> int:
        """Calculate consecutive days of successful CI builds"""
        from datetime import datetime, timedelta
        
        try:
            # Get all build nodes sorted by creation time (most recent first)
            build_nodes = [n for n in self.nodes.values() if n.type == NodeType.BUILD]
            if not build_nodes:
                return 0
            
            # Sort by created_at or updated_at timestamp
            build_nodes.sort(key=lambda n: self._get_build_timestamp(n), reverse=True)
            
            current_time = datetime.now()
            green_days = 0
            consecutive_success = True
            
            # Track days we've seen successful builds
            successful_days = set()
            
            for build in build_nodes:
                build_time = self._get_build_datetime(build)
                if not build_time:
                    continue
                
                # Check if build is successful
                conclusion = build.content.get("conclusion", "").lower()
                if conclusion not in ["success", "success"]:
                    # If we encounter a failure and we're still counting consecutive successes, stop
                    if consecutive_success:
                        break
                    continue
                
                # Calculate days ago
                days_ago = (current_time - build_time).days
                
                # Only count recent builds (within last 30 days to avoid very old data)
                if days_ago > 30:
                    break
                
                # Add successful day
                successful_days.add(days_ago)
            
            # Count consecutive days from today backwards
            for day in range(31):  # Check last 30 days
                if day in successful_days:
                    green_days += 1
                else:
                    break
            
            return green_days
            
        except Exception as e:
            print(f"⚠️  Error calculating CI green days: {e}")
            return 0
    
    def _get_build_timestamp(self, build_node) -> float:
        """Extract timestamp from build node for sorting"""
        from datetime import datetime
        # Try different timestamp fields
        for field in ["updated_at", "created_at", "timestamp"]:
            if field in build_node.content:
                try:
                    dt_str = build_node.content[field]
                    if isinstance(dt_str, str):
                        # Handle ISO format with Z
                        if dt_str.endswith('Z'):
                            dt_str = dt_str.replace('Z', '+00:00')
                        dt = datetime.fromisoformat(dt_str)
                        return dt.timestamp()
                except (ValueError, TypeError):
                    continue
        
        # Fallback to node state timestamp
        return build_node.state.last_changed
    
    def _get_build_datetime(self, build_node) -> Optional['datetime']:
        """Extract datetime from build node"""
        from datetime import datetime
        timestamp = self._get_build_timestamp(build_node)
        try:
            return datetime.fromtimestamp(timestamp)
        except (ValueError, OSError):
            return None
    
    def _auto_update_project_status(self, coverage_ratio: float, open_p0_issues: int, uncovered_requirements: int):
        """Auto-update project status based on phi signals"""
        try:
            # Find project nodes
            project_nodes = [n for n in self.nodes.values() if n.type == NodeType.PROJECT]
            
            for project in project_nodes:
                current_status = project.state.status
                new_status = self._determine_project_status(coverage_ratio, open_p0_issues, uncovered_requirements, current_status)
                
                if new_status != current_status:
                    print(f"🔄 Auto-updating project status: {current_status.value} → {new_status.value}")
                    project.state.status = new_status
                    
                    # Mark project as changed to trigger hot state update
                    self.hot_state.changed_nodes.add(project.id)
                    
        except Exception as e:
            print(f"⚠️  Error auto-updating project status: {e}")
    
    def _determine_project_status(self, coverage_ratio: float, open_p0_issues: int, uncovered_requirements: int, current_status: Status) -> Status:
        """Determine appropriate project status based on metrics"""
        
        # Don't downgrade from released/deprecated states
        if current_status in [Status.RELEASED, Status.DEPRECATED]:
            return current_status
        
        # If there are P0 issues or uncovered requirements, stay in draft or move to blocked
        if open_p0_issues > 0:
            return Status.BLOCKED
            
        if uncovered_requirements > 0:
            return Status.DRAFT
        
        # Full coverage and no P0 issues - ready for active development
        if coverage_ratio >= 1.0:
            if current_status in [Status.DRAFT]:
                return Status.IN_DEV  # Active development with full testing
            elif current_status == Status.IN_DEV:
                return Status.APPROVED  # Ready for review/release
                
        # Partial coverage - still in development
        if coverage_ratio >= 0.8:
            return Status.IN_DEV
            
        # Low coverage - stay in draft
        return Status.DRAFT

    def _calculate_entropy_hint(self) -> float:
        """Calculate 0-1 entropy/risk hint based on graph state"""
        entropy_factors = []
        
        # Version misalignment factor
        total_nodes = len(self.nodes)
        if total_nodes > 0:
            changed_ratio = len(self.hot_state.changed_nodes) / total_nodes
            entropy_factors.append(min(changed_ratio * 2, 1.0))  # Cap at 1.0
        
        # Tension severity factor
        tensions = self.detect_graph_tensions()
        if tensions:
            high_tensions = len([t for t in tensions if t.get("severity") == "high"])
            tension_factor = min(high_tensions / 5.0, 1.0)  # Normalize by 5 high tensions = max entropy
            entropy_factors.append(tension_factor)
        
        # P0 issues factor
        p0_count = len([
            n for n in self.nodes.values() 
            if (n.type in [NodeType.ISSUE, NodeType.BUG] and 
                n.state.criticality == Criticality.P0 and 
                n.state.status not in [Status.RELEASED, Status.DEPRECATED])
        ])
        if p0_count > 0:
            entropy_factors.append(min(p0_count / 3.0, 1.0))  # 3+ P0s = high entropy
        
        # Coverage gap factor
        coverage = self.coverage_ratio()
        test_coverage = coverage.get("test_requirement_coverage", 1.0)  # Default to 1.0 if no tests/reqs
        coverage_gap = 1.0 - test_coverage
        entropy_factors.append(coverage_gap)
        
        # Return average entropy, capped at 1.0
        if entropy_factors:
            return min(sum(entropy_factors) / len(entropy_factors), 1.0)
        else:
            return 0.0
    
    def mark_changed(self, node_id: str):
        """Mark node as changed in hot state"""
        if node_id in self.nodes:
            self.hot_state.changed_nodes.add(node_id)
            # Update hot cache
            if node_id in self.llm_cards:
                self.hot_state.hot_cards[node_id] = self.llm_cards[node_id]
            
            # Add pulse
            self.hot_state.add_pulse({
                "event": "node_changed",
                "node_id": node_id,
                "node_type": self.nodes[node_id].type.value
            })
    
    def pulse(self, phi_snapshot: Dict[str, Any] = None) -> GraphPulse:
        """Generate compressed GraphPulse for LLM consumption"""
        # Collect recent changes (delta-first)
        changed_nodes = list(self.hot_state.changed_nodes)[:5]
        
        # Key edges (recent relationships)
        key_edges = []
        for node_id in changed_nodes:
            if node_id in self.node_edges:
                for edge_id in list(self.node_edges[node_id])[:2]:  # Max 2 per node
                    if edge_id in self.edges:
                        edge = self.edges[edge_id]
                        key_edges.append(f"{edge.relation.value}:{edge.from_node}->{edge.to_node}")
        
        # Counters (type breakdown)
        counters = {}
        for node_type in [NodeType.REQUIREMENT, NodeType.TEST, NodeType.CODEMODULE, NodeType.ISSUE]:
            count = len([n for n in self.nodes.values() if n.type == node_type])
            counters[node_type.value.lower()] = count
        
        # Last events from pulse buffer
        last_events = []
        for pulse in list(self.hot_state.recent_pulses)[-3:]:  # Last 3 events
            event_data = pulse.get("data", {})
            if "event" in event_data:
                last_events.append(f"{event_data['event']}:{event_data.get('node_id', 'unknown')}")
        
        # Create and cache pulse
        pulse = GraphPulse(
            timestamp=time.time(),
            changed_nodes=changed_nodes,
            key_edges=key_edges[:3],  # Max 3 edges
            counters=counters,
            last_events=last_events,
            phi_snapshot=phi_snapshot or {}
        )
        
        self.hot_state.current_pulse = pulse
        return pulse
    
    def _create_restoration_pulse(self, phi_data: Dict[str, Any]) -> GraphPulse:
        """Create GraphPulse immediately after snapshot restoration for delta-first consciousness"""
        # Use restored hot state changed nodes (delta-first)
        changed_nodes = list(self.hot_state.changed_nodes)[:5]
        
        # Generate counters from restored Φ data
        counters = phi_data.get("counters", {})
        if not counters:
            # Fallback: compute current counters
            for node_type in [NodeType.REQUIREMENT, NodeType.TEST, NodeType.CODEMODULE, NodeType.ISSUE]:
                count = len([n for n in self.nodes.values() if n.type == node_type])
                counters[node_type.value.lower()] = count
        
        # Key relationships from changed nodes
        key_edges = []
        for node_id in changed_nodes[:3]:  # Limit to 3 for context
            if node_id in self.node_edges:
                for edge_id in list(self.node_edges[node_id])[:1]:  # 1 edge per node
                    if edge_id in self.edges:
                        edge = self.edges[edge_id]
                        key_edges.append(f"{edge.relation.value}:{edge.from_node}->{edge.to_node}")
        
        # Restoration events
        last_events = [
            "consciousness_restored",
            f"phi_restored:phi0={phi_data.get('phi0', 'N/A')}",
            f"identity_validated:{self.instance_id}"
        ]
        
        return GraphPulse(
            timestamp=time.time(),
            changed_nodes=changed_nodes,
            key_edges=key_edges,
            counters=counters,
            last_events=last_events,
            phi_snapshot=phi_data
        )
    
    def _create_snapshot_audit_node(self, snapshot_data: Dict[str, Any]):
        """Create Snapshot node in Ψ for audit trail (optional)"""
        try:
            snapshot_node = create_node(
                NodeType.ARTIFACT,  # Using ARTIFACT as closest existing type
                f"Ψ Snapshot {time.strftime('%Y%m%d_%H%M%S')}",
                content={
                    "type": "consciousness_snapshot",
                    "node_count": len(snapshot_data.get("nodes", {})),
                    "edge_count": len(snapshot_data.get("edges", {})),
                    "phi0": snapshot_data.get("phi", {}).get("phi0", "N/A"),
                    "species_id": snapshot_data.get("identity", {}).get("species_id"),
                    "instance_id": snapshot_data.get("identity", {}).get("instance_id"),
                    "snapshot_path": str(self.warm_snapshot_path) if self.warm_snapshot_path else None
                }
            )
            
            # Add to graph (will be validated by gene)
            success = self.add_node(snapshot_node)
            if success:
                print(f"📋 Created audit node: {snapshot_node.id}")
            
        except Exception as e:
            print(f"⚠️  Could not create snapshot audit node: {e}")
    
    def get_phi_from_snapshot(self) -> Dict[str, Any]:
        """Get last Φ projection from loaded snapshot"""
        return getattr(self, '_last_phi', {})
    
    def translate_tensions_to_actions(self, tool_registry=None) -> List[Dict[str, Any]]:
        """Map tensions/coverage gaps to concrete tool actions (Φ → Ω → act)"""
        actions = []
        
        # Get current tensions and coverage
        tensions = self.detect_graph_tensions()
        coverage = self.coverage_ratio()
        cleanup_actions = self.suggest_cleanup_actions()
        
        # Coverage gap → create test issues
        test_coverage = coverage.get("test_requirement_coverage", 0.0)
        if test_coverage < 0.8:  # 80% threshold
            gap_count = len([n for n in self.nodes.values() if n.type == NodeType.REQUIREMENT]) - \
                       len([n for n in self.nodes.values() if n.type == NodeType.TEST])
            
            if gap_count > 0:
                actions.append({
                    "type": "create_issue",
                    "priority": "high",
                    "title": f"Improve test coverage: {gap_count} requirements without tests",
                    "body": f"Current test coverage: {test_coverage:.1%}\nTarget: 80%+\n\nRequirements needing test coverage: {gap_count}",
                    "labels": ["testing", "coverage", "technical-debt"],
                    "tool_params": {
                        "capability": "create_issue",
                        "title": f"Improve test coverage ({test_coverage:.1%} → 80%+)",
                        "labels": "testing,coverage"
                    }
                })
        
        # High-severity tensions → create issues
        high_tensions = [t for t in tensions if t.get("severity") == "high"]
        for tension in high_tensions:
            if tension.get("type") == "broken_relationship":
                actions.append({
                    "type": "create_issue",
                    "priority": "high", 
                    "title": f"Fix broken relationship: {tension.get('description', 'Unknown')}",
                    "body": f"Graph tension detected:\n\nType: {tension['type']}\nDescription: {tension['description']}\nEdge: {tension.get('edge_id', 'N/A')}",
                    "labels": ["bug", "graph-integrity"],
                    "tool_params": {
                        "capability": "create_issue",
                        "title": f"Fix: {tension.get('description', 'Graph integrity issue')}",
                        "labels": "bug,graph-integrity"
                    }
                })
        
        # Cleanup actions → create issues or PRs
        for cleanup in cleanup_actions[:3]:  # Top 3 priority actions
            if cleanup["type"] == "unused_code_removal" and cleanup["priority"] == "medium":
                actions.append({
                    "type": "create_issue",
                    "priority": "medium",
                    "title": cleanup["description"],
                    "body": f"Cleanup opportunity detected:\n\n{cleanup['description']}\n\nAction: {cleanup['suggested_action']}\nRisk: {cleanup['risk_level']}\n\nItems: {len(cleanup['items'])}",
                    "labels": ["cleanup", "tech-debt"],
                    "tool_params": {
                        "capability": "create_issue",
                        "title": cleanup["description"],
                        "labels": "cleanup,technical-debt"
                    }
                })
            
            elif cleanup["type"] == "documentation_debt" and cleanup["priority"] == "high":
                actions.append({
                    "type": "create_issue", 
                    "priority": "high",
                    "title": cleanup["description"],
                    "body": f"Documentation gaps detected:\n\n{cleanup['suggested_action']}\n\nAPIs without docs: {len(cleanup['items'])}",
                    "labels": ["documentation", "api"],
                    "tool_params": {
                        "capability": "create_issue",
                        "title": cleanup["description"],
                        "labels": "documentation,api"
                    }
                })
        
        # Successful implementations → close related issues  
        implemented_reqs = []
        for node in self.nodes.values():
            if node.type == NodeType.REQUIREMENT and node.state.status == Status.RELEASED:
                implemented_reqs.append(node.id)
        
        if implemented_reqs and tool_registry:
            # This would query existing issues and close ones related to implemented requirements
            actions.append({
                "type": "query_and_close_issues",
                "priority": "low",
                "description": f"Close issues for {len(implemented_reqs)} implemented requirements",
                "tool_params": {
                    "capability": "list_issues", 
                    "state": "open",
                    "labels": "requirement"
                }
            })
        
        return actions
    
    def _serialize_node(self, node: GraphNode) -> Dict[str, Any]:
        """Serialize node for JSON storage"""
        data = asdict(node)
        # Convert enums to strings
        data['type'] = node.type.value
        data['state']['status'] = node.state.status.value  
        data['state']['criticality'] = node.state.criticality.value
        return data
    
    def _serialize_edge(self, edge: GraphEdge) -> Dict[str, Any]:
        """Serialize edge for JSON storage"""
        data = asdict(edge)
        data['relation'] = edge.relation.value
        return data
        
    def _serialize_card(self, card: LLMCard) -> Dict[str, Any]:
        """Serialize LLM card for JSON storage"""
        return asdict(card)
        
    def _deserialize_node(self, node_data: Dict[str, Any]) -> GraphNode:
        """Deserialize node from JSON data"""
        # Convert string back to enum
        node_data['type'] = NodeType(node_data['type'])
        node_data['state']['status'] = Status(node_data['state']['status'])
        node_data['state']['criticality'] = Criticality(node_data['state']['criticality'])
        
        # Reconstruct NodeState
        state_data = node_data.pop('state')
        state = NodeState(**state_data)
        
        # Reconstruct GraphNode
        return GraphNode(state=state, **node_data)
        
    def _deserialize_edge(self, edge_data: Dict[str, Any]) -> GraphEdge:
        """Deserialize edge from JSON data"""
        edge_data['relation'] = RelationType(edge_data['relation'])
        return GraphEdge(**edge_data)
        
    def _deserialize_card(self, card_data: Dict[str, Any]) -> LLMCard:
        """Deserialize LLM card from JSON data"""
        return LLMCard(**card_data)

# ============================================================================
# Compatibility & Integration
# ============================================================================

# Backward compatibility alias - OntologyGraph now points to phenotype
OntologyGraph = OntologyPhenotype

def create_ontology_with_gene(gene_template: str = "project_manager") -> OntologyPhenotype:
    """Factory function to create ontology with specific gene template"""
    # Try to load from schema file, fallback to default if not found
    gene = OntologyGene.from_schema_file(gene_template)
    return OntologyPhenotype(gene)

def list_available_gene_templates() -> List[str]:
    """List all available gene templates/schemas"""
    return OntologyGene.list_available_schemas()

# Factory functions
def create_node(node_type: NodeType, title: str, **kwargs) -> GraphNode:
    """Create a new graph node"""
    node_id = f"{node_type.value.lower()}:{uuid.uuid4().hex[:8]}"
    state = NodeState(**kwargs.get('state', {}))
    
    return GraphNode(
        id=node_id,
        type=node_type,
        title=title,
        state=state,
        meta=kwargs.get('meta', {}),
        content=kwargs.get('content', {})
    )

def create_edge(relation: RelationType, from_node: str, to_node: str, **kwargs) -> GraphEdge:
    """Create a new graph edge"""
    edge_id = f"{relation.value}:{uuid.uuid4().hex[:8]}"
    
    return GraphEdge(
        id=edge_id,
        relation=relation,
        from_node=from_node,
        to_node=to_node,
        meta=kwargs.get('meta', {})
    )