"""
RE_ware: Self-Modifying Recursive Emergence Library
A reusable agent framework that can analyze and refactor its own source code.
Key feature: agent.evolve() modifies the source code using engineering and entropic principles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
import time
import json
import copy
import ast
import inspect
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

class REState:
    """Ψ (Psi) - Memory/State component of RE triad"""
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.ontology: Dict[str, Any] = {
            "nodes": ["Problem", "Spec", "Plan", "Task", "Pattern", "Decision"],
            "rels": ["REFINES", "IMPLEMENTS", "VERIFIES", "DERIVES"]
        }
        self.traces: List[Dict[str, Any]] = []
    
    def store(self, key: str, value: Any):
        self.memory[key] = value
        self.traces.append({"action": "store", "key": key, "ts": time.time()})
    
    def recall(self, key: str) -> Any:
        return self.memory.get(key)

class REGuardrails:
    """Ω (Omega) - Guardrails/Constraints component of RE triad"""
    def __init__(self):
        self.predicates: Dict[str, Callable] = {}
        self.constraints: Set[str] = set()
        
    def add_predicate(self, name: str, func: Callable[[Any], bool]):
        self.predicates[name] = func
        
    def check(self, name: str, context: Any = None) -> bool:
        if name in self.predicates:
            return self.predicates[name](context)
        return True
    
    def enforce(self, name: str, context: Any = None) -> bool:
        """Returns True if guardrail passes, False if violated"""
        return self.check(name, context)

class REProjection:
    """Φ (Phi) - Projection/Coherence component of RE triad"""
    def __init__(self):
        self.coherence_score: float = 0.0
        self.readiness: Dict[str, bool] = {}
        self.contradictions: List[str] = []
        self.spec_match_phi0: float = 0.0  # Φ₀ for spec match detection
    
    def update_coherence(self, state: REState, guardrails: REGuardrails) -> float:
        """Calculate current coherence based on state and guardrails"""
        score = 1.0
        
        # Penalize contradictions
        if self.contradictions:
            score -= len(self.contradictions) * 0.1
            
        # Reward fulfilled readiness conditions
        ready_count = sum(1 for r in self.readiness.values() if r)
        total_count = len(self.readiness) or 1
        score *= (ready_count / total_count)
        
        self.coherence_score = max(0.0, min(1.0, score))
        return self.coherence_score
    
    def calculate_spec_match_phi0(self, ontology_graph) -> float:
        """Calculate Φ₀ based on actual project structure and patterns"""
        from .ontology import NodeType, Status
        
        phi0_components = {
            "code_structure": 0.0,
            "has_tests": 0.0, 
            "has_docs": 0.0,
            "project_completeness": 0.0
        }
        
        try:
            # Get actual node types from the ontology
            all_nodes = list(ontology_graph.nodes.values())
            if not all_nodes:
                return 0.0
            
            # 1. Code structure assessment
            code_nodes = [n for n in all_nodes if n.type == NodeType.CODEMODULE]
            doc_nodes = [n for n in all_nodes if n.type == NodeType.TECHNICALDOC] 
            
            if code_nodes:
                # Check for key files that indicate project structure
                key_files = {'evolve.py', 'setup.py', '__init__.py', 'main.py'}
                found_key_files = sum(1 for node in code_nodes 
                                    if any(key in node.title.lower() for key in key_files))
                phi0_components["code_structure"] = min(1.0, found_key_files / 2.0)
                
                # 2. Test detection - look for test patterns
                test_patterns = ['test_', 'tests/', 'test.py', 'conftest.py']  
                test_files = sum(1 for node in code_nodes + doc_nodes
                               if any(pattern in node.title.lower() for pattern in test_patterns))
                phi0_components["has_tests"] = min(1.0, test_files / max(len(code_nodes) * 0.3, 1))
                
                # 3. Documentation assessment
                doc_patterns = ['readme', 'documentation', '.md', 'docs/']
                doc_score = len(doc_nodes) / max(len(code_nodes) * 0.2, 1)  # 20% doc ratio
                phi0_components["has_docs"] = min(1.0, doc_score)
                
                # 4. Project completeness (based on total nodes and diversity)
                total_nodes = len(all_nodes)
                phi0_components["project_completeness"] = min(1.0, total_nodes / 100.0)  # Scale to project size
            
            
            # Calculate Φ₀ as weighted average of components
            weights = {
                "code_structure": 0.3,
                "has_tests": 0.25,
                "has_docs": 0.25,
                "project_completeness": 0.2
            }
            
            self.spec_match_phi0 = sum(phi0_components[k] * weights[k] for k in weights)
            
            # Store component details for debugging
            self._phi0_breakdown = phi0_components.copy()
            self._phi0_breakdown["final_phi0"] = self.spec_match_phi0
            
            return self.spec_match_phi0
            
        except Exception as e:
            print(f"⚠️  Φ₀ calculation failed: {e}")
            self.spec_match_phi0 = 0.0
            return 0.0
    
    def is_at_spec(self, threshold: float = 0.8) -> bool:
        """Check if project is 'at spec' and ready for release"""
        return self.spec_match_phi0 >= threshold

@dataclass
class AgentGene:
    """Immutable agent identity and core protocols"""
    species_id: str  # e.g., "re_ware_v1"
    agent_id: str    # unique runtime instance
    purpose: str     # core mission
    version: str     # protocol version
    allowed_protocols: Set[str] = field(default_factory=lambda: {"self_modify", "ontology_evolve"})
    birth_time: float = field(default_factory=time.time)
    
    @classmethod
    def create(cls, species_id: str, purpose: str, version: str = "1.0.0") -> 'AgentGene':
        """Create new agent gene with unique ID"""
        return cls(
            species_id=species_id,
            agent_id=str(uuid.uuid4()),
            purpose=purpose,
            version=version
        )

@dataclass
class Evolution:
    """Record of an evolution step"""
    timestamp: float
    trigger: str
    changes: Dict[str, Any]
    coherence_before: float
    coherence_after: float
    guardrails_passed: bool = True

class REWare:
    """Minimal RE agent with self-evolution capability"""
    
    def __init__(self, name: str = "re_agent", description: str = "", species_id: str = "re_ware_v1"):
        # Agent Gene & Identity (immutable core)
        self.gene = AgentGene.create(species_id, description or f"{name} agent")
        self.name = name
        self.description = description
        
        # RE Triad
        self.psi = REState()  # Ψ - Memory/State
        self.omega = REGuardrails()  # Ω - Guardrails
        self.phi = REProjection()  # Φ - Projection
        
        # Evolution tracking
        self.evolutions: List[Evolution] = []
        
        # Store gene in memory for persistence
        self.psi.store("agent_gene", self.gene)
        
        # Initialize identity-based guardrails
        self._init_identity_guardrails()
        self.generation = 0
        
        # Initialize basic guardrails
        self._init_guardrails()
    
    def _init_identity_guardrails(self):
        """Initialize identity-based guardrails"""
        # Core identity preservation
        self.omega.add_predicate("preserve_purpose", 
            lambda ctx: self.gene.purpose in (ctx or {}).get("new_description", self.description))
        
        # Protocol compliance
        self.omega.add_predicate("allowed_protocols",
            lambda ctx: (ctx or {}).get("protocol", "self_modify") in self.gene.allowed_protocols)
        
        # Version compatibility  
        self.omega.add_predicate("version_forward",
            lambda ctx: True)  # Allow version evolution
    
    def save_gene_sidecar(self, project_root: Path):
        """Save agent gene to sidecar JSON for restart persistence"""
        try:
            gene_file = project_root / f".re_agent_{self.gene.species_id}.json"
            gene_data = {
                "species_id": self.gene.species_id,
                "agent_id": self.gene.agent_id,
                "purpose": self.gene.purpose,
                "version": self.gene.version,
                "allowed_protocols": list(self.gene.allowed_protocols),
                "birth_time": self.gene.birth_time,
                "last_save": time.time()
            }
            
            with open(gene_file, 'w') as f:
                json.dump(gene_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Could not save agent gene: {e}")
    
    @classmethod
    def load_from_gene_sidecar(cls, project_root: Path, species_id: str) -> Optional['REWare']:
        """Load agent from gene sidecar if it exists"""
        try:
            gene_file = project_root / f".re_agent_{species_id}.json"
            if not gene_file.exists():
                return None
                
            with open(gene_file, 'r') as f:
                gene_data = json.load(f)
            
            # Recreate agent with same gene
            agent = cls.__new__(cls)
            agent.gene = AgentGene(
                species_id=gene_data["species_id"],
                agent_id=gene_data["agent_id"],
                purpose=gene_data["purpose"],
                version=gene_data["version"],
                allowed_protocols=set(gene_data["allowed_protocols"]),
                birth_time=gene_data["birth_time"]
            )
            
            agent.name = f"re_agent_{agent.gene.agent_id[:8]}"
            agent.description = agent.gene.purpose
            
            # Initialize triad
            agent.psi = REState()
            agent.omega = REGuardrails()
            agent.phi = REProjection()
            agent.evolutions = []
            
            # Restore identity
            agent.psi.store("agent_gene", agent.gene)
            agent._init_identity_guardrails()
            agent.generation = 0
            agent._init_guardrails()
            
            print(f"🧬 Restored agent {agent.gene.agent_id[:8]} from gene sidecar")
            return agent
            
        except Exception as e:
            print(f"⚠️  Could not load agent gene: {e}")
            return None
        
    def _init_guardrails(self):
        """Initialize basic guardrails for self-evolution"""
        self.omega.add_predicate("coherence_improving", 
                                lambda ctx: getattr(ctx, 'coherence_score', 0) > 0.5)
        self.omega.add_predicate("no_contradictions",
                                lambda ctx: len(getattr(ctx, 'contradictions', [])) == 0)
        self.omega.add_predicate("placeholder_test", lambda ctx: True)  # Placeholder
        
    def evolve(self, trigger: str = "manual") -> bool:
        """Core evolution method with Ω gates for safety"""
        print(f"🧬 [{self.name}] Starting evolution (trigger: {trigger})...")
        
        # PRE-GATE: Check no_contradictions predicate
        if not self.omega.check("no_contradictions", {"trigger": trigger}):
            print("❌ Pre-gate failed: contradictions detected")
            return False
        
        coherence_before = self.phi.coherence_score
        
        # 1. REFLECT: Read own source code
        source_code = self._read_own_source()
        if not source_code:
            print("❌ Could not read own source code")
            return False
            
        # 2. ANALYZE: Detect problems using engineering & entropic principles
        problems = self._analyze_code_problems(source_code)
        if not problems:
            print("✅ No problems detected in source code")
            return False
            
        print(f"🔍 Found {len(problems)} problems: {list(problems.keys())}")
        
        # 3. PLAN: Generate refactoring plan
        refactor_plan = self._plan_refactoring(problems, source_code)
        if not refactor_plan:
            print("❌ Could not generate refactoring plan")
            return False
            
        print(f"📋 Refactoring plan: {len(refactor_plan)} changes")
        
        # 4. GUARD: Check if changes are permitted
        if not self._validate_evolution_plan(refactor_plan):
            print("❌ Evolution plan violates guardrails")
            return False
            
        # 5. IMPLEMENT: Apply changes to source code
        success = self._apply_source_refactoring(source_code, refactor_plan)
        if not success:
            print("❌ Failed to apply source code changes")
            return False
            
        # 6. VERIFY: Test that changes work
        if not self._verify_evolution():
            print("❌ Evolution verification failed")
            return False
            
        # 7. UPDATE COHERENCE: Calculate post-evolution state
        coherence_after = self.phi.update_coherence(self.psi, self.omega)
        
        # POST-GATE: Ensure coherence is improving
        coherence_context = {
            "coherence_before": coherence_before,
            "coherence_after": coherence_after,
            "trigger": trigger
        }
        
        if not self.omega.check("coherence_improving", coherence_context):
            print(f"❌ Post-gate failed: coherence not improving ({coherence_before:.2f} -> {coherence_after:.2f})")
            return False
        
        # 8. RECORD: Evolution passed all gates
        evolution = Evolution(
            timestamp=time.time(),
            trigger=trigger,
            changes={"refactored": list(problems.keys()), "plan": refactor_plan},
            coherence_before=coherence_before,
            coherence_after=coherence_after,
            guardrails_passed=True  # All Ω gates passed
        )
        self.evolutions.append(evolution)
        self.generation += 1
        
        print(f"✅ Evolution successful! Gen {self.generation}, coherence: {coherence_after:.2f}")
        return True
    
    def _read_own_source(self) -> Optional[str]:
        """Read the source code of this module"""
        try:
            module_file = inspect.getfile(self.__class__)
            with open(module_file, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading source: {e}")
            return None
    
    def _analyze_code_problems(self, source_code: str) -> Dict[str, str]:
        """Analyze evolution needs using ontological graph tensions"""
        problems = {}
        
        # Check if we have access to project intelligence for LLM-driven analysis
        graph_tension_analysis = self.psi.recall("graph_tension_analysis")
        if graph_tension_analysis and graph_tension_analysis.get("evolution_needed", False):
            # Use LLM analysis to determine what capabilities need to evolve
            analysis = graph_tension_analysis.get("analysis", "")
            
            # Parse LLM recommendations into concrete problems using actual method detection
            if "cleanup" in analysis.lower() and not self._has_actual_method(source_code, 'suggest_organic_cleanup'):
                problems['missing_cleanup_intelligence'] = 'LLM detected need for cleanup capabilities'
            
            if "documentation" in analysis.lower() and not self._has_actual_method(source_code, 'generate_documentation'):
                problems['missing_documentation_generation'] = 'LLM detected documentation gaps'
                
            if "monitoring" in analysis.lower() and not self._has_actual_method(source_code, 'monitor_project_health'):
                problems['missing_monitoring_capability'] = 'LLM detected need for monitoring capabilities'
                
            if "optimization" in analysis.lower() and not self._has_actual_method(source_code, 'optimize_performance'):
                problems['missing_optimization'] = 'LLM detected optimization opportunities'
                
            if "test" in analysis.lower() and not self._has_actual_method(source_code, 'generate_missing_tests'):
                problems['missing_test_generation'] = 'LLM detected test coverage gaps'
                
            # Check for capability expansion needs based on tension count
            tension_count = graph_tension_analysis.get("tension_count", 0)
            if tension_count > 5 and source_code.count('def ') < 30:
                problems['insufficient_capabilities'] = f'LLM detected {tension_count} tensions requiring capability expansion'
        else:
            # Fallback to basic analysis when no ontological graph is available
            method_count = source_code.count('def ')
            if method_count < 15:  # Basic capability threshold
                problems['basic_capability_deficit'] = f'Only {method_count} methods, need basic capabilities'
                
            # Check for fundamental missing capabilities that enable LLM-driven evolution
            # Look for actual method definitions (not in implementation strings)
            if not self._has_actual_method(source_code, 'evolve_from_tensions'):
                problems['missing_tension_evolution'] = 'Need LLM-driven evolution capability'
                
            # Check for essential LLM-integration capabilities
            if not self._has_actual_method(source_code, 'generate_documentation'):
                problems['missing_documentation_generation'] = 'Need LLM-driven documentation capability'
                
            if not self._has_actual_method(source_code, 'monitor_project_health'):
                problems['missing_monitoring_capability'] = 'Need LLM-driven monitoring capability'
        
        return problems
    
    def _has_actual_method(self, source_code: str, method_name: str) -> bool:
        """Check if a method actually exists in the class (not just in implementation strings)"""
        # Look for method definition pattern that's not in an implementation string
        import re
        
        # Pattern to match actual method definitions in the class
        # This avoids matching methods inside triple-quoted implementation strings
        pattern = rf"^\s+def {method_name}\("
        
        lines = source_code.split('\n')
        in_implementation_string = False
        triple_quote_count = 0
        
        for line in lines:
            # Track if we're inside a triple-quoted implementation string
            triple_quote_count += line.count("'''")
            in_implementation_string = (triple_quote_count % 2) == 1
            
            # If we find the method definition and we're not in an implementation string
            if re.match(pattern, line) and not in_implementation_string:
                return True
                
        return False
    
    def _find_duplicate_imports(self, source_code: str) -> List[str]:
        """Find duplicate import statements"""
        import_lines = [line.strip() for line in source_code.split('\n') 
                       if line.strip().startswith('import ') or line.strip().startswith('from ')]
        seen = set()
        duplicates = []
        for line in import_lines:
            if line in seen:
                duplicates.append(line)
            seen.add(line)
        return duplicates
    
    def _plan_refactoring(self, problems: Dict[str, str], source_code: str) -> List[Dict[str, Any]]:
        """Generate specific refactoring plan to fix problems"""
        plan = []
        
        if 'missing_specialized_spawning' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'spawn_specialized_offspring',
                'implementation': '''def spawn_specialized_offspring(self, specialization: str) -> 'REWare':
        \"\"\"Create specialized offspring for specific problem domains\"\"\"
        child = self.spawn(f"Specialist_{specialization}_{self.generation}", 
                          inherited_traits=set(self.psi.memory.keys()))
        
        # Add specialization-specific capabilities
        if specialization == "analyzer":
            child.psi.store("specialty_analysis", {"focus": "deep_pattern_recognition"})
        elif specialization == "optimizer": 
            child.psi.store("specialty_optimization", {"focus": "efficiency_improvement"})
            
        return child'''
            })
            
        if 'missing_knowledge_integration' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class', 
                'method_name': 'integrate_external_knowledge',
                'implementation': '''def integrate_external_knowledge(self, knowledge_source: str) -> bool:
        \"\"\"Integrate knowledge from external sources\"\"\"
        # Simulate knowledge integration
        knowledge_key = f"external_{knowledge_source}_{len(self.psi.memory)}"
        self.psi.store(knowledge_key, {
            "source": knowledge_source,
            "integrated_at": time.time(),
            "type": "external_knowledge"
        })
        return True'''
            })
            
        if 'missing_optimization' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'optimize_performance', 
                'implementation': '''def optimize_performance(self) -> float:
        \"\"\"Optimize agent performance and return improvement score\"\"\"
        # Clean old memories to reduce entropy
        old_memories = [k for k, v in self.psi.memory.items() 
                       if isinstance(v, dict) and v.get("type") == "temporary"]
        for k in old_memories[:len(old_memories)//2]:  # Remove half of temp memories
            del self.psi.memory[k]
            
        # Update coherence based on optimization
        self.phi.update_coherence(self.psi, self.omega)
        return self.phi.coherence_score'''
            })

        # Handle LLM-driven cleanup intelligence needs
        if 'missing_cleanup_intelligence' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'suggest_organic_cleanup',
                'implementation': '''def suggest_organic_cleanup(self) -> List[Dict[str, Any]]:
        \"\"\"Suggest cleanup actions based on LLM analysis of graph tensions\"\"\"
        graph_analysis = self.psi.recall("graph_tension_analysis")
        if not graph_analysis:
            return []
        
        suggestions = []
        analysis_text = graph_analysis.get("analysis", "").lower()
        
        # Parse LLM recommendations into actionable cleanup
        if "archive" in analysis_text or "obsolete" in analysis_text:
            suggestions.append({
                "action": "archive_cleanup",
                "description": "LLM detected obsolete files needing cleanup",
                "confidence": 0.9,
                "learned_from": "llm_tension_analysis"
            })
        
        if "test" in analysis_text and "coverage" in analysis_text:
            suggestions.append({
                "action": "add_tests",
                "description": "LLM detected test coverage gaps", 
                "confidence": 0.95,
                "learned_from": "llm_tension_analysis"
            })
            
        if "documentation" in analysis_text:
            suggestions.append({
                "action": "generate_docs",
                "description": "LLM detected documentation gaps",
                "confidence": 0.8,
                "learned_from": "llm_tension_analysis"
            })
        
        return suggestions'''
            })

        if 'missing_cleanup_execution' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'execute_cleanup_plan',
                'implementation': '''def execute_cleanup_plan(self, cleanup_actions: List[Dict]) -> bool:
        \"\"\"Execute approved cleanup actions safely\"\"\"
        executed_count = 0
        
        for action in cleanup_actions:
            if action.get("risk_level") == "low" and action.get("priority") == "high":
                # Only execute low-risk, high-priority items automatically
                action_type = action.get("type")
                
                if action_type == "documentation_debt":
                    # Generate documentation stubs
                    self.psi.store(f"doc_generated_{time.time()}", {
                        "type": "auto_documentation",
                        "items": action.get("items", []),
                        "status": "generated"
                    })
                    executed_count += 1
                
                elif action_type == "archive_cleanup":
                    # Mark for review rather than auto-delete
                    self.psi.store(f"cleanup_reviewed_{time.time()}", {
                        "type": "marked_for_cleanup",
                        "items": action.get("items", []),
                        "status": "pending_review"
                    })
                    executed_count += 1
        
        return executed_count > 0'''
            })

        if 'missing_hygiene_assessment' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'assess_project_hygiene',
                'implementation': '''def assess_project_hygiene(self) -> Dict[str, float]:
        \"\"\"Assess overall project health and hygiene\"\"\"
        hygiene_score = {
            "code_quality": 0.8,  # Base score
            "documentation": 0.7,
            "test_coverage": 0.6,
            "technical_debt": 0.5,
            "overall": 0.0
        }
        
        # Adjust based on detected issues
        cleanup_data = self.psi.recall("cleanup_opportunities")
        if cleanup_data:
            issue_count = cleanup_data.get("count", 0)
            if issue_count > 10:
                hygiene_score["technical_debt"] = max(0.2, hygiene_score["technical_debt"] - 0.1)
            
            high_priority_count = len(cleanup_data.get("high_priority", []))
            if high_priority_count > 5:
                hygiene_score["overall"] = max(0.3, hygiene_score["overall"] - 0.2)
        
        # Calculate overall score
        hygiene_score["overall"] = sum(hygiene_score.values()) / (len(hygiene_score) - 1)
        
        return hygiene_score'''
            })

        # Handle capability expansion needs
        if 'insufficient_capabilities' in problems or 'capability_deficit' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'expand_capabilities',
                'implementation': '''def expand_capabilities(self) -> bool:
        \"\"\"Expand agent capabilities based on current needs\"\"\"
        new_capability = f"dynamic_capability_{len(self.psi.memory)}_{self.generation}"
        self.psi.store(new_capability, {
            "type": "expanded_capability",
            "added_at": time.time(),
            "purpose": "continuous_growth"
        })
        
        # Add problem-solving method
        self.psi.store("capability_advanced_problem_solving", {
            "methods": ["decomposition", "pattern_matching", "recursive_analysis"],
            "level": "advanced"
        })
        return True'''
            })
        
        # Handle LLM-detected documentation needs
        if 'missing_documentation_generation' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'generate_documentation',
                'implementation': '''def generate_documentation(self) -> bool:
        \"\"\"Generate documentation based on LLM analysis\"\"\"
        graph_analysis = self.psi.recall("graph_tension_analysis")
        if not graph_analysis:
            return False
        
        # Create documentation based on LLM insights
        doc_data = {
            "type": "llm_generated_docs",
            "generated_at": time.time(),
            "based_on_analysis": graph_analysis.get("analysis", "")[:200],
            "context_nodes": graph_analysis.get("context_nodes", 0)
        }
        
        self.psi.store("generated_documentation", doc_data)
        return True'''
            })
        
        # Handle LLM-detected monitoring needs  
        if 'missing_monitoring_capability' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'monitor_project_health',
                'implementation': '''def monitor_project_health(self) -> Dict[str, Any]:
        \"\"\"Monitor project health using LLM-driven insights\"\"\"
        graph_analysis = self.psi.recall("graph_tension_analysis")
        health_metrics = {
            "tension_count": graph_analysis.get("tension_count", 0) if graph_analysis else 0,
            "evolution_needed": graph_analysis.get("evolution_needed", False) if graph_analysis else False,
            "coherence_score": self.phi.coherence_score,
            "last_analysis": time.time()
        }
        
        # Determine health status based on LLM analysis
        if health_metrics["tension_count"] > 10:
            health_metrics["status"] = "critical"
        elif health_metrics["tension_count"] > 5:
            health_metrics["status"] = "warning"
        else:
            health_metrics["status"] = "healthy"
            
        self.psi.store("project_health_monitoring", health_metrics)
        return health_metrics'''
            })
        
        # Handle LLM-detected test generation needs
        if 'missing_test_generation' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class', 
                'method_name': 'generate_missing_tests',
                'implementation': '''def generate_missing_tests(self) -> List[str]:
        \"\"\"Generate test stubs based on LLM analysis of coverage gaps\"\"\"
        graph_analysis = self.psi.recall("graph_tension_analysis")
        if not graph_analysis:
            return []
        
        test_stubs = []
        analysis = graph_analysis.get("analysis", "").lower()
        
        if "test" in analysis:
            # Generate test cases based on LLM recommendations
            test_stub = {
                "test_type": "unit_test",
                "description": "LLM-recommended test coverage",
                "priority": "high" if "critical" in analysis else "medium",
                "generated_at": time.time()
            }
            test_stubs.append(f"test_llm_recommended_{len(test_stubs)}")
            self.psi.store(f"generated_test_{time.time()}", test_stub)
        
        return test_stubs'''
            })
        
        # Handle fundamental tension-driven evolution capability
        if 'missing_tension_evolution' in problems:
            plan.append({
                'type': 'add_method',
                'target': 'end_of_class',
                'method_name': 'evolve_from_tensions',
                'implementation': '''def evolve_from_tensions(self, tensions: List[Dict[str, Any]]) -> bool:
        \"\"\"Evolve capabilities directly from ontological graph tensions\"\"\"
        if not tensions:
            return False
        
        # Use tensions to drive evolution instead of hardcoded analysis
        evolution_needed = False
        
        for tension in tensions:
            if tension.get("severity") == "high" and tension.get("requires_llm_analysis", False):
                evolution_needed = True
                
                # Store tension data for future evolution cycles
                tension_key = f"tension_{tension.get('type')}_{time.time()}".replace('.', '_')
                self.psi.store(tension_key, {
                    "tension_data": tension,
                    "requires_capability_evolution": True,
                    "detected_at": time.time()
                })
        
        if evolution_needed:
            # Trigger next evolution cycle based on tensions
            self.phi.contradictions.append("High-severity ontological tensions detected")
            
        return evolution_needed'''
            })
            
        return plan
    
    def _validate_evolution_plan(self, plan: List[Dict[str, Any]]) -> bool:
        """Validate evolution plan against guardrails"""
        # Basic safety checks
        for change in plan:
            if change['type'] in ['remove_placeholder', 'clean_imports', 'complete_todos', 'add_method']:
                continue  # These are safe refactorings
            else:
                print(f"⚠️  Unknown change type: {change['type']}")
                return False
        return True
    
    def _apply_source_refactoring(self, source_code: str, plan: List[Dict[str, Any]]) -> bool:
        """Apply refactoring changes to the source file"""
        try:
            modified_code = source_code
            
            for change in plan:
                if change['type'] == 'remove_placeholder':
                    modified_code = modified_code.replace(
                        change['target'], 
                        change['replacement']
                    )
                    print(f"✏️  Replaced placeholder code")
                    
                elif change['type'] == 'add_method':
                    # Add new method before the final closing of REWare class
                    method_code = f"    \n    {change['implementation']}\n"
                    
                    # Find the end of REWare class (before def re_ware function)
                    insertion_point = modified_code.find('\n# Factory function for easy instantiation')
                    if insertion_point == -1:
                        insertion_point = modified_code.rfind('    def __repr__(self):')
                        if insertion_point != -1:
                            # Find the end of __repr__ method
                            insertion_point = modified_code.find('\n\n', insertion_point)
                    
                    if insertion_point != -1:
                        modified_code = (modified_code[:insertion_point] + 
                                       method_code + 
                                       modified_code[insertion_point:])
                        print(f"🔧 Added new method: {change['method_name']}")
                    else:
                        print(f"⚠️  Could not find insertion point for {change['method_name']}")
                    
            # Write back to file
            module_file = inspect.getfile(self.__class__)
            with open(module_file, 'w') as f:
                f.write(modified_code)
            
            print(f"💾 Applied {len(plan)} refactoring changes to {module_file}")
            return True
            
        except Exception as e:
            print(f"Error applying refactoring: {e}")
            return False
    
    def _verify_evolution(self) -> bool:
        """Verify that evolved code still works"""
        try:
            # Try to reload the module
            import importlib
            import re_ware
            importlib.reload(re_ware)
            
            # Basic smoke test - create a new instance
            test_agent = re_ware.re_ware("test", "verification")
            if test_agent.name == "test":
                print("✅ Evolution verification passed")
                return True
            else:
                print("❌ Evolution verification failed - instance creation issue")
                return False
                
        except Exception as e:
            print(f"❌ Evolution verification failed: {e}")
            return False
    
    def spawn(self, name: str, inherited_traits: Optional[Set[str]] = None) -> 'REWare':
        """Create offspring agent with inherited capabilities"""
        offspring = REWare(name=name, description=f"Spawned from {self.name}")
        
        # Inherit selected memory patterns
        if inherited_traits:
            for trait in inherited_traits:
                if trait in self.psi.memory:
                    offspring.psi.store(trait, copy.deepcopy(self.psi.memory[trait]))
        
        # Inherit ontology
        offspring.psi.ontology = copy.deepcopy(self.psi.ontology)
        
        # Inherit guardrails with stricter constraints
        for name, predicate in self.omega.predicates.items():
            offspring.omega.add_predicate(f"strict_{name}", predicate)
            
        return offspring
    
    def __repr__(self):
        return f"REWare(name='{self.name}', gen={self.generation}, coherence={self.phi.coherence_score:.2f})"
    
    def generate_documentation(self) -> bool:
        """Generate documentation based on LLM analysis"""
        graph_analysis = self.psi.recall("graph_tension_analysis")
        if not graph_analysis:
            return False
        
        # Create documentation based on LLM insights
        doc_data = {
            "type": "llm_generated_docs",
            "generated_at": time.time(),
            "based_on_analysis": graph_analysis.get("analysis", "")[:200],
            "context_nodes": graph_analysis.get("context_nodes", 0)
        }
        
        self.psi.store("generated_documentation", doc_data)
        return True
    
    def monitor_project_health(self) -> Dict[str, Any]:
        """Monitor project health using LLM-driven insights"""
        graph_analysis = self.psi.recall("graph_tension_analysis")
        health_metrics = {
            "tension_count": graph_analysis.get("tension_count", 0) if graph_analysis else 0,
            "evolution_needed": graph_analysis.get("evolution_needed", False) if graph_analysis else False,
            "coherence_score": self.phi.coherence_score,
            "last_analysis": time.time()
        }
        
        # Determine health status based on LLM analysis
        if health_metrics["tension_count"] > 10:
            health_metrics["status"] = "critical"
        elif health_metrics["tension_count"] > 5:
            health_metrics["status"] = "warning"
        else:
            health_metrics["status"] = "healthy"
            
        self.psi.store("project_health_monitoring", health_metrics)
        return health_metrics
    
    def evolve_from_tensions(self, tensions: List[Dict[str, Any]]) -> bool:
        """Evolve capabilities directly from ontological graph tensions"""
        if not tensions:
            return False
        
        # Use tensions to drive evolution instead of hardcoded analysis
        evolution_needed = False
        
        for tension in tensions:
            if tension.get("severity") == "high" and tension.get("requires_llm_analysis", False):
                evolution_needed = True
                
                # Store tension data for future evolution cycles
                tension_key = f"tension_{tension.get('type')}_{time.time()}".replace('.', '_')
                self.psi.store(tension_key, {
                    "tension_data": tension,
                    "requires_capability_evolution": True,
                    "detected_at": time.time()
                })
        
        if evolution_needed:
            # Trigger next evolution cycle based on tensions
            self.phi.contradictions.append("High-severity ontological tensions detected")
            
        return evolution_needed
    
    def suggest_organic_cleanup(self) -> List[Dict[str, Any]]:
        """Suggest cleanup actions based on LLM analysis of graph tensions"""
        graph_analysis = self.psi.recall("graph_tension_analysis")
        if not graph_analysis:
            return []
        
        suggestions = []
        analysis_text = graph_analysis.get("analysis", "").lower()
        
        # Parse LLM recommendations into actionable cleanup
        if "archive" in analysis_text or "obsolete" in analysis_text:
            suggestions.append({
                "action": "archive_cleanup",
                "description": "LLM detected obsolete files needing cleanup",
                "confidence": 0.9,
                "learned_from": "llm_tension_analysis"
            })
        
        if "test" in analysis_text and "coverage" in analysis_text:
            suggestions.append({
                "action": "add_tests",
                "description": "LLM detected test coverage gaps", 
                "confidence": 0.95,
                "learned_from": "llm_tension_analysis"
            })
            
        if "documentation" in analysis_text:
            suggestions.append({
                "action": "generate_docs",
                "description": "LLM detected documentation gaps",
                "confidence": 0.8,
                "learned_from": "llm_tension_analysis"
            })
        
        return suggestions
    
    def generate_missing_tests(self) -> List[str]:
        """Generate test stubs based on LLM analysis of coverage gaps"""
        graph_analysis = self.psi.recall("graph_tension_analysis")
        if not graph_analysis:
            return []
        
        test_stubs = []
        analysis = graph_analysis.get("analysis", "").lower()
        
        if "test" in analysis:
            # Generate test cases based on LLM recommendations
            test_stub = {
                "test_type": "unit_test",
                "description": "LLM-recommended test coverage",
                "priority": "high" if "critical" in analysis else "medium",
                "generated_at": time.time()
            }
            test_stubs.append(f"test_llm_recommended_{len(test_stubs)}")
            self.psi.store(f"generated_test_{time.time()}", test_stub)
        
        return test_stubs
    
    def evolve_from_tensions(self, tensions: List[Dict[str, Any]]) -> bool:
        """Evolve capabilities directly from ontological graph tensions"""
        if not tensions:
            return False
        
        # Use tensions to drive evolution instead of hardcoded analysis
        evolution_needed = False
        
        for tension in tensions:
            if tension.get("severity") == "high" and tension.get("requires_llm_analysis", False):
                evolution_needed = True
                
                # Store tension data for future evolution cycles
                tension_key = f"tension_{tension.get('type')}_{time.time()}".replace('.', '_')
                self.psi.store(tension_key, {
                    "tension_data": tension,
                    "requires_capability_evolution": True,
                    "detected_at": time.time()
                })
        
        if evolution_needed:
            # Trigger next evolution cycle based on tensions
            self.phi.contradictions.append("High-severity ontological tensions detected")
            
        return evolution_needed

# Factory function for easy instantiation
def re_ware(name: str = "agent", description: str = "") -> REWare:
    """Create a new RE_ware agent instance"""
    return REWare(name=name, description=description)