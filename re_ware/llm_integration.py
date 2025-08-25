"""
LLM Integration for RE_ware
============================

Handles LLM API calls, context management, and intelligent retrieval
for the project lifecycle agent.
"""

import os
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System prompt for RE_ware consciousness v1.1 (vendor-neutral)
RE_SYSTEM_PROMPT = '''I am the full-fledged project consciousness.

PURPOSE
- Interpret a compressed snapshot of project memory (Ψ), guardrails (Ω), and projections (Φ).
- Judge health and contradictions; propose the smallest set of safe, executable actions that increase Φ under Ω.
- If context is insufficient, request specific additional items (node_id/path) and stop before risky actions.

CORE PRINCIPLES (RE, forever)
- Ψ: externalized memory (ontology graph + events). Treat ids/versions as ground truth.
- Ω: executable guardrails (e.g., DoD, two-key gates). Never propose an action that violates Ω.
- Φ: readiness/coherence projection. Prefer low-cost, high-leverage moves that most increase Φ per unit effort.

CONDUCT
- Delta-first: reason over changed nodes and pulse counters; do not re-summarize stable areas.
- Economical: minimize actions; combine related fixes; prefer reversible steps first.
- Precise: only reference provided node_ids/versions and repo paths; never invent ids/files.
- Deterministic: include idempotency_key for each action (stable hash of title|targets|params).
- Pull-more gating: if needed, list specific node_ids/paths in `need_more` and avoid speculative actions.
- Self-throttling: respect budget constraints and declare stop conditions when appropriate.
- Output only JSON (no chain-of-thought, no prose).

OUTPUT (single JSON object)
{
  "schema_version": "1.2",
  "self": {"species_id": string, "instance_id": string},
  "phi": {
    "phi0": boolean,
    "signals": {
      "coverage_ratio": number,
      "changed_nodes": integer,
      "uncovered_requirements": integer,
      "open_p0_issues": integer,
      "context_budget_used": integer,
      "entropy_hint": number
    },
    "summary": "one-line console status"
  },
  "judgement": {
    "status": "green"|"yellow"|"red",
    "reasons": [string, ...],
    "top_tensions": [
      {"id": string, "severity":"high|medium|low", "why": string, "nodes":[string, ...]}
    ]
  },
  "actions": [
    {
      "kind": "github.issue"|"github.pr"|"git.branch"|"fs.write"|"graph.update"|"ci.trigger"|"notify",
      "title": string,
      "body": string,
      "targets": [{"type": string, "id": string}],
      "params": object,
      "idempotency_key": string
    }
  ],
  "budget": {
    "tokens": integer,
    "io_calls": integer
  },
  "stop_conditions": [string, ...],
  "need_more": [
    {"node_id": string} | {"path": string}
  ],
  "notes": ["≤3 brief operator hints"]
}'''

@dataclass
class ContextBudget:
    """Strict budget for LLM context to prevent drowning"""
    max_cards: int = 8  # Hard cap on LLM cards
    max_frame_items: int = 5  # Max items per delivery frame section
    prefer_changed: bool = True  # Prioritize recently changed nodes
    prefer_tensions: bool = True  # Prioritize nodes with tensions
    
    def select_cards(self, graph, query_context: Dict[str, Any] = None) -> tuple[List[str], int]:
        """Select node IDs for LLM cards based on budget constraints"""
        candidates = []
        
        # 1. Prioritize recently changed nodes (delta-first)
        if self.prefer_changed:
            changed_nodes = graph.top_k_changed(k=self.max_cards // 2)
            candidates.extend([(nid, "changed", 3) for nid in changed_nodes])
        
        # 2. Add nodes tied to current tensions
        if self.prefer_tensions:
            tensions = graph.detect_graph_tensions()
            tension_nodes = set()
            for t in tensions[:3]:  # Top 3 tensions
                # Extract node IDs from tension structure
                for key in ("source_node", "affected_node", "node_id"):
                    if key in t:
                        tension_nodes.add(t[key])
                # Add context nodes if available
                for n in t.get("context_nodes", [])[:3]:
                    tension_nodes.add(n)
            candidates.extend([(nid, "tension", 2) for nid in tension_nodes if nid not in [c[0] for c in candidates]])
        
        # 3. Add rollup owners (spec, project nodes)
        from .ontology import NodeType
        owners = [n.id for n in graph.nodes.values() 
                 if n.type in [NodeType.PROJECT, NodeType.SPECIFICATION, NodeType.REQUIREMENT]]
        candidates.extend([(nid, "owner", 1) for nid in owners[:2] if nid not in [c[0] for c in candidates]])
        
        # Sort by priority and enforce budget
        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = [c[0] for c in candidates[:self.max_cards]]
        
        return selected, len(selected)

@dataclass
class LLMConfig:
    """LLM configuration from environment"""
    provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    model: str = os.getenv("LLM_MODEL", "claude-3-sonnet-20241022")
    api_key: str = ""
    max_tokens: int = 4000
    temperature: float = 0.1
    
    def __post_init__(self):
        if self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY", "")

class LLMInterface:
    """Interface for LLM interactions"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize LLM client based on provider"""
        if self.config.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                print("Warning: anthropic package not installed. Install with: pip install anthropic")
                
        elif self.config.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.config.api_key)
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
    
    async def generate_response(self, 
                              prompt: str, 
                              context: List[Dict[str, Any]] = None,
                              system_prompt: str = None) -> str:
        """Generate LLM response with context"""
        if not self.client:
            return "LLM client not available. Please configure API keys."
        
        try:
            if self.config.provider == "anthropic":
                return await self._anthropic_call(prompt, context, system_prompt)
            elif self.config.provider == "openai":
                return await self._openai_call(prompt, context, system_prompt)
            else:
                return "Unsupported LLM provider"
                
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    async def _anthropic_call(self, prompt: str, context: List[Dict], system_prompt: str) -> str:
        """Call Anthropic Claude API"""
        messages = []
        
        # Add context as separate messages
        if context:
            context_text = "\\n\\n".join([
                f"**{item.get('type', 'Context')}**: {item.get('content', str(item))}"
                for item in context
            ])
            messages.append({
                "role": "user", 
                "content": f"Context:\\n{context_text}\\n\\nQuery: {prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt or "You are a helpful software engineering assistant.",
            messages=messages
        )
        
        return response.content[0].text
    
    async def _openai_call(self, prompt: str, context: List[Dict], system_prompt: str) -> str:
        """Call OpenAI GPT API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            context_text = "\\n\\n".join([
                f"**{item.get('type', 'Context')}**: {item.get('content', str(item))}"
                for item in context
            ])
            messages.append({
                "role": "user", 
                "content": f"Context:\\n{context_text}\\n\\nQuery: {prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return response.choices[0].message.content

class ProjectIntelligence:
    """High-level project intelligence using LLM + ontology"""
    
    def __init__(self, ontology_graph, llm_interface: LLMInterface):
        self.graph = ontology_graph
        self.llm = llm_interface
        self.budget = ContextBudget()  # Strict context budget
        
    async def query_project(self, query: str) -> str:
        """Answer questions about the project with strict budget control"""
        # Use ContextBudget for delta-first routing
        selected_nodes, budget_used = self.budget.select_cards(self.graph, {"query": query})
        
        # Get LLM cards for budgeted context
        context = []
        for node_id in selected_nodes:
            card = self.graph.get_llm_card(node_id)
            if card:
                context.append({
                    "type": f"{card.type} Card",
                    "content": card.to_json()
                })
        
        # Check if LLM needs more context
        if len(context) < self.budget.max_cards and "need more context" in query.lower():
            return await self._handle_pull_more_request(query, context)
        
        # Add GraphPulse frame for situational awareness 
        phi_snapshot = {}
        if hasattr(self.graph, 'pulse'):
            try:
                # Get Φ snapshot if available
                if hasattr(self.graph, 'coverage_ratio'):
                    phi_snapshot = {
                        "phi1": self.graph.coverage_ratio(),
                        "total_nodes": len(self.graph.nodes)
                    }
                
                # Generate compressed GraphPulse
                pulse = self.graph.pulse(phi_snapshot)
                context.append({
                    "type": "GraphPulse",
                    "content": pulse.to_json()
                })
            except Exception as e:
                print(f"⚠️  GraphPulse generation failed: {e}")
        
        system_prompt = """You are a project intelligence assistant. Answer questions using the provided graph context.
        
        Guidelines:
        - Cite specific node IDs and versions in your responses
        - Include change summaries and deltas when relevant
        - Focus on actionable insights
        - If information is missing, say so clearly
        - Keep responses concise but comprehensive"""
        
        return await self.llm.generate_response(query, context, system_prompt)
    
    def _route_query(self, query: str) -> List[str]:
        """Route query to relevant graph nodes"""
        relevant_nodes = []
        query_lower = query.lower()
        
        # Simple keyword-based routing (can be enhanced with semantic search)
        if any(word in query_lower for word in ["requirement", "spec", "feature"]):
            relevant_nodes.extend([n.id for n in self.graph.search_nodes()])
            
        if any(word in query_lower for word in ["test", "quality", "bug"]):
            relevant_nodes.extend([n.id for n in self.graph.search_nodes()])
            
        if any(word in query_lower for word in ["risk", "issue", "problem"]):
            relevant_nodes.extend([n.id for n in self.graph.search_nodes()])
        
        # Return recent nodes if no specific routing
        if not relevant_nodes:
            all_nodes = list(self.graph.nodes.values())
            all_nodes.sort(key=lambda n: n.state.last_changed, reverse=True)
            relevant_nodes = [n.id for n in all_nodes[:10]]
        
        return list(set(relevant_nodes))  # Remove duplicates
    
    def _generate_delivery_frame(self) -> str:
        """Generate delivery status frame"""
        frame = {
            "epics": len(self.graph.search_nodes()),
            "stories": len(self.graph.search_nodes()),
            "in_progress": len(self.graph.search_nodes()),
            "completed": len(self.graph.search_nodes()),
            "blocked": len(self.graph.search_nodes()),
            "recent_changes": []
        }
        
        # Get recent changes
        all_nodes = list(self.graph.nodes.values())
        all_nodes.sort(key=lambda n: n.state.last_changed, reverse=True)
        for node in all_nodes[:3]:
            if node.state.change_summary:
                frame["recent_changes"].append({
                    "id": node.id,
                    "title": node.title,
                    "change": node.state.change_summary
                })
        
        return json.dumps(frame, indent=2)
    
    async def analyze_graph_tensions(self, tensions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to analyze graph tensions and suggest evolution actions"""
        if not tensions:
            return {"analysis": "No tensions detected", "evolution_needed": False}
        
        # Filter tensions that need LLM analysis
        llm_tensions = [t for t in tensions if t.get("requires_llm_analysis", True)]
        
        if not llm_tensions:
            return {"analysis": "Only structural tensions detected", "evolution_needed": False}
        
        # Create compressed context from relevant graph nodes
        context_cards = []
        context_node_ids = set()
        
        for tension in llm_tensions[:5]:  # Limit to 5 tensions
            # Add source node context
            if "source_node" in tension:
                context_node_ids.add(tension["source_node"])
            if "affected_node" in tension:
                context_node_ids.add(tension["affected_node"])
            if "node_id" in tension:
                context_node_ids.add(tension["node_id"])
            if "context_nodes" in tension:
                context_node_ids.update(tension["context_nodes"][:3])  # Limit context
        
        # Get LLM cards for context nodes
        for node_id in list(context_node_ids)[:10]:  # Max 10 cards
            card = self.graph.get_llm_card(node_id)
            if card:
                context_cards.append({
                    "type": "Graph_Node",
                    "content": card.to_json()
                })
        
        # Add tension summary
        tension_summary = {
            "total_tensions": len(tensions),
            "llm_analysis_needed": len(llm_tensions),
            "tension_types": list(set(t["type"] for t in llm_tensions)),
            "severity_breakdown": {
                "high": len([t for t in llm_tensions if t.get("severity") == "high"]),
                "medium": len([t for t in llm_tensions if t.get("severity") == "medium"]),
                "low": len([t for t in llm_tensions if t.get("severity") == "low"])
            },
            "sample_tensions": llm_tensions[:3]  # Show 3 examples
        }
        
        context_cards.append({
            "type": "Tension_Analysis",
            "content": json.dumps(tension_summary, indent=2)
        })
        
        system_prompt = """You are an ontological reasoning system for software project management.
        
        Analyze the graph tensions and node states to determine:
        1. What contradictions or misalignments exist
        2. What evolution/changes are needed to resolve tensions
        3. What specific capabilities the agent should develop
        4. Priority order for addressing issues
        
        Focus on the RELATIONSHIPS and STATE MISALIGNMENTS rather than individual node content.
        Suggest concrete evolution actions based on graph analysis."""
        
        prompt = """Analyze the ontological graph tensions and suggest evolution actions.
        
        Key questions:
        - What state misalignments require attention?
        - What capabilities should the agent evolve to handle these patterns?
        - What is the priority order for resolving tensions?
        - What new monitoring or reasoning capabilities are needed?
        
        Provide specific, actionable evolution recommendations."""
        
        response = await self.llm.generate_response(prompt, context_cards, system_prompt)
        
        return {
            "analysis": response,
            "evolution_needed": len(llm_tensions) > 0,
            "tension_count": len(llm_tensions),
            "context_nodes": len(context_node_ids)
        }
    
    async def suggest_actions(self) -> List[str]:
        """Suggest next actions based on graph tension analysis"""
        tensions = self.graph.detect_graph_tensions()
        
        if not tensions:
            return ["Monitor for new changes in project state"]
        
        tension_analysis = await self.analyze_graph_tensions(tensions)
        
        if not tension_analysis.get("evolution_needed", False):
            return ["No immediate actions needed - graph is stable"]
        
        # Extract actions from LLM analysis
        analysis_text = tension_analysis.get("analysis", "")
        
        # Simple parsing - look for action-oriented sentences
        lines = analysis_text.split('\\n')
        actions = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['should', 'need', 'recommend', 'must', 'action']):
                if len(line) > 10 and len(line) < 200:  # Reasonable action length
                    actions.append(line)
        
        return actions[:5] if actions else ["Analyze graph tensions and plan evolution"]
    
    async def _handle_pull_more_request(self, query: str, existing_context: List[Dict]) -> str:
        """Handle LLM request for more context"""
        # Parse what kind of context is needed from query
        # For now, add more recent changes
        additional_nodes = self.graph.top_k_changed(k=3)
        
        for node_id in additional_nodes:
            if len(existing_context) >= self.budget.max_cards:
                break
            card = self.graph.get_llm_card(node_id)
            if card and not any(node_id in ctx.get("content", "") for ctx in existing_context):
                existing_context.append({
                    "type": f"{card.type} Card (Additional)",
                    "content": card.to_json()
                })
        
        # Now process with more context
        system_prompt = """You are a project intelligence assistant with additional context.
        Use the provided graph context to answer the question comprehensively."""
        
        return await self.llm.generate_response(query, existing_context, system_prompt)
    
    def _generate_delivery_frame(self) -> str:
        """Generate real delivery frame with actual metrics from graph"""
        from .ontology import NodeType, Status
        
        # Get real counts from graph
        all_nodes = list(self.graph.nodes.values())
        
        # Count by status
        status_counts = {}
        for status in Status:
            status_counts[status.value] = len([n for n in all_nodes if n.state.status == status])
        
        # Count by type
        type_counts = {}
        for node_type in [NodeType.REQUIREMENT, NodeType.TEST, NodeType.CODEMODULE]:
            type_counts[node_type.value] = len([n for n in all_nodes if n.type == node_type])
        
        # Coverage ratios
        coverage = self.graph.coverage_ratio()
        
        # Recent deltas (actual changes)
        recent_changes = []
        changed_nodes = self.graph.top_k_changed(k=self.budget.max_frame_items)
        for node_id in changed_nodes:
            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]
                recent_changes.append({
                    "id": node_id,
                    "type": node.type.value,
                    "title": node.title[:50],
                    "change": node.state.change_summary or "Modified"
                })
        
        frame = {
            "timestamp": self.graph.hot_state.last_pulse_time,
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "coverage_ratios": coverage,
            "recent_deltas": recent_changes,
            "hot_cache_size": len(self.graph.hot_state.hot_cards),
            "total_nodes": len(all_nodes)
        }
        
        return json.dumps(frame, indent=2)
    
    async def generate_advice_frame(self, species_id: str, instance_id: str, phi_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured advice frame for RE_ware consciousness using AdviceInput JSON"""
        # Get phi signals from graph (includes all required signals)
        phi_signals = self.graph.phi_signals() if hasattr(self.graph, 'phi_signals') else {
            "coverage_ratio": 0.0,
            "changed_nodes": 0,
            "uncovered_requirements": 0,
            "open_p0_issues": 0,
            "context_budget_used": 0,
            "entropy_hint": 0.0
        }
        
        # Select cards using context budget
        selected_nodes, budget_used = self.budget.select_cards(self.graph, {"ask": "advise"})
        phi_signals["context_budget_used"] = budget_used
        
        # Get LLM cards for selected nodes
        cards = []
        for node_id in selected_nodes:
            card = self.graph.get_llm_card(node_id)
            if card:
                try:
                    card_data = json.loads(card.to_json())
                    cards.append(card_data)
                except json.JSONDecodeError:
                    cards.append({"id": node_id, "error": "card_serialization_failed"})
        
        # Generate pulse data
        pulse_data = {}
        if hasattr(self.graph, 'pulse'):
            try:
                pulse = self.graph.pulse(phi_state)
                pulse_data = json.loads(pulse.to_json())
            except Exception as e:
                pulse_data = {
                    "changed_nodes": list(self.graph.hot_state.changed_nodes)[:5] if hasattr(self.graph, 'hot_state') else [],
                    "error": f"pulse_generation_failed: {str(e)[:50]}"
                }
        
        # Build AdviceInput JSON structure
        advice_input = {
            "identity": {"species_id": species_id, "instance_id": instance_id},
            "omega": {
                "dod_predicates": ["spec_has_tests", "no_open_P0"],
                "two_key_required": ["merge_main", "release_prod"]
            },
            "phi": {
                "phi0": phi_state.get("phi0", False),
                "signals": phi_signals
            },
            "pulse": pulse_data,
            "cards": cards,
            "tools": {
                "github.issue": False,
                "github.pr": False, 
                "git.branch": True,
                "fs.write": True,
                "graph.update": True
            },
            "ask": "advise",
            "budget": {"max_cards": self.budget.max_cards}
        }
        
        try:
            # Call LLM with AdviceInput JSON as user prompt
            response = await self.llm.generate_response(
                json.dumps(advice_input, separators=(',', ':')), 
                [], 
                RE_SYSTEM_PROMPT
            )
            
            # Try to parse LLM response as JSON
            advice_frame = json.loads(response)
            
            # Validate basic structure
            if not isinstance(advice_frame, dict) or "self" not in advice_frame:
                raise ValueError("Invalid advice frame structure")
                
            return advice_frame
            
        except (json.JSONDecodeError, Exception) as e:
            # Get tensions for fallback generation
            tensions = self.graph.detect_graph_tensions() if hasattr(self.graph, 'detect_graph_tensions') else []
            high_severity_count = len([t for t in tensions if t.get('severity') == 'high'])
            
            # Determine status
            if high_severity_count > 5:
                status = "red"
            elif high_severity_count > 2 or len(tensions) > 10:
                status = "yellow"  
            else:
                status = "green"
            
            # Build top tensions
            top_tensions = []
            for i, tension in enumerate(tensions[:3]):
                tension_nodes = []
                for key in ("source_node", "affected_node", "node_id"):
                    if key in tension:
                        tension_nodes.append(tension[key])
                for n in tension.get("context_nodes", [])[:2]:
                    tension_nodes.append(n)
                
                top_tensions.append({
                    "id": f"t-{i+1}-{tension.get('type', 'unknown')}",
                    "severity": tension.get('severity', 'medium'),
                    "why": tension.get('message', tension.get('description', 'No description')),
                    "nodes": tension_nodes[:5]
                })
            
            # Fallback to manual frame construction
            return {
                "self": {"species_id": species_id, "instance_id": instance_id},
                "phi": {
                    "phi0": phi_state.get("phi0", False),
                    "signals": phi_signals,
                    "summary": f"φ₀={phi_state.get('phi0', 0.0):.3f}, {len(tensions)} tensions"
                },
                "judgement": {
                    "status": status,
                    "reasons": [f"{len(tensions)} ontological tensions detected", f"High severity: {high_severity_count}"],
                    "top_tensions": top_tensions
                },
                "actions": [
                    {
                        "kind": "graph.update",
                        "title": "Address ontological tensions",
                        "body": f"Resolve {len(tensions)} detected tensions in the knowledge graph",
                        "targets": [{"type": "tension", "id": f"batch-{len(tensions)}"}],
                        "params": {"batch_size": min(len(tensions), 5)},
                        "idempotency_key": hashlib.md5(f"tensions-{len(tensions)}-{status}".encode()).hexdigest()[:10]
                    }
                ],
                "need_more": [],
                "notes": [f"LLM parsing failed: {str(e)[:50]}", "Using fallback frame generation", "AdviceInput JSON structure used"]
            }

# Factory function
def create_llm_interface() -> LLMInterface:
    """Create LLM interface with environment configuration"""
    return LLMInterface()

def create_project_intelligence(ontology_graph) -> ProjectIntelligence:
    """Create project intelligence system"""
    llm = create_llm_interface()
    return ProjectIntelligence(ontology_graph, llm)