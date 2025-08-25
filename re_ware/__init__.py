"""
RE_ware: Self-Evolving Software Lifecycle Management Library
============================================================

A reusable library for managing complete software project lifecycles through:
- Ontological graph representation
- LLM-powered intelligence 
- Self-evolving capabilities
- Autonomous project monitoring

Usage:
    # Use the main CLI interface:
    # python evolve.py status
    # python evolve.py advice  
    # python evolve.py tick
"""

# Core RE_ware library exports
from .ontology import (
    OntologyGraph, NodeType, RelationType, Status, Criticality,
    GraphNode, GraphEdge, LLMCard, create_node, create_edge
)
from .llm_integration import LLMInterface, ProjectIntelligence, create_llm_interface, create_project_intelligence
from .core import REWare, re_ware
from .tool_registry import ToolRegistry

__version__ = "0.1.0"
__author__ = "RE_ware Development Team"

__all__ = [
    # "ProjectAgent",  # Deprecated - use evolution CLI
    # "create_project_agent",  # Deprecated - use evolution CLI
    "OntologyGraph",
    "NodeType",
    "RelationType", 
    "Status",
    "Criticality",
    "GraphNode",
    "GraphEdge",
    "LLMCard",
    "create_node",
    "create_edge",
    "LLMInterface",
    "ProjectIntelligence",
    "create_llm_interface",
    "create_project_intelligence",
    "REWare",
    "re_ware",
    "ToolRegistry"
]