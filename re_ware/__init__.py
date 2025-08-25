"""
RE_ware: Self-Evolving Software Lifecycle Management Library
============================================================

A reusable library for managing complete software project lifecycles through:
- Ontological graph representation
- LLM-powered intelligence 
- Self-evolving capabilities
- Autonomous project monitoring

Usage:
    # Use the evolution CLI interface instead:
    # python -m re_ware.evolution monitor
    # python -m re_ware.evolution frames
    # python -m re_ware.evolution gates
"""

# ProjectAgent deprecated - use evolution CLI interface
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