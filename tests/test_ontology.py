"""
Basic tests for RE_ware ontology module
"""
import pytest
from re_ware.ontology import (
    NodeType, RelationType, Status,
    create_node, create_edge, create_ontology_with_gene
)


def test_create_node_basic():
    """Test basic node creation"""
    node = create_node(NodeType.REQUIREMENT, "Test Requirement")
    
    assert node.type == NodeType.REQUIREMENT
    assert node.title == "Test Requirement"
    assert node.state.status == Status.DRAFT
    assert node.id.startswith("requirement:")


def test_create_edge_basic():
    """Test basic edge creation"""
    edge = create_edge(
        RelationType.IMPLEMENTS, 
        "from_node_id", 
        "to_node_id"
    )
    
    assert edge.relation == RelationType.IMPLEMENTS
    assert edge.from_node == "from_node_id" 
    assert edge.to_node == "to_node_id"
    assert edge.id.startswith("implements:")


def test_ontology_creation():
    """Test ontology creation with gene schema"""
    ontology = create_ontology_with_gene("project_manager")
    
    assert ontology is not None
    assert hasattr(ontology, 'nodes')
    assert hasattr(ontology, 'edges')
    assert hasattr(ontology, 'gene')


def test_node_content_and_meta():
    """Test node with content and metadata"""
    content = {"priority": "high", "category": "functional"}
    meta = {"created_by": "test_system"}
    
    node = create_node(
        NodeType.REQUIREMENT,
        "Advanced Test Requirement", 
        content=content,
        meta=meta
    )
    
    assert node.content["priority"] == "high"
    assert node.content["category"] == "functional"
    assert node.meta["created_by"] == "test_system"


def test_node_state_management():
    """Test node state tracking"""
    node = create_node(NodeType.CODEMODULE, "test_module.py")
    
    # Initial state
    assert node.state.status == Status.DRAFT
    assert node.state.version == "0.1"
    
    # Update state
    node.touch("Added new functionality", by="developer")
    assert node.state.change_summary == "Added new functionality"
    assert node.state.provenance["by"] == "developer"