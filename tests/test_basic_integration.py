"""
Basic integration tests for RE_ware system
"""
import pytest
from re_ware.ontology import create_ontology_with_gene, NodeType, RelationType, create_node, create_edge


def test_ontology_node_integration():
    """Test creating nodes and adding to ontology"""
    ontology = create_ontology_with_gene("project_manager")
    
    # Create a requirement node
    req_node = create_node(NodeType.REQUIREMENT, "Test Integration Requirement")
    success = ontology.add_node(req_node)
    
    assert success is True
    assert req_node.id in ontology.nodes
    assert len(ontology.nodes) == 1


def test_ontology_edge_integration():
    """Test creating edges between nodes"""
    ontology = create_ontology_with_gene("project_manager")
    
    # Create two nodes
    req_node = create_node(NodeType.REQUIREMENT, "Test Requirement")
    test_node = create_node(NodeType.TEST, "Test Case")
    
    ontology.add_node(req_node)
    ontology.add_node(test_node)
    
    # Create edge
    edge = create_edge(RelationType.VERIFIES, test_node.id, req_node.id)
    success = ontology.add_edge(edge)
    
    assert success is True
    assert edge.id in ontology.edges
    assert len(ontology.edges) == 1


def test_phi_signals_basic():
    """Test phi signals generation"""
    ontology = create_ontology_with_gene("project_manager")
    
    # Add some nodes
    req_node = create_node(NodeType.REQUIREMENT, "Requirement 1")
    test_node = create_node(NodeType.TEST, "Test 1")
    
    ontology.add_node(req_node)
    ontology.add_node(test_node)
    
    # Test phi signals
    signals = ontology.phi_signals()
    
    assert isinstance(signals, dict)
    assert "coverage_ratio" in signals
    assert "changed_nodes" in signals
    assert signals["coverage_ratio"] >= 0.0
    assert signals["changed_nodes"] >= 0


def test_node_state_tracking():
    """Test node state is tracked correctly"""
    ontology = create_ontology_with_gene("project_manager")
    
    node = create_node(NodeType.CODEMODULE, "test_module.py")
    ontology.add_node(node)
    
    # Node should be in hot state as changed
    assert node.id in ontology.hot_state.changed_nodes


def test_coverage_calculation_empty():
    """Test coverage calculation with empty ontology"""
    ontology = create_ontology_with_gene("project_manager")
    
    coverage = ontology.coverage_ratio()
    
    # Empty ontology should return empty coverage dict
    assert isinstance(coverage, dict)


def test_search_nodes_by_type():
    """Test searching nodes by type"""
    ontology = create_ontology_with_gene("project_manager")
    
    # Add different types
    req_node = create_node(NodeType.REQUIREMENT, "Requirement 1") 
    test_node = create_node(NodeType.TEST, "Test 1")
    code_node = create_node(NodeType.CODEMODULE, "Module 1")
    
    ontology.add_node(req_node)
    ontology.add_node(test_node)
    ontology.add_node(code_node)
    
    # Search by type
    reqs = ontology.search_nodes(node_type=NodeType.REQUIREMENT)
    tests = ontology.search_nodes(node_type=NodeType.TEST)
    
    assert len(reqs) == 1
    assert len(tests) == 1
    assert reqs[0].type == NodeType.REQUIREMENT
    assert tests[0].type == NodeType.TEST