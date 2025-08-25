#!/usr/bin/env python3
"""
Quick test to verify coverage calculation fix
"""
import sys
sys.path.append('.')

from re_ware.ontology import (
    create_ontology_with_gene, NodeType, RelationType,
    create_node, create_edge
)

def test_coverage_fix():
    """Test that VERIFIES relationships work for coverage"""
    print("🔬 Testing coverage calculation fix...")
    
    # Create ontology
    ontology = create_ontology_with_gene("project_manager")
    
    # Create a requirement node
    req_node = create_node(NodeType.REQUIREMENT, "Test Coverage Requirement")
    print(f"Created requirement: {req_node.id}")
    ontology.add_node(req_node)
    
    # Create a test node  
    test_node = create_node(NodeType.TEST, "Test Coverage Test")
    print(f"Created test: {test_node.id}")
    ontology.add_node(test_node)
    
    # Create VERIFIES relationship
    verifies_edge = create_edge(RelationType.VERIFIES, test_node.id, req_node.id)
    print(f"Created VERIFIES edge: {verifies_edge.id}")
    success = ontology.add_edge(verifies_edge)
    print(f"Edge added successfully: {success}")
    
    # Check coverage calculation
    coverage = ontology.coverage_ratio()
    print(f"Coverage ratios: {coverage}")
    
    phi_signals = ontology.phi_signals()
    print(f"Phi signals coverage_ratio: {phi_signals['coverage_ratio']}")
    
    if phi_signals['coverage_ratio'] > 0:
        print("✅ COVERAGE FIX WORKS!")
        return True
    else:
        print("❌ Coverage still 0")
        return False

if __name__ == "__main__":
    test_coverage_fix()