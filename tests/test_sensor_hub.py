"""
Basic tests for RE_ware sensor hub
"""
import pytest
from re_ware.sensor_hub import SensorHub, DomainEvent
from re_ware.ontology import create_ontology_with_gene, NodeType


def test_domain_event_creation():
    """Test basic domain event creation"""
    event = DomainEvent(
        source="test",
        kind="create",
        path="test_file.py",
        actor="test_system"
    )
    
    assert event.source == "test"
    assert event.kind == "create"
    assert event.path == "test_file.py"
    assert event.actor == "test_system"
    assert event.event_id is not None
    assert len(event.event_id) == 12  # MD5 hash truncated to 12 chars


def test_sensor_hub_creation():
    """Test sensor hub can be created with ontology"""
    ontology = create_ontology_with_gene("project_manager")
    hub = SensorHub(ontology)
    
    assert hub.graph == ontology
    assert hub.sensors == {}
    assert hub.event_buffer == []
    assert hub.mapping_rules is not None


def test_deduplication_logic():
    """Test the new find_existing_node_for_path method"""
    ontology = create_ontology_with_gene("project_manager")
    hub = SensorHub(ontology)
    
    # Should return None when no nodes exist
    result = hub._find_existing_node_for_path("test.py", NodeType.CODEMODULE)
    assert result is None


def test_consolidate_duplicate_nodes():
    """Test duplicate node consolidation"""
    ontology = create_ontology_with_gene("project_manager") 
    hub = SensorHub(ontology)
    
    # Should handle empty graph gracefully
    stats = hub.consolidate_duplicate_nodes()
    assert stats["merged"] == 0
    assert stats["removed"] == 0


def test_event_id_uniqueness():
    """Test that different events generate different IDs"""
    event1 = DomainEvent(source="git", kind="modify", path="file1.py")
    event2 = DomainEvent(source="git", kind="modify", path="file2.py")
    event3 = DomainEvent(source="fs", kind="modify", path="file1.py")
    
    # Different paths should have different IDs
    assert event1.event_id != event2.event_id
    
    # Different sources should have different IDs
    assert event1.event_id != event3.event_id
    
    # Same parameters should have same ID (idempotency)
    event4 = DomainEvent(source="git", kind="modify", path="file1.py", sha="abc123")
    event5 = DomainEvent(source="git", kind="modify", path="file1.py", sha="abc123")
    assert event4.event_id == event5.event_id