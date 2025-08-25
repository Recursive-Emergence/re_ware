"""
Tests for RE_ware evolution engine
"""
import pytest
import asyncio
from pathlib import Path
from re_ware.evolution import REWareState, REWareInteractive
from re_ware.ontology import NodeType, create_ontology_with_gene


def test_reware_state_creation():
    """Test REWareState basic functionality"""
    state = REWareState()
    
    assert state.phi0 == 0.0
    assert state.phi_coherence == 0.0
    assert state.stability_check is False
    assert state.cycles_completed == 0


def test_reware_state_update():
    """Test REWareState update functionality"""
    state = REWareState()
    
    # Update with new values
    state.update(phi0=0.5, coherence=0.8, stability=True)
    
    assert state.phi0 == 0.5
    assert state.phi_coherence == 0.8
    assert state.stability_check is True
    assert state.cycles_completed == 1
    assert state.last_update > 0


def test_reware_interactive_creation():
    """Test REWareInteractive agent creation"""
    project_root = Path("/tmp/test_project")
    agent = REWareInteractive(project_root, schema_name="project_manager")
    
    assert agent.project_root == project_root
    assert agent.schema_name == "project_manager"
    assert agent.running is True
    assert agent.ontology is None  # Not initialized yet
    assert agent.sensor_hub is None  # Not initialized yet


def test_advice_display_formatting():
    """Test advice display helper method"""
    project_root = Path("/tmp/test_project")
    agent = REWareInteractive(project_root)
    
    # Test with simple advice
    advice_frame = {
        "judgement": "Test assessment",
        "actions": ["Action 1", "Action 2"]
    }
    
    # This should not raise an exception
    agent._display_advice(advice_frame)


def test_advice_display_with_dict_actions():
    """Test advice display with complex action dictionaries"""
    project_root = Path("/tmp/test_project")
    agent = REWareInteractive(project_root)
    
    # Test with dictionary actions
    advice_frame = {
        "judgement": "Test assessment",
        "actions": [
            {"title": "Test Action 1", "body": "Test description"},
            {"description": "Test Action 2"},
            "Simple string action"
        ]
    }
    
    # This should not raise an exception
    agent._display_advice(advice_frame)


@pytest.mark.asyncio
async def test_get_advice_with_caching_no_agent():
    """Test get_advice_with_caching when agent is not initialized"""
    project_root = Path("/tmp/test_project")
    agent = REWareInteractive(project_root)
    
    result = await agent.get_advice_with_caching()
    assert "error" in result
    assert "not initialized" in result["error"].lower()


def test_recent_advice_no_ontology():
    """Test _get_recent_advice when ontology is not set"""
    project_root = Path("/tmp/test_project")
    agent = REWareInteractive(project_root)
    
    result = agent._get_recent_advice()
    assert result is None


def test_store_advice_node_no_ontology():
    """Test _store_advice_node when ontology is not set"""
    project_root = Path("/tmp/test_project")
    agent = REWareInteractive(project_root)
    
    advice_frame = {"judgement": "test", "actions": []}
    phi_signals = {"coverage_ratio": 0.0}
    
    result = agent._store_advice_node(advice_frame, phi_signals)
    assert result is None