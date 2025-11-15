"""
Tests for SCAR algorithm
"""

import pytest
from src.core.network_topology import NetworkTopology
from src.core.scar_algorithm import SCARAlgorithm


def test_scar_initialization():
    """Test SCAR algorithm initialization."""
    config = {'topology_type': 'ring', 'num_nodes': 10}
    topology = NetworkTopology(config)
    
    scar = SCARAlgorithm(topology)
    assert scar is not None
    assert scar.topology == topology


def test_route_selection():
    """Test route selection."""
    config = {'topology_type': 'ring', 'num_nodes': 10}
    topology = NetworkTopology(config)
    scar = SCARAlgorithm(topology)
    
    nodes = topology.get_nodes()
    if len(nodes) >= 2:
        source = nodes[0]
        destination = nodes[5]
        
        route = scar.get_route(source, destination, current_time=0)
        assert route is not None
        assert len(route) >= 2
        assert route[0] == source
        assert route[-1] == destination


def test_route_table():
    """Test route table management."""
    config = {'topology_type': 'ring', 'num_nodes': 10}
    topology = NetworkTopology(config)
    scar = SCARAlgorithm(topology)
    
    nodes = topology.get_nodes()
    if len(nodes) >= 2:
        source = nodes[0]
        destination = nodes[5]
        
        route = scar.get_route(source, destination, current_time=0)
        route_table = scar.get_route_table()
        
        assert (source, destination) in route_table
        assert route_table[(source, destination)] == route


def test_link_load_update():
    """Test link load updates."""
    config = {'topology_type': 'ring', 'num_nodes': 10}
    topology = NetworkTopology(config)
    scar = SCARAlgorithm(topology)
    
    edges = topology.get_edges()
    if edges:
        u, v = edges[0]
        scar.update_link_load(u, v, 50.0)
        
        utilization = topology.get_link_utilization(u, v)
        assert utilization >= 0.0

