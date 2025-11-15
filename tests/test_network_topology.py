"""
Tests for network topology
"""

import pytest
from src.core.network_topology import NetworkTopology


def test_network_creation():
    """Test network topology creation."""
    config = {'topology_type': 'random', 'num_nodes': 10}
    topology = NetworkTopology(config)
    
    assert topology.graph.number_of_nodes() == 10
    assert topology.graph.number_of_edges() > 0


def test_link_attributes():
    """Test link attribute initialization."""
    config = {'topology_type': 'ring', 'num_nodes': 5}
    topology = NetworkTopology(config)
    
    edges = topology.get_edges()
    assert len(edges) > 0
    
    for u, v in edges:
        assert 'capacity' in topology.graph[u][v]
        assert 'utilization' in topology.graph[u][v]
        assert 'latency' in topology.graph[u][v]


def test_link_utilization():
    """Test link utilization updates."""
    config = {'topology_type': 'ring', 'num_nodes': 5}
    topology = NetworkTopology(config)
    
    edges = topology.get_edges()
    if edges:
        u, v = edges[0]
        topology.update_link_load(u, v, 50.0)
        
        utilization = topology.get_link_utilization(u, v)
        assert 0.0 <= utilization <= 1.0


def test_congestion_detection():
    """Test congestion detection."""
    config = {'topology_type': 'ring', 'num_nodes': 5}
    topology = NetworkTopology(config)
    
    edges = topology.get_edges()
    if edges:
        u, v = edges[0]
        capacity = topology.get_link_capacity(u, v)
        
        # Set load above threshold
        threshold = 0.7
        load = capacity * threshold * 1.1
        topology.update_link_load(u, v, load)
        
        assert topology.is_link_congested(u, v, threshold)


def test_path_cost():
    """Test path cost calculation."""
    config = {'topology_type': 'ring', 'num_nodes': 5}
    topology = NetworkTopology(config)
    
    nodes = topology.get_nodes()
    if len(nodes) >= 3:
        path = [nodes[0], nodes[1], nodes[2]]
        cost = topology.get_path_cost(path, metric='hop')
        assert cost == 2  # 2 hops

