"""
Network Topology Representation

This module provides network graph representation and management using NetworkX.
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
import yaml


class NetworkTopology:
    """Represents a network topology with nodes and links."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize network topology.
        
        Args:
            config: Configuration dictionary with network parameters
        """
        self.graph = nx.Graph()
        self.config = config or {}
        self._initialize_topology()
    
    def _initialize_topology(self):
        """Initialize the network topology based on configuration."""
        topology_type = self.config.get('topology_type', 'random')
        num_nodes = self.config.get('num_nodes', 20)
        
        if topology_type == 'random':
            self._create_random_topology(num_nodes)
        elif topology_type == 'grid':
            self._create_grid_topology(num_nodes)
        elif topology_type == 'ring':
            self._create_ring_topology(num_nodes)
        elif topology_type == 'star':
            self._create_star_topology(num_nodes)
        elif topology_type == 'file':
            self._load_from_file(self.config.get('topology_file'))
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
        
        # Initialize link attributes
        self._initialize_link_attributes()
    
    def _create_random_topology(self, num_nodes: int):
        """Create a random network topology."""
        # Create a connected random graph
        # Using Erdos-Renyi model with probability to ensure connectivity
        p = 2.0 / num_nodes  # Probability for edge creation
        self.graph = nx.erdos_renyi_graph(num_nodes, p, seed=42)
        
        # Ensure connectivity
        if not nx.is_connected(self.graph):
            # Add edges to make it connected
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                self.graph.add_edge(node1, node2)
    
    def _create_grid_topology(self, num_nodes: int):
        """Create a grid network topology."""
        # Calculate grid dimensions
        side = int(np.sqrt(num_nodes))
        if side * side != num_nodes:
            side = int(np.ceil(np.sqrt(num_nodes)))
        
        self.graph = nx.grid_2d_graph(side, side)
        # Convert to node-labeled graph
        mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        self.graph = nx.relabel_nodes(self.graph, mapping)
    
    def _create_ring_topology(self, num_nodes: int):
        """Create a ring network topology."""
        self.graph = nx.cycle_graph(num_nodes)
    
    def _create_star_topology(self, num_nodes: int):
        """Create a star network topology."""
        self.graph = nx.star_graph(num_nodes - 1)
    
    def _load_from_file(self, filepath: str):
        """Load topology from file."""
        # This would load from a file format (e.g., GraphML, JSON)
        # For now, raise NotImplementedError
        raise NotImplementedError("File loading not yet implemented")
    
    def _initialize_link_attributes(self):
        """Initialize link attributes (capacity, latency, load)."""
        capacity_min = self.config.get('link_capacity_min', 10)
        capacity_max = self.config.get('link_capacity_max', 100)
        initial_latency = self.config.get('initial_latency', 5)
        
        for u, v in self.graph.edges():
            # Random capacity within range
            capacity = random.uniform(capacity_min, capacity_max)
            
            # Set link attributes
            self.graph[u][v]['capacity'] = capacity  # Mbps
            self.graph[u][v]['current_load'] = 0.0  # Mbps
            self.graph[u][v]['utilization'] = 0.0  # 0.0 to 1.0
            self.graph[u][v]['latency'] = initial_latency  # ms
            self.graph[u][v]['queue_length'] = 0  # packets
            self.graph[u][v]['is_congested'] = False
    
    def add_node(self, node_id: int, **attributes):
        """Add a node to the network."""
        self.graph.add_node(node_id, **attributes)
    
    def add_link(self, node1: int, node2: int, capacity: float, latency: float = 5.0):
        """
        Add a link between two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
            capacity: Link capacity in Mbps
            latency: Link latency in ms
        """
        self.graph.add_edge(node1, node2)
        self.graph[node1][node2]['capacity'] = capacity
        self.graph[node1][node2]['current_load'] = 0.0
        self.graph[node1][node2]['utilization'] = 0.0
        self.graph[node1][node2]['latency'] = latency
        self.graph[node1][node2]['queue_length'] = 0
        self.graph[node1][node2]['is_congested'] = False
    
    def get_link_utilization(self, node1: int, node2: int) -> float:
        """Get utilization of a link."""
        if not self.graph.has_edge(node1, node2):
            return 0.0
        return self.graph[node1][node2].get('utilization', 0.0)
    
    def get_link_capacity(self, node1: int, node2: int) -> float:
        """Get capacity of a link."""
        if not self.graph.has_edge(node1, node2):
            return 0.0
        return self.graph[node1][node2].get('capacity', 0.0)
    
    def get_link_load(self, node1: int, node2: int) -> float:
        """Get current load on a link."""
        if not self.graph.has_edge(node1, node2):
            return 0.0
        return self.graph[node1][node2].get('current_load', 0.0)
    
    def update_link_load(self, node1: int, node2: int, load: float):
        """
        Update the load on a link.
        
        Args:
            node1: First node ID
            node2: Second node ID
            load: New load value in Mbps
        """
        if not self.graph.has_edge(node1, node2):
            return
        
        self.graph[node1][node2]['current_load'] = max(0.0, load)
        capacity = self.graph[node1][node2]['capacity']
        if capacity > 0:
            self.graph[node1][node2]['utilization'] = min(1.0, load / capacity)
        else:
            self.graph[node1][node2]['utilization'] = 0.0
    
    def is_link_congested(self, node1: int, node2: int, threshold: float = 0.7) -> bool:
        """
        Check if a link is congested.
        
        Args:
            node1: First node ID
            node2: Second node ID
            threshold: Congestion threshold (0.0 to 1.0)
        
        Returns:
            True if link utilization exceeds threshold
        """
        utilization = self.get_link_utilization(node1, node2)
        is_congested = utilization > threshold
        if self.graph.has_edge(node1, node2):
            self.graph[node1][node2]['is_congested'] = is_congested
        return is_congested
    
    def get_all_congested_links(self, threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        Get all congested links in the network.
        
        Args:
            threshold: Congestion threshold
        
        Returns:
            List of (node1, node2) tuples for congested links
        """
        congested = []
        for u, v in self.graph.edges():
            if self.is_link_congested(u, v, threshold):
                congested.append((u, v))
        return congested
    
    def get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node."""
        return list(self.graph.neighbors(node))
    
    def get_nodes(self) -> List[int]:
        """Get all node IDs."""
        return list(self.graph.nodes())
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """Get all edges."""
        return list(self.graph.edges())
    
    def get_path_cost(self, path: List[int], metric: str = 'hop') -> float:
        """
        Calculate cost of a path.
        
        Args:
            path: List of node IDs forming the path
            metric: Cost metric ('hop', 'latency', 'congestion', 'utilization')
        
        Returns:
            Path cost
        """
        if len(path) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self.graph.has_edge(u, v):
                return float('inf')
            
            if metric == 'hop':
                cost += 1
            elif metric == 'latency':
                cost += self.graph[u][v].get('latency', 0)
            elif metric == 'congestion':
                utilization = self.graph[u][v].get('utilization', 0.0)
                cost += utilization
            elif metric == 'utilization':
                cost += self.graph[u][v].get('utilization', 0.0)
            else:
                cost += 1
        
        return cost
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_links': self.graph.number_of_edges(),
            'average_degree': 2 * self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
            'is_connected': nx.is_connected(self.graph),
        }
        
        if stats['is_connected']:
            stats['diameter'] = nx.diameter(self.graph)
            stats['average_path_length'] = nx.average_shortest_path_length(self.graph)
        
        # Link statistics
        utilizations = [self.graph[u][v].get('utilization', 0.0) 
                       for u, v in self.graph.edges()]
        if utilizations:
            stats['avg_utilization'] = np.mean(utilizations)
            stats['max_utilization'] = np.max(utilizations)
            stats['min_utilization'] = np.min(utilizations)
        
        return stats


def load_topology_from_config(config_file: str) -> NetworkTopology:
    """
    Load network topology from configuration file.
    
    Args:
        config_file: Path to YAML configuration file
    
    Returns:
        NetworkTopology instance
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    network_config = config.get('network', {})
    return NetworkTopology(network_config)

