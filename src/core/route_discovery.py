"""
Route Discovery Mechanism

This module provides route discovery functionality to find candidate paths
between source and destination nodes.
"""

from typing import List, Dict, Optional, Tuple
import networkx as nx
from functools import lru_cache
from src.core.network_topology import NetworkTopology
from src.core.congestion_monitor import CongestionMonitor


class RouteDiscovery:
    """Discovers routes between nodes in the network."""
    
    def __init__(self, topology: NetworkTopology, congestion_monitor: CongestionMonitor):
        """
        Initialize route discovery.
        
        Args:
            topology: NetworkTopology instance
            congestion_monitor: CongestionMonitor instance
        """
        self.topology = topology
        self.congestion_monitor = congestion_monitor
        
        # Caching for performance
        self._path_cache: Dict[Tuple[int, int, str], Optional[List[int]]] = {}
        self._evaluation_cache: Dict[Tuple[Tuple[int, ...], float, float], Dict] = {}
        self._weighted_graph_cache = None
        self._weighted_graph_dirty = True
    
    def find_shortest_path(self, source: int, destination: int, 
                          metric: str = 'hop') -> Optional[List[int]]:
        """
        Find shortest path using specified metric.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            metric: Path metric ('hop', 'latency', 'congestion')
        
        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        try:
            if metric == 'hop':
                path = nx.shortest_path(self.topology.graph, source, destination)
            elif metric == 'latency':
                path = nx.shortest_path(self.topology.graph, source, destination, 
                                       weight='latency')
            elif metric == 'congestion':
                # Use utilization as weight
                path = self._find_least_congested_path(source, destination)
            else:
                path = nx.shortest_path(self.topology.graph, source, destination)
            
            return path
        except nx.NetworkXNoPath:
            return None
    
    def find_k_shortest_paths(self, source: int, destination: int, 
                              k: int = 3, metric: str = 'hop') -> List[List[int]]:
        """
        Find k shortest paths between source and destination.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            k: Number of paths to find
            metric: Path metric
        
        Returns:
            List of paths (each path is a list of node IDs)
        """
        # Limit k for performance
        k = min(k, 3)
        cache_key = (source, destination, metric, k)
        
        try:
            if metric == 'hop':
                paths = list(nx.shortest_simple_paths(self.topology.graph, 
                                                  source, destination))[:k]
            elif metric == 'latency':
                paths = list(nx.shortest_simple_paths(self.topology.graph, 
                                                     source, destination, 
                                                     weight='latency'))[:k]
            elif metric == 'congestion':
                paths = self._find_k_least_congested_paths(source, destination, k)
            else:
                paths = list(nx.shortest_simple_paths(self.topology.graph, 
                                                     source, destination))[:k]
            
            return paths
        except (nx.NetworkXNoPath, StopIteration):
            return []
    
    def _get_weighted_graph(self):
        """Get or create weighted graph with congestion scores (cached)."""
        if self._weighted_graph_cache is None or self._weighted_graph_dirty:
            # Create weighted graph with congestion scores
            self._weighted_graph_cache = self.topology.graph.copy()
            for u, v in self._weighted_graph_cache.edges():
                congestion_score = self.congestion_monitor.get_link_congestion_score(u, v)
                self._weighted_graph_cache[u][v]['congestion_weight'] = congestion_score
            self._weighted_graph_dirty = False
        return self._weighted_graph_cache
    
    def _invalidate_cache(self):
        """Invalidate cached weighted graph when congestion changes."""
        self._weighted_graph_dirty = True
        # Clear path cache for congestion-based paths
        keys_to_remove = [k for k in self._path_cache.keys() if k[2] == 'congestion']
        for k in keys_to_remove:
            del self._path_cache[k]
    
    def _find_least_congested_path(self, source: int, destination: int) -> Optional[List[int]]:
        """Find path with minimum congestion."""
        cache_key = (source, destination, 'congestion')
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        try:
            weighted_graph = self._get_weighted_graph()
            path = nx.shortest_path(weighted_graph, source, destination, 
                                   weight='congestion_weight')
            self._path_cache[cache_key] = path
            return path
        except nx.NetworkXNoPath:
            self._path_cache[cache_key] = None
            return None
    
    def _find_k_least_congested_paths(self, source: int, destination: int, 
                                      k: int) -> List[List[int]]:
        """Find k least congested paths."""
        try:
            weighted_graph = self._get_weighted_graph()
            # Limit k to 3 for performance (was 5)
            k = min(k, 3)
            paths = list(nx.shortest_simple_paths(weighted_graph, source, destination, 
                                                 weight='congestion_weight'))[:k]
            return paths
        except (nx.NetworkXNoPath, StopIteration):
            return []
    
    def evaluate_path(self, path: List[int], 
                     congestion_weight: float = 0.6,
                     stability_weight: float = 0.4) -> Dict:
        """
        Evaluate a path based on multiple criteria.
        
        Args:
            path: List of node IDs forming the path
            congestion_weight: Weight for congestion metric (0.0 to 1.0)
            stability_weight: Weight for stability metric (0.0 to 1.0)
        
        Returns:
            Dictionary with path evaluation metrics
        """
        if len(path) < 2:
            return {
                'path': path,
                'hop_count': 0,
                'total_latency': 0.0,
                'congestion_score': 0.0,
                'stability_score': 0.0,
                'combined_score': 0.0
            }
        
        hop_count = len(path) - 1
        total_latency = self.topology.get_path_cost(path, metric='latency')
        congestion_score = self.congestion_monitor.get_path_congestion_score(path)
        
        # Normalize scores (lower is better for congestion)
        # For combined score, we'll use inverse of congestion
        normalized_congestion = 1.0 - congestion_score
        
        # Stability score will be provided by stability mechanism
        # For now, use a default value
        stability_score = 1.0
        
        # Combined score (higher is better)
        combined_score = (normalized_congestion * congestion_weight + 
                         stability_score * stability_weight)
        
        return {
            'path': path,
            'hop_count': hop_count,
            'total_latency': total_latency,
            'congestion_score': congestion_score,
            'stability_score': stability_score,
            'combined_score': combined_score
        }
    
    def find_best_path(self, source: int, destination: int,
                      congestion_weight: float = 0.6,
                      stability_weight: float = 0.4,
                      stability_scores: Optional[Dict[Tuple[int, int], float]] = None) -> Optional[Dict]:
        """
        Find best path considering both congestion and stability.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            congestion_weight: Weight for congestion in path selection
            stability_weight: Weight for stability in path selection
            stability_scores: Dictionary mapping (node1, node2) to stability scores
        
        Returns:
            Dictionary with best path and evaluation metrics
        """
        # Find candidate paths - reduced from k=5 to k=3 for performance
        candidate_paths = self.find_k_shortest_paths(source, destination, k=3, metric='hop')
        
        if not candidate_paths:
            return None
        
        # Evaluate each candidate path
        best_path = None
        best_score = -1.0
        
        for path in candidate_paths:
            # Cache evaluation results
            path_tuple = tuple(path)
            cache_key = (path_tuple, congestion_weight, stability_weight)
            
            if cache_key in self._evaluation_cache:
                evaluation = self._evaluation_cache[cache_key].copy()
            else:
                evaluation = self.evaluate_path(path, congestion_weight, stability_weight)
                self._evaluation_cache[cache_key] = evaluation.copy()
            
            # Update stability score if provided
            if stability_scores:
                path_stability = self._calculate_path_stability(path, stability_scores)
                evaluation['stability_score'] = path_stability
                evaluation['combined_score'] = (
                    (1.0 - evaluation['congestion_score']) * congestion_weight +
                    path_stability * stability_weight
                )
            
            if evaluation['combined_score'] > best_score:
                best_score = evaluation['combined_score']
                best_path = evaluation
        
        return best_path
    
    def _calculate_path_stability(self, path: List[int], 
                                  stability_scores: Dict[Tuple[int, int], float]) -> float:
        """Calculate average stability score for a path."""
        if len(path) < 2:
            return 0.0
        
        scores = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = (min(u, v), max(u, v))
            if link in stability_scores:
                scores.append(stability_scores[link])
        
        return sum(scores) / len(scores) if scores else 0.0

