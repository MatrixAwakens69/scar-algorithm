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

# Forward declaration for simulator (to avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.simulation.network_simulator import NetworkSimulator


class RouteDiscovery:
    """Discovers routes between nodes in the network."""
    
    def __init__(self, topology: NetworkTopology, congestion_monitor: CongestionMonitor, simulator: Optional['NetworkSimulator'] = None):
        """
        Initialize route discovery.
        
        Args:
            topology: NetworkTopology instance
            congestion_monitor: CongestionMonitor instance
            simulator: Optional NetworkSimulator instance for queueing delay calculation
        """
        self.topology = topology
        self.congestion_monitor = congestion_monitor
        self.simulator = simulator  # For accessing node queues and calculating queueing delay
        
        # Caching for performance
        self._path_cache: Dict[Tuple[int, int, str], Optional[List[int]]] = {}
        self._evaluation_cache: Dict[Tuple[Tuple[int, ...], float, float], Dict] = {}
        self._weighted_graph_cache = None
        self._weighted_graph_dirty = True
        
        # Track when to invalidate evaluation cache (when congestion changes significantly)
        self._last_congestion_check: Dict[Tuple[int, int], float] = {}
    
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
    
    def _calculate_link_delay(self, u: int, v: int, packet_size: float = 15000.0) -> float:
        """
        Calculate total delay for a link (transmission + queueing).
        
        Args:
            u: Source node
            v: Destination node
            packet_size: Packet size in bytes
            
        Returns:
            Total delay in seconds (transmission time + queueing delay)
        """
        # Get link properties
        link_capacity = self.topology.get_link_capacity(u, v)
        link_latency = self.topology.graph[u][v].get('latency', 5.0)
        
        # Transmission time = packet_size / capacity + latency
        transmission_time = (packet_size * 8) / (link_capacity * 1e6) + (link_latency / 1000.0)
        
        # Calculate queueing delay at node u
        queueing_delay = 0.0
        if self.simulator and u in self.simulator.node_queues:
            queue_length = len(self.simulator.node_queues[u])
            if queue_length > 0 and link_capacity > 0:
                # Estimate queueing delay: packets in queue * transmission time per packet
                # This is a simplified model - in reality, queueing depends on arrival/service rates
                avg_packet_size = 15000.0  # bytes
                service_time = (avg_packet_size * 8) / (link_capacity * 1e6)
                queueing_delay = queue_length * service_time
        
        return transmission_time + queueing_delay
    
    def _get_weighted_graph(self):
        """Get or create weighted graph with total delay (transmission + queueing) as cost.
        
        Uses a cost function that considers:
        1. Transmission delay (packet_size / capacity + latency)
        2. Queueing delay (estimated from packets waiting at node)
        3. This ensures paths with lower total delay are preferred, even if longer
        """
        # Always rebuild weighted graph to get current delay values (dynamic routing)
        weighted_graph = self.topology.graph.copy()
        packet_size = 4000.0  # Standard packet size (increased by 10x)
        
        for u, v in weighted_graph.edges():
            # Calculate total delay (transmission + queueing)
            total_delay = self._calculate_link_delay(u, v, packet_size)
            
            # Use delay as cost (lower delay = lower cost = preferred)
            # Scale delay to reasonable range (seconds to cost units)
            weighted_graph[u][v]['congestion_weight'] = total_delay * 1000.0  # Convert to ms for better scaling
        return weighted_graph
    
    def _invalidate_cache(self):
        """Invalidate cached weighted graph when congestion changes."""
        self._weighted_graph_dirty = True
        # Clear path cache for congestion-based paths
        keys_to_remove = [k for k in self._path_cache.keys() if k[2] == 'congestion']
        for k in keys_to_remove:
            del self._path_cache[k]
        # Also clear evaluation cache since congestion affects path evaluation
        self._evaluation_cache.clear()
    
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
        
        # Calculate total path delay (transmission + queueing) - this is the key metric
        # Paths with lower total delay are preferred, even if longer
        total_path_delay = 0.0
        packet_size = 15000.0  # Standard packet size (increased by 10x)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_delay = self._calculate_link_delay(u, v, packet_size)
            total_path_delay += link_delay
        
        # Calculate path capacity (minimum capacity along the path - bottleneck)
        path_capacity = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_capacity = self.topology.get_link_capacity(u, v)
            path_capacity = min(path_capacity, link_capacity)
        
        # Normalize capacity (0.0 to 1.0, higher is better)
        max_capacity = 10.0  # Reduced from 100.0 (capacities are now 10x smaller)
        normalized_capacity = min(path_capacity / max_capacity, 1.0) if max_capacity > 0 else 0.0
        
        # Normalize delay (lower is better) - convert to score (higher is better)
        # Assume max reasonable delay is 1 second (1000ms)
        max_delay = 1.0  # seconds
        normalized_delay_score = 1.0 - min(total_path_delay / max_delay, 1.0)
        
        # Normalize scores (lower is better for congestion)
        # For combined score, we'll use inverse of congestion
        normalized_congestion = 1.0 - congestion_score
        
        # Stability score will be provided by stability mechanism
        # For now, use a default value
        stability_score = 1.0
        
        # Combined score (higher is better)
        # Prioritize congestion heavily (60%) - avoid congested paths
        # Total delay (25%), capacity (10%), stability (5%)
        # When congestion is high, it dominates the score
        combined_score = (normalized_congestion * 0.6 +   # Congestion is most important (avoid congested paths)
                         normalized_delay_score * 0.25 +  # Total delay matters
                         normalized_capacity * 0.1 +      # Capacity matters
                         stability_score * 0.05)          # Stability matters less
        
        # For load balancing, also consider path length (shorter paths preferred if delay/congestion are similar)
        # Normalize hop count (inverse - shorter is better)
        max_hops = 10  # Assume max 10 hops for normalization
        normalized_hops = 1.0 - (min(hop_count, max_hops) / max_hops)
        # Add small weight for path length (10% weight)
        combined_score = combined_score * 0.9 + normalized_hops * 0.1
        
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
        # Find candidate paths - use k=5 to ensure we get all reasonable paths including direct paths
        # This is important to find alternative routes when congestion occurs
        candidate_paths = self.find_k_shortest_paths(source, destination, k=5, metric='hop')
        
        # Also try to find paths by congestion metric to ensure we consider less congested paths
        congestion_paths = self.find_k_shortest_paths(source, destination, k=3, metric='congestion')
        
        # Combine and deduplicate paths
        all_paths = candidate_paths.copy()
        for path in congestion_paths:
            if path not in all_paths:
                all_paths.append(path)
        
        candidate_paths = all_paths[:5]  # Limit to 5 paths for performance
        
        if not candidate_paths:
            return None
        
        # Evaluate each candidate path
        best_path = None
        best_score = -1.0
        
        for path in candidate_paths:
            # Always evaluate fresh to get current congestion (dynamic load-aware routing)
            # Don't use cache - congestion changes in real-time
            evaluation = self.evaluate_path(path, congestion_weight, stability_weight)
            
            # Update stability score if provided
            if stability_scores:
                path_stability = self._calculate_path_stability(path, stability_scores)
                evaluation['stability_score'] = path_stability
                # Recalculate combined score with stability, capacity, and path length
                normalized_congestion = 1.0 - evaluation['congestion_score']
                
                # Get path capacity for this path
                path_capacity = float('inf')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    link_capacity = self.topology.get_link_capacity(u, v)
                    path_capacity = min(path_capacity, link_capacity)
                max_capacity = 100.0
                normalized_capacity = min(path_capacity / max_capacity, 1.0) if max_capacity > 0 else 0.0
                
                # Combined score: congestion (60%) + stability (40%)
                base_score = (normalized_congestion * congestion_weight +
                             path_stability * stability_weight)
                # Add capacity (20% weight)
                base_score = base_score * 0.7 + normalized_capacity * 0.2
                # Add path length (10% weight)
                max_hops = 10
                normalized_hops = 1.0 - (min(evaluation['hop_count'], max_hops) / max_hops)
                evaluation['combined_score'] = base_score * 0.9 + normalized_hops * 0.1
            
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

