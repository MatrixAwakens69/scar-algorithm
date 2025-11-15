"""
SCAR Algorithm - Stable, Congestion-Aware Routing

This module implements the core SCAR algorithm that combines congestion awareness
with route stability maintenance.
"""

from typing import Dict, List, Optional, Tuple
from src.core.network_topology import NetworkTopology
from src.core.congestion_monitor import CongestionMonitor
from src.core.route_discovery import RouteDiscovery
from src.core.stability_mechanism import StabilityMechanism


class SCARAlgorithm:
    """Implements the SCAR (Stable, Congestion-Aware Routing) algorithm."""
    
    def __init__(self, topology: NetworkTopology,
                 congestion_threshold: float = 0.7,
                 stability_threshold: float = 0.6,
                 congestion_weight: float = 0.6,
                 stability_weight: float = 0.4,
                 route_update_interval: int = 10,
                 max_route_changes_per_window: int = 3,
                 route_change_window: int = 50):
        """
        Initialize SCAR algorithm.
        
        Args:
            topology: NetworkTopology instance
            congestion_threshold: Threshold for congestion detection
            stability_threshold: Threshold for route stability
            congestion_weight: Weight for congestion in route selection
            stability_weight: Weight for stability in route selection
            route_update_interval: Interval for route updates
            max_route_changes_per_window: Maximum route changes per time window
            route_change_window: Time window for tracking route changes
        """
        self.topology = topology
        self.congestion_monitor = CongestionMonitor(topology, congestion_threshold)
        self.route_discovery = RouteDiscovery(topology, self.congestion_monitor)
        self.stability_mechanism = StabilityMechanism(
            topology, stability_threshold, max_route_changes_per_window, route_change_window
        )
        
        self.congestion_threshold = congestion_threshold
        self.stability_threshold = stability_threshold
        self.congestion_weight = congestion_weight
        self.stability_weight = stability_weight
        self.route_update_interval = route_update_interval
        
        # Track last update time for each route
        self.last_update_time: Dict[Tuple[int, int], int] = {}
        
        # Route table: (source, dest) -> route
        self.route_table: Dict[Tuple[int, int], List[int]] = {}
        
        # Cache for route congestion evaluations
        self._route_congestion_cache: Dict[Tuple[Tuple[int, ...]], float] = {}
        
        # Link route discovery to congestion monitor for cache invalidation
        self.congestion_monitor.route_discovery = self.route_discovery
    
    def get_route(self, source: int, destination: int, 
                 current_time: int = 0, force_update: bool = False) -> Optional[List[int]]:
        """
        Get route for source-destination pair using SCAR algorithm.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            current_time: Current simulation time
            force_update: Force route update regardless of timing
        
        Returns:
            Route (list of node IDs) or None if no route exists
        """
        key = (source, destination)
        
        # Check if route exists and is still valid
        if key in self.route_table and not force_update:
            # Check if update is needed
            last_update = self.last_update_time.get(key, 0)
            time_since_update = current_time - last_update
            
            # Update route if:
            # 1. Update interval has passed, OR
            # 2. Current route is highly congested, OR
            # 3. Stability allows update and better route exists
            
            current_route = self.route_table[key]
            route_congestion = self._evaluate_route_congestion(current_route)
            
            # Check if route update is needed
            needs_update = (
                time_since_update >= self.route_update_interval or
                route_congestion > self.congestion_threshold * 1.2  # Highly congested
            )
            
            if needs_update:
                # Check if stability allows update
                if self.stability_mechanism.can_change_route(source, destination, current_time):
                    new_route = self._select_route(source, destination, current_time)
                    if new_route and new_route != current_route:
                        # Only update if new route is significantly better
                        if self._should_update_route(current_route, new_route, current_time):
                            self._update_route(key, new_route, current_time)
                            return new_route
                
                # Update last check time even if route didn't change
                self.last_update_time[key] = current_time
            
            return current_route
        
        # No route exists or force update - find new route
        new_route = self._select_route(source, destination, current_time)
        if new_route:
            self._update_route(key, new_route, current_time)
        return new_route
    
    def _select_route(self, source: int, destination: int, 
                     current_time: int) -> Optional[List[int]]:
        """
        Select best route considering both congestion and stability.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            current_time: Current simulation time
        
        Returns:
            Best route (list of node IDs) or None
        """
        # Get link stability scores
        link_stability_scores = self.stability_mechanism.get_link_stability_scores(current_time)
        
        # Find best path considering both congestion and stability
        best_path_eval = self.route_discovery.find_best_path(
            source, destination,
            congestion_weight=self.congestion_weight,
            stability_weight=self.stability_weight,
            stability_scores=link_stability_scores
        )
        
        if best_path_eval:
            return best_path_eval['path']
        return None
    
    def _should_update_route(self, current_route: List[int], new_route: List[int],
                            current_time: int) -> bool:
        """
        Determine if route should be updated.
        
        Args:
            current_route: Current route
            new_route: Proposed new route
            current_time: Current simulation time
        
        Returns:
            True if route should be updated
        """
        # Evaluate both routes
        current_eval = self.route_discovery.evaluate_path(
            current_route, self.congestion_weight, self.stability_weight
        )
        new_eval = self.route_discovery.evaluate_path(
            new_route, self.congestion_weight, self.stability_weight
        )
        
        # Get stability scores
        current_stability = self.stability_mechanism.get_stability_score(
            current_route[0], current_route[-1], current_time
        )
        
        # Update if:
        # 1. New route is significantly better (threshold: 10% improvement)
        improvement_threshold = 0.1
        score_improvement = (new_eval['combined_score'] - current_eval['combined_score']) / max(0.01, current_eval['combined_score'])
        
        if score_improvement > improvement_threshold:
            return True
        
        # 2. Current route is highly congested and new route is less congested
        if (current_eval['congestion_score'] > self.congestion_threshold and
            new_eval['congestion_score'] < current_eval['congestion_score'] * 0.8):
            return True
        
        return False
    
    def _update_route(self, key: Tuple[int, int], route: List[int], current_time: int):
        """
        Update route in route table and stability mechanism.
        
        Args:
            key: (source, destination) tuple
            route: New route
            current_time: Current simulation time
        """
        source, destination = key
        self.route_table[key] = route
        self.last_update_time[key] = current_time
        self.stability_mechanism.set_route(source, destination, route, current_time)
    
    def _evaluate_route_congestion(self, route: List[int]) -> float:
        """
        Evaluate congestion level of a route (cached).
        
        Args:
            route: Route (list of node IDs)
        
        Returns:
            Average congestion score (0.0 to 1.0)
        """
        route_tuple = tuple(route)
        if route_tuple in self._route_congestion_cache:
            return self._route_congestion_cache[route_tuple]
        
        score = self.congestion_monitor.get_path_congestion_score(route)
        self._route_congestion_cache[route_tuple] = score
        return score
    
    def update_link_load(self, node1: int, node2: int, load: float):
        """
        Update load on a link.
        
        Args:
            node1: First node ID
            node2: Second node ID
            load: New load value in Mbps
        """
        self.congestion_monitor.update_link_load(node1, node2, load)
    
    def get_route_table(self) -> Dict[Tuple[int, int], List[int]]:
        """Get current route table."""
        return self.route_table.copy()
    
    def get_algorithm_stats(self, current_time: int = 0) -> Dict:
        """
        Get algorithm statistics.
        
        Args:
            current_time: Current simulation time
        
        Returns:
            Dictionary with algorithm statistics
        """
        stability_summary = self.stability_mechanism.get_route_stability_summary()
        congestion_summary = self.congestion_monitor.get_network_congestion_summary()
        
        stats = {
            'num_routes': len(self.route_table),
            'stability': stability_summary,
            'congestion': congestion_summary,
            'avg_route_age': stability_summary.get('avg_route_age', 0.0),
            'total_route_changes': stability_summary.get('total_changes', 0),
        }
        
        return stats
    
    def update_time(self, current_time: int):
        """
        Update time-based tracking.
        
        Args:
            current_time: Current simulation time
        """
        self.stability_mechanism.update_time(current_time)
    
    def reset(self):
        """Reset algorithm state."""
        self.route_table.clear()
        self.last_update_time.clear()
        self.stability_mechanism = StabilityMechanism(
            self.topology, self.stability_threshold,
            self.stability_mechanism.max_changes_per_window,
            self.stability_mechanism.route_change_window
        )

