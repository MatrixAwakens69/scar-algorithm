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
        self.route_discovery = RouteDiscovery(topology, self.congestion_monitor, simulator=None)
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
        Uses dynamic load-aware routing - always selects best path based on current link loads.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            current_time: Current simulation time
            force_update: Force route update regardless of timing
        
        Returns:
            Route (list of node IDs) or None if no route exists
        """
        key = (source, destination)
        
        # Always evaluate routes dynamically based on current congestion (load-aware routing)
        # This ensures new packets use less congested paths even if previous packets used a different route
        if key in self.route_table and not force_update:
            current_route = self.route_table[key]
            
            # Check if current route has any highly congested links (>50% utilization)
            # If so, aggressively look for alternatives
            current_route_has_congestion = False
            max_congestion_on_route = 0.0
            for i in range(len(current_route) - 1):
                u, v = current_route[i], current_route[i + 1]
                link_congestion = self.congestion_monitor.get_link_congestion_score(u, v)
                max_congestion_on_route = max(max_congestion_on_route, link_congestion)
                if link_congestion > 0.5:  # 50% utilization threshold
                    current_route_has_congestion = True
                    break
            
            # Always find the best current route based on real-time congestion and capacity
            new_route = self._select_route(source, destination, current_time)
            if new_route and new_route != current_route:
                # Evaluate both routes using the full evaluation (congestion + capacity + stability)
                current_eval = self.route_discovery.evaluate_path(
                    current_route, self.congestion_weight, self.stability_weight
                )
                new_eval = self.route_discovery.evaluate_path(
                    new_route, self.congestion_weight, self.stability_weight
                )
                
                # Aggressive switching logic for load balancing:
                # 1. If current route has congestion > 50%, switch to any less congested route
                if current_route_has_congestion:
                    if new_eval['congestion_score'] < max_congestion_on_route * 0.9:  # 10% improvement
                        self._update_route(key, new_route, current_time)
                        return new_route
                    # Also switch if new route has any link with congestion < 0.3 and current route has > 0.5
                    new_route_max_congestion = 0.0
                    for i in range(len(new_route) - 1):
                        u, v = new_route[i], new_route[i + 1]
                        link_congestion = self.congestion_monitor.get_link_congestion_score(u, v)
                        new_route_max_congestion = max(new_route_max_congestion, link_congestion)
                    if new_route_max_congestion < 0.3 and max_congestion_on_route > 0.5:
                        self._update_route(key, new_route, current_time)
                        return new_route
                
                # 2. Switch to new route if it's better (dynamic load balancing)
                # Use combined score which considers congestion, capacity, and path length
                if new_eval['combined_score'] > current_eval['combined_score']:
                    # New route is better - switch immediately for dynamic routing
                    self._update_route(key, new_route, current_time)
                    return new_route
                
                # 3. Switch if new route has lower congestion (even small improvements matter)
                elif new_eval['congestion_score'] < current_eval['congestion_score'] * 0.98:  # 2% improvement
                    # New route is less congested - switch for load balancing
                    self._update_route(key, new_route, current_time)
                    return new_route
            
            # If current route is highly congested but no better route found, still return it
            # (but next packet will re-evaluate)
            return current_route
        
        # No route exists or force update - find best route based on current loads
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
        # 1. New route is better (threshold: 5% improvement - more responsive)
        improvement_threshold = 0.05
        score_improvement = (new_eval['combined_score'] - current_eval['combined_score']) / max(0.01, current_eval['combined_score'])
        
        if score_improvement > improvement_threshold:
            return True
        
        # 2. Current route is congested and new route is less congested (more lenient)
        if (current_eval['congestion_score'] > self.congestion_threshold and
            new_eval['congestion_score'] < current_eval['congestion_score'] * 0.9):  # 10% improvement
            return True
        
        # 3. New route has significantly lower congestion (even if current isn't highly congested)
        if new_eval['congestion_score'] < current_eval['congestion_score'] * 0.7:  # 30% improvement
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

