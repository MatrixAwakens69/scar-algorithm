"""
Route Stability Mechanism

This module tracks route stability and calculates stability metrics.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from src.core.network_topology import NetworkTopology


class StabilityMechanism:
    """Tracks route stability and manages route changes."""
    
    def __init__(self, topology: NetworkTopology, 
                 stability_threshold: float = 0.6,
                 max_changes_per_window: int = 3,
                 route_change_window: int = 50):
        """
        Initialize stability mechanism.
        
        Args:
            topology: NetworkTopology instance
            stability_threshold: Threshold for stability (0.0 to 1.0)
            max_changes_per_window: Maximum route changes allowed per time window
            route_change_window: Time window size for tracking route changes
        """
        self.topology = topology
        self.stability_threshold = stability_threshold
        self.max_changes_per_window = max_changes_per_window
        self.route_change_window = route_change_window
        
        # Track route history for each source-destination pair
        self.route_history: Dict[Tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Track route change events
        self.route_changes: Dict[Tuple[int, int], deque] = defaultdict(lambda: deque())
        
        # Track current routes
        self.current_routes: Dict[Tuple[int, int], List[int]] = {}
        
        # Track route ages (how long a route has been active)
        self.route_ages: Dict[Tuple[int, int], int] = defaultdict(int)
    
    def get_current_route(self, source: int, destination: int) -> Optional[List[int]]:
        """
        Get current route for a source-destination pair.
        
        Args:
            source: Source node ID
            destination: Destination node ID
        
        Returns:
            Current route (list of node IDs) or None
        """
        key = (source, destination)
        return self.current_routes.get(key)
    
    def set_route(self, source: int, destination: int, route: List[int], 
                  current_time: int = 0):
        """
        Set route for a source-destination pair.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            route: Route (list of node IDs)
            current_time: Current simulation time
        """
        key = (source, destination)
        old_route = self.current_routes.get(key)
        
        # Check if route changed
        if old_route != route:
            # Record route change
            self.route_changes[key].append(current_time)
            # Remove old change events outside the window
            while (self.route_changes[key] and 
                   current_time - self.route_changes[key][0] > self.route_change_window):
                self.route_changes[key].popleft()
            
            # Reset route age
            self.route_ages[key] = 0
        else:
            # Increment route age if route is the same
            self.route_ages[key] += 1
        
        # Update current route
        self.current_routes[key] = route
        
        # Record in history
        self.route_history[key].append({
            'route': route,
            'time': current_time,
            'age': self.route_ages[key]
        })
    
    def can_change_route(self, source: int, destination: int, 
                        current_time: int = 0) -> bool:
        """
        Check if route can be changed based on stability constraints.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            current_time: Current simulation time
        
        Returns:
            True if route change is allowed
        """
        key = (source, destination)
        
        # Count route changes in current window
        changes_in_window = sum(1 for change_time in self.route_changes[key]
                               if current_time - change_time <= self.route_change_window)
        
        # Allow change if under the limit
        return changes_in_window < self.max_changes_per_window
    
    def get_stability_score(self, source: int, destination: int, 
                           current_time: int = 0) -> float:
        """
        Calculate stability score for a route (0.0 to 1.0).
        Higher score means more stable.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            current_time: Current simulation time
        
        Returns:
            Stability score
        """
        key = (source, destination)
        
        # Get route age
        route_age = self.route_ages.get(key, 0)
        
        # Get number of changes in window
        changes_in_window = sum(1 for change_time in self.route_changes[key]
                               if current_time - change_time <= self.route_change_window)
        
        # Calculate stability based on:
        # 1. Route age (older routes are more stable)
        # 2. Number of recent changes (fewer changes = more stable)
        
        # Normalize route age (assuming max age of route_change_window)
        age_score = min(1.0, route_age / self.route_change_window)
        
        # Normalize change count (fewer changes = higher score)
        max_changes = self.max_changes_per_window
        change_score = 1.0 - (changes_in_window / max(1, max_changes))
        
        # Combined stability score
        stability = (age_score * 0.5 + change_score * 0.5)
        
        return max(0.0, min(1.0, stability))
    
    def get_link_stability_scores(self, current_time: int = 0) -> Dict[Tuple[int, int], float]:
        """
        Get stability scores for all links based on route usage.
        
        Args:
            current_time: Current simulation time
        
        Returns:
            Dictionary mapping (node1, node2) to stability score
        """
        link_usage_count: Dict[Tuple[int, int], int] = defaultdict(int)
        link_route_ages: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # Count how many routes use each link and track route ages
        for (source, dest), route in self.current_routes.items():
            route_age = self.route_ages.get((source, dest), 0)
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                link = (min(u, v), max(u, v))
                link_usage_count[link] += 1
                link_route_ages[link].append(route_age)
        
        # Calculate stability scores for links
        link_stability: Dict[Tuple[int, int], float] = {}
        for link in self.topology.get_edges():
            normalized_link = (min(link[0], link[1]), max(link[0], link[1]))
            
            usage_count = link_usage_count.get(normalized_link, 0)
            route_ages = link_route_ages.get(normalized_link, [])
            
            # Stability based on:
            # 1. How many routes use this link (more = more stable)
            # 2. Average age of routes using this link (older = more stable)
            
            usage_score = min(1.0, usage_count / 5.0)  # Normalize to 5 routes max
            
            avg_age = sum(route_ages) / len(route_ages) if route_ages else 0
            age_score = min(1.0, avg_age / self.route_change_window)
            
            stability = (usage_score * 0.4 + age_score * 0.6)
            link_stability[normalized_link] = max(0.0, min(1.0, stability))
        
        return link_stability
    
    def get_route_stability_summary(self) -> Dict:
        """
        Get summary of route stability across the network.
        
        Returns:
            Dictionary with stability statistics
        """
        if not self.current_routes:
            return {
                'total_routes': 0,
                'avg_stability': 0.0,
                'avg_route_age': 0.0,
                'total_changes': 0
            }
        
        stability_scores = []
        route_ages = []
        total_changes = 0
        
        for key in self.current_routes.keys():
            # Use a default time for stability calculation
            stability = self.get_stability_score(key[0], key[1], current_time=100)
            stability_scores.append(stability)
            route_ages.append(self.route_ages.get(key, 0))
            total_changes += len(self.route_changes.get(key, []))
        
        return {
            'total_routes': len(self.current_routes),
            'avg_stability': sum(stability_scores) / len(stability_scores) if stability_scores else 0.0,
            'avg_route_age': sum(route_ages) / len(route_ages) if route_ages else 0.0,
            'total_changes': total_changes
        }
    
    def reset(self):
        """Reset all stability tracking data."""
        self.route_history.clear()
        self.route_changes.clear()
        self.current_routes.clear()
        self.route_ages.clear()
    
    def update_time(self, current_time: int):
        """
        Update time-based tracking (cleanup old events).
        
        Args:
            current_time: Current simulation time
        """
        # Clean up old route change events
        for key in list(self.route_changes.keys()):
            while (self.route_changes[key] and 
                   current_time - self.route_changes[key][0] > self.route_change_window):
                self.route_changes[key].popleft()
            
            # Remove empty deques
            if not self.route_changes[key]:
                del self.route_changes[key]

