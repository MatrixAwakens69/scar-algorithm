"""
Congestion Detection and Monitoring

This module provides real-time congestion detection and monitoring capabilities.
"""

from typing import Dict, List, Tuple, Optional
from src.core.network_topology import NetworkTopology
import numpy as np


class CongestionMonitor:
    """Monitors network congestion and detects congested links."""
    
    def __init__(self, topology: NetworkTopology, congestion_threshold: float = 0.7):
        """
        Initialize congestion monitor.
        
        Args:
            topology: NetworkTopology instance
            congestion_threshold: Threshold for congestion detection (0.0 to 1.0)
        """
        self.topology = topology
        self.congestion_threshold = congestion_threshold
        self.congestion_history: Dict[Tuple[int, int], List[float]] = {}
        self.congestion_events: List[Dict] = []
    
    def update_link_load(self, node1: int, node2: int, load: float):
        """
        Update link load and check for congestion.
        
        Args:
            node1: First node ID
            node2: Second node ID
            load: New load value in Mbps
        """
        self.topology.update_link_load(node1, node2, load)
        
        # Track congestion history
        link = self._normalize_link(node1, node2)
        if link not in self.congestion_history:
            self.congestion_history[link] = []
        self.congestion_history[link].append(self.topology.get_link_utilization(node1, node2))
        
        # Keep only recent history (last 100 updates)
        if len(self.congestion_history[link]) > 100:
            self.congestion_history[link] = self.congestion_history[link][-100:]
        
        # Invalidate route discovery cache when congestion changes
        if hasattr(self, 'route_discovery'):
            if hasattr(self.route_discovery, '_invalidate_cache'):
                self.route_discovery._invalidate_cache()
    
    def check_congestion(self, node1: int, node2: int) -> bool:
        """
        Check if a link is congested.
        
        Args:
            node1: First node ID
            node2: Second node ID
        
        Returns:
            True if link is congested
        """
        return self.topology.is_link_congested(node1, node2, self.congestion_threshold)
    
    def get_congestion_level(self, node1: int, node2: int) -> str:
        """
        Get congestion level as string.
        
        Args:
            node1: First node ID
            node2: Second node ID
        
        Returns:
            'low', 'medium', or 'high'
        """
        utilization = self.topology.get_link_utilization(node1, node2)
        
        if utilization < self.congestion_threshold * 0.5:
            return 'low'
        elif utilization < self.congestion_threshold:
            return 'medium'
        else:
            return 'high'
    
    def get_all_congested_links(self) -> List[Tuple[int, int]]:
        """
        Get all currently congested links.
        
        Returns:
            List of (node1, node2) tuples
        """
        return self.topology.get_all_congested_links(self.congestion_threshold)
    
    def get_link_congestion_score(self, node1: int, node2: int) -> float:
        """
        Get congestion score for a link (0.0 to 1.0).
        Higher score means more congested.
        
        Args:
            node1: First node ID
            node2: Second node ID
        
        Returns:
            Congestion score
        """
        utilization = self.topology.get_link_utilization(node1, node2)
        return min(1.0, utilization / self.congestion_threshold) if self.congestion_threshold > 0 else 0.0
    
    def get_path_congestion_score(self, path: List[int]) -> float:
        """
        Calculate average congestion score for a path.
        
        Args:
            path: List of node IDs forming the path
        
        Returns:
            Average congestion score
        """
        if len(path) < 2:
            return 0.0
        
        scores = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.topology.graph.has_edge(u, v):
                scores.append(self.get_link_congestion_score(u, v))
        
        return np.mean(scores) if scores else 0.0
    
    def get_network_congestion_summary(self) -> Dict:
        """
        Get summary of network congestion.
        
        Returns:
            Dictionary with congestion statistics
        """
        all_links = self.topology.get_edges()
        congested_links = self.get_all_congested_links()
        
        utilizations = [self.topology.get_link_utilization(u, v) 
                       for u, v in all_links]
        
        summary = {
            'total_links': len(all_links),
            'congested_links': len(congested_links),
            'congestion_ratio': len(congested_links) / max(1, len(all_links)),
            'avg_utilization': np.mean(utilizations) if utilizations else 0.0,
            'max_utilization': np.max(utilizations) if utilizations else 0.0,
            'min_utilization': np.min(utilizations) if utilizations else 0.0,
        }
        
        return summary
    
    def _normalize_link(self, node1: int, node2: int) -> Tuple[int, int]:
        """Normalize link representation (smaller node first)."""
        return (min(node1, node2), max(node1, node2))
    
    def set_threshold(self, threshold: float):
        """
        Update congestion threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        self.congestion_threshold = max(0.0, min(1.0, threshold))

