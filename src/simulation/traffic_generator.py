"""
Traffic Generation

This module generates network traffic patterns for simulation.
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from src.simulation.network_simulator import NetworkSimulator, SimulationEvent, EventType


class TrafficGenerator:
    """Generates traffic patterns for network simulation."""
    
    def __init__(self, simulator: NetworkSimulator, topology):
        """
        Initialize traffic generator.
        
        Args:
            simulator: NetworkSimulator instance
            topology: NetworkTopology instance
        """
        self.simulator = simulator
        self.topology = topology
        self.nodes = list(topology.get_nodes())
    
    def generate_constant_traffic(self, source: int, destination: int,
                                  packet_rate: float, packet_size: float = 1500.0,
                                  duration: float = 1000.0):
        """
        Generate constant rate traffic.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            packet_rate: Packets per time unit
            packet_size: Packet size in bytes
            duration: Duration of traffic generation
        """
        interval = 1.0 / packet_rate if packet_rate > 0 else float('inf')
        current_time = 0.0
        
        while current_time < duration:
            event = SimulationEvent(
                event_type=EventType.PACKET_GENERATE,
                time=current_time,
                data={
                    'source': source,
                    'destination': destination,
                    'size': packet_size
                },
                priority=0
            )
            self.simulator.schedule_event(event)
            current_time += interval
    
    def generate_bursty_traffic(self, source: int, destination: int,
                                avg_packet_rate: float, burst_size: int = 10,
                                packet_size: float = 1500.0, duration: float = 1000.0):
        """
        Generate bursty traffic pattern.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            avg_packet_rate: Average packets per time unit
            burst_size: Number of packets in a burst
            packet_size: Packet size in bytes
            duration: Duration of traffic generation
        """
        interval = 1.0 / avg_packet_rate if avg_packet_rate > 0 else float('inf')
        current_time = 0.0
        
        while current_time < duration:
            # Generate burst
            for _ in range(burst_size):
                event = SimulationEvent(
                    event_type=EventType.PACKET_GENERATE,
                    time=current_time,
                    data={
                        'source': source,
                        'destination': destination,
                        'size': packet_size
                    },
                    priority=0
                )
                self.simulator.schedule_event(event)
                current_time += 0.01  # Small interval between packets in burst
            
            # Wait before next burst
            current_time += interval - (burst_size * 0.01)
    
    def generate_random_traffic(self, num_packets: int, packet_size: float = 1500.0,
                               time_range: Tuple[float, float] = (0.0, 1000.0),
                               source: Optional[int] = None, destination: Optional[int] = None):
        """
        Generate random traffic pattern.
        
        Args:
            num_packets: Number of packets to generate
            packet_size: Packet size in bytes
            time_range: (start_time, end_time) tuple
            source: Fixed source node (if None, uses first node)
            destination: Fixed destination node (if None, uses last node)
        """
        start_time, end_time = time_range
        
        # Use fixed source and destination if not provided
        if source is None:
            source = self.nodes[0]  # First node as source
        if destination is None:
            destination = self.nodes[-1]  # Last node as destination
        
        # Ensure source and destination are different
        if source == destination and len(self.nodes) > 1:
            destination = self.nodes[-1] if source != self.nodes[-1] else self.nodes[0]
        
        for _ in range(num_packets):
            # Random generation time
            gen_time = random.uniform(start_time, end_time)
            
            event = SimulationEvent(
                event_type=EventType.PACKET_GENERATE,
                time=gen_time,
                data={
                    'source': source,
                    'destination': destination,
                    'size': packet_size
                },
                priority=0
            )
            self.simulator.schedule_event(event)
    
    def generate_traffic_pattern(self, pattern: str, **kwargs):
        """
        Generate traffic based on pattern type.
        
        Args:
            pattern: Traffic pattern ('constant', 'bursty', 'random')
            **kwargs: Pattern-specific parameters
        """
        if pattern == 'constant':
            self.generate_constant_traffic(**kwargs)
        elif pattern == 'bursty':
            self.generate_bursty_traffic(**kwargs)
        elif pattern == 'random':
            self.generate_random_traffic(**kwargs)
        else:
            raise ValueError(f"Unknown traffic pattern: {pattern}")
    
    def generate_multiple_flows(self, num_flows: int, pattern: str = 'random',
                                **kwargs):
        """
        Generate multiple traffic flows.
        
        Args:
            num_flows: Number of flows to generate
            pattern: Traffic pattern for each flow
            **kwargs: Pattern-specific parameters (can include fixed source/destination)
        """
        # Use fixed source and destination if provided, otherwise use defaults
        source = kwargs.get('source', self.nodes[0])
        destination = kwargs.get('destination', self.nodes[-1])
        
        for _ in range(num_flows):
            flow_kwargs = kwargs.copy()
            flow_kwargs['source'] = source
            flow_kwargs['destination'] = destination
            
            self.generate_traffic_pattern(pattern, **flow_kwargs)

