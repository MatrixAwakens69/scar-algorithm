"""
Network Simulation Engine

This module provides event-driven network simulation capabilities.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import heapq
from src.core.network_topology import NetworkTopology
from src.core.scar_algorithm import SCARAlgorithm


class EventType(Enum):
    """Types of simulation events."""
    PACKET_GENERATE = "packet_generate"
    PACKET_ARRIVE = "packet_arrive"
    PACKET_DEPART = "packet_depart"
    ROUTE_UPDATE = "route_update"
    LINK_UPDATE = "link_update"


@dataclass
class SimulationEvent:
    """Represents a simulation event."""
    event_type: EventType
    time: float
    data: Dict[str, Any]
    priority: int = 0  # Lower number = higher priority
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.time != other.time:
            return self.time < other.time
        return self.priority < other.priority


@dataclass
class Packet:
    """Represents a network packet."""
    packet_id: int
    source: int
    destination: int
    size: float  # bytes
    generation_time: float
    route: Optional[List[int]] = None
    current_node: Optional[int] = None
    next_hop: Optional[int] = None
    arrival_time: Optional[float] = None
    delivery_time: Optional[float] = None
    hops_traveled: int = 0


class NetworkSimulator:
    """Event-driven network simulator."""
    
    def __init__(self, topology: NetworkTopology, scar_algorithm: SCARAlgorithm):
        """
        Initialize network simulator.
        
        Args:
            topology: NetworkTopology instance
            scar_algorithm: SCARAlgorithm instance
        """
        self.topology = topology
        self.scar_algorithm = scar_algorithm
        
        # Event queue (priority queue)
        self.event_queue: List[SimulationEvent] = []
        
        # Current simulation time
        self.current_time = 0.0
        
        # Packets in the network
        self.active_packets: Dict[int, Packet] = {}
        self.delivered_packets: List[Packet] = []
        self.lost_packets: List[Packet] = []
        
        # Packet counter
        self.packet_counter = 0
        
        # Statistics
        self.stats = {
            'packets_generated': 0,
            'packets_delivered': 0,
            'packets_lost': 0,
            'total_delay': 0.0,
            'total_hops': 0,
        }
        
        # Event tracking for status display
        self.last_event_info = "No events yet"
    
    def schedule_event(self, event: SimulationEvent):
        """
        Schedule an event.
        
        Args:
            event: SimulationEvent to schedule
        """
        heapq.heappush(self.event_queue, event)
    
    def generate_packet(self, source: int, destination: int, size: float = 1500.0):
        """
        Generate a packet and schedule its transmission.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            size: Packet size in bytes
        """
        self.packet_counter += 1
        packet = Packet(
            packet_id=self.packet_counter,
            source=source,
            destination=destination,
            size=size,
            generation_time=self.current_time,
            current_node=source
        )
        
        # Get route using SCAR algorithm
        route = self.scar_algorithm.get_route(source, destination, int(self.current_time))
        
        if not route or len(route) < 2:
            # No route available - packet is lost
            self.lost_packets.append(packet)
            self.stats['packets_lost'] += 1
            return
        
        packet.route = route
        packet.next_hop = route[1] if len(route) > 1 else None
        
        # Add to active packets
        self.active_packets[packet.packet_id] = packet
        
        # Schedule packet transmission to next hop
        self._schedule_packet_transmission(packet)
        
        self.stats['packets_generated'] += 1
    
    def _schedule_packet_transmission(self, packet: Packet):
        """Schedule packet transmission to next hop."""
        if not packet.next_hop or not packet.route:
            return
        
        current_node = packet.current_node
        next_node = packet.next_hop
        
        # Calculate transmission time based on link capacity and latency
        link_capacity = self.topology.get_link_capacity(current_node, next_node)
        link_latency = self.topology.graph[current_node][next_node].get('latency', 5.0)
        
        # Transmission time = packet_size / capacity + latency
        # Convert bytes to bits and capacity from Mbps to bps
        transmission_time = (packet.size * 8) / (link_capacity * 1e6) + (link_latency / 1000.0)
        
        # Update link load
        packet_load = (packet.size * 8) / 1e6  # Convert to Mbps
        current_load = self.topology.get_link_load(current_node, next_node)
        new_load = current_load + packet_load
        self.scar_algorithm.update_link_load(current_node, next_node, new_load)
        
        # Schedule arrival at next node
        arrival_time = self.current_time + transmission_time
        
        event = SimulationEvent(
            event_type=EventType.PACKET_ARRIVE,
            time=arrival_time,
            data={'packet_id': packet.packet_id},
            priority=1
        )
        self.schedule_event(event)
    
    def _handle_packet_arrival(self, packet_id: int):
        """Handle packet arrival at a node."""
        if packet_id not in self.active_packets:
            return
        
        packet = self.active_packets[packet_id]
        
        if not packet.route:
            return
        
        # Update current node
        packet.current_node = packet.next_hop
        packet.hops_traveled += 1
        
        # Check if packet reached destination
        if packet.current_node == packet.destination:
            # Packet delivered
            packet.arrival_time = self.current_time
            packet.delivery_time = self.current_time - packet.generation_time
            
            self.delivered_packets.append(packet)
            del self.active_packets[packet_id]
            
            self.stats['packets_delivered'] += 1
            self.stats['total_delay'] += packet.delivery_time
            self.stats['total_hops'] += packet.hops_traveled
            
            # Update link load (packet left the link)
            if packet.hops_traveled > 0:
                # Find the previous link
                route = packet.route
                if len(route) >= 2:
                    prev_idx = route.index(packet.current_node) - 1
                    if prev_idx >= 0:
                        prev_node = route[prev_idx]
                        packet_load = (packet.size * 8) / 1e6
                        current_load = self.topology.get_link_load(prev_node, packet.current_node)
                        new_load = max(0.0, current_load - packet_load)
                        self.scar_algorithm.update_link_load(prev_node, packet.current_node, new_load)
        else:
            # Continue to next hop
            route = packet.route
            current_idx = route.index(packet.current_node)
            if current_idx < len(route) - 1:
                packet.next_hop = route[current_idx + 1]
                self._schedule_packet_transmission(packet)
            else:
                # Route error - packet lost
                self.lost_packets.append(packet)
                del self.active_packets[packet_id]
                self.stats['packets_lost'] += 1
    
    def step(self) -> bool:
        """
        Execute one simulation step.
        
        Returns:
            True if event was processed, False if no events
        """
        if not self.event_queue:
            return False
        
        # Get next event
        event = heapq.heappop(self.event_queue)
        self.current_time = event.time
        
        # Process event and update status
        if event.event_type == EventType.PACKET_ARRIVE:
            packet_id = event.data['packet_id']
            if packet_id in self.active_packets:
                packet = self.active_packets[packet_id]
                self.last_event_info = f"Packet {packet_id} arrived at node {packet.next_hop}"
            self._handle_packet_arrival(packet_id)
        elif event.event_type == EventType.PACKET_GENERATE:
            source = event.data['source']
            destination = event.data['destination']
            size = event.data.get('size', 1500.0)
            self.last_event_info = f"Packet generated: {source} -> {destination}"
            self.generate_packet(source, destination, size)
        elif event.event_type == EventType.ROUTE_UPDATE:
            # Route update event
            self.last_event_info = "Route update triggered"
            self.scar_algorithm.update_time(int(self.current_time))
        
        return True
    
    def run(self, duration: float):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration
        """
        end_time = self.current_time + duration
        
        while self.current_time < end_time:
            if not self.step():
                # No more events, advance time
                if self.event_queue:
                    next_event_time = self.event_queue[0].time
                    self.current_time = min(next_event_time, end_time)
                else:
                    self.current_time = end_time
                    break
    
    def get_stats(self) -> Dict:
        """Get simulation statistics."""
        delivered = self.stats['packets_delivered']
        generated = self.stats['packets_generated']
        
        stats = {
            'current_time': self.current_time,
            'packets_generated': generated,
            'packets_delivered': delivered,
            'packets_lost': self.stats['packets_lost'],
            'packet_delivery_ratio': delivered / max(1, generated),
            'avg_delay': self.stats['total_delay'] / max(1, delivered),
            'avg_hops': self.stats['total_hops'] / max(1, delivered),
            'active_packets': len(self.active_packets),
        }
        
        return stats
    
    def reset(self):
        """Reset simulator state."""
        self.event_queue.clear()
        self.current_time = 0.0
        self.active_packets.clear()
        self.delivered_packets.clear()
        self.lost_packets.clear()
        self.packet_counter = 0
        self.stats = {
            'packets_generated': 0,
            'packets_delivered': 0,
            'packets_lost': 0,
            'total_delay': 0.0,
            'total_hops': 0,
        }

