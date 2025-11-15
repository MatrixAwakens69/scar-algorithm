"""
Packet Visualization

This module provides animated packet movement visualization.
"""

from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.simulation.network_simulator import NetworkSimulator, Packet
from src.visualization.network_renderer import NetworkRenderer


class PacketVisualizer:
    """Visualizes packet movement along routes."""
    
    def __init__(self, simulator: NetworkSimulator, renderer: NetworkRenderer):
        """
        Initialize packet visualizer.
        
        Args:
            simulator: NetworkSimulator instance
            renderer: NetworkRenderer instance
        """
        self.simulator = simulator
        self.renderer = renderer
        self.packet_markers: Dict[int, plt.Line2D] = {}
    
    def draw_packets(self, ax):
        """
        Draw packets on the network visualization.
        
        Args:
            ax: Matplotlib axes
        """
        if self.renderer.pos is None:
            return
        
        # Create a snapshot of active_packets to avoid thread-safety issues
        # The simulation thread may modify the dictionary while we're iterating
        try:
            # Get a copy of items to iterate safely
            active_packets_snapshot = list(self.simulator.active_packets.items())
        except (RuntimeError, KeyError):
            # Dictionary was modified during snapshot, skip this frame
            return
        
        # Draw active packets from snapshot
        for packet_id, packet in active_packets_snapshot:
            # Double-check packet still exists (may have been removed)
            if packet_id not in self.simulator.active_packets:
                continue
                
            if packet.current_node is None or packet.next_hop is None:
                continue
            
            # Get positions
            if packet.current_node not in self.renderer.pos:
                continue
            
            x1, y1 = self.renderer.pos[packet.current_node]
            
            # Draw packet as a circle
            circle = plt.Circle((x1, y1), 0.02, color='red', zorder=5)
            ax.add_patch(circle)
            
            # Draw arrow to next hop if available
            if packet.next_hop and packet.next_hop in self.renderer.pos:
                x2, y2 = self.renderer.pos[packet.next_hop]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7),
                           zorder=4)
    
    def update_packet_positions(self):
        """Update packet positions for animation."""
        # This would be used for smooth animation
        # For now, packets are drawn at their current node positions
        pass

