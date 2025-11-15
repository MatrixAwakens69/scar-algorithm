"""
Network Topology Visualization

This module provides network topology rendering with congestion visualization.
"""

from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from src.core.network_topology import NetworkTopology
from src.core.congestion_monitor import CongestionMonitor


class NetworkRenderer:
    """Renders network topology with congestion visualization."""
    
    def __init__(self, topology: NetworkTopology, congestion_monitor: CongestionMonitor,
                 figsize: Tuple[int, int] = (12, 8), node_size: int = 300):
        """
        Initialize network renderer.
        
        Args:
            topology: NetworkTopology instance
            congestion_monitor: CongestionMonitor instance
            figsize: Figure size (width, height)
            node_size: Size of nodes in visualization
        """
        self.topology = topology
        self.congestion_monitor = congestion_monitor
        self.figsize = figsize
        self.node_size = node_size
        
        self.fig = None
        self.ax = None
        self.pos = None  # Node positions
        
        # Color scheme for congestion
        self.congestion_colors = {
            'low': '#00FF00',      # Green
            'medium': '#FFFF00',   # Yellow
            'high': '#FF0000'      # Red
        }
        
        # Cache for link colors and widths to avoid recalculation
        self._link_color_cache: Dict[Tuple[int, int], str] = {}
        self._link_width_cache: Dict[Tuple[int, int], float] = {}
        self._cache_dirty = True
    
    def _calculate_layout(self, layout_type: str = 'spring'):
        """
        Calculate node positions for visualization.
        
        Args:
            layout_type: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        """
        if layout_type == 'spring':
            self.pos = nx.spring_layout(self.topology.graph, seed=42, k=1, iterations=50)
        elif layout_type == 'circular':
            self.pos = nx.circular_layout(self.topology.graph)
        elif layout_type == 'kamada_kawai':
            self.pos = nx.kamada_kawai_layout(self.topology.graph)
        else:
            self.pos = nx.spring_layout(self.topology.graph, seed=42)
    
    def _get_link_color(self, node1: int, node2: int) -> str:
        """
        Get color for a link based on congestion level (cached).
        
        Args:
            node1: First node ID
            node2: Second node ID
        
        Returns:
            Color string
        """
        link_key = (min(node1, node2), max(node1, node2))
        
        # Check cache first
        if not self._cache_dirty and link_key in self._link_color_cache:
            return self._link_color_cache[link_key]
        
        congestion_level = self.congestion_monitor.get_congestion_level(node1, node2)
        color = self.congestion_colors.get(congestion_level, '#808080')
        
        # Cache the result
        self._link_color_cache[link_key] = color
        return color
    
    def _get_link_width(self, node1: int, node2: int, base_width: float = 1.0) -> float:
        """
        Get width for a link based on capacity (cached).
        
        Args:
            node1: First node ID
            node2: Second node ID
            base_width: Base width multiplier
        
        Returns:
            Link width
        """
        link_key = (min(node1, node2), max(node1, node2))
        
        # Check cache first (widths don't change, so cache is permanent)
        if link_key in self._link_width_cache:
            return self._link_width_cache[link_key]
        
        capacity = self.topology.get_link_capacity(node1, node2)
        # Normalize capacity to width (assuming max capacity of 100)
        width = base_width * (capacity / 100.0)
        width = max(0.5, min(3.0, width))
        
        # Cache the result
        self._link_width_cache[link_key] = width
        return width
    
    def render(self, highlight_routes: Optional[List[List[int]]] = None,
              highlight_nodes: Optional[List[int]] = None,
              title: str = "Network Topology"):
        """
        Render network topology.
        
        Args:
            highlight_routes: List of routes to highlight
            highlight_nodes: List of nodes to highlight
            title: Plot title
        """
        if self.pos is None:
            self._calculate_layout()
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        self.ax.clear()
        
        # Draw links
        for u, v in self.topology.get_edges():
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            
            color = self._get_link_color(u, v)
            width = self._get_link_width(u, v)
            
            # Make blue links faint, congestion-colored links bold
            # Check if it's a congestion-colored link (green, yellow, red)
            color_upper = color.upper() if isinstance(color, str) else str(color).upper()
            is_congestion_link = (color_upper in ['#00FF00', '#FFFF00', '#FF0000', '#FFA500'] or
                                 color_upper in ['GREEN', 'YELLOW', 'RED', 'ORANGE'])
            
            if is_congestion_link:
                # Congestion-colored links (green, yellow, red) - make bold
                alpha = 1.0
                linewidth = max(3.0, width * 2.0)
            else:
                # Blue/gray links (low or no congestion) - make faint
                alpha = 0.2
                linewidth = max(0.3, width * 0.3)
            
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha, zorder=1)
        
        # Mark cache as clean after rendering (will be marked dirty on next render if congestion changes)
        self._cache_dirty = False
        
        # Highlight routes
        if highlight_routes:
            for route in highlight_routes:
                if len(route) < 2:
                    continue
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    if self.topology.graph.has_edge(u, v):
                        x1, y1 = self.pos[u]
                        x2, y2 = self.pos[v]
                        self.ax.plot([x1, x2], [y1, y2], color='blue', 
                                    linewidth=3, alpha=0.8, zorder=2, linestyle='--')
        
        # Draw nodes
        node_colors = []
        for node in self.topology.get_nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('cyan')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(self.topology.graph, self.pos, 
                              node_color=node_colors,
                              node_size=self.node_size,
                              ax=self.ax)
        
        # Draw node labels
        nx.draw_networkx_labels(self.topology.graph, self.pos,
                               font_size=8,
                               ax=self.ax)
        
        # Add comprehensive legend
        legend_elements = [
            mpatches.Patch(color='#00FF00', label='Low Congestion (Links)'),
            mpatches.Patch(color='#FFFF00', label='Medium Congestion (Links)'),
            mpatches.Patch(color='#FF0000', label='High Congestion (Links)'),
        ]
        
        # Add line style explanations
        from matplotlib.lines import Line2D
        line_legend = [
            Line2D([0], [0], color='blue', linewidth=2, linestyle='-', label='Regular Network Links'),
            Line2D([0], [0], color='blue', linewidth=3, linestyle='--', label='Active Routes (Dashed)'),
        ]
        
        # Combine legends
        all_legend = legend_elements + line_legend
        self.ax.legend(handles=all_legend, loc='upper left', fontsize=8)
        
        self.ax.set_title(title)
        self.ax.axis('off')
        
        plt.tight_layout()
    
    def show(self):
        """Display the plot."""
        if self.fig:
            plt.show()
    
    def save(self, filename: str, dpi: int = 300):
        """
        Save plot to file.
        
        Args:
            filename: Output filename
            dpi: Resolution
        """
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    def close(self):
        """Close the figure."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

