"""
Metrics Dashboard

This module provides real-time performance metrics display.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import deque


class MetricsDashboard:
    """Displays real-time performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics dashboard.
        
        Args:
            max_history: Maximum number of data points to keep
        """
        self.max_history = max_history
        
        # Metrics history
        self.time_history: deque = deque(maxlen=max_history)
        self.packet_delivery_history: deque = deque(maxlen=max_history)
        self.avg_delay_history: deque = deque(maxlen=max_history)
        self.congestion_history: deque = deque(maxlen=max_history)
        self.stability_history: deque = deque(maxlen=max_history)
        
        self.fig = None
        self.axes = None
    
    def update(self, stats: Dict):
        """
        Update metrics with new statistics.
        
        Args:
            stats: Dictionary with simulation and algorithm statistics
        """
        sim_stats = stats.get('simulation', {})
        algo_stats = stats.get('algorithm', {})
        current_time = stats.get('current_time', 0.0)
        
        # Update history
        self.time_history.append(current_time)
        self.packet_delivery_history.append(sim_stats.get('packet_delivery_ratio', 0.0))
        self.avg_delay_history.append(sim_stats.get('avg_delay', 0.0))
        
        congestion_summary = algo_stats.get('congestion', {})
        self.congestion_history.append(congestion_summary.get('avg_utilization', 0.0))
        
        stability_summary = algo_stats.get('stability', {})
        self.stability_history.append(stability_summary.get('avg_stability', 0.0))
    
    def render(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Render metrics dashboard.
        
        Args:
            figsize: Figure size
        """
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        
        axes = self.axes.flatten()
        
        # Plot 1: Packet Delivery Ratio
        axes[0].clear()
        if len(self.time_history) > 0:
            axes[0].plot(list(self.time_history), list(self.packet_delivery_history), 'b-')
        axes[0].set_title('Packet Delivery Ratio')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Ratio')
        axes[0].grid(True)
        axes[0].set_ylim(0, 1)
        
        # Plot 2: Average Delay
        axes[1].clear()
        if len(self.time_history) > 0:
            axes[1].plot(list(self.time_history), list(self.avg_delay_history), 'r-')
        axes[1].set_title('Average End-to-End Delay')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Delay (time units)')
        axes[1].grid(True)
        
        # Plot 3: Network Congestion
        axes[2].clear()
        if len(self.time_history) > 0:
            axes[2].plot(list(self.time_history), list(self.congestion_history), 'g-')
        axes[2].set_title('Average Network Utilization')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Utilization')
        axes[2].grid(True)
        axes[2].set_ylim(0, 1)
        
        # Plot 4: Route Stability
        axes[3].clear()
        if len(self.time_history) > 0:
            axes[3].plot(list(self.time_history), list(self.stability_history), 'm-')
        axes[3].set_title('Route Stability')
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Stability Score')
        axes[3].grid(True)
        axes[3].set_ylim(0, 1)
        
        plt.tight_layout()
    
    def show(self):
        """Display the dashboard."""
        if self.fig:
            plt.show()
    
    def save(self, filename: str, dpi: int = 300):
        """
        Save dashboard to file.
        
        Args:
            filename: Output filename
            dpi: Resolution
        """
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    def close(self):
        """Close the dashboard."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
    
    def render_single_plot(self, ax):
        """
        Render metrics in a single combined plot.
        
        Args:
            ax: Matplotlib axis to render on
        """
        ax.clear()
        
        if len(self.time_history) == 0:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Performance Metrics')
            return
        
        time_list = list(self.time_history)
        
        # Only plot if we have data
        if len(time_list) == 0:
            return
        
        # Plot multiple metrics on same axis with different y-scales
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # Primary y-axis: Packet Delivery Ratio
        if len(self.packet_delivery_history) > 0 and len(self.packet_delivery_history) == len(time_list):
            ax.plot(time_list, list(self.packet_delivery_history), 'b-', 
                   label='Delivery Ratio', linewidth=2, alpha=0.8)
        ax.set_ylabel('Delivery Ratio', color='b', fontsize=9)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))  # Limit to ~5 ticks
        ax.tick_params(axis='y', labelcolor='b', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Secondary y-axis: Average Delay
        if len(self.avg_delay_history) > 0 and len(self.avg_delay_history) == len(time_list):
            ax2.plot(time_list, list(self.avg_delay_history), 'r-', 
                    label='Avg Delay', linewidth=2, alpha=0.8)
        ax2.set_ylabel('Delay', color='r', fontsize=9)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))  # Limit to ~5 ticks
        ax2.tick_params(axis='y', labelcolor='r', labelsize=8)
        
        # Tertiary y-axis: Network Utilization
        if len(self.congestion_history) > 0 and len(self.congestion_history) == len(time_list):
            ax3.plot(time_list, list(self.congestion_history), 'g-', 
                    label='Utilization', linewidth=2, alpha=0.8)
        ax3.set_ylabel('Utilization', color='g', fontsize=9)
        ax3.set_ylim(0, 1)
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))  # Limit to ~5 ticks
        ax3.tick_params(axis='y', labelcolor='g', labelsize=8)
        
        ax.set_xlabel('Time', fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title('Performance Metrics Over Time', fontsize=10, pad=10)
        
        # Add legend - only include lines that have data
        lines = []
        labels = []
        if len(self.packet_delivery_history) > 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines.extend(lines1)
            labels.extend(labels1)
        if len(self.avg_delay_history) > 0:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines.extend(lines2)
            labels.extend(labels2)
        if len(self.congestion_history) > 0:
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines.extend(lines3)
            labels.extend(labels3)
        
        if lines:
            ax.legend(lines, labels, loc='upper left', fontsize=7, framealpha=0.9)
    
    def reset(self):
        """Reset all metrics history."""
        self.time_history.clear()
        self.packet_delivery_history.clear()
        self.avg_delay_history.clear()
        self.congestion_history.clear()
        self.stability_history.clear()
    
    def get_current_metrics(self) -> Dict:
        """Get current metric values."""
        return {
            'packet_delivery_ratio': self.packet_delivery_history[-1] if self.packet_delivery_history else 0.0,
            'avg_delay': self.avg_delay_history[-1] if self.avg_delay_history else 0.0,
            'avg_congestion': self.congestion_history[-1] if self.congestion_history else 0.0,
            'avg_stability': self.stability_history[-1] if self.stability_history else 0.0,
        }

