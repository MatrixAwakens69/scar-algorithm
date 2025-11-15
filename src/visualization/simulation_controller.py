"""
Simulation Controller

This module provides step-by-step simulation control with state management.
"""

from typing import Dict, List, Optional, Callable
from enum import Enum
from src.simulation.network_simulator import NetworkSimulator
from src.visualization.network_renderer import NetworkRenderer


class SimulationState(Enum):
    """Simulation states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"


class SimulationController:
    """Controls step-by-step simulation with play/pause/step functionality."""
    
    def __init__(self, simulator: NetworkSimulator, renderer: NetworkRenderer):
        """
        Initialize simulation controller.
        
        Args:
            simulator: NetworkSimulator instance
            renderer: NetworkRenderer instance
        """
        self.simulator = simulator
        self.renderer = renderer
        
        self.state = SimulationState.STOPPED
        self.speed_multiplier = 1.0
        self.step_size = 1.0  # Time units per step
        
        # Callbacks
        self.on_step_callback: Optional[Callable] = None
        self.on_state_change_callback: Optional[Callable] = None
        
        # Simulation history for step-back
        self.history: List[Dict] = []
        self.history_index = -1
        self.max_history = 100
    
    def play(self):
        """Start or resume simulation."""
        if self.state == SimulationState.STOPPED:
            self._save_state()
        
        self.state = SimulationState.RUNNING
        self._notify_state_change()
    
    def pause(self):
        """Pause simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self._save_state()
            self._notify_state_change()
    
    def stop(self):
        """Stop simulation."""
        self.state = SimulationState.STOPPED
        self._save_state()
        self._notify_state_change()
    
    def step_forward(self):
        """Execute one simulation step."""
        if self.state == SimulationState.STOPPED:
            self._save_state()
        
        self.state = SimulationState.STEPPING
        
        # Execute step
        executed = self.simulator.step()
        
        if executed:
            self._save_state()
            self._notify_step()
        
        # Auto-pause after step
        if self.state == SimulationState.STEPPING:
            self.state = SimulationState.PAUSED
            self._notify_state_change()
        
        return executed
    
    def step_backward(self):
        """Step backward in simulation history."""
        if self.history_index > 0:
            self.history_index -= 1
            self._restore_state(self.history[self.history_index])
            self._notify_step()
            return True
        return False
    
    def run_for_time(self, duration: float):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Duration to simulate
        """
        if self.state == SimulationState.STOPPED:
            self._save_state()
        
        self.state = SimulationState.RUNNING
        self._notify_state_change()
        
        end_time = self.simulator.current_time + duration
        
        while (self.state == SimulationState.RUNNING and 
               self.simulator.current_time < end_time):
            self.simulator.step()
            self._notify_step()
        
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self._save_state()
            self._notify_state_change()
    
    def update(self):
        """Update visualization."""
        # Get current routes
        route_table = self.simulator.scar_algorithm.get_route_table()
        routes = list(route_table.values()) if route_table else []
        
        # Get active packet locations
        active_nodes = []
        for packet in self.simulator.active_packets.values():
            if packet.current_node is not None:
                active_nodes.append(packet.current_node)
        
        # Render network
        self.renderer.render(
            highlight_routes=routes,
            highlight_nodes=active_nodes,
            title=f"SCAR Simulation - Time: {self.simulator.current_time:.2f}"
        )
    
    def set_speed(self, multiplier: float):
        """
        Set simulation speed multiplier.
        
        Args:
            multiplier: Speed multiplier (> 0)
        """
        self.speed_multiplier = max(0.1, min(10.0, multiplier))
    
    def set_step_size(self, step_size: float):
        """
        Set step size for stepping.
        
        Args:
            step_size: Time units per step
        """
        self.step_size = max(0.1, step_size)
    
    def get_stats(self) -> Dict:
        """Get current simulation statistics."""
        sim_stats = self.simulator.get_stats()
        algo_stats = self.simulator.scar_algorithm.get_algorithm_stats(
            int(self.simulator.current_time)
        )
        
        return {
            'simulation': sim_stats,
            'algorithm': algo_stats,
            'state': self.state.value,
            'current_time': self.simulator.current_time,
        }
    
    def _save_state(self):
        """Save current simulation state to history."""
        state = {
            'time': self.simulator.current_time,
            'stats': self.simulator.get_stats(),
            'route_table': self.simulator.scar_algorithm.get_route_table().copy(),
            'active_packets': {pid: {
                'current_node': p.current_node,
                'next_hop': p.next_hop,
                'route': p.route
            } for pid, p in self.simulator.active_packets.items()}
        }
        
        # Remove future history if we're going back
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(state)
        self.history_index = len(self.history) - 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.history_index = len(self.history) - 1
    
    def _restore_state(self, state: Dict):
        """Restore simulation state from history."""
        # Note: Full state restoration would require more complex implementation
        # For now, we'll just update what we can
        pass
    
    def _notify_step(self):
        """Notify step callback."""
        if self.on_step_callback:
            self.on_step_callback()
    
    def _notify_state_change(self):
        """Notify state change callback."""
        if self.on_state_change_callback:
            self.on_state_change_callback(self.state)
    
    def set_on_step_callback(self, callback: Callable):
        """Set callback for step events."""
        self.on_step_callback = callback
    
    def set_on_state_change_callback(self, callback: Callable):
        """Set callback for state change events."""
        self.on_state_change_callback = callback

