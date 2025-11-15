"""
Threaded Simulation Controller

This module provides a thread-safe simulation controller that runs
simulation steps in a background thread to keep the UI responsive.
"""

import threading
import queue
import time
from typing import Optional, Callable
from src.simulation.network_simulator import NetworkSimulator
from src.visualization.simulation_controller import SimulationController, SimulationState


class ThreadedSimulationController(SimulationController):
    """Thread-safe simulation controller with background processing."""
    
    def __init__(self, simulator: NetworkSimulator, renderer):
        """
        Initialize threaded simulation controller.
        
        Args:
            simulator: NetworkSimulator instance
            renderer: NetworkRenderer instance
        """
        super().__init__(simulator, renderer)
        
        # Threading components
        self.simulation_thread: Optional[threading.Thread] = None
        self.update_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.running = False
        self.lock = threading.Lock()
        
        # Performance settings
        # At 1x speed: 1 step per 1 second real time (20 steps over 20 seconds)
        self.base_steps_per_update = 1  # Base steps before updating UI
        self.max_steps_per_cycle = 20  # Maximum steps per animation cycle
        self.speed_multiplier = 1.0  # Speed multiplier
        self.last_step_time = None  # Track real time for step rate control
        self.steps_per_second = 1.0  # At 1x: 1 step per 1 second real time
    
    def play(self):
        """Start or resume simulation in background thread."""
        with self.lock:
            if self.state == SimulationState.STOPPED:
                self._save_state()
            
            self.state = SimulationState.RUNNING
            self._notify_state_change()
            
            if not self.running or (self.simulation_thread and not self.simulation_thread.is_alive()):
                self.running = True
                if self.simulation_thread and self.simulation_thread.is_alive():
                    # Thread already running, just continue
                    return
                self.simulation_thread = threading.Thread(
                    target=self._simulation_worker,
                    daemon=True
                )
                self.simulation_thread.start()
    
    def pause(self):
        """Pause simulation."""
        with self.lock:
            if self.state == SimulationState.RUNNING:
                self.state = SimulationState.PAUSED
                self._save_state()
                self._notify_state_change()
    
    def stop(self):
        """Stop simulation."""
        with self.lock:
            self.state = SimulationState.STOPPED
            self.running = False
            self._save_state()
            self._notify_state_change()
    
    def step_forward(self):
        """Execute one simulation step (synchronous for step button)."""
        with self.lock:
            if self.state == SimulationState.STOPPED:
                self._save_state()
            
            self.state = SimulationState.STEPPING
            
            # Execute step directly (no threading for step)
            executed = self.simulator.step()
            
            if executed:
                self._save_state()
                self._notify_step()
            
            # Auto-pause after step
            if self.state == SimulationState.STEPPING:
                self.state = SimulationState.PAUSED
                self._notify_state_change()
            
            return executed
    
    def _simulation_worker(self):
        """Background worker thread that processes simulation steps."""
        steps_processed = 0
        self.last_step_time = time.time()
        
        while self.running:
            with self.lock:
                should_run = (self.state == SimulationState.RUNNING)
            
            if not should_run:
                time.sleep(0.01)  # Small delay when paused
                continue
            
            # Step-based rate control: At 1x speed, 1 step per 1 second real time
            # Calculate target steps per second based on speed multiplier
            target_steps_per_second = self.steps_per_second * self.speed_multiplier
            
            # Calculate time since last step
            current_time = time.time()
            elapsed_time = current_time - self.last_step_time
            
            # Calculate how many steps we should have processed by now
            target_steps = elapsed_time * target_steps_per_second
            
            # Process steps (at most 1 step per iteration to maintain smooth rate)
            steps_this_cycle = 0
            if target_steps >= 1.0:
                with self.lock:
                    if self.state != SimulationState.RUNNING:
                        continue
                    
                    executed = self.simulator.step()
                    if not executed:
                        # No more events, pause
                        self.state = SimulationState.PAUSED
                        self.running = False
                        self._notify_state_change()
                        break
                    
                    steps_processed += 1
                    steps_this_cycle = 1
                    self.last_step_time = current_time
                    
                    # Queue update for UI thread
                    try:
                        self.update_queue.put_nowait({
                            'type': 'update',
                            'steps': steps_this_cycle,
                            'time': self.simulator.current_time
                        })
                    except queue.Full:
                        pass  # Skip if queue is full
            else:
                # Not time for next step yet, wait a bit
                time.sleep(0.01)
    
    def get_pending_updates(self):
        """Get all pending updates from simulation thread (non-blocking)."""
        updates = []
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                updates.append(update)
            except queue.Empty:
                break
        return updates
    
    def update(self):
        """Update visualization (called from main thread)."""
        # Always update (not just when updates are queued) for smooth animation
        # Get current routes
        route_table = self.simulator.scar_algorithm.get_route_table()
        routes = list(route_table.values()) if route_table else []
        
        # Get active packet locations (thread-safe snapshot)
        active_nodes = []
        try:
            # Create snapshot to avoid thread-safety issues
            active_packets_snapshot = list(self.simulator.active_packets.values())
            for packet in active_packets_snapshot:
                if packet.current_node is not None:
                    active_nodes.append(packet.current_node)
        except (RuntimeError, KeyError):
            # Dictionary was modified during snapshot, use empty list
            active_nodes = []
        
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
                     1.0 = 1 step per 1 second real time (20 steps over 20 seconds)
                     2.0 = 2 steps per 1 second real time (2x faster)
                     0.5 = 0.5 steps per 1 second real time (2x slower)
        """
        with self.lock:
            self.speed_multiplier = max(0.1, min(10.0, multiplier))
            # Reset timing when speed changes
            self.last_step_time = time.time()
    
    def cleanup(self):
        """Clean up threads and resources."""
        self.running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)

