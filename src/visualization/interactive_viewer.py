"""
Interactive Viewer

This module provides an interactive GUI for controlling the simulation.
"""

from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import threading
import time
from src.visualization.simulation_controller import SimulationController, SimulationState
from src.visualization.threaded_simulation import ThreadedSimulationController
from src.visualization.metrics_dashboard import MetricsDashboard
from src.visualization.packet_visualizer import PacketVisualizer


class InteractiveViewer:
    """Interactive GUI for simulation control."""
    
    def __init__(self, controller: SimulationController,
                 metrics_dashboard: MetricsDashboard,
                 packet_visualizer: PacketVisualizer,
                 simulator=None, traffic_generator=None, config=None):
        """
        Initialize interactive viewer.
        
        Args:
            controller: SimulationController instance
            metrics_dashboard: MetricsDashboard instance
            packet_visualizer: PacketVisualizer instance
            simulator: NetworkSimulator instance (for restart)
            traffic_generator: TrafficGenerator instance (for restart)
            config: Configuration dict (for restart)
        """
        self.controller = controller
        self.metrics_dashboard = metrics_dashboard
        self.packet_visualizer = packet_visualizer
        self.simulator = simulator
        self.traffic_generator = traffic_generator
        self.config = config
        
        # Get simulator reference for log updates
        if hasattr(controller, 'simulator'):
            self.simulator = controller.simulator
        
        self.fig = None
        self.ax_network = None
        self.ax_metrics = None
        self.ax_status = None
        self.animation = None  # Animation timer for continuous simulation
        self._active_animations = []  # Keep references to prevent garbage collection
        self._stopped_animations = []  # Keep references to stopped animations
        self._animation_keepalive = None  # Keep-alive function for animation
        self.running = False
        self.status_text = None
        self.log_text = None
        self.log_entries = []
        self.max_log_entries = 6  # Maximum number of log entries to display
        self.log_scroll_offset = 0  # Scroll offset for log window
        self.speed_selector = None
        self.current_speed = 1.0  # Default speed multiplier
        
        # Set up callbacks
        self.controller.set_on_step_callback(self._on_step)
        self.controller.set_on_state_change_callback(self._on_state_change)
    
    def create_ui(self):
        """Create the user interface."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(18, 10))
        
        # Network visualization (left side) - larger
        self.ax_network = plt.subplot2grid((5, 4), (0, 0), colspan=2, rowspan=5)
        
        # Set renderer to use this axis
        self.controller.renderer.fig = self.fig
        self.controller.renderer.ax = self.ax_network
        
        # Metrics dashboard (top right) - ensure proper sizing
        self.ax_metrics = plt.subplot2grid((5, 4), (0, 2), colspan=2, rowspan=2)
        self.metrics_dashboard.fig = self.fig
        # Don't set ax directly, pass it to render method
        
        # Status text area (middle right)
        self.ax_status = plt.subplot2grid((5, 4), (2, 2), colspan=2, rowspan=1)
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.05, 0.95, 'Simulation Ready', 
                                               transform=self.ax_status.transAxes,
                                               fontsize=10, verticalalignment='top',
                                               family='monospace',
                                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Log area (bottom right) - scrollable log of network events
        # Shifted up and right to avoid overlap
        self.ax_log = plt.subplot2grid((5, 4), (3, 2), colspan=2, rowspan=1)
        self.ax_log.axis('off')
        self.log_text = self.ax_log.text(0.02, 0.98, '', 
                                         transform=self.ax_log.transAxes,
                                         fontsize=16, verticalalignment='top',  # 2x original (8 -> 16)
                                         family='monospace',
                                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        # Add scroll buttons for log
        ax_scroll_up = plt.axes([0.90, 0.22, 0.02, 0.03])
        ax_scroll_down = plt.axes([0.90, 0.19, 0.02, 0.03])
        self.btn_scroll_up = Button(ax_scroll_up, '▲')
        self.btn_scroll_down = Button(ax_scroll_down, '▼')
        self.btn_scroll_up.on_clicked(self._on_log_scroll_up)
        self.btn_scroll_down.on_clicked(self._on_log_scroll_down)
        
        # Control buttons
        ax_play = plt.axes([0.52, 0.05, 0.08, 0.04])
        ax_pause = plt.axes([0.61, 0.05, 0.08, 0.04])
        ax_stop = plt.axes([0.70, 0.05, 0.08, 0.04])
        ax_step = plt.axes([0.52, 0.1, 0.08, 0.04])
        ax_restart = plt.axes([0.61, 0.1, 0.08, 0.04])
        
        self.btn_play = Button(ax_play, 'Play')
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_step = Button(ax_step, 'Step')
        self.btn_restart = Button(ax_restart, 'Restart')
        
        self.btn_play.on_clicked(self._on_play_clicked)
        self.btn_pause.on_clicked(self._on_pause_clicked)
        self.btn_stop.on_clicked(self._on_stop_clicked)
        self.btn_step.on_clicked(self._on_step_clicked)
        self.btn_restart.on_clicked(self._on_restart_clicked)
        
        # Speed control - moved up above controls
        ax_speed_label = plt.axes([0.79, 0.32, 0.12, 0.02])
        ax_speed_label.axis('off')
        ax_speed_label.text(0.5, 0.5, 'Speed:', ha='center', va='center', 
                           transform=ax_speed_label.transAxes, fontsize=10, weight='bold')
        
        # Create radio buttons for speed selection - bigger area with better spacing
        ax_speed = plt.axes([0.79, 0.22, 0.12, 0.10])
        ax_speed.axis('off')
        speed_options = ['0.25x', '0.5x', '1x', '2x', '4x']
        self.speed_selector = RadioButtons(ax_speed, speed_options, active=2)  # Default to 1x
        
        # Make radio buttons bigger and better spaced
        try:
            if hasattr(self.speed_selector, 'circles'):
                for circle in self.speed_selector.circles:
                    circle.set_radius(0.10)  # Bigger circles
                    circle.set_linewidth(2)  # Thicker border
            if hasattr(self.speed_selector, 'labels'):
                for text in self.speed_selector.labels:
                    text.set_fontsize(11)  # Bigger text
                    text.set_fontweight('bold')  # Bold text
        except Exception as e:
            print(f"Warning: Could not customize speed selector appearance: {e}")
        
        # Set up callback with error handling
        try:
            self.speed_selector.on_clicked(self._on_speed_changed)
        except Exception as e:
            print(f"Warning: Could not set up speed selector callback: {e}")
        
        # Initialize speed on controller
        try:
            if hasattr(self.controller, 'set_speed'):
                self.controller.set_speed(self.current_speed)
        except Exception as e:
            print(f"Warning: Could not initialize simulation speed: {e}")
        
        plt.tight_layout()
    
    def update(self):
        """Update the visualization (non-blocking)."""
        if self.fig is None:
            self.create_ui()
        
        try:
            # Update network visualization (non-blocking)
            self.controller.update()
            
            # Update metrics
            stats = self.controller.get_stats()
            self.metrics_dashboard.update(stats)
            
            # Draw packets (with thread-safe access)
            if self.controller.renderer.ax:
                try:
                    self.packet_visualizer.draw_packets(self.controller.renderer.ax)
                except (RuntimeError, KeyError) as e:
                    # Dictionary was modified during iteration, skip this frame
                    # This is expected in multi-threaded environment
                    pass
            
            # Update metrics dashboard
            self.metrics_dashboard.render_single_plot(self.ax_metrics)
            
            # Update status text
            self._update_status_text()
            
            # Update log
            self._update_log()
            
            # Use draw_idle for non-blocking updates (prevents mouse cursor from going to loading state)
            if self.fig and self.fig.canvas:
                self.fig.canvas.draw_idle()
        except Exception as e:
            # Silently handle any update errors to prevent blocking
            pass
    
    def _update_status_text(self):
        """Update the status text display."""
        if self.status_text is None:
            return
        
        stats = self.controller.get_stats()
        sim_stats = stats.get('simulation', {})
        state = self.controller.state.value.upper()
        
        # Get algorithm stats for utilization
        algo_stats = stats.get('algorithm', {})
        congestion_summary = algo_stats.get('congestion', {})
        avg_utilization = congestion_summary.get('avg_utilization', 0.0)
        
        status_lines = [
            f"State: {state}",
            f"Time: {self.controller.simulator.current_time:.2f}",
            f"Packets: {sim_stats.get('packets_generated', 0)} gen, "
            f"{sim_stats.get('packets_delivered', 0)} del, "
            f"{sim_stats.get('active_packets', 0)} active",
            f"Delivery Ratio: {sim_stats.get('packet_delivery_ratio', 0):.2%}",
            f"Avg Delay: {sim_stats.get('avg_delay', 0):.2f}",
            f"Utilization: {avg_utilization:.2%}",
        ]
        
        # Get last event info
        if hasattr(self.controller.simulator, 'last_event_info'):
            status_lines.append(f"Last Event: {self.controller.simulator.last_event_info}")
        
        self.status_text.set_text('\n'.join(status_lines))
    
    def _update_log(self):
        """Update the log display with network events."""
        if self.log_text is None or self.simulator is None:
            return
        
        # Get latest event info from simulator
        event_info = getattr(self.simulator, 'last_event_info', '')
        if event_info and event_info != 'No events yet':
            # Add timestamp
            time_str = f"{self.simulator.current_time:.2f}"
            log_entry = f"[{time_str}] {event_info}"
            
            # Add to log entries if it's new
            if not self.log_entries or self.log_entries[-1] != log_entry:
                self.log_entries.append(log_entry)
                
                # Keep more entries for scrolling (store up to 100 entries for scrollable log)
                max_stored_entries = 100
                if len(self.log_entries) > max_stored_entries:
                    self.log_entries = self.log_entries[-max_stored_entries:]
                    # Adjust scroll offset if we removed entries
                    removed_count = len(self.log_entries) - max_stored_entries
                    if removed_count > 0:
                        self.log_scroll_offset = max(0, self.log_scroll_offset - removed_count)
        
        # Update log text (show entries based on scroll offset, scrollable)
        if self.log_entries:
            # Calculate which entries to show based on scroll offset
            total_entries = len(self.log_entries)
            start_idx = max(0, total_entries - self.max_log_entries - self.log_scroll_offset)
            end_idx = total_entries - self.log_scroll_offset
            display_entries = self.log_entries[start_idx:end_idx] if end_idx > 0 else []
            
            if display_entries:
                log_text = '\n'.join(display_entries)
            else:
                log_text = 'No more entries...'
        else:
            log_text = 'No events yet...'
        
        self.log_text.set_text(log_text)
    
    def show(self):
        """Display the interactive viewer."""
        if self.fig is None:
            self.create_ui()
        
        self.update()
        # Use interactive mode to allow animations to run
        plt.ion()
        # Show the figure and keep it alive
        # Ensure figure is displayed and canvas is ready
        if self.fig.canvas:
            self.fig.canvas.draw()
        # Show the figure (this registers it with the backend)
        self.fig.show()
        # Block to keep window open - this ensures animations can run
        plt.show(block=True)
    
    def _on_step(self):
        """Callback for step events."""
        self.update()
    
    def _on_step_clicked(self, event):
        """Handle Step button click."""
        executed = self.controller.step_forward()
        if executed:
            self._update_status_text()
            self.update()
    
    def _on_play_clicked(self, event):
        """Handle Play button click."""
        try:
            # Ensure speed is set before starting
            if hasattr(self.controller, 'set_speed'):
                self.controller.set_speed(self.current_speed)
            
            self.controller.play()
            self._start_animation()
        except Exception as e:
            print(f"Error starting simulation: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_pause_clicked(self, event):
        """Handle Pause button click."""
        self.controller.pause()
        self._stop_animation()
    
    def _on_stop_clicked(self, event):
        """Handle Stop button click."""
        self.controller.stop()
        self._stop_animation()
    
    def _on_restart_clicked(self, event):
        """Handle Restart button click."""
        if self.simulator is None or self.traffic_generator is None or self.config is None:
            print("Warning: Cannot restart - missing simulator, traffic_generator, or config")
            return
        
        # Stop current simulation
        self.controller.stop()
        self._stop_animation()
        
        # Reset simulator
        self.simulator.reset()
        
        # Reset SCAR algorithm
        if hasattr(self.controller, 'simulator') and hasattr(self.controller.simulator, 'scar_algorithm'):
            scar_algorithm = self.controller.simulator.scar_algorithm
            # Reset route table and stability mechanism
            if hasattr(scar_algorithm, 'route_table'):
                scar_algorithm.route_table.clear()
            if hasattr(scar_algorithm, 'stability_mechanism'):
                scar_algorithm.stability_mechanism.reset()
            if hasattr(scar_algorithm, 'congestion_monitor'):
                # Reset congestion history
                if hasattr(scar_algorithm.congestion_monitor, 'congestion_history'):
                    scar_algorithm.congestion_monitor.congestion_history.clear()
        
        # Reset metrics dashboard (completely clear plot)
        self.metrics_dashboard.reset()
        if self.ax_metrics:
            # Properly clear all axes including twin axes
            self.ax_metrics.clear()
            # Remove any existing twin axes by clearing the figure's children
            for child in list(self.ax_metrics.get_children()):
                if hasattr(child, 'remove'):
                    try:
                        child.remove()
                    except:
                        pass
            # Reinitialize with clean state
            self.ax_metrics.text(0.5, 0.5, 'No data yet', ha='center', va='center', 
                                transform=self.ax_metrics.transAxes, fontsize=12)
            self.ax_metrics.set_title('Performance Metrics')
            self.ax_metrics.set_xlabel('Time', fontsize=9)
            # Force redraw to ensure clean state
            if self.fig and self.fig.canvas:
                self.fig.canvas.draw_idle()
        
        # Clear log
        self.log_entries.clear()
        if self.log_text:
            self.log_text.set_text('Simulation restarted...')
        
        # Regenerate traffic
        sim_config = self.config.get('simulation', {})
        num_packets = sim_config.get('num_packets', 100)
        time_range = (0.0, sim_config.get('duration', 1000.0))
        
        self.traffic_generator.generate_random_traffic(
            num_packets=num_packets,
            packet_size=sim_config.get('packet_size', 1500.0),
            time_range=time_range
        )
        
        # Update visualization (non-blocking)
        self.controller.update()
        if self.fig and self.fig.canvas:
            self.fig.canvas.draw_idle()
        
        print("Simulation restarted")
    
    def _start_animation(self):
        """Start the animation timer for continuous simulation."""
        if self.fig is None:
            return
        
        # Stop any existing animation first
        self._stop_animation()
        
        # Calculate interval based on speed (faster speed = shorter interval)
        # Base interval is 1000ms (10x slower than before for better observation)
        # Divide by speed multiplier to get actual interval
        base_interval = 1000  # 1 second base interval (10x slower than original 100ms)
        animation_interval = max(50, int(base_interval / self.current_speed))
        
        # Create animation that runs every interval ms for smoother updates
        # Use repeat=True to keep it running
        try:
            # Ensure figure is shown and interactive
            plt.ion()
            
            # Ensure figure canvas is ready
            if not hasattr(self.fig, 'canvas') or self.fig.canvas is None:
                # Canvas not ready, can't create animation
                print("Warning: Figure canvas not ready, cannot start animation")
                return
            
            # Ensure figure is displayed
            if not self.fig.canvas.manager:
                self.fig.show()
            
            # Draw once to ensure figure is registered
            self.fig.canvas.draw()
            
            # Store animation in a way that prevents garbage collection
            # CRITICAL: Store reference BEFORE creating to prevent immediate deletion
            anim_refs = []
            if not hasattr(self, '_active_animations'):
                self._active_animations = []
            
            # Create the animation
            self.animation = FuncAnimation(
                self.fig,
                self._animation_step,
                interval=animation_interval,  # Update interval based on speed
                blit=False,
                cache_frame_data=False,
                repeat=True
            )
            
            # Immediately store reference to prevent garbage collection
            self._active_animations.append(self.animation)
            anim_refs.append(self.animation)
            
            # Store reference on figure as well
            if not hasattr(self.fig, '_animations'):
                self.fig._animations = []
            self.fig._animations.append(self.animation)
            
            # Store on class instance to keep alive
            if not hasattr(InteractiveViewer, '_all_animations'):
                InteractiveViewer._all_animations = []
            InteractiveViewer._all_animations.append(self.animation)
            
            # Force canvas to draw to ensure animation is registered
            self.fig.canvas.draw_idle()
            
            # Ensure animation is not immediately garbage collected
            # by keeping a reference in the closure
            def keep_alive():
                return self.animation
            self._animation_keepalive = keep_alive
            
        except Exception as e:
            print(f"Warning: Could not start animation: {e}")
            import traceback
            traceback.print_exc()
            self.animation = None
    
    def _stop_animation(self):
        """Stop the animation timer."""
        if self.animation is not None:
            try:
                # Stop the event source
                if hasattr(self.animation, 'event_source') and self.animation.event_source:
                    self.animation.event_source.stop()
                # Pause the animation
                if hasattr(self.animation, 'pause'):
                    self.animation.pause()
            except Exception as e:
                # Ignore errors when stopping
                pass
            
            # Remove from figure's animation list
            if self.fig and hasattr(self.fig, '_animations'):
                if self.animation in self.fig._animations:
                    self.fig._animations.remove(self.animation)
            
            # Remove from viewer's animation list
            if hasattr(self, '_active_animations'):
                if self.animation in self._active_animations:
                    self._active_animations.remove(self.animation)
            
            # Don't delete the animation object, just clear the reference
            # This prevents the warning about deletion without rendering
            anim_ref = self.animation
            self.animation = None
            
            # Only delete after a short delay to ensure it was rendered
            # Actually, we'll keep a reference to prevent immediate deletion
            if not hasattr(self, '_stopped_animations'):
                self._stopped_animations = []
            self._stopped_animations.append(anim_ref)
    
    def _animation_step(self, frame):
        """Animation step callback - runs continuously when playing (non-blocking)."""
        # Ensure we have a valid figure and animation
        if self.fig is None or self.animation is None:
            return
        
        # For threaded controller, just check for updates (non-blocking)
        if isinstance(self.controller, ThreadedSimulationController):
            # Check for pending updates from simulation thread
            if self.controller.state == SimulationState.RUNNING:
                # Update visualization with latest state (non-blocking)
                try:
                    self.update()  # This now uses draw_idle internally
                except:
                    pass  # Silently handle errors to prevent blocking
            elif self.controller.state != SimulationState.RUNNING:
                # Stop animation if state changed
                self._stop_animation()
        else:
            # Original synchronous behavior for non-threaded controller
            if self.controller.state == SimulationState.RUNNING:
                # Step the simulation
                executed = self.controller.simulator.step()
                if executed:
                    # Update visualization
                    self.update()
                else:
                    # No more events, pause simulation
                    self.controller.pause()
            elif self.controller.state != SimulationState.RUNNING:
                # Stop animation if state changed
                self._stop_animation()
    
    def _on_speed_changed(self, label):
        """Handle speed selection change."""
        try:
            # Parse speed from label (e.g., "2x" -> 2.0)
            speed_str = str(label).replace('x', '')
            self.current_speed = float(speed_str)
        except (ValueError, AttributeError, TypeError):
            self.current_speed = 1.0
        
        # Update simulation controller speed if it supports it
        try:
            if hasattr(self.controller, 'set_speed'):
                self.controller.set_speed(self.current_speed)
        except Exception as e:
            print(f"Warning: Could not set simulation speed: {e}")
        
        # If animation is running, restart it with new speed
        try:
            if self.animation is not None and self.controller.state == SimulationState.RUNNING:
                self._stop_animation()
                self._start_animation()
        except Exception as e:
            print(f"Warning: Could not restart animation with new speed: {e}")
    
    def _on_state_change(self, state: SimulationState):
        """Callback for state change events."""
        # Update button states based on simulation state
        if state == SimulationState.RUNNING:
            self._start_animation()
        else:
            self._stop_animation()
    
    def _on_log_scroll_up(self, event):
        """Scroll log window up (show older entries)."""
        if self.log_entries:
            max_scroll = max(0, len(self.log_entries) - self.max_log_entries)
            self.log_scroll_offset = min(self.log_scroll_offset + 1, max_scroll)
            self._update_log()
    
    def _on_log_scroll_down(self, event):
        """Scroll log window down (show newer entries)."""
        self.log_scroll_offset = max(0, self.log_scroll_offset - 1)
        self._update_log()
