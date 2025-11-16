"""
Main Entry Point for SCAR Algorithm Simulation

This script runs the SCAR algorithm simulation with visualization.
"""

import yaml
import sys
import random
from src.core.network_topology import NetworkTopology, load_topology_from_config
from src.core.scar_algorithm import SCARAlgorithm
from src.simulation.network_simulator import NetworkSimulator, SimulationEvent, EventType
from src.simulation.traffic_generator import TrafficGenerator
from src.core.congestion_monitor import CongestionMonitor
from src.visualization.network_renderer import NetworkRenderer
from src.visualization.simulation_controller import SimulationController
from src.visualization.threaded_simulation import ThreadedSimulationController
from src.visualization.packet_visualizer import PacketVisualizer
from src.visualization.metrics_dashboard import MetricsDashboard
from src.visualization.interactive_viewer import InteractiveViewer


def load_config(config_file: str = 'config/config.yaml'):
    """Load configuration from file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function."""
    print("SCAR Algorithm Simulation")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        print(f"Warning: Config file not found. Using defaults.")
        config = {}
    
    # Initialize network topology
    print("Initializing network topology...")
    network_config = config.get('network', {})
    topology = NetworkTopology(network_config)
    print(f"Network created: {topology.graph.number_of_nodes()} nodes, "
          f"{topology.graph.number_of_edges()} links")
    
    # Initialize SCAR algorithm
    print("Initializing SCAR algorithm...")
    algo_config = config.get('algorithm', {})
    scar_algorithm = SCARAlgorithm(
        topology,
        congestion_threshold=algo_config.get('congestion_threshold', 0.7),
        stability_threshold=algo_config.get('stability_threshold', 0.6),
        congestion_weight=algo_config.get('congestion_weight', 0.6),
        stability_weight=algo_config.get('stability_weight', 0.4),
        route_update_interval=algo_config.get('route_update_interval', 10),
        max_route_changes_per_window=algo_config.get('max_route_changes_per_window', 3),
        route_change_window=algo_config.get('route_change_window', 50)
    )
    
    # Initialize simulator
    print("Initializing network simulator...")
    simulator = NetworkSimulator(topology, scar_algorithm)
    
    # Link simulator to route discovery for queueing delay calculation
    scar_algorithm.route_discovery.simulator = simulator
    
    # Initialize visualization components
    print("Initializing visualization...")
    # Use the congestion monitor from SCAR algorithm (they share the same instance)
    renderer = NetworkRenderer(topology, scar_algorithm.congestion_monitor)
    # Use threaded controller for responsive UI
    controller = ThreadedSimulationController(simulator, renderer)
    packet_visualizer = PacketVisualizer(simulator, renderer)
    metrics_dashboard = MetricsDashboard()
    
    # Generate traffic
    print("Generating traffic...")
    sim_config = config.get('simulation', {})
    traffic_generator = TrafficGenerator(simulator, topology)
    
    traffic_pattern = sim_config.get('traffic_pattern', 'random')
    num_packets = sim_config.get('num_packets', 100)
    time_range = (0.0, sim_config.get('duration', 1000.0))
    
    # Generate traffic in bursts - ALL packets must be in bursts (4-10 packets per burst)
    # Distribute bursts with maximum interval constraint to avoid large gaps
    start_time, end_time = time_range
    time_span = end_time - start_time
    
    # Maximum interval between bursts (in time units) - ensures no large gaps
    max_burst_interval = 0.3  # Maximum 0.3ms between bursts
    
    # Calculate minimum number of bursts needed to avoid gaps
    # Use minimum burst size of 4 to calculate maximum number of bursts
    min_burst_size = 4
    max_num_bursts = num_packets // min_burst_size  # Maximum bursts if all are size 4
    
    # Calculate required number of bursts based on max interval
    required_bursts_by_interval = int(time_span * 0.95 / max_burst_interval)  # Use 95% of duration
    
    # Use the larger of the two to ensure we have enough bursts
    num_bursts = max(1, min(max_num_bursts, required_bursts_by_interval))
    
    # Distribute bursts evenly across the entire duration
    active_duration = time_span * 0.95
    burst_interval = active_duration / max(num_bursts, 1)
    
    # Ensure burst interval doesn't exceed maximum
    if burst_interval > max_burst_interval:
        # Recalculate with more bursts
        num_bursts = max(num_bursts, int(active_duration / max_burst_interval))
        burst_interval = active_duration / max(num_bursts, 1)
    
    packets_generated = 0
    burst_count = 0
    
    current_time = start_time
    
    while packets_generated < num_packets and current_time < end_time - 0.5:
        # Calculate how many packets we still need
        remaining_packets = num_packets - packets_generated
        
        # ALWAYS use burst size between 4-10, never exceed this range
        if remaining_packets <= 10:
            # If remaining packets are 10 or less, use all of them (but at least 4 if possible)
            burst_size = max(1, remaining_packets)  # Use remaining, but allow 1-3 if that's all that's left
        else:
            # Random burst size between 4-10
            burst_size = random.randint(4, 10)
        
        # Generate burst event with all packets
        event = SimulationEvent(
            event_type=EventType.PACKET_GENERATE,
            time=current_time,
            data={
                'source': 0,  # Fixed source
                'destination': 7,  # Fixed destination (sink)
                'size': sim_config.get('packet_size', 1500.0),
                'burst_size': burst_size  # All packets in this burst
            },
            priority=0
        )
        simulator.schedule_event(event)
        packets_generated += burst_size
        burst_count += 1
        
        # Move to next burst time
        # Use calculated interval with small random variation (±10%)
        next_interval = burst_interval * random.uniform(0.9, 1.1)
        current_time += next_interval
        
        # Stop if we've generated enough packets
        if packets_generated >= num_packets:
            break
    
    print(f"Generated {packets_generated} packets in {burst_count} bursts")
    
    # Initialize interactive viewer (pass simulator, traffic_generator, and config for restart)
    viewer = InteractiveViewer(controller, metrics_dashboard, packet_visualizer,
                              simulator=simulator, traffic_generator=traffic_generator, config=config)
    
    # Check if visualization is enabled
    viz_config = config.get('visualization', {})
    if viz_config.get('enabled', True):
        print("Starting interactive simulation...")
        print("Controls:")
        print("  - Play: Start/resume simulation")
        print("  - Pause: Pause simulation")
        print("  - Stop: Stop simulation")
        print("  - Step: Execute one simulation step")
        
        # Create UI
        viewer.create_ui()
        
        # Initial render
        controller.update()
        viewer.update()
        
        # Run simulation with visualization
        try:
            # You can run the simulation programmatically or interactively
            # For now, we'll show the initial state and let user control via UI
            print("\nSimulation ready. Use the UI controls to run the simulation.")
            viewer.show()
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
    else:
        # Run simulation without visualization
        print("Running simulation (no visualization)...")
        duration = sim_config.get('duration', 1000.0)
        simulator.run(duration)
        
        # Print statistics
        stats = simulator.get_stats()
        print("\nSimulation Statistics:")
        print(f"  Packets Generated: {stats['packets_generated']}")
        print(f"  Packets Delivered: {stats['packets_delivered']}")
        print(f"  Packets Lost: {stats['packets_lost']}")
        print(f"  Packet Delivery Ratio: {stats['packet_delivery_ratio']:.2%}")
        print(f"  Average Delay: {stats['avg_delay']:.2f} time units")
        print(f"  Average Hops: {stats['avg_hops']:.2f}")
    
    print("\nSimulation complete.")


if __name__ == '__main__':
    main()

