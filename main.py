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
    
    # Generate traffic with bursts - each event generates multiple packets
    # This creates congestion quickly so the algorithm can demonstrate path switching
    burst_size = sim_config.get('burst_size', 3)  # Packets per burst
    num_bursts = max(1, num_packets // burst_size)  # Number of bursts
    
    for _ in range(num_bursts):
        # Random generation time
        gen_time = random.uniform(time_range[0], time_range[1])
        
        event = SimulationEvent(
            event_type=EventType.PACKET_GENERATE,
            time=gen_time,
            data={
                'source': 0,  # Fixed source
                'destination': 5,  # Fixed destination
                'size': sim_config.get('packet_size', 1500.0),
                'burst_size': burst_size
            },
            priority=0
        )
        simulator.schedule_event(event)
    print(f"Generated {num_packets} packets")
    
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

