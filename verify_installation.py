"""
Verification script to ensure all dependencies are available
"""

def verify_imports():
    """Verify all required imports."""
    print("Verifying Python library imports...")
    print("-" * 50)
    
    libraries = {
        'networkx': 'NetworkX',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'simpy': 'SimPy',
        'PyQt5': 'PyQt5',
        'plotly': 'Plotly',
        'yaml': 'PyYAML',
        'pytest': 'Pytest',
    }
    
    all_ok = True
    for module_name, display_name in libraries.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'OK')
            print(f"[OK] {display_name:15} - {version}")
        except ImportError as e:
            print(f"[FAIL] {display_name:15} - MISSING: {e}")
            all_ok = False
    
    print("-" * 50)
    
    if all_ok:
        print("[OK] All core libraries are available!")
    else:
        print("[FAIL] Some libraries are missing. Please install them.")
    
    return all_ok


def verify_scar_modules():
    """Verify SCAR algorithm modules can be imported."""
    print("\nVerifying SCAR algorithm modules...")
    print("-" * 50)
    
    modules = [
        'src.core.network_topology',
        'src.core.congestion_monitor',
        'src.core.route_discovery',
        'src.core.stability_mechanism',
        'src.core.scar_algorithm',
        'src.simulation.network_simulator',
        'src.simulation.traffic_generator',
        'src.visualization.network_renderer',
        'src.visualization.simulation_controller',
        'src.visualization.packet_visualizer',
        'src.visualization.metrics_dashboard',
        'src.visualization.interactive_viewer',
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
        except ImportError as e:
            print(f"[FAIL] {module_name} - ERROR: {e}")
            all_ok = False
        except Exception as e:
            print(f"[FAIL] {module_name} - ERROR: {e}")
            all_ok = False
    
    print("-" * 50)
    
    if all_ok:
        print("[OK] All SCAR modules are importable!")
    else:
        print("[FAIL] Some modules have errors.")
    
    return all_ok


def verify_basic_functionality():
    """Verify basic functionality works."""
    print("\nVerifying basic functionality...")
    print("-" * 50)
    
    try:
        from src.core.network_topology import NetworkTopology
        from src.core.scar_algorithm import SCARAlgorithm
        
        # Create a simple network
        config = {'topology_type': 'ring', 'num_nodes': 5}
        topology = NetworkTopology(config)
        print(f"[OK] Network topology created: {topology.graph.number_of_nodes()} nodes")
        
        # Initialize SCAR algorithm
        scar = SCARAlgorithm(topology)
        print("[OK] SCAR algorithm initialized")
        
        # Test route selection
        nodes = topology.get_nodes()
        if len(nodes) >= 2:
            route = scar.get_route(nodes[0], nodes[2], current_time=0)
            if route:
                print(f"[OK] Route selection works: {route}")
            else:
                print("[WARN] Route selection returned None (may be expected)")
        
        print("-" * 50)
        print("[OK] Basic functionality verified!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 50)
    print("SCAR Algorithm Installation Verification")
    print("=" * 50)
    
    libs_ok = verify_imports()
    modules_ok = verify_scar_modules()
    func_ok = verify_basic_functionality()
    
    print("\n" + "=" * 50)
    if libs_ok and modules_ok and func_ok:
        print("[OK] ALL CHECKS PASSED - Installation is complete!")
        print("=" * 50)
        print("\nYou can now run the simulation with:")
        print("  python main.py")
    else:
        print("[FAIL] SOME CHECKS FAILED - Please review errors above")
        print("=" * 50)

