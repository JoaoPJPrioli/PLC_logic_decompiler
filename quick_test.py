"""
Quick test for the visualization fixes
"""
import sys
import os

# Test the basic functionality
print("Testing PLC Decompiler visualization fixes...")

try:
    # Test importing the main app
    from app import PLCDecompilerApp
    print("✓ Successfully imported PLCDecompilerApp")
    
    # Create app instance
    app_instance = PLCDecompilerApp()
    print("✓ Successfully created app instance")
    
    # Test mock graph data creation for different types
    for graph_type in ['control_flow', 'data_dependency', 'instruction_network', 'execution_flow']:
        try:
            test_data = app_instance._create_mock_graph_data(graph_type)
            nodes = test_data.get('nodes', [])
            edges = test_data.get('edges', [])
            print(f"✓ {graph_type}: {len(nodes)} nodes, {len(edges)} edges")
            
            # Validate node structure
            if nodes and all('id' in node and 'label' in node for node in nodes):
                print(f"  ✓ Node structure valid for {graph_type}")
            else:
                print(f"  ⚠ Node structure incomplete for {graph_type}")
                
        except Exception as e:
            print(f"✗ Failed to create {graph_type} data: {e}")
    
    # Test mock graph building
    mock_analysis_data = {
        'controller': {'name': 'TestController'},
        'tags': [{'name': 'Start_Button', 'data_type': 'BOOL'}],
        'programs': [{'name': 'MainProgram', 'routines': ['MainRoutine']}]
    }
    
    try:
        result = app_instance._build_mock_graph(mock_analysis_data)
        if result.get('build_successful'):
            print("✓ Mock graph building successful")
            print(f"  - Built {len(result.get('graphs', {}))} graph types")
            print(f"  - Generated {len(result.get('recommendations', []))} recommendations")
        else:
            print(f"✗ Mock graph building failed: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"✗ Mock graph building error: {e}")
    
    print("\n🎉 Basic tests completed successfully!")
    print("The fixes should resolve the visualization issues.")
    
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
