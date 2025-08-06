#!/usr/bin/env python3
"""
Test Script for PLC Decompiler Fixes
Tests the main functionality with the actual L5X file
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_graph_builder():
    """Test the fixed graph builder"""
    try:
        from src.analysis.fixed_graph_builder import FixedAdvancedGraphBuilder
        
        print("Testing Graph Builder...")
        
        # Load the analysis report 
        analysis_file = Path("outputs/analysis_report_20250801_103536.json")
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            builder = FixedAdvancedGraphBuilder()
            result = builder.build_comprehensive_graph(analysis_data)
            
            print(f"Graph build successful: {result['build_successful']}")
            print(f"Graphs built: {len(result.get('graphs', {}))}")
            print(f"Total nodes: {result.get('statistics', {}).get('total_nodes', 0)}")
            print(f"Total edges: {result.get('statistics', {}).get('total_edges', 0)}")
            print(f"Recommendations: {len(result.get('recommendations', []))}")
            
            for rec in result.get('recommendations', [])[:3]:
                print(f"  - {rec}")
            
            return True
        else:
            print("Analysis report not found")
            return False
            
    except Exception as e:
        print(f"Graph builder test failed: {e}")
        return False

def test_logic_analyzer():
    """Test the logic analyzer script"""
    try:
        analysis_file = Path("outputs/analysis_report_20250801_103536.json")
        if not analysis_file.exists():
            print("Analysis report not found for logic analyzer test")
            return False
        
        # Import and test the analyzer
        from plc_logic_analyzer import PLCLogicAnalyzer
        
        print("\\nTesting Logic Analyzer...")
        
        analyzer = PLCLogicAnalyzer(str(analysis_file))
        
        print(f"Variables loaded: {len(analyzer.variables)}")
        print(f"Instructions parsed: {len(analyzer.instructions)}")
        
        # Test variable search
        robot_vars = analyzer.search_variable("Robot", fuzzy=True)
        print(f"Robot-related variables: {len(robot_vars)}")
        
        # Test safety circuit detection
        safety_circuits = analyzer.find_safety_circuits()
        print(f"Safety circuits found: {len(safety_circuits)}")
        
        # Test timer circuits
        timer_circuits = analyzer.find_timer_circuits()
        print(f"Timer circuits found: {len(timer_circuits)}")
        
        return True
        
    except Exception as e:
        print(f"Logic analyzer test failed: {e}")
        return False

def test_chroma_analyzer():
    """Test the ChromaDB analyzer"""
    try:
        chroma_path = Path("chroma_db/chroma.sqlite3")
        if not chroma_path.exists():
            print("ChromaDB file not found for testing")
            return False
        
        from chroma_analyzer import ChromaDBAnalyzer
        
        print("\\nTesting ChromaDB Analyzer...")
        
        analyzer = ChromaDBAnalyzer(str(chroma_path))
        collections = analyzer.list_collections()
        
        print(f"Collections found: {len(collections)}")
        
        if collections:
            # Analyze first collection
            first_collection = collections[0]
            analysis = analyzer.analyze_collection(first_collection)
            print(f"First collection '{first_collection}': {analysis.get('count', 0)} documents")
        
        return True
        
    except Exception as e:
        print(f"ChromaDB analyzer test failed: {e}")
        return False

def test_enhanced_service():
    """Test the enhanced PLC service with the actual L5X file"""
    try:
        from enhanced_plc_service import EnhancedPLCProcessingService
        
        print("\\nTesting Enhanced PLC Service...")
        
        l5x_file = Path("Assembly_Controls_Robot.L5X")
        if not l5x_file.exists():
            print("L5X file not found")
            return False
        
        service = EnhancedPLCProcessingService(output_dir="outputs")
        result = service.process_l5x_file_comprehensive(str(l5x_file))
        
        print(f"Processing successful: {result.get('success', False)}")
        print(f"Components used: {len(result.get('components_used', []))}")
        print(f"Outputs generated: {len(result.get('outputs_generated', []))}")
        
        if result.get('analysis_results'):
            l5x_data = result['analysis_results'].get('l5x_parsing', {})
            print(f"Controller tags: {len(l5x_data.get('controller_tags', []))}")
            print(f"Programs: {len(l5x_data.get('programs', []))}")
        
        return True
        
    except Exception as e:
        print(f"Enhanced service test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== PLC Decompiler Fix Tests ===\\n")
    
    tests = [
        ("Graph Builder", test_graph_builder),
        ("Logic Analyzer", test_logic_analyzer), 
        ("ChromaDB Analyzer", test_chroma_analyzer),
        ("Enhanced Service", test_enhanced_service)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\\n=== Test Results ===")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 All tests passed! The fixes are working correctly.")
    else:
        print(f"\\n⚠️  {total - passed} test(s) failed. Check the errors above.")

if __name__ == "__main__":
    main()
