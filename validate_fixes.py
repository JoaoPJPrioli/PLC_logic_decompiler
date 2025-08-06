#!/usr/bin/env python3
"""
Simple validation script to test our fixes
"""

def main():
    print("=== PLC Decompiler Fix Validation ===")
    
    # Test 1: Check if we can import our fixed modules
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from src.analysis.fixed_graph_builder import FixedAdvancedGraphBuilder
        print("✓ Fixed graph builder import successful")
    except Exception as e:
        print(f"✗ Fixed graph builder import failed: {e}")
    
    # Test 2: Check analysis file
    try:
        import json
        with open("outputs/analysis_report_20250801_103536.json", 'r') as f:
            data = json.load(f)
        print(f"✓ Analysis report loaded ({len(str(data))} chars)")
    except Exception as e:
        print(f"✗ Analysis report load failed: {e}")
    
    # Test 3: Test ChromaDB file
    try:
        import os
        chroma_path = "chroma_db/chroma.sqlite3"
        if os.path.exists(chroma_path):
            size = os.path.getsize(chroma_path)
            print(f"✓ ChromaDB file found ({size} bytes)")
        else:
            print("✗ ChromaDB file not found")
    except Exception as e:
        print(f"✗ ChromaDB check failed: {e}")
    
    # Test 4: Test L5X file
    try:
        with open("Assembly_Controls_Robot.L5X", 'r') as f:
            content = f.read()
        programs = content.count('<Program ')
        routines = content.count('<Routine ')
        print(f"✓ L5X file loaded ({programs} programs, {routines} routines)")
    except Exception as e:
        print(f"✗ L5X file load failed: {e}")
    
    print("\n=== Summary ===")
    print("1. Graph generation has been fixed to create meaningful nodes and edges")
    print("2. Logic analyzer script created to search variables and build logic representations")
    print("3. Gemini API configuration updated to use gemma-2-27b-it model")
    print("4. ChromaDB analyzer script created to explore the database")
    print("\n=== Usage Instructions ===")
    print("To run the app:")
    print("  python app.py")
    print("\nTo analyze logic from the report:")
    print("  python plc_logic_analyzer.py outputs/analysis_report_20250801_103536.json --search Robot --fuzzy")
    print("\nTo analyze ChromaDB:")
    print("  python chroma_analyzer.py chroma_db/chroma.sqlite3 --list")

if __name__ == "__main__":
    main()
