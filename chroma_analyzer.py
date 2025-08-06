#!/usr/bin/env python3
"""
ChromaDB Analysis Tool
This script analyzes the ChromaDB database to understand what data is stored and how to query it
"""

import chromadb
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from collections import defaultdict

class ChromaDBAnalyzer:
    """Analyzes ChromaDB database content and structure"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.client = None
        self.collections = {}
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=str(self.db_path.parent))
            print(f"Connected to ChromaDB at: {self.db_path}")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            sys.exit(1)
    
    def list_collections(self) -> List[str]:
        """List all collections in the database"""
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            print(f"Found {len(collection_names)} collections:")
            for name in collection_names:
                print(f"  - {name}")
            return collection_names
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def analyze_collection(self, collection_name: str) -> Dict[str, Any]:
        """Analyze a specific collection"""
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get collection statistics
            count = collection.count()
            print(f"\\nAnalyzing collection '{collection_name}':")
            print(f"  Total documents: {count}")
            
            if count == 0:
                return {
                    'name': collection_name,
                    'count': 0,
                    'documents': [],
                    'metadata_fields': [],
                    'sample_documents': []
                }
            
            # Get all documents (limit to reasonable number for analysis)
            limit = min(count, 1000)
            results = collection.get(limit=limit)
            
            # Analyze metadata fields
            metadata_fields = set()
            metadata_types = defaultdict(set)
            
            for metadata in results.get('metadatas', []):
                if metadata:
                    for key, value in metadata.items():
                        metadata_fields.add(key)
                        metadata_types[key].add(type(value).__name__)
            
            print(f"  Metadata fields: {len(metadata_fields)}")
            for field in sorted(metadata_fields):
                types = list(metadata_types[field])
                print(f"    {field}: {', '.join(types)}")
            
            # Sample documents
            sample_size = min(5, len(results.get('documents', [])))
            sample_documents = []
            
            for i in range(sample_size):
                doc = {
                    'id': results['ids'][i] if results.get('ids') else None,
                    'document': results['documents'][i] if results.get('documents') else None,
                    'metadata': results['metadatas'][i] if results.get('metadatas') else None,
                    'embedding_size': len(results['embeddings'][i]) if results.get('embeddings') and results['embeddings'][i] else 0
                }
                sample_documents.append(doc)
                
                print(f"\\n  Sample document {i+1}:")
                print(f"    ID: {doc['id']}")
                print(f"    Document length: {len(doc['document']) if doc['document'] else 0} chars")
                print(f"    Metadata: {doc['metadata']}")
                print(f"    Embedding size: {doc['embedding_size']}")
                if doc['document']:
                    preview = doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document']
                    print(f"    Content preview: {preview}")
            
            analysis = {
                'name': collection_name,
                'count': count,
                'metadata_fields': sorted(metadata_fields),
                'metadata_types': {k: list(v) for k, v in metadata_types.items()},
                'sample_documents': sample_documents,
                'has_embeddings': bool(results.get('embeddings') and any(results['embeddings']))
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing collection '{collection_name}': {e}")
            return {'name': collection_name, 'error': str(e)}
    
    def search_collection(self, collection_name: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search a collection using semantic similarity"""
        try:
            collection = self.client.get_collection(collection_name)
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results.get('distances') else None
                }
                search_results.append(result)
            
            print(f"\\nSearch results for '{query}' in '{collection_name}':")
            for i, result in enumerate(search_results):
                print(f"  Result {i+1} (distance: {result['distance']:.4f} if result['distance'] else 'N/A'):")
                print(f"    ID: {result['id']}")
                print(f"    Metadata: {result['metadata']}")
                preview = result['document'][:150] + "..." if len(result['document']) > 150 else result['document']
                print(f"    Content: {preview}")
                print()
            
            return search_results
            
        except Exception as e:
            print(f"Error searching collection '{collection_name}': {e}")
            return []
    
    def find_plc_content(self, collection_name: str) -> List[Dict[str, Any]]:
        """Find PLC-specific content in a collection"""
        plc_keywords = [
            "ladder logic", "rung", "instruction", "XIC", "XIO", "OTE", "timer", "counter",
            "program", "routine", "tag", "variable", "controller", "PLC", "automation"
        ]
        
        all_results = []
        
        for keyword in plc_keywords:
            try:
                results = self.search_collection(collection_name, keyword, n_results=3)
                for result in results:
                    result['search_keyword'] = keyword
                    all_results.append(result)
            except:
                continue
        
        # Remove duplicates based on ID
        unique_results = {}
        for result in all_results:
            doc_id = result['id']
            if doc_id not in unique_results or result['distance'] < unique_results[doc_id]['distance']:
                unique_results[doc_id] = result
        
        return list(unique_results.values())
    
    def generate_usage_examples(self, collection_name: str) -> List[str]:
        """Generate usage examples for the collection"""
        try:
            analysis = self.analyze_collection(collection_name)
            
            examples = [
                f"# Search for PLC instructions",
                f"python chroma_analyzer.py {self.db_path} --search '{collection_name}' 'ladder logic instructions'",
                f"",
                f"# Search for specific variables", 
                f"python chroma_analyzer.py {self.db_path} --search '{collection_name}' 'timer variables'",
                f"",
                f"# Search for safety circuits",
                f"python chroma_analyzer.py {self.db_path} --search '{collection_name}' 'emergency stop safety'",
            ]
            
            # Add examples based on metadata fields
            if 'program_name' in analysis.get('metadata_fields', []):
                examples.extend([
                    f"",
                    f"# Filter by program (using metadata)",
                    f"# You can modify the search to filter by metadata fields like 'program_name'"
                ])
            
            if 'routine_name' in analysis.get('metadata_fields', []):
                examples.extend([
                    f"# Filter by routine (using metadata)",
                    f"# You can modify the search to filter by metadata fields like 'routine_name'"
                ])
            
            return examples
            
        except Exception as e:
            return [f"# Error generating examples: {e}"]
    
    def export_analysis(self, output_file: str):
        """Export complete database analysis to file"""
        analysis_report = {
            'database_path': str(self.db_path),
            'timestamp': str(self.client._system_time if hasattr(self.client, '_system_time') else 'unknown'),
            'collections': {},
            'usage_recommendations': []
        }
        
        collection_names = self.list_collections()
        
        for name in collection_names:
            print(f"\\nAnalyzing collection '{name}'...")
            collection_analysis = self.analyze_collection(name)
            analysis_report['collections'][name] = collection_analysis
            
            # Add usage examples
            examples = self.generate_usage_examples(name)
            analysis_report['collections'][name]['usage_examples'] = examples
            
            # Find PLC-specific content
            if collection_analysis.get('count', 0) > 0:
                plc_content = self.find_plc_content(name)
                analysis_report['collections'][name]['plc_content_samples'] = plc_content[:5]  # Top 5
        
        # Generate recommendations
        recommendations = [
            "Use semantic search to find related PLC concepts",
            "Combine multiple search terms for better results",
            "Use metadata filters to narrow down results by program/routine",
            "Try different phrasings of technical terms",
            "Search for specific instruction types (XIC, OTE, TON, etc.)"
        ]
        
        analysis_report['usage_recommendations'] = recommendations
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, default=str)
        
        print(f"\\nComplete analysis exported to: {output_file}")
        
        # Print summary
        print("\\n=== ChromaDB Analysis Summary ===")
        total_docs = sum(col.get('count', 0) for col in analysis_report['collections'].values())
        print(f"Total collections: {len(collection_names)}")
        print(f"Total documents: {total_docs}")
        
        if total_docs > 0:
            print("\\nCollections:")
            for name, analysis in analysis_report['collections'].items():
                print(f"  {name}: {analysis.get('count', 0)} documents")
                if analysis.get('metadata_fields'):
                    print(f"    Metadata: {', '.join(analysis['metadata_fields'])}")
            
            print("\\nUsage Examples:")
            for rec in recommendations:
                print(f"  • {rec}")
        else:
            print("No documents found in database.")

def main():
    parser = argparse.ArgumentParser(description="ChromaDB Analysis Tool")
    parser.add_argument("db_path", help="Path to ChromaDB database file or directory")
    parser.add_argument("--list", "-l", action="store_true", help="List all collections")
    parser.add_argument("--analyze", "-a", help="Analyze specific collection")
    parser.add_argument("--search", "-s", nargs=2, metavar=('COLLECTION', 'QUERY'), 
                       help="Search collection with query")
    parser.add_argument("--plc", "-p", help="Find PLC-specific content in collection") 
    parser.add_argument("--export", "-e", help="Export complete analysis to file")
    parser.add_argument("--results", "-n", type=int, default=5, 
                       help="Number of search results to return (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ChromaDBAnalyzer(args.db_path)
    
    if args.list:
        analyzer.list_collections()
    
    elif args.analyze:
        analyzer.analyze_collection(args.analyze)
    
    elif args.search:
        collection_name, query = args.search
        analyzer.search_collection(collection_name, query, args.results)
    
    elif args.plc:
        results = analyzer.find_plc_content(args.plc)
        print(f"\\nFound {len(results)} PLC-related documents in '{args.plc}':")
        for i, result in enumerate(results[:10]):  # Show top 10
            print(f"  {i+1}. {result['id']} (keyword: {result['search_keyword']})")
            preview = result['document'][:100] + "..." if len(result['document']) > 100 else result['document']
            print(f"     {preview}")
    
    elif args.export:
        analyzer.export_analysis(args.export)
    
    else:
        # Default: show overview
        collection_names = analyzer.list_collections()
        if collection_names:
            print("\\n=== Quick Analysis ===")
            for name in collection_names[:3]:  # Analyze first 3 collections
                analyzer.analyze_collection(name)
        
        print("\\n=== Usage Examples ===")
        print("List collections:")
        print(f"  python chroma_analyzer.py {args.db_path} --list")
        print("\\nAnalyze a collection:")
        print(f"  python chroma_analyzer.py {args.db_path} --analyze COLLECTION_NAME")
        print("\\nSearch for content:")
        print(f"  python chroma_analyzer.py {args.db_path} --search COLLECTION_NAME 'your query'")
        print("\\nExport full analysis:")
        print(f"  python chroma_analyzer.py {args.db_path} --export analysis_report.json")

if __name__ == "__main__":
    main()
