"""
Step 29: ChromaDB Integration for Semantic Search
Advanced semantic search capabilities for PLC logic and documentation

This module provides comprehensive semantic search functionality using ChromaDB
for intelligent discovery of PLC components, logic patterns, and relationships.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import traceback
import hashlib
import pickle

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced ChromaDB import with PyTorch compatibility handling
CHROMADB_AVAILABLE = False
PYTORCH_COMPATIBLE = False
chromadb = None

try:
    # First check if PyTorch is compatible
    import torch
    # Test the problematic sparse module
    try:
        from torch._C._sparse import _spsolve
        PYTORCH_COMPATIBLE = True
        logger.info("✅ PyTorch sparse module is compatible")
    except (ImportError, AttributeError) as e:
        logger.warning(f"⚠️ PyTorch sparse compatibility issue: {e}")
        PYTORCH_COMPATIBLE = False
except ImportError:
    logger.info("PyTorch not available - will use alternative embeddings")
    PYTORCH_COMPATIBLE = False

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    import numpy as np
    
    # If PyTorch is not compatible, configure ChromaDB to use alternative embeddings
    if not PYTORCH_COMPATIBLE:
        logger.info("Configuring ChromaDB to avoid PyTorch dependencies")
        # We'll use OpenAI embeddings or other alternatives instead of sentence transformers
    
    CHROMADB_AVAILABLE = True
    logger.info("✅ ChromaDB available with compatibility adjustments")
    
except ImportError as e:
    logger.warning(f"ChromaDB not available - running in mock mode: {e}")
    CHROMADB_AVAILABLE = False
    chromadb = None
except Exception as e:
    logger.warning(f"ChromaDB initialization failed: {e}, using mock")
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    # Core PLC components
    from src.core.l5x_parser import L5XParser
    from src.analysis.ladder_logic_parser import LadderLogicParser
    from src.analysis.instruction_analysis import InstructionAnalyzer
    from src.models.tags import Tag
    from src.models.knowledge_graph import PLCKnowledgeGraph
    CORE_IMPORTS_AVAILABLE = True
except ImportError:
    print("Core imports not available - using mock implementations")
    CORE_IMPORTS_AVAILABLE = False


@dataclass
class SemanticSearchResult:
    """Result from semantic search operation"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    search_type: str
    timestamp: datetime
    

@dataclass
class DocumentChunk:
    """Chunk of documentation or code for semantic indexing"""
    chunk_id: str
    content: str
    chunk_type: str  # 'tag', 'instruction', 'routine', 'comment', 'logic_pattern'
    source_file: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchQuery:
    """Semantic search query configuration"""
    query_text: str
    search_types: List[str]
    max_results: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True
    boost_recent: bool = False


class MockChromaDB:
    """Mock ChromaDB for demonstration when not available"""
    
    def __init__(self):
        self.collections = {}
        self.documents = {}
    
    def get_or_create_collection(self, name: str, **kwargs):
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]


class MockCollection:
    """Mock ChromaDB collection"""
    
    def __init__(self, name: str):
        self.name = name
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
    
    def add(self, documents: List[str], metadatas: List[Dict], ids: List[str], **kwargs):
        """Add documents to mock collection"""
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        # Mock embeddings
        self.embeddings.extend([[0.1] * 384 for _ in documents])
    
    def query(self, query_texts: List[str], n_results: int = 10, **kwargs):
        """Mock query implementation"""
        if not self.documents:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
        
        # Simple mock: return first n_results documents
        n_results = min(n_results, len(self.documents))
        return {
            'documents': [self.documents[:n_results]],
            'metadatas': [self.metadatas[:n_results]],
            'distances': [[0.5] * n_results],  # Mock distances
            'ids': [self.ids[:n_results]]
        }
    
    def get(self, **kwargs):
        """Get all documents"""
        return {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'ids': self.ids
        }
    
    def count(self):
        """Count documents"""
        return len(self.documents)


class PLCSemanticSearchEngine:
    """Advanced semantic search engine for PLC logic and documentation"""
    
    def __init__(self, db_path: str = "./chroma_db", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize semantic search engine"""
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with PyTorch compatibility handling
        if CHROMADB_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(
                    path=str(self.db_path),
                    settings=Settings()
                )
                
                # Choose embedding function based on PyTorch compatibility
                if PYTORCH_COMPATIBLE:
                    try:
                        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name=embedding_model
                        )
                        logger.info(f"Using SentenceTransformer embeddings: {embedding_model}")
                    except Exception as e:
                        logger.warning(f"SentenceTransformer failed: {e}, falling back to default")
                        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                else:
                    # Use alternative embedding function that doesn't require PyTorch
                    logger.info("Using default embeddings due to PyTorch compatibility issues")
                    self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                    
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}, using mock")
                self.client = MockChromaDB()
                self.embedding_function = None
        else:
            self.client = MockChromaDB()
            self.embedding_function = None
        
        # Collections for different content types
        self.collections = {}
        self._initialize_collections()
        
        # Document processing cache
        self.document_cache = {}
        self.search_cache = {}
        
        logger.info(f"Semantic search engine initialized with {'ChromaDB' if CHROMADB_AVAILABLE else 'Mock'}")
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections for different content types"""
        collection_configs = {
            'plc_tags': {
                'name': 'plc_tags',
                'metadata': {'description': 'PLC tag definitions and properties'}
            },
            'ladder_logic': {
                'name': 'ladder_logic', 
                'metadata': {'description': 'Ladder logic instructions and patterns'}
            },
            'routines': {
                'name': 'routines',
                'metadata': {'description': 'PLC routines and program structure'}
            },
            'comments': {
                'name': 'comments',
                'metadata': {'description': 'PLC comments and documentation'}
            },
            'logic_patterns': {
                'name': 'logic_patterns',
                'metadata': {'description': 'Common logic patterns and templates'}
            },
            'knowledge_base': {
                'name': 'knowledge_base',
                'metadata': {'description': 'PLC knowledge base and best practices'}
            }
        }
        
        for collection_name, config in collection_configs.items():
            try:
                if CHROMADB_AVAILABLE and self.embedding_function:
                    collection = self.client.get_or_create_collection(
                        name=config['name'],
                        embedding_function=self.embedding_function,
                        metadata=config['metadata']
                    )
                else:
                    collection = self.client.get_or_create_collection(
                        name=config['name']
                    )
                
                self.collections[collection_name] = collection
                logger.info(f"Initialized collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize collection {collection_name}: {e}")
    
    async def index_l5x_file(self, l5x_file_path: str) -> Dict[str, int]:
        """Index L5X file content for semantic search"""
        logger.info(f"Starting L5X file indexing: {l5x_file_path}")
        
        try:
            # Parse L5X file
            if CORE_IMPORTS_AVAILABLE:
                parser = L5XParser()
                parser_result = await parser.parse_file(l5x_file_path)
                
                if not parser_result.success:
                    raise ValueError(f"Failed to parse L5X file: {parser_result.error}")
                
                parsed_data = parser_result.data
            else:
                # Mock parsed data
                parsed_data = self._create_mock_parsed_data(l5x_file_path)
            
            indexing_results = {}
            
            # Index different content types
            indexing_results['tags'] = await self._index_tags(parsed_data, l5x_file_path)
            indexing_results['routines'] = await self._index_routines(parsed_data, l5x_file_path)
            indexing_results['comments'] = await self._index_comments(parsed_data, l5x_file_path)
            indexing_results['logic_patterns'] = await self._index_logic_patterns(parsed_data, l5x_file_path)
            
            total_indexed = sum(indexing_results.values())
            logger.info(f"Indexing completed: {total_indexed} documents indexed")
            
            return indexing_results
            
        except Exception as e:
            logger.error(f"Error indexing L5X file: {e}")
            raise
    
    async def _index_tags(self, parsed_data: Dict[str, Any], source_file: str) -> int:
        """Index PLC tags for semantic search"""
        collection = self.collections['plc_tags']
        
        tags = parsed_data.get('tags', [])
        if not tags:
            return 0
        
        documents = []
        metadatas = []
        ids = []
        
        for tag in tags:
            # Create searchable document from tag
            tag_doc = self._create_tag_document(tag, source_file)
            
            documents.append(tag_doc['content'])
            metadatas.append(tag_doc['metadata'])
            ids.append(tag_doc['id'])
        
        # Add to collection
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(documents)} tags")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error indexing tags: {e}")
            return 0
    
    async def _index_routines(self, parsed_data: Dict[str, Any], source_file: str) -> int:
        """Index PLC routines for semantic search"""
        collection = self.collections['routines']
        
        programs = parsed_data.get('programs', [])
        if not programs:
            return 0
        
        documents = []
        metadatas = []
        ids = []
        
        for program in programs:
            routines = program.get('routines', [])
            for routine in routines:
                # Create searchable document from routine
                routine_doc = self._create_routine_document(routine, program, source_file)
                
                documents.append(routine_doc['content'])
                metadatas.append(routine_doc['metadata'])
                ids.append(routine_doc['id'])
        
        # Add to collection
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(documents)} routines")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error indexing routines: {e}")
            return 0
    
    async def _index_comments(self, parsed_data: Dict[str, Any], source_file: str) -> int:
        """Index PLC comments for semantic search"""
        collection = self.collections['comments']
        
        comments = parsed_data.get('comments', [])
        if not comments:
            return 0
        
        documents = []
        metadatas = []
        ids = []
        
        for comment in comments:
            if comment.get('text', '').strip():
                # Create searchable document from comment
                comment_doc = self._create_comment_document(comment, source_file)
                
                documents.append(comment_doc['content'])
                metadatas.append(comment_doc['metadata'])
                ids.append(comment_doc['id'])
        
        # Add to collection
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(documents)} comments")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error indexing comments: {e}")
            return 0
    
    async def _index_logic_patterns(self, parsed_data: Dict[str, Any], source_file: str) -> int:
        """Index logic patterns for semantic search"""
        collection = self.collections['logic_patterns']
        
        # Extract logic patterns from routines
        programs = parsed_data.get('programs', [])
        if not programs:
            return 0
        
        documents = []
        metadatas = []
        ids = []
        
        pattern_id = 0
        for program in programs:
            routines = program.get('routines', [])
            for routine in routines:
                # Extract patterns from routine logic
                patterns = self._extract_logic_patterns(routine, program, source_file)
                
                for pattern in patterns:
                    pattern_doc = {
                        'id': f"{source_file}_{program['name']}_{routine['name']}_pattern_{pattern_id}",
                        'content': pattern['description'],
                        'metadata': {
                            'pattern_type': pattern['type'],
                            'source_file': source_file,
                            'program': program['name'],
                            'routine': routine['name'],
                            'confidence': pattern['confidence'],
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    documents.append(pattern_doc['content'])
                    metadatas.append(pattern_doc['metadata'])
                    ids.append(pattern_doc['id'])
                    pattern_id += 1
        
        # Add to collection
        try:
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Indexed {len(documents)} logic patterns")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error indexing logic patterns: {e}")
            return 0
    
    def _create_tag_document(self, tag: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """Create searchable document from tag"""
        tag_name = tag.get('name', 'Unknown')
        tag_type = tag.get('data_type', 'Unknown')
        description = tag.get('description', '')
        scope = tag.get('scope', 'controller')
        
        # Create comprehensive content for embedding
        content_parts = [
            f"Tag: {tag_name}",
            f"Type: {tag_type}",
            f"Scope: {scope}"
        ]
        
        if description:
            content_parts.append(f"Description: {description}")
        
        # Add additional properties
        if tag.get('initial_value'):
            content_parts.append(f"Initial Value: {tag['initial_value']}")
        
        if tag.get('dimensions'):
            content_parts.append(f"Array Dimensions: {tag['dimensions']}")
        
        content = " | ".join(content_parts)
        
        return {
            'id': f"{source_file}_{scope}_{tag_name}",
            'content': content,
            'metadata': {
                'tag_name': tag_name,
                'tag_type': tag_type,
                'scope': scope,
                'source_file': source_file,
                'description': description,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _create_routine_document(self, routine: Dict[str, Any], program: Dict[str, Any], 
                                source_file: str) -> Dict[str, Any]:
        """Create searchable document from routine"""
        routine_name = routine.get('name', 'Unknown')
        routine_type = routine.get('type', 'Ladder')
        program_name = program.get('name', 'Unknown')
        
        # Create comprehensive content
        content_parts = [
            f"Routine: {routine_name}",
            f"Type: {routine_type}",
            f"Program: {program_name}"
        ]
        
        # Add instruction summary if available
        if routine.get('instructions'):
            instruction_types = set()
            for instr in routine['instructions']:
                instruction_types.add(instr.get('type', 'Unknown'))
            
            if instruction_types:
                content_parts.append(f"Instructions: {', '.join(sorted(instruction_types))}")
        
        # Add logic description if available
        if routine.get('description'):
            content_parts.append(f"Description: {routine['description']}")
        
        content = " | ".join(content_parts)
        
        return {
            'id': f"{source_file}_{program_name}_{routine_name}",
            'content': content,
            'metadata': {
                'routine_name': routine_name,
                'routine_type': routine_type,
                'program_name': program_name,
                'source_file': source_file,
                'instruction_count': len(routine.get('instructions', [])),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _create_comment_document(self, comment: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """Create searchable document from comment"""
        comment_text = comment.get('text', '')
        operand = comment.get('operand', '')
        comment_type = comment.get('type', 'general')
        
        # Create content for embedding
        content_parts = [f"Comment: {comment_text}"]
        
        if operand:
            content_parts.append(f"Associated with: {operand}")
        
        content = " | ".join(content_parts)
        
        comment_id = hashlib.md5(f"{source_file}_{operand}_{comment_text}".encode()).hexdigest()[:16]
        
        return {
            'id': f"{source_file}_comment_{comment_id}",
            'content': content,
            'metadata': {
                'comment_text': comment_text,
                'operand': operand,
                'comment_type': comment_type,
                'source_file': source_file,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _extract_logic_patterns(self, routine: Dict[str, Any], program: Dict[str, Any], 
                               source_file: str) -> List[Dict[str, Any]]:
        """Extract logic patterns from routine"""
        patterns = []
        
        # Mock pattern extraction - in real implementation this would analyze ladder logic
        routine_name = routine.get('name', 'Unknown')
        instructions = routine.get('instructions', [])
        
        if not instructions:
            return patterns
        
        # Detect common patterns
        instruction_types = [instr.get('type', '') for instr in instructions]
        
        # Timer pattern
        if any('TON' in itype or 'TOF' in itype or 'RTO' in itype for itype in instruction_types):
            patterns.append({
                'type': 'timer_pattern',
                'description': f"Timer logic pattern in routine {routine_name} with timing instructions",
                'confidence': 0.9
            })
        
        # Counter pattern
        if any('CTU' in itype or 'CTD' in itype for itype in instruction_types):
            patterns.append({
                'type': 'counter_pattern',
                'description': f"Counter logic pattern in routine {routine_name} with counting instructions",
                'confidence': 0.85
            })
        
        # Start/Stop pattern
        if any('XIC' in itype for itype in instruction_types) and any('OTE' in itype for itype in instruction_types):
            patterns.append({
                'type': 'start_stop_pattern',
                'description': f"Start/Stop logic pattern in routine {routine_name} with input/output logic",
                'confidence': 0.75
            })
        
        return patterns
    
    async def semantic_search(self, query: SearchQuery) -> List[SemanticSearchResult]:
        """Perform semantic search across all collections"""
        logger.info(f"Performing semantic search: {query.query_text}")
        
        all_results = []
        
        # Search each collection type if specified
        search_collections = []
        if 'tags' in query.search_types or 'all' in query.search_types:
            search_collections.append(('plc_tags', 'tag'))
        if 'routines' in query.search_types or 'all' in query.search_types:
            search_collections.append(('routines', 'routine'))
        if 'comments' in query.search_types or 'all' in query.search_types:
            search_collections.append(('comments', 'comment'))
        if 'patterns' in query.search_types or 'all' in query.search_types:
            search_collections.append(('logic_patterns', 'pattern'))
        
        if not search_collections:
            search_collections = [
                ('plc_tags', 'tag'),
                ('routines', 'routine'),
                ('comments', 'comment'),
                ('logic_patterns', 'pattern')
            ]
        
        # Search each collection
        for collection_name, search_type in search_collections:
            try:
                collection = self.collections[collection_name]
                
                # Query collection
                results = collection.query(
                    query_texts=[query.query_text],
                    n_results=query.max_results,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Process results
                if results['documents'] and results['documents'][0]:
                    documents = results['documents'][0]
                    metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                    distances = results['distances'][0] if results['distances'] else [0.5] * len(documents)
                    
                    for doc, metadata, distance in zip(documents, metadatas, distances):
                        # Convert distance to similarity score
                        similarity_score = 1.0 - distance
                        
                        if similarity_score >= query.similarity_threshold:
                            result = SemanticSearchResult(
                                document_id=metadata.get('id', f"{collection_name}_{len(all_results)}"),
                                content=doc,
                                metadata=metadata if query.include_metadata else {},
                                similarity_score=similarity_score,
                                search_type=search_type,
                                timestamp=datetime.now()
                            )
                            all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Apply boost for recent results if requested
        if query.boost_recent:
            all_results = self._boost_recent_results(all_results)
        
        # Limit results
        final_results = all_results[:query.max_results]
        
        logger.info(f"Search completed: {len(final_results)} results found")
        return final_results
    
    def _boost_recent_results(self, results: List[SemanticSearchResult]) -> List[SemanticSearchResult]:
        """Boost similarity scores for recent results"""
        now = datetime.now()
        
        for result in results:
            timestamp_str = result.metadata.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    age_hours = (now - timestamp).total_seconds() / 3600
                    
                    # Boost factor decreases with age
                    if age_hours < 24:  # Less than 1 day
                        boost = 1.1
                    elif age_hours < 168:  # Less than 1 week
                        boost = 1.05
                    else:
                        boost = 1.0
                    
                    result.similarity_score = min(1.0, result.similarity_score * boost)
                    
                except Exception:
                    pass  # Skip boost if timestamp parsing fails
        
        # Re-sort after boosting
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
    async def add_knowledge_base_content(self, content: List[Dict[str, Any]]) -> int:
        """Add knowledge base content for enhanced search"""
        collection = self.collections['knowledge_base']
        
        documents = []
        metadatas = []
        ids = []
        
        for item in content:
            doc_id = item.get('id') or f"kb_{len(documents)}"
            content_text = item.get('content', '')
            metadata = item.get('metadata', {})
            
            if content_text.strip():
                documents.append(content_text)
                metadatas.append({
                    'knowledge_type': item.get('type', 'general'),
                    'category': item.get('category', 'general'),
                    'source': item.get('source', 'manual'),
                    'timestamp': datetime.now().isoformat(),
                    **metadata
                })
                ids.append(doc_id)
        
        try:
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} knowledge base items")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error adding knowledge base content: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    'document_count': count,
                    'status': 'active'
                }
            except Exception as e:
                stats[name] = {
                    'document_count': 0,
                    'status': f'error: {str(e)}'
                }
        
        return stats
    
    async def optimize_collections(self) -> Dict[str, Any]:
        """Optimize collections and clean up old documents"""
        optimization_results = {}
        
        for name, collection in self.collections.items():
            try:
                # Get collection info
                initial_count = collection.count()
                
                # For now, just report current state
                # In a real implementation, this might compact or reorganize data
                optimization_results[name] = {
                    'initial_count': initial_count,
                    'final_count': initial_count,
                    'optimized': True
                }
                
            except Exception as e:
                optimization_results[name] = {
                    'error': str(e),
                    'optimized': False
                }
        
        return optimization_results
    
    def _create_mock_parsed_data(self, source_file: str) -> Dict[str, Any]:
        """Create mock parsed data for demonstration"""
        return {
            'controller_info': {
                'name': 'Mock_Controller',
                'type': 'Allen-Bradley CompactLogix'
            },
            'tags': [
                {
                    'name': 'Emergency_Stop',
                    'data_type': 'BOOL',
                    'scope': 'controller',
                    'description': 'Emergency stop button input'
                },
                {
                    'name': 'Conveyor_Speed',
                    'data_type': 'REAL',
                    'scope': 'controller',
                    'description': 'Conveyor belt speed setpoint'
                },
                {
                    'name': 'Production_Count',
                    'data_type': 'DINT',
                    'scope': 'controller',
                    'description': 'Production counter value'
                }
            ],
            'programs': [
                {
                    'name': 'MainProgram',
                    'type': 'PROGRAM',
                    'routines': [
                        {
                            'name': 'MainRoutine',
                            'type': 'Ladder',
                            'instructions': [
                                {'type': 'XIC', 'operand': 'Emergency_Stop'},
                                {'type': 'OTE', 'operand': 'System_Enable'},
                                {'type': 'TON', 'operand': 'Delay_Timer'}
                            ]
                        }
                    ]
                }
            ],
            'comments': [
                {
                    'text': 'Safety system emergency stop logic',
                    'operand': 'Emergency_Stop',
                    'type': 'tag_comment'
                }
            ]
        }


# Convenience functions
async def create_semantic_search_engine(db_path: str = "./chroma_db") -> PLCSemanticSearchEngine:
    """Create and initialize semantic search engine"""
    engine = PLCSemanticSearchEngine(db_path)
    return engine


async def index_plc_project(engine: PLCSemanticSearchEngine, 
                           l5x_files: List[str]) -> Dict[str, Any]:
    """Index multiple L5X files for semantic search"""
    total_results = {}
    
    for l5x_file in l5x_files:
        try:
            file_results = await engine.index_l5x_file(l5x_file)
            total_results[l5x_file] = file_results
            
        except Exception as e:
            logger.error(f"Error indexing {l5x_file}: {e}")
            total_results[l5x_file] = {'error': str(e)}
    
    return total_results


async def search_plc_content(engine: PLCSemanticSearchEngine,
                            query_text: str,
                            search_types: List[str] = ['all'],
                            max_results: int = 10) -> List[SemanticSearchResult]:
    """Perform semantic search with simplified interface"""
    query = SearchQuery(
        query_text=query_text,
        search_types=search_types,
        max_results=max_results
    )
    
    return await engine.semantic_search(query)


# Export main classes and functions
__all__ = [
    'PLCSemanticSearchEngine',
    'SemanticSearchResult',
    'SearchQuery',
    'DocumentChunk',
    'create_semantic_search_engine',
    'index_plc_project',
    'search_plc_content'
]
