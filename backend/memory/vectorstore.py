"""
Vector Store using FAISS for WhatsApp AI Second Brain Assistant
Handles document embedding, storage, and semantic search

Example Usage:
INPUT: 
await vector_store.add_document("user123", "Meeting about AI project timeline and requirements")

SEARCH:
results = await vector_store.search("AI project", "user123", top_k=5)

OUTPUT:
[
    {
        "doc_id": "doc_001",
        "content": "Meeting about AI project timeline...",
        "score": 0.85,
        "metadata": {"source": "whatsapp", "timestamp": "2025-06-29"}
    }
]
"""

import os
import json
import pickle
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from backend.utils.config import config
from backend.models.schema import VectorDocument

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight, fast embedding model
        self.embedding_model = None
        self.index = None
        self.documents = {}  # doc_id -> document metadata
        self.user_docs = {}  # user_id -> [doc_ids]
        self.vector_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.index_file = config.VECTOR_STORE_PATH / "faiss_index.bin"
        self.docs_file = config.VECTOR_STORE_PATH / "documents.json"
        self.user_docs_file = config.VECTOR_STORE_PATH / "user_docs.json"
        
        # Initialize in async context
        self._initialized = False
    
    async def initialize(self):
        """Initialize the vector store (call this first)"""
        if self._initialized:
            return
        
        try:
            # Load embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Create or load FAISS index
            await self._load_or_create_index()
            
            # Load document metadata
            await self._load_documents()
            
            self._initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise
    
    async def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        try:
            if self.index_file.exists():
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(str(self.index_file))
            else:
                logger.info("Creating new FAISS index...")
                # Create flat L2 index (good for smaller datasets)
                self.index = faiss.IndexFlatL2(self.vector_dim)
                await self._save_index()
        except Exception as e:
            logger.warning(f"Failed to load index, creating new one: {e}")
            self.index = faiss.IndexFlatL2(self.vector_dim)
    
    async def _load_documents(self):
        """Load document metadata"""
        try:
            if self.docs_file.exists():
                with open(self.docs_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                    
            if self.user_docs_file.exists():
                with open(self.user_docs_file, 'r', encoding='utf-8') as f:
                    self.user_docs = json.load(f)
                    
            logger.info(f"Loaded {len(self.documents)} documents for {len(self.user_docs)} users")
            
        except Exception as e:
            logger.warning(f"Failed to load documents: {e}")
            self.documents = {}
            self.user_docs = {}
    
    async def _save_index(self):
        """Save FAISS index to disk"""
        try:
            faiss.write_index(self.index, str(self.index_file))
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    async def _save_documents(self):
        """Save document metadata to disk"""
        try:
            with open(self.docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, default=str)
                
            with open(self.user_docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_docs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
            
        # Clean and truncate text
        clean_text = text.strip()[:1000]  # Limit to 1000 chars
        
        embedding = self.embedding_model.encode([clean_text])
        return embedding[0].astype('float32')
    
    async def add_document(
        self, 
        user_id: str, 
        content: str, 
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add document to vector store
        
        Args:
            user_id: User identifier
            content: Document content
            doc_id: Optional document ID (will generate if not provided)
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate doc_id if not provided
            if not doc_id:
                import uuid
                doc_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self._generate_embedding(content)
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            
            # Store document metadata
            doc_metadata = {
                "doc_id": doc_id,
                "user_id": user_id,
                "content": content,
                "metadata": metadata or {},
                "created_at": str(asyncio.get_event_loop().time()),
                "index_position": self.index.ntotal - 1  # Position in FAISS index
            }
            
            self.documents[doc_id] = doc_metadata
            
            # Update user document mapping
            if user_id not in self.user_docs:
                self.user_docs[user_id] = []
            self.user_docs[user_id].append(doc_id)
            
            # Save to disk
            await self._save_index()
            await self._save_documents()
            
            logger.info(f"Added document {doc_id} for user {user_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            user_id: Optional user filter
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.reshape(1, -1), min(top_k * 2, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                # Find document by index position
                doc = None
                for doc_id, doc_data in self.documents.items():
                    if doc_data.get("index_position") == idx:
                        doc = doc_data
                        break
                
                if not doc:
                    continue
                
                # Filter by user if specified
                if user_id and doc.get("user_id") != user_id:
                    continue
                
                # Convert FAISS L2 distance to similarity score (0-1)
                similarity = 1 / (1 + score)
                
                results.append({
                    "doc_id": doc["doc_id"],
                    "content": doc["content"],
                    "score": float(similarity),
                    "metadata": doc.get("metadata", {}),
                    "user_id": doc["user_id"]
                })
            
            # Sort by score and limit results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user"""
        if not self._initialized:
            await self.initialize()
        
        user_doc_ids = self.user_docs.get(user_id, [])
        documents = []
        
        for doc_id in user_doc_ids:
            if doc_id in self.documents:
                documents.append(self.documents[doc_id])
        
        return documents
    
    async def delete_document(self, doc_id: str, user_id: str) -> bool:
        """Delete a document (soft delete - marks as deleted)"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if doc_id not in self.documents:
                return False
            
            doc = self.documents[doc_id]
            if doc["user_id"] != user_id:
                return False  # User can only delete their own docs
            
            # Mark as deleted (we can't easily remove from FAISS index)
            doc["deleted"] = True
            
            # Remove from user docs
            if user_id in self.user_docs and doc_id in self.user_docs[user_id]:
                self.user_docs[user_id].remove(doc_id)
            
            await self._save_documents()
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    async def get_document_count(self, user_id: Optional[str] = None) -> int:
        """Get total document count"""
        if not self._initialized:
            await self.initialize()
        
        if user_id:
            return len([d for d in self.documents.values() 
                       if d.get("user_id") == user_id and not d.get("deleted", False)])
        else:
            return len([d for d in self.documents.values() if not d.get("deleted", False)])
    
    async def similar_documents(
        self, 
        doc_id: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find documents similar to given document"""
        if not self._initialized:
            await self.initialize()
        
        if doc_id not in self.documents:
            return []
        
        doc = self.documents[doc_id]
        return await self.search(doc["content"], doc["user_id"], top_k + 1)[1:]  # Exclude self
    
    async def cleanup_deleted(self):
        """Clean up deleted documents (maintenance operation)"""
        if not self._initialized:
            await self.initialize()
        
        # Remove deleted documents from memory
        deleted_count = 0
        for doc_id in list(self.documents.keys()):
            if self.documents[doc_id].get("deleted", False):
                del self.documents[doc_id]
                deleted_count += 1
        
        if deleted_count > 0:
            await self._save_documents()
            logger.info(f"Cleaned up {deleted_count} deleted documents")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self._initialized:
            await self.initialize()
        
        total_docs = len([d for d in self.documents.values() if not d.get("deleted", False)])
        total_users = len(self.user_docs)
        
        user_stats = {}
        for user_id, doc_ids in self.user_docs.items():
            active_docs = len([doc_id for doc_id in doc_ids 
                             if doc_id in self.documents and not self.documents[doc_id].get("deleted", False)])
            user_stats[user_id] = active_docs
        
        return {
            "total_documents": total_docs,
            "total_users": total_users,
            "index_size": self.index.ntotal if self.index else 0,
            "user_document_counts": user_stats,
            "vector_dimension": self.vector_dim,
            "model_name": self.model_name
        }

# Global vector store instance
vector_store = VectorStore()

# Test function
async def test_vector_store():
    """Test vector store functionality"""
    
    print("🧪 Testing Vector Store")
    print("=" * 50)
    
    try:
        # Initialize
        await vector_store.initialize()
        print("✅ Vector store initialized")
        
        # Add test documents
        test_docs = [
            ("user1", "AI and machine learning are revolutionizing business processes"),
            ("user1", "Meeting notes: Discussed quarterly targets and team performance"),
            ("user2", "Reminder to complete project proposal by Friday"),
            ("user1", "Benefits of automation include cost reduction and efficiency gains")
        ]
        
        doc_ids = []
        for user_id, content in test_docs:
            doc_id = await vector_store.add_document(
                user_id, 
                content, 
                metadata={"source": "test", "type": "note"}
            )
            doc_ids.append(doc_id)
            print(f"✅ Added document: {doc_id[:8]}...")
        
        # Test search
        search_results = await vector_store.search("AI benefits", "user1", top_k=3)
        print(f"✅ Search results: {len(search_results)} documents found")
        for result in search_results:
            print(f"   Score: {result['score']:.3f} - {result['content'][:50]}...")
        
        # Test user documents
        user_docs = await vector_store.get_user_documents("user1")
        print(f"✅ User1 documents: {len(user_docs)}")
        
        # Test stats
        stats = await vector_store.get_stats()
        print(f"✅ Stats: {stats}")
        
        print("✅ All vector store tests passed!")
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_vector_store())
