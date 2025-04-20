# src/retrieval/vector_store.py
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalVectorStore:
    """
    A simple local vector store implementation for storing and retrieving embeddings.
    For production, consider using dedicated vector databases like Pinecone, Weaviate, or Chroma.
    """
    
    def __init__(self, 
                vector_store_dir: str,
                embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local vector store.
        
        Args:
            vector_store_dir: Directory containing vector store data
            embedding_model: Name of the sentence-transformers model to use for queries
        """
        self.vector_store_dir = vector_store_dir
        
        # Load embedding model for query encoding
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Storage for embeddings and chunks
        self.embeddings = None
        self.chunks = []
        self.section_map = {}
        
        # Report metadata
        self.report_name = os.path.basename(os.path.dirname(os.path.dirname(vector_store_dir)))
        
        # Initialize by loading data
        self._load_data()
        
        logger.info(f"Initialized local vector store for {self.report_name}")
        
    def _load_data(self):
        """Load embeddings and chunks from disk."""
        # Load embeddings
        embeddings_path = os.path.join(self.vector_store_dir, "embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        else:
            logger.error(f"Embeddings file not found: {embeddings_path}")
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        # Load chunks
        chunks_path = os.path.join(self.vector_store_dir, "chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            logger.info(f"Loaded {len(self.chunks)} chunks")
        else:
            logger.error(f"Chunks file not found: {chunks_path}")
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        # Load section map
        section_map_path = os.path.join(self.vector_store_dir, "section_map.json")
        if os.path.exists(section_map_path):
            with open(section_map_path, 'r', encoding='utf-8') as f:
                self.section_map = json.load(f)
            logger.info(f"Loaded section map with {len(self.section_map)} sections")
        else:
            logger.warning(f"Section map file not found: {section_map_path}")
    
    def similarity_search(self, 
                          query: str, 
                          top_k: int = 5, 
                          threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Compute similarities
        similarities = self._calculate_similarities(query_embedding)
        
        # Get top-k results above threshold
        top_indices = np.argsort(-similarities)[:top_k*2]  # Get more than needed to filter by threshold
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                # Get the chunk and add the similarity score
                chunk = self.chunks[idx].copy()
                chunk["similarity"] = float(similarity)
                results.append(chunk)
            
            # Stop once we have top_k results
            if len(results) >= top_k:
                break
        
        logger.info(f"Found {len(results)} results for query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        return results
    
    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query and all embeddings.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            Array of similarity scores
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Normalize document embeddings (if not already normalized)
        embedding_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = np.divide(self.embeddings, embedding_norms, 
                                         out=np.zeros_like(self.embeddings), 
                                         where=embedding_norms > 0)
        
        # Calculate cosine similarities
        similarities = np.dot(normalized_embeddings, query_embedding)
        
        return similarities
    
    def search_by_section(self, 
                          section_name: str, 
                          query: str = None, 
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific section, optionally filtered by query.
        
        Args:
            section_name: Name of the section to search in
            query: Optional query text to search within the section
            top_k: Number of top results to return
            
        Returns:
            List of chunk dictionaries
        """
        # Check if section exists
        if section_name not in self.section_map:
            logger.warning(f"Section '{section_name}' not found")
            return []
        
        # Get chunk IDs for the section
        chunk_ids = self.section_map[section_name]
        section_chunks = [chunk for chunk in self.chunks if chunk["id"] in chunk_ids]
        
        # If no query, return the first top_k chunks from the section
        if not query:
            logger.info(f"Returning {min(top_k, len(section_chunks))} chunks from section '{section_name}'")
            return section_chunks[:top_k]
        
        # Otherwise, perform similarity search within the section
        query_embedding = self.embedding_model.encode(query)
        
        # Get indices of section chunks in the main chunks list
        section_indices = [i for i, chunk in enumerate(self.chunks) if chunk["id"] in chunk_ids]
        
        # Calculate similarities for section chunks only
        section_embeddings = self.embeddings[section_indices]
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Normalize section embeddings
        embedding_norms = np.linalg.norm(section_embeddings, axis=1, keepdims=True)
        normalized_embeddings = np.divide(section_embeddings, embedding_norms, 
                                         out=np.zeros_like(section_embeddings), 
                                         where=embedding_norms > 0)
        
        # Calculate cosine similarities
        similarities = np.dot(normalized_embeddings, query_embedding)
        
        # Get top-k results
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in top_indices:
            orig_idx = section_indices[idx]
            chunk = self.chunks[orig_idx].copy()
            chunk["similarity"] = float(similarities[idx])
            results.append(chunk)
        
        logger.info(f"Found {len(results)} results for query in section '{section_name}'")
        
        return results
    
    def list_sections(self) -> List[str]:
        """
        List all available sections in the report.
        
        Returns:
            List of section names
        """
        return list(self.section_map.keys())
    
    def get_table_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all table chunks.
        
        Returns:
            List of table chunk dictionaries
        """
        table_chunks = [chunk for chunk in self.chunks 
                       if chunk["metadata"].get("source") == "table"]
        
        logger.info(f"Found {len(table_chunks)} table chunks")
        return table_chunks
    
    def hybrid_search(self, 
                     query: str, 
                     top_k: int = 5,
                     include_tables: bool = True) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search that combines similarity search with keyword filtering.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            include_tables: Whether to include table chunks in the search
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Extract potential keywords from the query
        keywords = self._extract_keywords(query)
        
        # First do a regular similarity search
        similarity_results = self.similarity_search(query, top_k=top_k*2)
        
        # Filter chunks by source if needed
        if not include_tables:
            similarity_results = [chunk for chunk in similarity_results 
                                if chunk["metadata"].get("source") != "table"]
        
        # Boost scores for chunks that contain keywords
        for chunk in similarity_results:
            keyword_matches = sum(1 for keyword in keywords 
                                if keyword.lower() in chunk["text"].lower())
            boost = min(0.2, 0.05 * keyword_matches)  # Cap the boost at 0.2
            chunk["similarity"] = min(1.0, chunk["similarity"] + boost)
        
        # Re-sort by adjusted similarity
        similarity_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarity_results[:top_k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract potential keywords from the query.
        
        Args:
            query: Query text
            
        Returns:
            List of keyword strings
        """
        # Simple keyword extraction based on word length and common stopwords
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                    'when', 'where', 'how', 'which', 'who', 'whom', 'is', 'are', 'was',
                    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                    'did', 'to', 'at', 'in', 'on', 'by', 'for', 'with', 'about', 'against',
                    'between', 'into', 'through', 'during', 'before', 'after', 'above',
                    'below', 'from', 'up', 'down', 'of', 'off', 'over', 'under', 'again',
                    'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
                    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'}
        
        words = [word.strip('.,?!()[]{}":;') for word in query.split()]
        keywords = [word for word in words 
                   if word.lower() not in stopwords and len(word) > 3]
        
        return keywords


class PineconeVectorStore:
    """
    Pinecone vector store implementation for production use.
    This allows scaling to much larger documents and collections of reports.
    """
    
    def __init__(self, 
                 api_key: str,
                 environment: str,
                 index_name: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 namespace: str = "default"):
        """
        Initialize the Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            embedding_model: Name of the sentence-transformers model to use
            namespace: Namespace in the Pinecone index
        """
        try:
            import pinecone
            pinecone.init(api_key=api_key, environment=environment)
            
            # Check if index exists, if not create it
            if index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {index_name}")
                pinecone.create_index(
                    name=index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine"
                )
                
            self.index = pinecone.Index(index_name)
            self.namespace = namespace
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(embedding_model)
            
            logger.info(f"Initialized Pinecone vector store: {index_name}")
            
        except ImportError:
            logger.error("Pinecone package not installed. Install with 'pip install pinecone-client'")
            raise
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def upload_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> int:
        """
        Upload chunks and embeddings to Pinecone.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
            
        Returns:
            Number of chunks uploaded
        """
        import pinecone
        
        logger.info(f"Uploading {len(chunks)} chunks to Pinecone")
        
        # Prepare vector data
        vectors = []
        for i, chunk in enumerate(chunks):
            vector = {
                "id": chunk["id"],
                "values": embeddings[i].tolist(),
                "metadata": {
                    "text": chunk["text"],
                    "section": chunk["metadata"].get("section", ""),
                    "source": chunk["metadata"].get("source", "text"),
                    "chunk_index": chunk["metadata"].get("chunk_index", 0)
                }
            }
            vectors.append(vector)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            
            # Log progress
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(vectors):
                logger.info(f"Uploaded {min(i+batch_size, len(vectors))}/{len(vectors)} vectors")
        
        logger.info(f"Completed uploading {len(chunks)} chunks to Pinecone")
        return len(chunks)
    
    def similarity_search(self, 
                         query: str, 
                         top_k: int = 5, 
                         filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search in Pinecone.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            filter: Optional filter to apply to search
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Perform the search
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results["matches"]:
            chunk = {
                "id": match["id"],
                "text": match["metadata"]["text"],
                "metadata": {
                    "section": match["metadata"]["section"],
                    "source": match["metadata"]["source"],
                    "chunk_index": match["metadata"]["chunk_index"]
                },
                "similarity": match["score"]
            }
            formatted_results.append(chunk)
        
        logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        return formatted_results
    
    def search_by_section(self, 
                         section_name: str, 
                         query: str = None, 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a specific section.
        
        Args:
            section_name: Name of the section to search in
            query: Optional query text
            top_k: Number of top results to return
            
        Returns:
            List of chunk dictionaries
        """
        # Create filter for section
        filter = {"section": {"$eq": section_name}}
        
        # If no query, get random documents from the section
        if not query:
            # Simple solution - use a generic query that will match most documents
            return self.similarity_search("this section", top_k=top_k, filter=filter)
        
        # With query, perform filtered search
        return self.similarity_search(query, top_k=top_k, filter=filter)


# Helper function to create appropriate vector store
def create_vector_store(vector_store_type: str, **kwargs) -> Any:
    """
    Factory function to create the appropriate vector store.
    
    Args:
        vector_store_type: Type of vector store ("local" or "pinecone")
        **kwargs: Additional arguments for the specific vector store
        
    Returns:
        Vector store instance
    """
    if vector_store_type.lower() == "local":
        return LocalVectorStore(**kwargs)
    elif vector_store_type.lower() == "pinecone":
        return PineconeVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <path_to_vector_store_dir>")
        sys.exit(1)
    
    vector_store_dir = sys.argv[1]
    store = LocalVectorStore(vector_store_dir)
    
    # Test search
    results = store.similarity_search("What are the company's primary risk factors?", top_k=3)
    
    print(f"Search results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (similarity: {result['similarity']:.4f}):")
        print(f"Section: {result['metadata']['section']}")
        print(f"Text: {result['text'][:200]}...")