import os
import uuid
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(".env")

class VectorDBService:
    def __init__(self, 
                 collection_name: str = "youtube_transcriptions",
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None):
        """
        Initialize Vector DB Service with Qdrant Cloud
        
        Args:
            collection_name: Name of the Qdrant collection
            model_name: Sentence transformer model for embeddings
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            qdrant_url: Qdrant cloud URL (if None, reads from env)
            qdrant_api_key: Qdrant API key (if None, reads from env)
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize Qdrant client
        self._initialize_qdrant_client(qdrant_url, qdrant_api_key)
        
        # Create collection if it doesn't exist
        # self._create_collection_if_not_exists()
    
    def _initialize_qdrant_client(self, qdrant_url: Optional[str], qdrant_api_key: Optional[str]):
        """Initialize Qdrant client with cloud credentials"""
        try:
            # Get credentials from environment if not provided
            if not qdrant_url:
                qdrant_url = os.getenv("QDRANT_HOST")
                print(f"Qdrant URL: {qdrant_url}")
            if not qdrant_api_key:
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if not qdrant_url or not qdrant_api_key:
                raise ValueError(
                    "Qdrant credentials not found. Please provide QDRANT_URL and QDRANT_API_KEY "
                    "either as parameters or environment variables."
                )
            
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
            
            # Test connection
            collections = self.client.get_collections()
            print("✓ Successfully connected to Qdrant Cloud")
            
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant Cloud: {str(e)}")
            print("\nTo fix this issue:")
            print("1. Make sure you have a Qdrant Cloud account")
            print("2. Set QDRANT_URL environment variable (e.g., https://your-cluster.qdrant.tech)")
            print("3. Set QDRANT_API_KEY environment variable")
            print("4. Or pass them as parameters to the constructor")
            raise RuntimeError("Qdrant connection failed. Please check your credentials.")
    
    def _create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"✓ Created collection '{self.collection_name}'")
            else:
                print(f"✓ Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            print(f"❌ Failed to create collection: {str(e)}")
            raise RuntimeError(f"Collection creation failed: {str(e)}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        if not text or not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        print(f"✓ Split text into {len(chunks)} chunks")
        return chunks
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            print(f"✓ Generated embeddings for {len(texts)} texts")
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            print(f"❌ Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    def ingest_transcription(self, 
                           transcription_text: str, 
                           video_title: str,
                           video_url: str = "",
                           metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Ingest a YouTube transcription into the vector database
        
        Args:
            transcription_text: The transcribed text
            video_title: Title of the YouTube video
            video_url: URL of the YouTube video
            metadata: Additional metadata
            
        Returns:
            List of point IDs that were inserted
        """
        try:
            # Chunk the transcription
            chunks = self.chunk_text(transcription_text)
            if not chunks:
                print("⚠️ No chunks generated from transcription")
                return []
            
            # Generate embeddings
            embeddings = self.embed_texts(chunks)
            
            # Prepare points for insertion
            points = []
            point_ids = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Prepare metadata
                chunk_metadata = {
                    "video_title": video_title,
                    "video_url": video_url,
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "total_chunks": len(chunks)
                }
                
                # Add custom metadata if provided
                if metadata:
                    chunk_metadata.update(metadata)
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=chunk_metadata
                    )
                )
            
            # Insert points into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"✓ Successfully ingested {len(chunks)} chunks for video: {video_title}")
            return point_ids
            
        except Exception as e:
            print(f"❌ Failed to ingest transcription: {str(e)}")
            raise RuntimeError(f"Transcription ingestion failed: {str(e)}")
    
    def search_similar(self, 
                      query: str, 
                      limit: int = 5,
                      video_title_filter: Optional[str] = None,
                      score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar content in the vector database
        
        Args:
            query: Search query
            limit: Number of results to return
            video_title_filter: Filter by specific video title
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self.embed_texts([query])[0]
            
            # Prepare filter if video title is specified
            search_filter = None
            if video_title_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="video_title",
                            match=MatchValue(value=video_title_filter)
                        )
                    ]
                )
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("chunk_text", ""),
                    "video_title": result.payload.get("video_title", ""),
                    "video_url": result.payload.get("video_url", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "metadata": result.payload
                })
            
            print(f"✓ Found {len(formatted_results)} similar results for query: '{query[:50]}...'")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
            raise RuntimeError(f"Search operation failed: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            print(f"❌ Failed to get collection info: {str(e)}")
            return {}
    
    def delete_by_video_title(self, video_title: str) -> bool:
        """Delete all chunks for a specific video"""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="video_title",
                        match=MatchValue(value=video_title)
                    )
                ]
            )
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            
            print(f"✓ Deleted all chunks for video: {video_title}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete video chunks: {str(e)}")
            return False


# Example usage for testing
# if __name__ == "__main__":
#     # Initialize service
#     vector_service = VectorDBService()
# #     
# #     # Test with sample transcription
#     sample_text = "This is a sample transcription from a YouTube video about data science..."
#     video_title = "Introduction to Data Science"
#     video_url = "https://youtube.com/watch?v=example"
#     
# #     # Ingest transcription
#     point_ids = vector_service.ingest_transcription(
#         transcription_text=sample_text,
#         video_title=video_title,
#         video_url=video_url
#     )
    
#     # Search for similar content
    # results = vector_service.search_similar("What is data science?", limit=3)
#     
    # # Print results
    # for result in results:
    #     print(f"Score: {result['score']:.3f}")
    #     print(f"Text: {result['text'][:100]}...")
    #     print("---")
