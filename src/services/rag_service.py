"""
RAG Service - Orchestrates Vector DB and LLM services for complete RAG pipeline
"""
import os
from typing import List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv

from src.services.vector_db_service import VectorDBService
from src.services.llm_service import LLMService
from src.services.yt_service import YtService
from src.services.stt_service import STTService

load_dotenv(".env")


class RAGService:
    """
    Complete RAG (Retrieval Augmented Generation) service that orchestrates
    YouTube downloading, transcription, vector storage, and LLM responses
    """
    
    def __init__(self,
                 vector_service: Optional[VectorDBService] = None,
                 llm_service: Optional[LLMService] = None,
                 yt_service: Optional[YtService] = None,
                 stt_service: Optional[STTService] = None,
                 retrieval_k: int = 5,
                 score_threshold: float = 0.3):
        """
        Initialize RAG Service
        
        Args:
            vector_service: VectorDBService instance (creates new if None)
            llm_service: LLMService instance (creates new if None)
            yt_service: YtService instance (creates new if None)
            stt_service: STTService instance (creates new if None)
            retrieval_k: Number of chunks to retrieve for context
            score_threshold: Minimum similarity score for retrieval
        """
        self.retrieval_k = retrieval_k
        self.score_threshold = score_threshold
        
        # Initialize services
        self.vector_service = vector_service or VectorDBService()
        self.llm_service = llm_service or LLMService()
        self.yt_service = yt_service or YtService()
        self.stt_service = stt_service or STTService()
        
        print("✓ RAG Service initialized with all components")
    
    def process_youtube_video(self, 
                            video_url: str, 
                            custom_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: Download YouTube video, transcribe, and ingest to vector DB
        
        Args:
            video_url: YouTube video URL
            custom_title: Custom title for the video (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            print(f"🚀 Starting complete processing for: {video_url}")
            
            # Step 1: Download audio
            print("📥 Downloading audio...")
            audio_path = self.yt_service.download_audio(video_url)
            
            # Extract video title from the downloaded file path
            if custom_title:
                video_title = custom_title
            else:
                # Extract title from file path
                import os
                filename = os.path.basename(audio_path)
                video_title = filename.replace('.m4a', '').replace('_', ' ')
            
            # Step 2: Transcribe audio
            print("🎤 Transcribing audio...")
            transcription_result = self.stt_service.convert_audio_to_text(
                filepath=audio_path,
                filename=video_title.replace(' ', '_')
            )
            transcription_text = transcription_result['text']
            
            # Step 3: Ingest to vector database
            print("🗄️ Ingesting to vector database...")
            point_ids = self.vector_service.ingest_transcription(
                transcription_text=transcription_text,
                video_title=video_title,
                video_url=video_url,
                metadata={
                    "audio_path": audio_path,
                    "processing_timestamp": None  # Could add timestamp
                }
            )
            
            # Step 4: Generate summary
            print("📝 Generating summary...")
            summary = self.llm_service.generate_summary(
                content=transcription_text,
                title=video_title
            )
            
            result = {
                "success": True,
                "video_title": video_title,
                "video_url": video_url,
                "audio_path": audio_path,
                "transcription": transcription_text,
                "summary": summary,
                "point_ids": point_ids,
                "chunks_count": len(point_ids)
            }
            
            print(f"✅ Successfully processed video: {video_title}")
            print(f"   - Chunks created: {len(point_ids)}")
            print(f"   - Summary generated: {len(summary)} characters")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "video_url": video_url
            }
            print(f"❌ Failed to process video: {str(e)}")
            return error_result
    
    def ask_question(self, 
                    question: str, 
                    video_title_filter: Optional[str] = None,
                    use_conversation_history: bool = True) -> Dict[str, Any]:
        """
        Ask a question using RAG pipeline
        
        Args:
            question: User question
            video_title_filter: Filter results to specific video
            use_conversation_history: Whether to use conversation history
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            print(f"🤔 Processing question: '{question[:50]}...'")
            
            # Step 1: Retrieve relevant chunks
            print("🔍 Retrieving relevant content...")
            retrieved_chunks = self.vector_service.search_similar(
                query=question,
                limit=self.retrieval_k,
                video_title_filter=video_title_filter,
                score_threshold=self.score_threshold
            )
            
            if not retrieved_chunks:
                print("⚠️ No relevant content found")
                response = self.llm_service.send_message(
                    question, 
                    use_rag=False
                )
                return {
                    "success": True,
                    "question": question,
                    "answer": response,
                    "sources": [],
                    "retrieved_chunks": 0,
                    "used_rag": False
                }
            
            # Step 2: Generate response with context
            print(f"🧠 Generating response with {len(retrieved_chunks)} chunks...")
            response, source_titles = self.llm_service.ask_with_context(
                question=question,
                retrieved_chunks=retrieved_chunks
            )
            
            result = {
                "success": True,
                "question": question,
                "answer": response,
                "sources": source_titles,
                "retrieved_chunks": len(retrieved_chunks),
                "used_rag": True,
                "chunk_details": [
                    {
                        "video_title": chunk["video_title"],
                        "score": chunk["score"],
                        "text_preview": chunk["text"][:100] + "..."
                    }
                    for chunk in retrieved_chunks
                ]
            }
            
            print(f"✅ Generated response using {len(retrieved_chunks)} chunks from {len(source_titles)} videos")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "question": question
            }
            print(f"❌ Failed to process question: {str(e)}")
            return error_result
    
    def chat(self, message: str, video_filter: Optional[str] = None) -> str:
        """
        Simple chat interface that returns just the response
        
        Args:
            message: User message
            video_filter: Optional video title filter
            
        Returns:
            AI response as string
        """
        result = self.ask_question(message, video_filter)
        return result.get("answer", "I'm sorry, I couldn't process your question.")
    
    def get_video_summary(self, video_title: str) -> Optional[str]:
        """
        Get a summary for a specific video
        
        Args:
            video_title: Title of the video
            
        Returns:
            Summary string or None if not found
        """
        try:
            # Search for chunks from the specific video
            chunks = self.vector_service.search_similar(
                query="summary overview main points",  # Generic query to get representative chunks
                limit=10,  # Get more chunks for better summary
                video_title_filter=video_title,
                score_threshold=0.0  # Lower threshold for summary
            )
            
            if not chunks:
                print(f"⚠️ No content found for video: {video_title}")
                return None
            
            # Combine chunks to recreate approximate full content
            full_content = " ".join([chunk["text"] for chunk in chunks])
            
            # Generate summary
            summary = self.llm_service.generate_summary(
                content=full_content,
                title=video_title
            )
            
            return summary
            
        except Exception as e:
            print(f"❌ Failed to generate summary for {video_title}: {str(e)}")
            return None
    
    def list_available_videos(self) -> List[str]:
        """
        Get list of available video titles in the database
        
        Returns:
            List of video titles
        """
        try:
            # This is a workaround since VectorDBService doesn't have a direct method
            # We'll search with a generic query and extract unique video titles
            results = self.vector_service.search_similar(
                query="video content",
                limit=100,  # Get many results
                score_threshold=0.0
            )
            
            video_titles = list(set([result["video_title"] for result in results]))
            return sorted(video_titles)
            
        except Exception as e:
            print(f"❌ Failed to list videos: {str(e)}")
            return []
    
    def delete_video(self, video_title: str) -> bool:
        """
        Delete all content for a specific video
        
        Args:
            video_title: Title of the video to delete
            
        Returns:
            True if successful, False otherwise
        """
        return self.vector_service.delete_by_video_title(video_title)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history"""
        return self.llm_service.get_conversation_history()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.llm_service.clear_history()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            Dictionary with system stats
        """
        try:
            collection_info = self.vector_service.get_collection_info()
            model_info = self.llm_service.get_model_info()
            available_videos = self.list_available_videos()
            
            return {
                "vector_db": collection_info,
                "llm_model": model_info,
                "available_videos": len(available_videos),
                "video_titles": available_videos
            }
        except Exception as e:
            return {"error": str(e)}


# # Example usage for testing
# if __name__ == "__main__":
#     # Initialize RAG service
#     rag = RAGService()
# #     
#     # Process a YouTube video (uncomment to test)
#     # result = rag.process_youtube_video("https://youtu.be/your-video-id")
#     # print(f"Processing result: {result}")
# #     
#     # Ask questions
#     response = rag.ask_question("What is data science?")
#     print(f"Question: {response['question']}")
#     print(f"Answer: {response['answer']}")
#     print(f"Sources: {response['sources']}")
    
#     # Simple chat
#     answer = rag.chat("Tell me about machine learning")
#     print(f"Chat response: {answer}")
# #     
#     # Get system stats
#     stats = rag.get_system_stats()
#     print(f"System stats: {stats}")
