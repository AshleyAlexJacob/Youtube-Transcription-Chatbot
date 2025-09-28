"""
FastAPI REST API for YouTube Transcription Chatbot
"""
import os
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Import services
from src.services.rag_service import RAGService
from src.services.vector_db_service import VectorDBService
from src.services.llm_service import LLMService
from src.services.yt_service import YtService
from src.services.stt_service import STTService

# Import models
from src.models.api_models import (
    VideoProcessRequest, VideoProcessResponse,
    AudioDownloadResponse, TranscriptionResponse,
    ChatRequest, ChatResponse,
    VectorSearchRequest, VectorSearchResponse, VectorSearchResult,
    VectorIngestRequest, VectorIngestResponse,
    VideoSummaryRequest, VideoSummaryResponse,
    VideoListResponse, ConversationHistoryResponse,
    SystemStatsResponse, HealthCheckResponse,
    BaseResponse
)

# Global service instances
rag_service: Optional[RAGService] = None
vector_service: Optional[VectorDBService] = None
llm_service: Optional[LLMService] = None
yt_service: Optional[YtService] = None
stt_service: Optional[STTService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global rag_service, vector_service, llm_service, yt_service, stt_service
    
    # Startup
    print("🚀 Starting YouTube Transcription Chatbot API...")
    try:
        # Initialize services
        print("📊 Initializing services...")
        vector_service = VectorDBService()
        llm_service = LLMService()
        yt_service = YtService()
        stt_service = STTService()
        rag_service = RAGService(
            vector_service=vector_service,
            llm_service=llm_service,
            yt_service=yt_service,
            stt_service=stt_service
        )
        print("✅ All services initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {str(e)}")
        raise e
    
    yield
    
    # Shutdown
    print("🛑 Shutting down YouTube Transcription Chatbot API...")


# Initialize FastAPI app
app = FastAPI(
    title="YouTube Transcription Chatbot API",
    description="REST API for processing YouTube videos, transcribing content, and providing AI-powered chat responses",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    error_message = f"Internal server error: {str(exc)}"
    print(f"❌ {error_message}")
    print(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": error_message,
            "message": "An unexpected error occurred"
        }
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Check the health status of all services"""
    try:
        services_status = {}
        
        # Check each service
        try:
            if vector_service:
                collection_info = vector_service.get_collection_info()
                services_status["vector_db"] = "healthy" if collection_info else "unhealthy"
            else:
                services_status["vector_db"] = "not_initialized"
        except Exception as e:
            services_status["vector_db"] = f"error: {str(e)}"
        
        try:
            if llm_service:
                model_info = llm_service.get_model_info()
                services_status["llm"] = "healthy" if model_info else "unhealthy"
            else:
                services_status["llm"] = "not_initialized"
        except Exception as e:
            services_status["llm"] = f"error: {str(e)}"
        
        try:
            if yt_service:
                services_status["youtube"] = "healthy"
            else:
                services_status["youtube"] = "not_initialized"
        except Exception as e:
            services_status["youtube"] = f"error: {str(e)}"
        
        try:
            if stt_service:
                services_status["transcription"] = "healthy"
            else:
                services_status["transcription"] = "not_initialized"
        except Exception as e:
            services_status["transcription"] = f"error: {str(e)}"
        
        return HealthCheckResponse(
            status="healthy" if all("healthy" in status for status in services_status.values()) else "degraded",
            timestamp=datetime.now().isoformat(),
            services=services_status,
            version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


# Video Processing Endpoints
@app.post("/api/v1/videos/process", response_model=VideoProcessResponse, tags=["Video Processing"])
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    Complete video processing pipeline: Download, transcribe, and ingest to vector DB
    """
    try:
        if not rag_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not available"
            )
        
        # Process video
        result = rag_service.process_youtube_video(
            video_url=str(request.video_url),
            custom_title=request.custom_title
        )
        
        if result["success"]:
            return VideoProcessResponse(
                success=True,
                message="Video processed successfully",
                video_title=result["video_title"],
                video_url=result["video_url"],
                audio_path=result["audio_path"],
                transcription=result["transcription"],
                summary=result["summary"],
                chunks_count=result["chunks_count"],
                point_ids=result["point_ids"]
            )
        else:
            return VideoProcessResponse(
                success=False,
                message="Video processing failed",
                error=result["error"]
            )
            
    except Exception as e:
        return VideoProcessResponse(
            success=False,
            message="Video processing failed",
            error=str(e)
        )


@app.post("/api/v1/videos/download", response_model=AudioDownloadResponse, tags=["Video Processing"])
async def download_audio(request: VideoProcessRequest):
    """
    Download audio from YouTube video
    """
    try:
        if not yt_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="YouTube service not available"
            )
        
        # Download audio
        audio_path = yt_service.download_audio(str(request.video_url))
        
        # Get file info
        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        video_title = os.path.basename(audio_path).replace('.m4a', '').replace('_', ' ')
        
        return AudioDownloadResponse(
            success=True,
            message="Audio downloaded successfully",
            audio_path=audio_path,
            video_title=video_title,
            file_size=file_size
        )
        
    except Exception as e:
        return AudioDownloadResponse(
            success=False,
            message="Audio download failed",
            error=str(e)
        )


@app.post("/api/v1/audio/transcribe", response_model=TranscriptionResponse, tags=["Audio Processing"])
async def transcribe_audio(request: AudioTranscribeRequest):
    """
    Transcribe audio file to text
    """
    try:
        if not stt_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Speech-to-text service not available"
            )
        
        # Check if audio file exists
        if not os.path.exists(request.audio_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audio file not found: {request.audio_path}"
            )
        
        # Transcribe audio
        result = stt_service.convert_audio_to_text(
            filepath=request.audio_path,
            filename=request.filename
        )
        
        return TranscriptionResponse(
            success=True,
            message="Audio transcribed successfully",
            transcription=result['text'],
            filename=request.filename,
            audio_path=request.audio_path
        )
        
    except Exception as e:
        return TranscriptionResponse(
            success=False,
            message="Audio transcription failed",
            error=str(e)
        )


# Chat and Q&A Endpoints
@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat & Q&A"])
async def chat(request: ChatRequest):
    """
    Chat with AI using RAG (Retrieval Augmented Generation)
    """
    try:
        if not rag_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not available"
            )
        
        # Process chat request
        if request.use_rag:
            result = rag_service.ask_question(
                question=request.message,
                video_title_filter=request.video_filter
            )
            
            return ChatResponse(
                success=result["success"],
                message="Response generated successfully" if result["success"] else "Failed to generate response",
                question=result.get("question"),
                answer=result.get("answer"),
                sources=result.get("sources"),
                retrieved_chunks=result.get("retrieved_chunks"),
                used_rag=result.get("used_rag"),
                chunk_details=result.get("chunk_details"),
                error=result.get("error")
            )
        else:
            # Simple chat without RAG
            response = rag_service.chat(request.message)
            return ChatResponse(
                success=True,
                message="Response generated successfully",
                question=request.message,
                answer=response,
                sources=[],
                retrieved_chunks=0,
                used_rag=False
            )
            
    except Exception as e:
        return ChatResponse(
            success=False,
            message="Chat request failed",
            error=str(e)
        )


# Vector Database Endpoints
@app.post("/api/v1/vector/search", response_model=VectorSearchResponse, tags=["Vector Database"])
async def search_vectors(request: VectorSearchRequest):
    """
    Search for similar content in the vector database
    """
    try:
        if not vector_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database service not available"
            )
        
        # Perform search
        results = vector_service.search_similar(
            query=request.query,
            limit=request.limit,
            video_title_filter=request.video_title_filter,
            score_threshold=request.score_threshold
        )
        
        # Convert to response model
        search_results = [
            VectorSearchResult(
                id=result["id"],
                score=result["score"],
                text=result["text"],
                video_title=result["video_title"],
                video_url=result.get("video_url"),
                chunk_index=result.get("chunk_index"),
                metadata=result.get("metadata")
            )
            for result in results
        ]
        
        return VectorSearchResponse(
            success=True,
            message=f"Found {len(search_results)} results",
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        return VectorSearchResponse(
            success=False,
            message="Vector search failed",
            error=str(e)
        )


@app.post("/api/v1/vector/ingest", response_model=VectorIngestResponse, tags=["Vector Database"])
async def ingest_to_vector_db(request: VectorIngestRequest):
    """
    Ingest text content into the vector database
    """
    try:
        if not vector_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database service not available"
            )
        
        # Ingest content
        point_ids = vector_service.ingest_transcription(
            transcription_text=request.transcription_text,
            video_title=request.video_title,
            video_url=request.video_url or "",
            metadata=request.metadata
        )
        
        return VectorIngestResponse(
            success=True,
            message="Content ingested successfully",
            video_title=request.video_title,
            chunks_created=len(point_ids),
            point_ids=point_ids
        )
        
    except Exception as e:
        return VectorIngestResponse(
            success=False,
            message="Vector ingestion failed",
            error=str(e)
        )


# Video Management Endpoints
@app.get("/api/v1/videos", response_model=VideoListResponse, tags=["Video Management"])
async def list_videos():
    """
    Get list of all available videos in the database
    """
    try:
        if not rag_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not available"
            )
        
        videos = rag_service.list_available_videos()
        
        return VideoListResponse(
            success=True,
            message=f"Found {len(videos)} videos",
            videos=videos,
            total_videos=len(videos)
        )
        
    except Exception as e:
        return VideoListResponse(
            success=False,
            message="Failed to list videos",
            error=str(e)
        )


@app.post("/api/v1/videos/summary", response_model=VideoSummaryResponse, tags=["Video Management"])
async def get_video_summary(request: VideoSummaryRequest):
    """
    Generate or retrieve summary for a specific video
    """
    try:
        if not rag_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not available"
            )
        
        summary = rag_service.get_video_summary(request.video_title)
        
        if summary:
            return VideoSummaryResponse(
                success=True,
                message="Summary generated successfully",
                video_title=request.video_title,
                summary=summary
            )
        else:
            return VideoSummaryResponse(
                success=False,
                message="Video not found or summary generation failed",
                video_title=request.video_title,
                error="Video not found in database"
            )
            
    except Exception as e:
        return VideoSummaryResponse(
            success=False,
            message="Summary generation failed",
            error=str(e)
        )


@app.delete("/api/v1/videos/{video_title}", response_model=BaseResponse, tags=["Video Management"])
async def delete_video(video_title: str):
    """
    Delete all content for a specific video
    """
    try:
        if not rag_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not available"
            )
        
        success = rag_service.delete_video(video_title)
        
        if success:
            return BaseResponse(
                success=True,
                message=f"Video '{video_title}' deleted successfully"
            )
        else:
            return BaseResponse(
                success=False,
                message=f"Failed to delete video '{video_title}'"
            )
            
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f"Video deletion failed: {str(e)}"
        )


# Conversation Management Endpoints
@app.get("/api/v1/conversation/history", response_model=ConversationHistoryResponse, tags=["Conversation"])
async def get_conversation_history():
    """
    Get current conversation history
    """
    try:
        if not rag_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not available"
            )
        
        history = rag_service.get_conversation_history()
        
        return ConversationHistoryResponse(
            success=True,
            message=f"Retrieved {len(history)} conversation messages",
            history=history,
            total_messages=len(history)
        )
        
    except Exception as e:
        return ConversationHistoryResponse(
            success=False,
            message="Failed to retrieve conversation history",
            error=str(e)
        )


@app.delete("/api/v1/conversation/history", response_model=BaseResponse, tags=["Conversation"])
async def clear_conversation_history():
    """
    Clear conversation history
    """
    try:
        if not rag_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not available"
            )
        
        rag_service.clear_conversation_history()
        
        return BaseResponse(
            success=True,
            message="Conversation history cleared successfully"
        )
        
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f"Failed to clear conversation history: {str(e)}"
        )


# System Information Endpoints
@app.get("/api/v1/system/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_system_stats():
    """
    Get system statistics and information
    """
    try:
        if not rag_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not available"
            )
        
        stats = rag_service.get_system_stats()
        
        if "error" in stats:
            return SystemStatsResponse(
                success=False,
                message="Failed to retrieve system stats",
                error=stats["error"]
            )
        
        return SystemStatsResponse(
            success=True,
            message="System stats retrieved successfully",
            vector_db=stats.get("vector_db"),
            llm_model=stats.get("llm_model"),
            available_videos=stats.get("available_videos"),
            video_titles=stats.get("video_titles")
        )
        
    except Exception as e:
        return SystemStatsResponse(
            success=False,
            message="Failed to retrieve system stats",
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
