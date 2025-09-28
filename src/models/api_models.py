"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status enum for processing operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# Request Models
class VideoProcessRequest(BaseModel):
    """Request model for video processing"""
    video_url: HttpUrl = Field(..., description="YouTube video URL")
    custom_title: Optional[str] = Field(None, description="Custom title for the video")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://youtu.be/dQw4w9WgXcQ",
                "custom_title": "My Custom Video Title"
            }
        }


class AudioTranscribeRequest(BaseModel):
    """Request model for audio transcription"""
    audio_path: str = Field(..., description="Path to the audio file")
    filename: Optional[str] = Field(None, description="Custom filename for transcription")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_path": "artifacts/audios/sample_audio.m4a",
                "filename": "sample_transcription"
            }
        }


class ChatRequest(BaseModel):
    """Request model for chat/question answering"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message or question")
    video_filter: Optional[str] = Field(None, description="Filter responses to specific video title")
    use_rag: bool = Field(True, description="Whether to use RAG (Retrieval Augmented Generation)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is data science?",
                "video_filter": "Introduction to Data Science",
                "use_rag": True
            }
        }


class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: int = Field(5, ge=1, le=20, description="Number of results to return")
    video_title_filter: Optional[str] = Field(None, description="Filter by specific video title")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "limit": 5,
                "video_title_filter": "ML Tutorial",
                "score_threshold": 0.3
            }
        }


class VectorIngestRequest(BaseModel):
    """Request model for vector database ingestion"""
    transcription_text: str = Field(..., min_length=10, description="Text content to ingest")
    video_title: str = Field(..., min_length=1, max_length=200, description="Title of the video")
    video_url: Optional[str] = Field(None, description="URL of the source video")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transcription_text": "This is a sample transcription of a video about data science...",
                "video_title": "Introduction to Data Science",
                "video_url": "https://youtu.be/example",
                "metadata": {"duration": "10:30", "category": "education"}
            }
        }


class VideoSummaryRequest(BaseModel):
    """Request model for video summary generation"""
    video_title: str = Field(..., min_length=1, description="Title of the video to summarize")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_title": "Introduction to Machine Learning"
            }
        }


# Response Models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: Optional[str] = Field(None, description="Additional message or error description")


class VideoProcessResponse(BaseResponse):
    """Response model for video processing"""
    video_title: Optional[str] = Field(None, description="Title of the processed video")
    video_url: Optional[str] = Field(None, description="URL of the processed video")
    audio_path: Optional[str] = Field(None, description="Path to the downloaded audio file")
    transcription: Optional[str] = Field(None, description="Full transcription text")
    summary: Optional[str] = Field(None, description="AI-generated summary")
    chunks_count: Optional[int] = Field(None, description="Number of chunks created in vector DB")
    point_ids: Optional[List[str]] = Field(None, description="Vector DB point IDs")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "video_title": "Introduction to Data Science",
                "video_url": "https://youtu.be/example",
                "audio_path": "artifacts/audios/intro_data_science.m4a",
                "transcription": "Data science is a field that combines...",
                "summary": "This video introduces the fundamentals of data science...",
                "chunks_count": 15,
                "point_ids": ["uuid1", "uuid2", "uuid3"]
            }
        }


class AudioDownloadResponse(BaseResponse):
    """Response model for audio download"""
    audio_path: Optional[str] = Field(None, description="Path to the downloaded audio file")
    video_title: Optional[str] = Field(None, description="Title of the video")
    file_size: Optional[int] = Field(None, description="Size of the audio file in bytes")
    error: Optional[str] = Field(None, description="Error message if download failed")


class TranscriptionResponse(BaseResponse):
    """Response model for transcription"""
    transcription: Optional[str] = Field(None, description="Transcribed text")
    filename: Optional[str] = Field(None, description="Name of the transcription file")
    audio_path: Optional[str] = Field(None, description="Path to the source audio file")
    error: Optional[str] = Field(None, description="Error message if transcription failed")


class ChatResponse(BaseResponse):
    """Response model for chat/QA"""
    question: Optional[str] = Field(None, description="Original user question")
    answer: Optional[str] = Field(None, description="AI-generated response")
    sources: Optional[List[str]] = Field(None, description="Source video titles used for response")
    retrieved_chunks: Optional[int] = Field(None, description="Number of chunks retrieved from vector DB")
    used_rag: Optional[bool] = Field(None, description="Whether RAG was used for the response")
    chunk_details: Optional[List[Dict[str, Any]]] = Field(None, description="Details of retrieved chunks")
    error: Optional[str] = Field(None, description="Error message if chat failed")


class VectorSearchResult(BaseModel):
    """Model for individual vector search result"""
    id: str = Field(..., description="Unique ID of the chunk")
    score: float = Field(..., description="Similarity score")
    text: str = Field(..., description="Text content of the chunk")
    video_title: str = Field(..., description="Title of the source video")
    video_url: Optional[str] = Field(None, description="URL of the source video")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk in the video")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class VectorSearchResponse(BaseResponse):
    """Response model for vector search"""
    query: Optional[str] = Field(None, description="Original search query")
    results: Optional[List[VectorSearchResult]] = Field(None, description="Search results")
    total_results: Optional[int] = Field(None, description="Total number of results found")
    error: Optional[str] = Field(None, description="Error message if search failed")


class VectorIngestResponse(BaseResponse):
    """Response model for vector ingestion"""
    video_title: Optional[str] = Field(None, description="Title of the ingested video")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created")
    point_ids: Optional[List[str]] = Field(None, description="Vector DB point IDs")
    error: Optional[str] = Field(None, description="Error message if ingestion failed")


class VideoSummaryResponse(BaseResponse):
    """Response model for video summary"""
    video_title: Optional[str] = Field(None, description="Title of the summarized video")
    summary: Optional[str] = Field(None, description="AI-generated summary")
    error: Optional[str] = Field(None, description="Error message if summary generation failed")


class VideoListResponse(BaseResponse):
    """Response model for video list"""
    videos: Optional[List[str]] = Field(None, description="List of available video titles")
    total_videos: Optional[int] = Field(None, description="Total number of videos")
    error: Optional[str] = Field(None, description="Error message if listing failed")


class ConversationHistoryResponse(BaseResponse):
    """Response model for conversation history"""
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Conversation history")
    total_messages: Optional[int] = Field(None, description="Total number of messages")
    error: Optional[str] = Field(None, description="Error message if retrieval failed")


class SystemStatsResponse(BaseResponse):
    """Response model for system statistics"""
    vector_db: Optional[Dict[str, Any]] = Field(None, description="Vector database statistics")
    llm_model: Optional[Dict[str, Any]] = Field(None, description="LLM model information")
    available_videos: Optional[int] = Field(None, description="Number of available videos")
    video_titles: Optional[List[str]] = Field(None, description="List of video titles")
    error: Optional[str] = Field(None, description="Error message if stats retrieval failed")


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    version: str = Field(..., description="API version")
