"""
Main application entry point for YouTube Transcription Chatbot
"""
import uvicorn
from api import app

if __name__ == "__main__":
    print("🚀 Starting YouTube Transcription Chatbot API Server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
