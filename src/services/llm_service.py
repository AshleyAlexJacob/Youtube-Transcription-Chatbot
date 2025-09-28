import os
from typing import List, Dict, Optional, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv(".env")


class LLMService:
    def __init__(self, 
                 model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 google_api_key: Optional[str] = None,
                 max_history_length: int = 10):
        """
        Initialize LLM Service with Google Gemini
        
        Args:
            model_name: Google Gemini model name
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum tokens in response
            google_api_key: Google API key (if None, reads from env)
            max_history_length: Maximum conversation history to maintain
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history_length = max_history_length
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize Google Gemini client
        self._initialize_llm_client(google_api_key)
        
        # Initialize prompt templates
        self._initialize_prompt_templates()
        
        # Initialize output parser
        self.output_parser = StrOutputParser()
    
    def _initialize_llm_client(self, google_api_key: Optional[str]):
        """Initialize Google Gemini client"""
        try:
            # Get API key from environment if not provided
            if not google_api_key:
                google_api_key = os.getenv("GOOGLE_API_KEY")
            
            if not google_api_key:
                raise ValueError(
                    "Google API key not found. Please provide GOOGLE_API_KEY "
                    "either as parameter or environment variable."
                )
            
            # Initialize the LLM
            llm_config = {
                "model": self.model_name,
                "temperature": self.temperature,
                "google_api_key": google_api_key
            }
            
            if self.max_tokens:
                llm_config["max_output_tokens"] = self.max_tokens
            
            self.llm = ChatGoogleGenerativeAI(**llm_config)
            
            # Test connection with a simple query
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            print("✓ Successfully connected to Google Gemini")
            
        except Exception as e:
            print(f"❌ Failed to connect to Google Gemini: {str(e)}")
            print("\nTo fix this issue:")
            print("1. Make sure you have a Google Cloud account with Gemini API access")
            print("2. Set GOOGLE_API_KEY environment variable")
            print("3. Or pass the API key as parameter to the constructor")
            print("4. Ensure the API key has proper permissions")
            raise RuntimeError("Google Gemini connection failed. Please check your API key.")
    
    def _initialize_prompt_templates(self):
        """Initialize prompt templates for different use cases"""
        
        # RAG-based Q&A template
        self.rag_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an intelligent assistant that helps users understand YouTube video content. 
You have access to transcribed content from YouTube videos and can answer questions based on that content.

Instructions:
- Use the provided context from video transcriptions to answer questions
- If the context doesn't contain relevant information, say so honestly
- Provide specific, detailed answers when possible
- Include video titles when referencing specific content
- Be conversational and helpful
- If asked about multiple videos, clearly distinguish between them"""),
            
            MessagesPlaceholder(variable_name="chat_history"),
            
            HumanMessage(content="""Context from YouTube videos:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above.""")
        ])
        
        # General conversation template
        self.general_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant specializing in YouTube content analysis and general conversation. 
You can help users with questions about video content, provide summaries, and engage in natural conversation.

Be friendly, informative, and concise in your responses."""),
            
            MessagesPlaceholder(variable_name="chat_history"),
            
            HumanMessage(content="{question}")
        ])
        
        # Summarization template
        self.summary_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert at creating concise, informative summaries of YouTube video content.
Create summaries that capture the main points, key insights, and important details.

Guidelines:
- Focus on the most important information
- Use clear, structured formatting
- Include key topics and main arguments
- Keep it comprehensive but concise"""),
            
            HumanMessage(content="""Please create a summary of this YouTube video content:

Title: {title}
Content: {content}

Provide a well-structured summary with key points and main insights.""")
        ])
    
    def add_to_history(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to conversation history"""
        message = {
            "role": role,  # "human", "ai", or "system"
            "content": content,
            "timestamp": None,  # Could add timestamp if needed
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Trim history if it exceeds max length
        if len(self.conversation_history) > self.max_history_length * 2:  # *2 for human+ai pairs
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def _format_history_for_langchain(self) -> List:
        """Convert conversation history to LangChain message format"""
        messages = []
        for msg in self.conversation_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
        return messages
    
    def send_message(self, 
                    message: str, 
                    context: Optional[str] = None,
                    use_rag: bool = True) -> str:
        """
        Send a message and get response
        
        Args:
            message: User message/question
            context: Retrieved context from vector database (for RAG)
            use_rag: Whether to use RAG template or general conversation
            
        Returns:
            AI response as string
        """
        try:
            # Add user message to history
            self.add_to_history("human", message)
            
            # Choose appropriate template and create chain
            if use_rag and context:
                template = self.rag_template
                chain = template | self.llm | self.output_parser
                
                response = chain.invoke({
                    "question": message,
                    "context": context,
                    "chat_history": self._format_history_for_langchain()
                })
            else:
                template = self.general_template
                chain = template | self.llm | self.output_parser
                
                response = chain.invoke({
                    "question": message,
                    "chat_history": self._format_history_for_langchain()
                })
            
            # Add AI response to history
            self.add_to_history("ai", response)
            
            print(f"✓ Generated response for: '{message[:50]}...'")
            return response
            
        except Exception as e:
            error_msg = f"❌ Failed to generate response: {str(e)}"
            print(error_msg)
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def generate_summary(self, content: str, title: str = "YouTube Video") -> str:
        """
        Generate a summary of video content
        
        Args:
            content: Video transcription content
            title: Video title
            
        Returns:
            Summary as string
        """
        try:
            chain = self.summary_template | self.llm | self.output_parser
            
            summary = chain.invoke({
                "content": content,
                "title": title
            })
            
            print(f"✓ Generated summary for: {title}")
            return summary
            
        except Exception as e:
            error_msg = f"❌ Failed to generate summary: {str(e)}"
            print(error_msg)
            return f"I apologize, but I encountered an error while generating the summary: {str(e)}"
    
    def ask_with_context(self, 
                        question: str, 
                        retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Ask a question with retrieved context chunks
        
        Args:
            question: User question
            retrieved_chunks: List of retrieved chunks from vector database
            
        Returns:
            Tuple of (response, list of source video titles)
        """
        if not retrieved_chunks:
            return self.send_message(question, use_rag=False), []
        
        # Format context from retrieved chunks
        context_parts = []
        source_titles = set()
        
        for i, chunk in enumerate(retrieved_chunks):
            video_title = chunk.get("video_title", "Unknown Video")
            text = chunk.get("text", "")
            score = chunk.get("score", 0)
            
            source_titles.add(video_title)
            context_parts.append(f"""
[Source {i+1}: {video_title} (Relevance: {score:.3f})]
{text}
""")
        
        context = "\n".join(context_parts)
        response = self.send_message(question, context=context, use_rag=True)
        
        return response, list(source_titles)
    
    def create_rag_chain(self, vector_service):
        """
        Create a LangChain RAG chain using the vector service
        
        Args:
            vector_service: VectorDBService instance
            
        Returns:
            LangChain chain for RAG operations
        """
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        def format_docs(docs):
            """Format retrieved documents for context"""
            context_parts = []
            for i, doc in enumerate(docs):
                video_title = doc.metadata.get("video_title", "Unknown Video")
                score = doc.metadata.get("score", 0)
                context_parts.append(f"""
[Source {i+1}: {video_title} (Relevance: {score:.3f})]
{doc.page_content}
""")
            return "\n".join(context_parts)
        
        def retrieve_docs(question: str):
            """Retrieve documents using vector service"""
            results = vector_service.search_similar(
                query=question,
                limit=5,
                score_threshold=0.3
            )
            
            # Convert to Document-like objects
            class SimpleDoc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata
            
            docs = [
                SimpleDoc(result["text"], {
                    "video_title": result["video_title"],
                    "score": result["score"],
                    "video_url": result["video_url"]
                })
                for result in results
            ]
            return docs
        
        # Create the RAG chain
        rag_chain = (
            {"context": lambda x: format_docs(retrieve_docs(x["question"])), "question": RunnablePassthrough()}
            | self.rag_template
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_history_length": self.max_history_length,
            "history_count": len(self.conversation_history)
        }
    
    def update_system_prompt(self, new_system_prompt: str):
        """Update the system prompt for general conversations"""
        self.general_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=new_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{question}")
        ])
        print("✓ System prompt updated")


# # Example usage for testing
# if __name__ == "__main__":
#     # Initialize service
#     llm_service = LLMService()
# #     
#     # Test general conversation
#     response = llm_service.send_message("Hello! How are you?", use_rag=False)
#     print(f"Response: {response}")
#     
#     # Test with context (simulating RAG)
#     sample_context = """
#     Video: "Introduction to Data Science"
#     Content: Data science is a field that combines statistics, programming, and domain expertise...
#     """
#     
#     response = llm_service.send_message(
#         "What is data science?", 
#         context=sample_context, 
#         use_rag=True
#     )
#     print(f"RAG Response: {response}")
#     
#     # Test summary generation
#     summary = llm_service.generate_summary(
#         content="Long video content about machine learning...",
#         title="Machine Learning Basics"
#     )
#     print(f"Summary: {summary}")
#     
#     # Check conversation history
#     history = llm_service.get_conversation_history()
#     print(f"History length: {len(history)}")
