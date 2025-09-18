import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

# Add the backend directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson
from search_tools import CourseSearchTool
from vector_store import SearchResults, VectorStore
from rag_system import RAGSystem

# FastAPI testing imports
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def temp_chroma_path():
    """Create temporary directory for ChromaDB during tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_chroma_path):
    """Create test configuration"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_path
    config.ANTHROPIC_API_KEY = "test-key"
    config.MAX_RESULTS = 3
    return config


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for testing"""
    mock_store = Mock(spec=VectorStore)

    # Mock successful search results
    mock_store.search.return_value = SearchResults(
        documents=["Sample course content about Python programming"],
        metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
        distances=[0.1],
        error=None,
    )

    # Mock get_lesson_link method
    mock_store.get_lesson_link.return_value = "https://example.com/lesson/1"

    return mock_store


@pytest.fixture
def mock_vector_store_empty():
    """Create a mock VectorStore that returns empty results"""
    mock_store = Mock(spec=VectorStore)

    # Mock empty search results
    mock_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[], error=None
    )

    return mock_store


@pytest.fixture
def mock_vector_store_error():
    """Create a mock VectorStore that returns error"""
    mock_store = Mock(spec=VectorStore)

    # Mock error search results
    mock_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[], error="ChromaDB connection failed"
    )

    return mock_store


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Python Basics",
        instructor="John Doe",
        course_link="https://example.com/python-basics",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction to Python",
                lesson_link="https://example.com/lesson/1",
            ),
            Lesson(
                lesson_number=2,
                title="Variables and Data Types",
                lesson_link="https://example.com/lesson/2",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a high-level programming language",
            course_title="Python Basics",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Variables in Python can store different types of data",
            course_title="Python Basics",
            lesson_number=2,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()

    # Mock successful response without tool use
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response"
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tool_use():
    """Create a mock Anthropic client that triggers tool use"""
    mock_client = Mock()

    # Mock initial response with tool use
    mock_initial_response = Mock()
    mock_initial_response.stop_reason = "tool_use"

    # Mock tool use content
    mock_tool_use = Mock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "search_course_content"
    mock_tool_use.id = "tool_123"
    mock_tool_use.input = {"query": "test query"}

    mock_initial_response.content = [mock_tool_use]

    # Mock final response after tool use
    mock_final_response = Mock()
    mock_final_response.content = [Mock()]
    mock_final_response.content[0].text = "Response after tool use"
    mock_final_response.stop_reason = "end_turn"

    # Setup client to return different responses for different calls
    mock_client.messages.create.side_effect = [
        mock_initial_response,
        mock_final_response,
    ]

    return mock_client


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_search_tool_empty(mock_vector_store_empty):
    """Create CourseSearchTool with mock vector store that returns empty results"""
    return CourseSearchTool(mock_vector_store_empty)


@pytest.fixture
def course_search_tool_error(mock_vector_store_error):
    """Create CourseSearchTool with mock vector store that returns errors"""
    return CourseSearchTool(mock_vector_store_error)


@pytest.fixture
def mock_anthropic_client_multi_round():
    """Create a mock Anthropic client that simulates 2-round tool execution"""
    mock_client = Mock()

    # Mock first response with tool use
    mock_first_response = Mock()
    mock_first_response.stop_reason = "tool_use"

    mock_first_tool = Mock()
    mock_first_tool.type = "tool_use"
    mock_first_tool.name = "get_course_outline"
    mock_first_tool.id = "tool_1"
    mock_first_tool.input = {"course_title": "Python Basics"}
    mock_first_response.content = [mock_first_tool]

    # Mock second response with another tool use
    mock_second_response = Mock()
    mock_second_response.stop_reason = "tool_use"

    mock_second_tool = Mock()
    mock_second_tool.type = "tool_use"
    mock_second_tool.name = "search_course_content"
    mock_second_tool.id = "tool_2"
    mock_second_tool.input = {"query": "variables and data types"}
    mock_second_response.content = [mock_second_tool]

    # Mock final response without tool use
    mock_final_response = Mock()
    mock_final_response.content = [Mock()]
    mock_final_response.content[0].text = (
        "Based on the course outline and search, here is the answer"
    )
    mock_final_response.stop_reason = "end_turn"

    # Setup client to return different responses for different calls
    mock_client.messages.create.side_effect = [
        mock_first_response,
        mock_second_response,
        mock_final_response,
    ]

    return mock_client


@pytest.fixture
def mock_anthropic_client_single_round_stop():
    """Create a mock Anthropic client that stops after one round (no second tool call)"""
    mock_client = Mock()

    # Mock first response with tool use
    mock_first_response = Mock()
    mock_first_response.stop_reason = "tool_use"

    mock_first_tool = Mock()
    mock_first_tool.type = "tool_use"
    mock_first_tool.name = "search_course_content"
    mock_first_tool.id = "tool_1"
    mock_first_tool.input = {"query": "Python basics"}
    mock_first_response.content = [mock_first_tool]

    # Mock second response without tool use (stops after first round)
    mock_second_response = Mock()
    mock_second_response.content = [Mock()]
    mock_second_response.content[0].text = (
        "Here is the complete answer from the first search"
    )
    mock_second_response.stop_reason = "end_turn"

    mock_client.messages.create.side_effect = [
        mock_first_response,
        mock_second_response,
    ]

    return mock_client


@pytest.fixture
def mock_tool_manager_with_sources():
    """Create a mock tool manager that tracks sources across rounds"""
    mock_manager = Mock()

    # Track sources from multiple rounds

    def mock_execute_tool(tool_name, **kwargs):
        if tool_name == "get_course_outline":
            return "Course: Python Basics\nLessons: 1. Introduction, 2. Variables"
        elif tool_name == "search_course_content":
            return "Found content about Python variables and data types"
        return "Mock tool result"

    def mock_get_last_sources():
        if mock_manager.execute_tool.call_count == 1:
            return [
                {"text": "Python Basics Course", "link": "https://example.com/course"}
            ]
        elif mock_manager.execute_tool.call_count == 2:
            return [
                {
                    "text": "Python Basics - Lesson 2",
                    "link": "https://example.com/lesson/2",
                }
            ]
        return []

    def mock_reset_sources():
        pass  # No-op for mock

    mock_manager.execute_tool.side_effect = mock_execute_tool
    mock_manager.get_last_sources.side_effect = mock_get_last_sources
    mock_manager.reset_sources.side_effect = mock_reset_sources
    mock_manager.get_tool_definitions.return_value = [
        {"name": "search_course_content", "description": "Search course content"},
        {"name": "get_course_outline", "description": "Get course outline"},
    ]

    return mock_manager


# API Testing Fixtures

@pytest.fixture
def test_frontend_dir():
    """Create a temporary frontend directory with test files"""
    temp_dir = tempfile.mkdtemp()

    # Create test HTML file
    html_content = '''<!DOCTYPE html>
<html>
<head><title>Test App</title></head>
<body><h1>Test RAG System</h1></body>
</html>'''

    with open(os.path.join(temp_dir, 'index.html'), 'w') as f:
        f.write(html_content)

    # Create test CSS file
    css_content = 'body { font-family: Arial, sans-serif; }'
    with open(os.path.join(temp_dir, 'styles.css'), 'w') as f:
        f.write(css_content)

    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_system = Mock(spec=RAGSystem)

    # Mock successful query response
    mock_system.query.return_value = (
        "This is a test response about Python programming",
        [{"text": "Python Basics - Lesson 1", "link": "https://example.com/lesson/1"}]
    )

    # Mock course analytics
    mock_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "JavaScript Fundamentals"]
    }

    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test-session-123"
    mock_session_manager.clear_session.return_value = None
    mock_system.session_manager = mock_session_manager

    return mock_system


@pytest.fixture
def test_app(mock_rag_system, test_frontend_dir):
    """Create a test FastAPI app with mocked dependencies"""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app
    app = FastAPI(title="Test RAG System")

    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define models inline to avoid import issues
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # API endpoints with mock system
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        from fastapi import HTTPException
        try:
            # Use provided session_id or create new one
            if request.session_id:
                session_id = request.session_id
            else:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        from fastapi import HTTPException
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}")
    async def clear_session(session_id: str):
        from fastapi import HTTPException
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"message": f"Session {session_id} cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Mount static files
    app.mount("/", StaticFiles(directory=test_frontend_dir, html=True), name="static")

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data for testing"""
    return {
        "query": "What is Python programming?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_request_no_session():
    """Sample query request without session ID"""
    return {
        "query": "What are variables in Python?"
    }
