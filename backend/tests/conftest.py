import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
import os
import sys

# Add the backend directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


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
        error=None
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
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )

    return mock_store


@pytest.fixture
def mock_vector_store_error():
    """Create a mock VectorStore that returns error"""
    mock_store = Mock(spec=VectorStore)

    # Mock error search results
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="ChromaDB connection failed"
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
            Lesson(lesson_number=1, title="Introduction to Python", lesson_link="https://example.com/lesson/1"),
            Lesson(lesson_number=2, title="Variables and Data Types", lesson_link="https://example.com/lesson/2")
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a high-level programming language",
            course_title="Python Basics",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables in Python can store different types of data",
            course_title="Python Basics",
            lesson_number=2,
            chunk_index=1
        )
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
    mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]

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