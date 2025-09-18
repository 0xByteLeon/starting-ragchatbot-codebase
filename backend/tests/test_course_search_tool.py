import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool execute method and functionality"""

    def test_execute_successful_search(self, course_search_tool):
        """Test execute method with successful search results"""
        result = course_search_tool.execute("Python programming")

        # Verify the vector store search was called
        course_search_tool.store.search.assert_called_once_with(
            query="Python programming",
            course_name=None,
            lesson_number=None
        )

        # Verify the result contains formatted content
        assert "[Python Basics - Lesson 1]" in result
        assert "Sample course content about Python programming" in result

        # Verify sources were tracked
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0]["text"] == "Python Basics - Lesson 1"
        assert course_search_tool.last_sources[0]["link"] == "https://example.com/lesson/1"

    def test_execute_with_course_name_filter(self, course_search_tool):
        """Test execute method with course name filter"""
        result = course_search_tool.execute("variables", course_name="Python Basics")

        # Verify the search was called with course name
        course_search_tool.store.search.assert_called_once_with(
            query="variables",
            course_name="Python Basics",
            lesson_number=None
        )

        # Verify result is formatted correctly
        assert "[Python Basics - Lesson 1]" in result

    def test_execute_with_lesson_number_filter(self, course_search_tool):
        """Test execute method with lesson number filter"""
        result = course_search_tool.execute("functions", lesson_number=2)

        # Verify the search was called with lesson number
        course_search_tool.store.search.assert_called_once_with(
            query="functions",
            course_name=None,
            lesson_number=2
        )

        # Verify result is formatted correctly
        assert "[Python Basics - Lesson 1]" in result

    def test_execute_with_both_filters(self, course_search_tool):
        """Test execute method with both course name and lesson number filters"""
        result = course_search_tool.execute(
            "loops",
            course_name="Python Basics",
            lesson_number=3
        )

        # Verify the search was called with both filters
        course_search_tool.store.search.assert_called_once_with(
            query="loops",
            course_name="Python Basics",
            lesson_number=3
        )

    def test_execute_empty_results(self, course_search_tool_empty):
        """Test execute method when no results are found"""
        result = course_search_tool_empty.execute("nonexistent topic")

        # Should return no results message
        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, course_search_tool_empty):
        """Test execute method when no results found with filters"""
        result = course_search_tool_empty.execute(
            "test",
            course_name="Test Course",
            lesson_number=1
        )

        # Should include filter information in the message
        assert "No relevant content found in course 'Test Course' in lesson 1" in result

    def test_execute_error_handling(self, course_search_tool_error):
        """Test execute method when vector store returns an error"""
        result = course_search_tool_error.execute("test query")

        # Should return the error message
        assert "ChromaDB connection failed" in result

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly formatted"""
        definition = course_search_tool.get_tool_definition()

        # Verify structure
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Verify schema properties
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

        # Verify required fields
        assert "query" in definition["input_schema"]["required"]

    def test_format_results_with_lesson_link(self, mock_vector_store):
        """Test result formatting with lesson links"""
        # Setup mock to return lesson link
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/5"

        # Setup search results with lesson number
        mock_results = SearchResults(
            documents=["Advanced Python concepts"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 5}],
            distances=[0.2]
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("advanced concepts")

        # Verify lesson link was requested
        mock_vector_store.get_lesson_link.assert_called_once_with("Advanced Python", 5)

        # Verify sources include link
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["link"] == "https://example.com/lesson/5"

    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test result formatting when no lesson number is present"""
        # Setup search results without lesson number
        mock_results = SearchResults(
            documents=["Course overview content"],
            metadata=[{"course_title": "Overview Course", "lesson_number": None}],
            distances=[0.3]
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("overview")

        # Verify header format without lesson number
        assert "[Overview Course]" in result
        assert "Course overview content" in result

        # Verify sources don't include lesson reference
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Overview Course"
        assert tool.last_sources[0]["link"] is None

    def test_multiple_results_formatting(self, mock_vector_store):
        """Test formatting when multiple results are returned"""
        # Setup multiple search results
        mock_results = SearchResults(
            documents=["First result content", "Second result content"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/course-a/lesson/1",
            "https://example.com/course-b/lesson/2"
        ]

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Verify both results are formatted
        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "First result content" in result
        assert "Second result content" in result

        # Verify multiple sources are tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[1]["text"] == "Course B - Lesson 2"

    def test_sources_reset_between_searches(self, course_search_tool):
        """Test that sources are reset between different searches"""
        # First search
        course_search_tool.execute("first query")
        first_sources = course_search_tool.last_sources.copy()

        # Second search with different mock data
        mock_results = SearchResults(
            documents=["Different content"],
            metadata=[{"course_title": "Different Course", "lesson_number": 3}],
            distances=[0.4]
        )
        course_search_tool.store.search.return_value = mock_results
        course_search_tool.store.get_lesson_link.return_value = "https://example.com/different/3"

        course_search_tool.execute("second query")

        # Verify sources changed
        assert course_search_tool.last_sources != first_sources
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0]["text"] == "Different Course - Lesson 3"

    def test_edge_case_empty_metadata(self, mock_vector_store):
        """Test handling of edge case where metadata might be missing fields"""
        # Setup search results with incomplete metadata
        mock_results = SearchResults(
            documents=["Content with incomplete metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.5]
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("edge case")

        # Should handle gracefully with default values
        assert "[unknown]" in result
        assert "Content with incomplete metadata" in result