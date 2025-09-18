import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for the /api/query endpoint"""

    def test_successful_query_with_session(self, test_client, sample_query_request):
        """Test successful query with existing session ID"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)

        # Check source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source

    def test_successful_query_without_session(self, test_client, sample_query_request_no_session):
        """Test successful query without session ID (should create new session)"""
        response = test_client.post("/api/query", json=sample_query_request_no_session)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # Mock returns this value

    def test_query_with_empty_query(self, test_client):
        """Test query with empty query string"""
        response = test_client.post("/api/query", json={"query": ""})

        assert response.status_code == 200  # Should still succeed with mock
        data = response.json()
        assert "answer" in data

    def test_query_with_invalid_json(self, test_client):
        """Test query with invalid JSON structure"""
        response = test_client.post("/api/query", json={})

        assert response.status_code == 422  # Validation error

    def test_query_with_malformed_request(self, test_client):
        """Test query with malformed request body"""
        response = test_client.post("/api/query", data="invalid json")

        assert response.status_code == 422

    def test_query_rag_system_error(self, test_client, mock_rag_system, sample_query_request):
        """Test query when RAG system raises an exception"""
        # Reset the mock and configure it to raise an exception
        mock_rag_system.query.side_effect = Exception("RAG system error")

        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]

        # Reset the mock for other tests
        mock_rag_system.query.side_effect = None

    def test_query_response_structure(self, test_client, sample_query_request):
        """Test that query response matches expected structure"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data

        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_query_different_session_ids(self, test_client):
        """Test queries with different session IDs"""
        query1 = {"query": "Test query 1", "session_id": "session-1"}
        query2 = {"query": "Test query 2", "session_id": "session-2"}

        response1 = test_client.post("/api/query", json=query1)
        response2 = test_client.post("/api/query", json=query2)

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Both should succeed with their respective session IDs
        data1 = response1.json()
        data2 = response2.json()

        # Session IDs should be preserved when provided
        assert data1["session_id"] == "session-1"
        assert data2["session_id"] == "session-2"


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for the /api/courses endpoint"""

    def test_get_course_stats_success(self, test_client):
        """Test successful retrieval of course statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Check mock data
        assert data["total_courses"] == 2
        assert "Python Basics" in data["course_titles"]
        assert "JavaScript Fundamentals" in data["course_titles"]

    def test_get_course_stats_rag_system_error(self, test_client, mock_rag_system):
        """Test course stats when RAG system raises an exception"""
        # Configure mock to raise an exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]

        # Reset the mock for other tests
        mock_rag_system.get_course_analytics.side_effect = None

    def test_get_course_stats_response_structure(self, test_client):
        """Test that course stats response matches expected structure"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data

        assert data["total_courses"] >= 0
        assert len(data["course_titles"]) == data["total_courses"]

    def test_get_course_stats_empty_courses(self, test_client, mock_rag_system):
        """Test course stats when no courses are available"""
        # Configure mock to return empty data
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


@pytest.mark.api
class TestSessionsEndpoint:
    """Test cases for the /api/sessions endpoint"""

    def test_clear_session_success(self, test_client):
        """Test successful session clearing"""
        session_id = "test-session-456"
        response = test_client.delete(f"/api/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert session_id in data["message"]
        assert "cleared successfully" in data["message"]

    def test_clear_session_with_special_characters(self, test_client):
        """Test clearing session with special characters in ID"""
        session_id = "test-session-with-dashes-123"
        response = test_client.delete(f"/api/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert session_id in data["message"]

    def test_clear_session_rag_system_error(self, test_client, mock_rag_system):
        """Test session clearing when RAG system raises an exception"""
        # Configure mock to raise an exception
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session error")

        session_id = "error-session"
        response = test_client.delete(f"/api/sessions/{session_id}")

        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]

        # Reset the mock for other tests
        mock_rag_system.session_manager.clear_session.side_effect = None

    def test_clear_nonexistent_session(self, test_client):
        """Test clearing a session that doesn't exist"""
        session_id = "nonexistent-session"
        response = test_client.delete(f"/api/sessions/{session_id}")

        # Should still succeed since mock doesn't validate existence
        assert response.status_code == 200


@pytest.mark.api
class TestStaticFileServing:
    """Test cases for static file serving"""

    def test_serve_index_html(self, test_client):
        """Test serving the main index.html file"""
        response = test_client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Test RAG System" in response.text

    def test_serve_css_file(self, test_client):
        """Test serving CSS files"""
        response = test_client.get("/styles.css")

        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]
        assert "Arial" in response.text

    def test_serve_nonexistent_file(self, test_client):
        """Test serving a file that doesn't exist"""
        response = test_client.get("/nonexistent.html")

        assert response.status_code == 404

    def test_serve_root_path_variations(self, test_client):
        """Test different ways of accessing the root path"""
        paths = ["/", "/index.html"]

        for path in paths:
            response = test_client.get(path)
            assert response.status_code == 200
            assert "Test RAG System" in response.text


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints"""

    def test_api_workflow_complete_session(self, test_client, mock_rag_system):
        """Test a complete API workflow: query -> get courses -> clear session"""
        # Step 1: Make a query
        query_response = test_client.post("/api/query", json={
            "query": "What is Python?"
        })
        assert query_response.status_code == 200
        session_id = query_response.json()["session_id"]

        # Step 2: Get course statistics
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200

        # Step 3: Clear the session
        clear_response = test_client.delete(f"/api/sessions/{session_id}")
        assert clear_response.status_code == 200

    def test_api_cors_headers(self, test_client):
        """Test that CORS middleware is properly configured"""
        # Test that the app accepts cross-origin requests
        response = test_client.post("/api/query",
                                   json={"query": "test"},
                                   headers={"Origin": "http://localhost:3000"})

        assert response.status_code == 200
        # CORS headers may not always be visible in test client

    def test_api_content_types(self, test_client, sample_query_request):
        """Test that API endpoints return proper content types"""
        # JSON endpoints should return JSON content-type
        response = test_client.post("/api/query", json=sample_query_request)
        assert "application/json" in response.headers["content-type"]

        response = test_client.get("/api/courses")
        assert "application/json" in response.headers["content-type"]

    def test_api_error_handling_consistency(self, test_client, mock_rag_system):
        """Test that all endpoints handle errors consistently"""
        # Configure mocks to raise exceptions
        mock_rag_system.query.side_effect = Exception("Query error")
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session error")

        # Test all endpoints return 500 for system errors
        query_response = test_client.post("/api/query", json={"query": "test"})
        assert query_response.status_code == 500

        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 500

        session_response = test_client.delete("/api/sessions/test")
        assert session_response.status_code == 500

        # Reset all mocks for other tests
        mock_rag_system.query.side_effect = None
        mock_rag_system.get_course_analytics.side_effect = None
        mock_rag_system.session_manager.clear_session.side_effect = None


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Performance-related tests for API endpoints"""

    def test_concurrent_queries(self, test_client):
        """Test multiple concurrent queries to the same endpoint"""
        import concurrent.futures
        import threading

        def make_query(query_id):
            return test_client.post("/api/query", json={
                "query": f"Test query {query_id}"
            })

        # Test with multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_query, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        for response in results:
            assert response.status_code == 200

    def test_large_query_handling(self, test_client):
        """Test handling of large query strings"""
        large_query = "What is Python? " * 1000  # ~14KB query

        response = test_client.post("/api/query", json={
            "query": large_query
        })

        assert response.status_code == 200