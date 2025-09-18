from unittest.mock import Mock, patch

from config import Config
from rag_system import RAGSystem


class TestErrorHandling:
    """Test improved error handling in RAG system"""

    def test_missing_api_key_error(self):
        """Test error handling when API key is missing"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            config = Config()
            config.ANTHROPIC_API_KEY = ""  # Empty API key

            rag_system = RAGSystem(config)
            response, sources = rag_system.query("Test query")

            assert "Error: Anthropic API key not configured" in response
            assert "Please set ANTHROPIC_API_KEY in your .env file" in response
            assert sources == []

    def test_authentication_error_handling(self):
        """Test error handling for authentication errors"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.SessionManager"),
        ):

            # Mock AI generator to raise authentication error
            mock_ai.return_value.generate_response.side_effect = Exception(
                "authentication_error: invalid x-api-key"
            )

            config = Config()
            config.ANTHROPIC_API_KEY = "invalid-key"

            rag_system = RAGSystem(config)
            response, sources = rag_system.query("Test query")

            assert "Error: Invalid Anthropic API key" in response
            assert "Please check your ANTHROPIC_API_KEY" in response
            assert sources == []

    def test_rate_limit_error_handling(self):
        """Test error handling for rate limit errors"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.SessionManager"),
        ):

            # Mock AI generator to raise rate limit error
            mock_ai.return_value.generate_response.side_effect = Exception(
                "rate_limit exceeded"
            )

            config = Config()
            config.ANTHROPIC_API_KEY = "valid-key"

            rag_system = RAGSystem(config)
            response, sources = rag_system.query("Test query")

            assert "Error: API rate limit exceeded" in response
            assert "Please try again later" in response
            assert sources == []

    def test_network_error_handling(self):
        """Test error handling for network errors"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.SessionManager"),
        ):

            # Mock AI generator to raise network error
            mock_ai.return_value.generate_response.side_effect = Exception(
                "network connection failed"
            )

            config = Config()
            config.ANTHROPIC_API_KEY = "valid-key"

            rag_system = RAGSystem(config)
            response, sources = rag_system.query("Test query")

            assert "Error: Network connection issue" in response
            assert "Please check your internet connection" in response
            assert sources == []

    def test_generic_error_handling(self):
        """Test error handling for other generic errors"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.SessionManager"),
        ):

            # Mock AI generator to raise generic error
            mock_ai.return_value.generate_response.side_effect = Exception(
                "Unknown error occurred"
            )

            config = Config()
            config.ANTHROPIC_API_KEY = "valid-key"

            rag_system = RAGSystem(config)
            response, sources = rag_system.query("Test query")

            assert "Error: Query processing failed" in response
            assert "Unknown error occurred" in response
            assert sources == []

    def test_successful_query_after_fix(self):
        """Test that queries work successfully when properly configured"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.SessionManager") as mock_sm,
        ):

            # Mock successful AI response
            mock_ai.return_value.generate_response.return_value = "Successful response"
            mock_sm.return_value.get_conversation_history.return_value = None

            config = Config()
            config.ANTHROPIC_API_KEY = "valid-api-key"

            rag_system = RAGSystem(config)

            # Mock tool manager for sources
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.reset_sources = Mock()

            response, sources = rag_system.query("Test query")

            assert response == "Successful response"
            assert sources == []
            # Verify AI generator was called
            mock_ai.return_value.generate_response.assert_called_once()
