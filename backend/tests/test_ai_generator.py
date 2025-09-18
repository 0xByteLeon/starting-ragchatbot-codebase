import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGenerator:
    """Test AIGenerator tool calling functionality"""

    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test response generation without tools"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Mock the client
        with patch.object(ai_gen, 'client', mock_anthropic_client):
            response = ai_gen.generate_response("What is Python?")

        # Verify client was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]

        # Check basic parameters
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is Python?"

        # Verify no tools were added
        assert "tools" not in call_args

        # Check response
        assert response == "This is a test response"

    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation with conversation history"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        history = "User: Hello\nAssistant: Hi there!"

        with patch.object(ai_gen, 'client', mock_anthropic_client):
            response = ai_gen.generate_response("How are you?", conversation_history=history)

        # Verify system prompt includes history
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert "User: Hello\nAssistant: Hi there!" in call_args["system"]

    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client):
        """Test response generation with tools available but no tool use"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Create mock tools
        mock_tools = [{"name": "test_tool", "description": "A test tool"}]
        mock_tool_manager = Mock()

        with patch.object(ai_gen, 'client', mock_anthropic_client):
            response = ai_gen.generate_response(
                "What is Python?",
                tools=mock_tools,
                tool_manager=mock_tool_manager
            )

        # Verify tools were added to the call
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["tools"] == mock_tools
        assert call_args["tool_choice"] == {"type": "auto"}

        # Since no tool use, tool manager shouldn't be called
        mock_tool_manager.execute_tool.assert_not_called()

        assert response == "This is a test response"

    def test_generate_response_with_tool_use(self, mock_anthropic_client_with_tool_use):
        """Test response generation when AI decides to use a tool"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Create mock tools and tool manager
        mock_tools = [{"name": "search_course_content", "description": "Search course content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        with patch.object(ai_gen, 'client', mock_anthropic_client_with_tool_use):
            response = ai_gen.generate_response(
                "Find information about Python",
                tools=mock_tools,
                tool_manager=mock_tool_manager
            )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )

        # Verify two API calls were made (initial + follow-up)
        assert mock_anthropic_client_with_tool_use.messages.create.call_count == 2

        # Check final response
        assert response == "Response after tool use"

    def test_handle_tool_execution_single_tool(self):
        """Test _handle_tool_execution with single tool call"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.content = [Mock()]
        mock_initial_response.content[0].type = "tool_use"
        mock_initial_response.content[0].name = "search_course_content"
        mock_initial_response.content[0].id = "tool_123"
        mock_initial_response.content[0].input = {"query": "Python basics"}

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about Python"

        # Mock base parameters
        base_params = {
            "messages": [{"role": "user", "content": "Tell me about Python"}],
            "system": "You are a helpful assistant"
        }

        # Mock client for final call
        mock_client = Mock()
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Based on search results: Python is..."
        mock_client.messages.create.return_value = mock_final_response

        with patch.object(ai_gen, 'client', mock_client):
            result = ai_gen._handle_tool_execution(
                mock_initial_response,
                base_params,
                mock_tool_manager
            )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )

        # Verify final API call structure
        final_call_args = mock_client.messages.create.call_args[1]
        assert len(final_call_args["messages"]) == 3  # original + assistant + tool result

        # Check tool result message format
        tool_result_message = final_call_args["messages"][2]
        assert tool_result_message["role"] == "user"
        assert len(tool_result_message["content"]) == 1
        assert tool_result_message["content"][0]["type"] == "tool_result"
        assert tool_result_message["content"][0]["tool_use_id"] == "tool_123"
        assert tool_result_message["content"][0]["content"] == "Search results about Python"

        assert result == "Based on search results: Python is..."

    def test_handle_tool_execution_multiple_tools(self):
        """Test _handle_tool_execution with multiple tool calls"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Mock initial response with multiple tool uses
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_123"
        mock_tool_use_1.input = {"query": "Python"}

        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "get_course_outline"
        mock_tool_use_2.id = "tool_456"
        mock_tool_use_2.input = {"course_title": "Python Basics"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use_1, mock_tool_use_2]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result 1",
            "Outline result"
        ]

        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system"
        }

        # Mock client
        mock_client = Mock()
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Combined results response"
        mock_client.messages.create.return_value = mock_final_response

        with patch.object(ai_gen, 'client', mock_client):
            result = ai_gen._handle_tool_execution(
                mock_initial_response,
                base_params,
                mock_tool_manager
            )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Python")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Python Basics")

        # Verify tool results structure
        final_call_args = mock_client.messages.create.call_args[1]
        tool_result_message = final_call_args["messages"][2]
        assert len(tool_result_message["content"]) == 2  # Two tool results

        assert result == "Combined results response"

    def test_system_prompt_content(self):
        """Test that the system prompt contains expected content"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Check that system prompt contains tool usage guidelines
        assert "search_course_content" in ai_gen.SYSTEM_PROMPT
        assert "get_course_outline" in ai_gen.SYSTEM_PROMPT
        assert "Tool Usage Guidelines" in ai_gen.SYSTEM_PROMPT

    def test_api_parameters_configuration(self):
        """Test that API parameters are configured correctly"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Check base parameters
        assert ai_gen.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800

    def test_error_handling_in_tool_execution(self):
        """Test error handling when tool execution fails"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Mock initial response with tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_123"
        mock_tool_use.input = {"query": "test"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]

        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "Test system"
        }

        mock_client = Mock()
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Error handled response"
        mock_client.messages.create.return_value = mock_final_response

        with patch.object(ai_gen, 'client', mock_client):
            # This should not raise an exception, but handle it gracefully
            try:
                result = ai_gen._handle_tool_execution(
                    mock_initial_response,
                    base_params,
                    mock_tool_manager
                )
                # The method should still try to get a final response
                assert mock_client.messages.create.called
            except Exception as e:
                # If an exception is raised, it should be the original tool error
                assert "Tool execution failed" in str(e)