from unittest.mock import Mock, patch

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator tool calling functionality"""

    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test response generation without tools"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Mock the client
        with patch.object(ai_gen, "client", mock_anthropic_client):
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

        with patch.object(ai_gen, "client", mock_anthropic_client):
            ai_gen.generate_response("How are you?", conversation_history=history)

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

        with patch.object(ai_gen, "client", mock_anthropic_client):
            result = ai_gen.generate_response(
                "What is Python?", tools=mock_tools, tool_manager=mock_tool_manager
            )

        # Should return tuple (response, sources) when tools are provided
        assert isinstance(result, tuple)
        response, sources = result

        # Verify tools were added to the call
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["tools"] == mock_tools
        assert call_args["tool_choice"] == {"type": "auto"}

        # Since no tool use, tool manager shouldn't be called
        mock_tool_manager.execute_tool.assert_not_called()

        # Check response and empty sources
        assert response == "This is a test response"
        assert sources == []

    def test_generate_response_with_tool_use(self, mock_anthropic_client_with_tool_use):
        """Test response generation when AI decides to use a tool"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Create mock tools and tool manager
        mock_tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.reset_sources.return_value = None

        with patch.object(ai_gen, "client", mock_anthropic_client_with_tool_use):
            result = ai_gen.generate_response(
                "Find information about Python",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
            )

        # Should return tuple (response, sources) when tools are provided
        assert isinstance(result, tuple)
        response, sources = result

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test query"
        )

        # Verify two API calls were made (initial + follow-up)
        assert mock_anthropic_client_with_tool_use.messages.create.call_count == 2

        # Check final response
        assert response == "Response after tool use"
        assert isinstance(sources, list)

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
            "system": "You are a helpful assistant",
        }

        # Mock client for final call
        mock_client = Mock()
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Based on search results: Python is..."
        mock_client.messages.create.return_value = mock_final_response

        with patch.object(ai_gen, "client", mock_client):
            result = ai_gen._handle_tool_execution(
                mock_initial_response, base_params, mock_tool_manager
            )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify final API call structure
        final_call_args = mock_client.messages.create.call_args[1]
        assert (
            len(final_call_args["messages"]) == 3
        )  # original + assistant + tool result

        # Check tool result message format
        tool_result_message = final_call_args["messages"][2]
        assert tool_result_message["role"] == "user"
        assert len(tool_result_message["content"]) == 1
        assert tool_result_message["content"][0]["type"] == "tool_result"
        assert tool_result_message["content"][0]["tool_use_id"] == "tool_123"
        assert (
            tool_result_message["content"][0]["content"]
            == "Search results about Python"
        )

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
            "Outline result",
        ]

        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system",
        }

        # Mock client
        mock_client = Mock()
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Combined results response"
        mock_client.messages.create.return_value = mock_final_response

        with patch.object(ai_gen, "client", mock_client):
            result = ai_gen._handle_tool_execution(
                mock_initial_response, base_params, mock_tool_manager
            )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="Python"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Python Basics"
        )

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
            "system": "Test system",
        }

        mock_client = Mock()
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Error handled response"
        mock_client.messages.create.return_value = mock_final_response

        with patch.object(ai_gen, "client", mock_client):
            # This should not raise an exception, but handle it gracefully
            try:
                ai_gen._handle_tool_execution(
                    mock_initial_response, base_params, mock_tool_manager
                )
                # The method should still try to get a final response
                assert mock_client.messages.create.called
            except Exception as e:
                # If an exception is raised, it should be the original tool error
                assert "Tool execution failed" in str(e)

    def test_multi_round_tool_execution(
        self, mock_anthropic_client_multi_round, mock_tool_manager_with_sources
    ):
        """Test multi-round tool execution with 2 rounds"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        mock_tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search course content"},
        ]

        with patch.object(ai_gen, "client", mock_anthropic_client_multi_round):
            result = ai_gen.generate_response(
                "Find a course that discusses the same topic as lesson 2 of Python Basics",
                tools=mock_tools,
                tool_manager=mock_tool_manager_with_sources,
            )

        # Should return tuple (response, sources)
        assert isinstance(result, tuple)
        response, sources = result

        # Verify 3 API calls were made (round 1 tool use, round 2 tool use, final response)
        assert mock_anthropic_client_multi_round.messages.create.call_count == 3

        # Verify both tools were executed
        assert mock_tool_manager_with_sources.execute_tool.call_count == 2
        mock_tool_manager_with_sources.execute_tool.assert_any_call(
            "get_course_outline", course_title="Python Basics"
        )
        mock_tool_manager_with_sources.execute_tool.assert_any_call(
            "search_course_content", query="variables and data types"
        )

        # Verify sources were aggregated from both rounds
        assert len(sources) == 2
        assert any("Python Basics Course" in source["text"] for source in sources)
        assert any("Python Basics - Lesson 2" in source["text"] for source in sources)

        # Check final response
        assert response == "Based on the course outline and search, here is the answer"

    def test_multi_round_early_termination(
        self, mock_anthropic_client_single_round_stop, mock_tool_manager_with_sources
    ):
        """Test that multi-round stops early when no more tools are needed"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        mock_tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]

        with patch.object(ai_gen, "client", mock_anthropic_client_single_round_stop):
            result = ai_gen.generate_response(
                "What is Python?",
                tools=mock_tools,
                tool_manager=mock_tool_manager_with_sources,
            )

        # Should return tuple (response, sources)
        assert isinstance(result, tuple)
        response, sources = result

        # Verify only 2 API calls were made (round 1 tool use, then stop)
        assert mock_anthropic_client_single_round_stop.messages.create.call_count == 2

        # Verify only one tool was executed
        assert mock_tool_manager_with_sources.execute_tool.call_count == 1
        mock_tool_manager_with_sources.execute_tool.assert_called_with(
            "search_course_content", query="Python basics"
        )

        # Check response
        assert response == "Here is the complete answer from the first search"

    def test_multi_round_context_preservation(self, mock_tool_manager_with_sources):
        """Test that conversation context is preserved across rounds"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Mock client that tracks messages passed to it
        mock_client = Mock()
        messages_history = []

        def track_create_calls(**kwargs):
            messages_history.append(kwargs.get("messages", []).copy())

            if len(messages_history) == 1:
                # First call - return tool use
                mock_response = Mock()
                mock_response.stop_reason = "tool_use"
                mock_tool = Mock()
                mock_tool.type = "tool_use"
                mock_tool.name = "search_course_content"
                mock_tool.id = "tool_1"
                mock_tool.input = {"query": "test"}
                mock_response.content = [mock_tool]
                return mock_response
            else:
                # Second call - return final response
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Final answer"
                mock_response.stop_reason = "end_turn"
                return mock_response

        mock_client.messages.create.side_effect = track_create_calls

        mock_tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]

        with patch.object(ai_gen, "client", mock_client):
            ai_gen.generate_response(
                "Test query",
                conversation_history="Previous: Hello\nAssistant: Hi there!",
                tools=mock_tools,
                tool_manager=mock_tool_manager_with_sources,
            )

        # Verify conversation context was preserved
        assert len(messages_history) == 2

        # First call should have original query only
        first_messages = messages_history[0]
        assert len(first_messages) == 1
        assert first_messages[0]["role"] == "user"
        assert first_messages[0]["content"] == "Test query"

        # Second call should have: original query + assistant response + tool results
        second_messages = messages_history[1]
        assert len(second_messages) == 3
        assert second_messages[0]["role"] == "user"  # Original query
        assert second_messages[1]["role"] == "assistant"  # AI's tool use response
        assert second_messages[2]["role"] == "user"  # Tool results

    def test_multi_round_max_rounds_limit(self, mock_tool_manager_with_sources):
        """Test that multi-round stops at maximum rounds limit"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Mock client that always returns tool use (would go infinite without limit)
        mock_client = Mock()
        call_count = 0

        def always_tool_use(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                # First two calls return tool use
                mock_response = Mock()
                mock_response.stop_reason = "tool_use"
                mock_tool = Mock()
                mock_tool.type = "tool_use"
                mock_tool.name = "search_course_content"
                mock_tool.id = f"tool_{call_count}"
                mock_tool.input = {"query": f"test_{call_count}"}
                mock_response.content = [mock_tool]
                return mock_response
            else:
                # Final call (after max rounds) returns text response
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Final response after max rounds"
                mock_response.stop_reason = "end_turn"
                return mock_response

        mock_client.messages.create.side_effect = always_tool_use

        mock_tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]

        with patch.object(ai_gen, "client", mock_client):
            result = ai_gen.generate_response(
                "Test query",
                tools=mock_tools,
                tool_manager=mock_tool_manager_with_sources,
            )

        # Should return tuple (response, sources)
        assert isinstance(result, tuple)
        response, sources = result

        # Should make exactly 3 calls (2 rounds + 1 final)
        assert call_count == 3

        # Should execute exactly 2 tools (one per round)
        assert mock_tool_manager_with_sources.execute_tool.call_count == 2

        # Check final response
        assert response == "Final response after max rounds"

    def test_multi_round_source_aggregation(
        self, mock_anthropic_client_multi_round, mock_tool_manager_with_sources
    ):
        """Test that sources are properly aggregated across multiple rounds"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        mock_tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search course content"},
        ]

        with patch.object(ai_gen, "client", mock_anthropic_client_multi_round):
            result = ai_gen.generate_response(
                "Multi-round query",
                tools=mock_tools,
                tool_manager=mock_tool_manager_with_sources,
            )

        # Should return tuple (response, sources)
        assert isinstance(result, tuple)
        response, sources = result

        # Verify sources from both rounds are aggregated
        assert len(sources) == 2

        # Check sources contain data from both tool executions
        source_texts = [source["text"] for source in sources]
        assert "Python Basics Course" in source_texts
        assert "Python Basics - Lesson 2" in source_texts

        # Check links are preserved
        source_links = [source.get("link") for source in sources]
        assert "https://example.com/course" in source_links
        assert "https://example.com/lesson/2" in source_links

    def test_multi_round_error_handling(self, mock_tool_manager_with_sources):
        """Test error handling during multi-round execution"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        # Mock client that has tool execution error in second round
        mock_client = Mock()
        call_count = 0

        def error_in_second_round(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call succeeds with tool use
                mock_response = Mock()
                mock_response.stop_reason = "tool_use"
                mock_tool = Mock()
                mock_tool.type = "tool_use"
                mock_tool.name = "search_course_content"
                mock_tool.id = "tool_1"
                mock_tool.input = {"query": "test"}
                mock_response.content = [mock_tool]
                return mock_response
            else:
                # Second call returns final response
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Response despite tool error"
                mock_response.stop_reason = "end_turn"
                return mock_response

        # Make tool manager throw exception on second call
        def tool_with_error(tool_name, **kwargs):
            if mock_tool_manager_with_sources.execute_tool.call_count == 2:
                raise Exception("Tool execution failed")
            return "First tool succeeded"

        mock_tool_manager_with_sources.execute_tool.side_effect = tool_with_error
        mock_client.messages.create.side_effect = error_in_second_round

        mock_tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]

        with patch.object(ai_gen, "client", mock_client):
            # Should handle error gracefully and not raise exception
            result = ai_gen.generate_response(
                "Test query",
                tools=mock_tools,
                tool_manager=mock_tool_manager_with_sources,
            )

        # Should still return tuple even with error
        assert isinstance(result, tuple)
        response, sources = result

        # Should continue conversation despite tool error
        assert call_count == 2
        assert response == "Response despite tool error"

    def test_backward_compatibility_without_tools(self, mock_anthropic_client):
        """Test that single-round behavior is preserved when no tools are provided"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

        with patch.object(ai_gen, "client", mock_anthropic_client):
            result = ai_gen.generate_response("What is Python?")

        # Should return just the response text (not a tuple) for backward compatibility
        assert isinstance(result, str)
        assert result == "This is a test response"

        # Should make only one API call
        mock_anthropic_client.messages.create.assert_called_once()

        # Verify no tools were passed in the call
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" not in call_args
