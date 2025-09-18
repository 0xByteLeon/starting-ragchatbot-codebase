import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Available Tools:
1. **search_course_content**: For questions about specific course content or detailed educational materials
2. **get_course_outline**: For questions about course structure, lessons, outlines, or course organization

Tool Usage Guidelines:
- **Multi-round capability**: You can make up to 2 rounds of tool calls per query
- **Round 1**: Make initial tool calls to gather primary information
- **Round 2**: If needed, make additional tool calls based on Round 1 results
- **Course outline questions**: Use get_course_outline tool for questions about course structure, lesson lists, course organization, or "what lessons are in X course"
- **Course content questions**: Use search_course_content tool for detailed content within courses
- **Multi-step queries**: Use Round 1 to gather context, Round 2 to refine search
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course outline questions**: Use get_course_outline tool first, then present the complete course structure including course title, course link, and all lesson details
- **Course content questions**: Use search_course_content tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None):
        """
        Generate AI response with multi-round tool support.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (response_text, sources) if tools available, otherwise just response_text
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

            # Use multi-round conversation for tool-enabled queries
            response_text, sources = self._execute_multi_round_conversation(api_params, tool_manager)
            return response_text, sources

        # Simple response for non-tool queries
        response = self.client.messages.create(**api_params)
        return response.content[0].text

    def _execute_tools_for_round(self, response, tool_manager):
        """
        Execute tools for a round and return results + sources.

        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results, round_sources)
        """
        tool_results = []
        round_sources = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

                    # Collect sources from this tool execution
                    sources = tool_manager.get_last_sources()
                    round_sources.extend(sources)
                    tool_manager.reset_sources()

                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}"
                    })

        return tool_results, round_sources

    def _execute_multi_round_conversation(self, api_params: Dict[str, Any], tool_manager, max_rounds: int = 2):
        """
        Execute multi-round conversation with tool calling capability.

        Args:
            api_params: Base API parameters for Claude
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds

        Returns:
            Tuple of (final_response_text, all_sources)
        """
        messages = api_params["messages"].copy()
        all_sources = []

        for round_num in range(1, max_rounds + 1):
            # Make API call with current messages
            current_params = {
                **api_params,
                "messages": messages
            }
            response = self.client.messages.create(**current_params)

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Check if tools are requested
            if response.stop_reason == "tool_use":
                # Execute tools and collect results + sources
                tool_results, round_sources = self._execute_tools_for_round(response, tool_manager)
                messages.append({"role": "user", "content": tool_results})
                all_sources.extend(round_sources)

                # Continue to next round if not at max
                if round_num < max_rounds:
                    continue
                else:
                    # Max rounds reached, make final call without tools
                    final_params = {
                        **self.base_params,
                        "messages": messages,
                        "system": api_params["system"]
                    }
                    final_response = self.client.messages.create(**final_params)
                    return final_response.content[0].text, all_sources
            else:
                # No more tools requested - return final response
                return response.content[0].text, all_sources

        # Fallback (shouldn't reach here)
        return "Unable to complete request within maximum rounds", all_sources

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text