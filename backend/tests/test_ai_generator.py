import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGenerator(unittest.TestCase):
    """Test suite for AIGenerator"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"

        with patch("ai_generator.anthropic.Anthropic"):
            self.ai_generator = AIGenerator(self.api_key, self.model)

    def test_init(self):
        """Test AIGenerator initialization"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            ai_gen = AIGenerator("test_key", "test_model")

            mock_anthropic.assert_called_once_with(api_key="test_key")
            self.assertEqual(ai_gen.model, "test_model")
            self.assertEqual(ai_gen.base_params["model"], "test_model")
            self.assertEqual(ai_gen.base_params["temperature"], 0)
            self.assertEqual(ai_gen.base_params["max_tokens"], 800)

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test response generation without tools"""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a direct response")]
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator("test_key", "test_model")

        result = ai_gen.generate_response("What is Python?")

        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]

        self.assertEqual(call_args["model"], "test_model")
        self.assertEqual(call_args["temperature"], 0)
        self.assertEqual(call_args["max_tokens"], 800)
        self.assertEqual(len(call_args["messages"]), 1)
        self.assertEqual(call_args["messages"][0]["content"], "What is Python?")
        self.assertIn("course materials", call_args["system"])

        # Verify result
        self.assertEqual(result, "This is a direct response")

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator("test_key", "test_model")

        history = "User: Hello\nAssistant: Hi there!"
        result = ai_gen.generate_response(
            "Follow up question", conversation_history=history
        )

        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        self.assertIn("Previous conversation:", call_args["system"])
        self.assertIn("Hello", call_args["system"])
        self.assertIn("Hi there!", call_args["system"])

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test response generation with tools available but not used"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct answer without tools")]
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "search_course_content", "description": "Search courses"}]
        result = ai_gen.generate_response("General knowledge question", tools=tools)

        # Verify tools were provided to API
        call_args = mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["tools"], tools)
        self.assertEqual(call_args["tool_choice"], {"type": "auto"})

        self.assertEqual(result, "Direct answer without tools")

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test response generation when Claude uses tools"""
        # Mock initial response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "Python functions"}
        mock_tool_block.id = "tool_use_123"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"

        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Here's what I found about Python functions...")
        ]

        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]
        mock_anthropic.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "search_course_content", "description": "Search courses"}]
        result = ai_gen.generate_response(
            "Tell me about Python functions",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python functions"
        )

        # Verify two API calls were made (initial + follow-up)
        self.assertEqual(mock_client.messages.create.call_count, 2)

        # Verify final result
        self.assertEqual(result, "Here's what I found about Python functions...")

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution(self, mock_anthropic):
        """Test tool execution handling in detail"""
        # Create mock tool use blocks
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "test query"}
        tool_block1.id = "tool_1"

        text_block = Mock()
        text_block.type = "text"

        mock_initial_response = Mock()
        mock_initial_response.content = [text_block, tool_block1]

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response
        mock_anthropic.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        ai_gen = AIGenerator("test_key", "test_model")

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
        }

        result = ai_gen._handle_tool_execution(
            mock_initial_response, base_params, mock_tool_manager
        )

        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test query"
        )

        # Verify final API call structure
        final_call_args = mock_client.messages.create.call_args[1]

        # Should have 3 messages: original user, assistant tool use, user tool results
        self.assertEqual(len(final_call_args["messages"]), 3)

        # First message: original user message
        self.assertEqual(final_call_args["messages"][0]["role"], "user")

        # Second message: assistant's tool use
        self.assertEqual(final_call_args["messages"][1]["role"], "assistant")
        self.assertEqual(
            final_call_args["messages"][1]["content"], mock_initial_response.content
        )

        # Third message: tool results
        self.assertEqual(final_call_args["messages"][2]["role"], "user")
        tool_results = final_call_args["messages"][2]["content"]
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0]["type"], "tool_result")
        self.assertEqual(tool_results[0]["tool_use_id"], "tool_1")
        self.assertEqual(tool_results[0]["content"], "Tool result")

        self.assertEqual(result, "Final answer")

    @patch("ai_generator.anthropic.Anthropic")
    def test_multiple_tool_calls(self, mock_anthropic):
        """Test handling multiple tool calls in one response"""
        # Create multiple mock tool use blocks
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "query1"}
        tool_block1.id = "tool_1"

        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.input = {"course_name": "Python"}
        tool_block2.id = "tool_2"

        mock_initial_response = Mock()
        mock_initial_response.content = [tool_block1, tool_block2]

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Combined results")]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response
        mock_anthropic.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        ai_gen = AIGenerator("test_key", "test_model")

        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "system",
        }

        result = ai_gen._handle_tool_execution(
            mock_initial_response, base_params, mock_tool_manager
        )

        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify tool results message contains both results
        final_call_args = mock_client.messages.create.call_args[1]
        tool_results = final_call_args["messages"][2]["content"]
        self.assertEqual(len(tool_results), 2)

    def test_system_prompt_content(self):
        """Test that system prompt contains expected guidance"""
        system_prompt = AIGenerator.SYSTEM_PROMPT

        # Check for key components
        self.assertIn("course materials", system_prompt)
        self.assertIn("Course Content Search", system_prompt)
        self.assertIn("Course Outline", system_prompt)
        self.assertIn("structural questions", system_prompt)
        self.assertIn("course organization", system_prompt)
        self.assertIn("lesson lists", system_prompt)
        self.assertIn("Multi-round tool usage allowed", system_prompt)

        # Check response protocol
        self.assertIn("General knowledge questions", system_prompt)
        self.assertIn("Course-specific questions", system_prompt)
        self.assertIn("No meta-commentary", system_prompt)

    @patch("ai_generator.anthropic.Anthropic")
    def test_sequential_tool_calls_two_rounds(self, mock_anthropic):
        """Test complete 2-round sequential tool calling flow"""
        # Mock Round 1: Tool use response
        mock_tool_block_1 = Mock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "get_course_outline"
        mock_tool_block_1.input = {"course_name": "Python"}
        mock_tool_block_1.id = "tool_1"

        mock_round1_response = Mock()
        mock_round1_response.content = [mock_tool_block_1]
        mock_round1_response.stop_reason = "tool_use"

        # Mock Round 2: Tool use response
        mock_tool_block_2 = Mock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.input = {"query": "functions", "course_name": "Python"}
        mock_tool_block_2.id = "tool_2"

        mock_round2_response = Mock()
        mock_round2_response.content = [mock_tool_block_2]
        mock_round2_response.stop_reason = "tool_use"

        # Mock Final response (Round 3 would be, but we stop at 2)
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="Based on the course outline and search results, here's what I found..."
            )
        ]
        mock_final_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_round2_response,
            mock_final_response,
        ]
        mock_anthropic.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline result",
            "Search result",
        ]

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        result = ai_gen.generate_response(
            "What functions are covered in the Python course?",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify exactly 2 tool executions
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify exactly 2 API calls (max rounds reached)
        self.assertEqual(mock_client.messages.create.call_count, 2)

        # Since we hit max rounds, should return the text from round 2
        # The mock round 2 response has tool use, so get_text_content() should work
        # But since we're returning after max rounds, we get the final text

    @patch("ai_generator.anthropic.Anthropic")
    def test_sequential_tool_calls_early_termination(self, mock_anthropic):
        """Test early termination when Claude doesn't use tools in round 1"""
        # Mock Round 1: No tool use (direct answer)
        mock_round1_response = Mock()
        mock_round1_response.content = [
            Mock(text="This is a general knowledge answer that doesn't require tools.")
        ]
        mock_round1_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_round1_response
        mock_anthropic.return_value = mock_client

        mock_tool_manager = Mock()

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "search_course_content"}]
        result = ai_gen.generate_response(
            "What is Python?", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify no tools were executed
        mock_tool_manager.execute_tool.assert_not_called()

        # Verify only 1 API call (early termination)
        self.assertEqual(mock_client.messages.create.call_count, 1)

        # Verify result
        self.assertEqual(
            result, "This is a general knowledge answer that doesn't require tools."
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_sequential_tool_calls_with_tool_failure(self, mock_anthropic):
        """Test graceful handling of tool execution failure"""
        # Mock Round 1: Tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_1"

        mock_round1_response = Mock()
        mock_round1_response.content = [mock_tool_block]
        mock_round1_response.stop_reason = "tool_use"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_round1_response
        mock_anthropic.return_value = mock_client

        # Mock tool manager that fails
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "search_course_content"}]
        result = ai_gen.generate_response(
            "Search for something", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool execution was attempted
        mock_tool_manager.execute_tool.assert_called_once()

        # Verify graceful error handling
        self.assertEqual(
            result, "I encountered an error while searching for information."
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_context_preservation_across_rounds(self, mock_anthropic):
        """Test that conversation context is preserved between rounds"""
        # Mock Round 1: Tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.input = {"course_name": "Python"}
        mock_tool_block.id = "tool_1"

        mock_round1_response = Mock()
        mock_round1_response.content = [mock_tool_block]
        mock_round1_response.stop_reason = "tool_use"

        # Mock Round 2: No tool use (final answer)
        mock_round2_response = Mock()
        mock_round2_response.content = [Mock(text="Final answer based on context")]
        mock_round2_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_round2_response,
        ]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Course outline result"

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "get_course_outline"}]
        result = ai_gen.generate_response(
            "Tell me about Python course structure",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify 2 API calls were made
        self.assertEqual(mock_client.messages.create.call_count, 2)

        # Check that Round 2 call includes context from Round 1
        round2_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = round2_call_args["messages"]

        # Should have: original user query + assistant tool use + user tool results
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "user")  # Original query
        self.assertEqual(messages[1]["role"], "assistant")  # Tool use response
        self.assertEqual(messages[2]["role"], "user")  # Tool results

    @patch("ai_generator.anthropic.Anthropic")
    def test_conversation_history_with_sequential_rounds(self, mock_anthropic):
        """Test that conversation history works with sequential rounds"""
        # Mock single round response (no tools needed)
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with conversation context")]
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "search_course_content"}]
        mock_tool_manager = Mock()

        conversation_history = "User: Previous question\nAssistant: Previous answer"
        result = ai_gen.generate_response(
            "Follow up question",
            conversation_history=conversation_history,
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify system prompt includes conversation history
        call_args = mock_client.messages.create.call_args[1]
        self.assertIn("Previous conversation:", call_args["system"])
        self.assertIn("Previous question", call_args["system"])
        self.assertIn("Previous answer", call_args["system"])

    @patch("ai_generator.anthropic.Anthropic")
    def test_api_error_handling_in_sequential_rounds(self, mock_anthropic):
        """Test API error handling during sequential rounds"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "search_course_content"}]
        mock_tool_manager = Mock()

        result = ai_gen.generate_response(
            "Test query", tools=tools, tool_manager=mock_tool_manager
        )

        # Should get error message
        self.assertIn("I apologize, but I encountered an error", result)

    @patch("ai_generator.anthropic.Anthropic")
    def test_backward_compatibility_no_tools(self, mock_anthropic):
        """Test that old behavior is preserved when no tools provided"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct response without tools")]
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator("test_key", "test_model")

        # Call without tools (old behavior)
        result = ai_gen.generate_response("What is Python?")

        # Verify single API call
        self.assertEqual(mock_client.messages.create.call_count, 1)

        # Verify no tools in API call
        call_args = mock_client.messages.create.call_args[1]
        self.assertNotIn("tools", call_args)

        self.assertEqual(result, "Direct response without tools")

    @patch("ai_generator.anthropic.Anthropic")
    def test_backward_compatibility_no_tool_manager(self, mock_anthropic):
        """Test that old behavior is preserved when tool_manager is None"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response without tool manager")]
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator("test_key", "test_model")

        tools = [{"name": "search_course_content"}]

        # Call with tools but no tool_manager (old edge case)
        result = ai_gen.generate_response(
            "What is Python?", tools=tools, tool_manager=None
        )

        # Should use single round logic
        self.assertEqual(mock_client.messages.create.call_count, 1)
        self.assertEqual(result, "Response without tool manager")


if __name__ == "__main__":
    unittest.main()
