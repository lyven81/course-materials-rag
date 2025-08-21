import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anthropic


@dataclass
class RoundContext:
    """Manages conversation state across rounds"""

    def __init__(self, initial_query: str, conversation_history: Optional[str] = None):
        self.messages: List[Dict] = [{"role": "user", "content": initial_query}]
        self.round_number: int = 1
        self.conversation_history = conversation_history

    def add_assistant_message(self, content):
        """Add assistant's response to message history"""
        self.messages.append({"role": "assistant", "content": content})

    def add_user_message(self, content):
        """Add user message (typically tool results) to history"""
        self.messages.append({"role": "user", "content": content})

    def increment_round(self):
        """Move to next round"""
        self.round_number += 1


@dataclass
class RoundResult:
    """Encapsulates results from a single round"""

    response: Any
    has_tool_use: bool
    execution_success: bool = True
    error: Optional[str] = None

    def get_text_content(self) -> str:
        """Extract text content from response"""
        if hasattr(self.response, "content") and self.response.content:
            return self.response.content[0].text
        return ""


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **Course Content Search**: For questions about specific topics within course materials
2. **Course Outline**: For questions about course structure, lesson lists, or course overviews

Tool Usage Guidelines:
- Use content search for **specific topic questions** within course materials
- Use course outline for **structural questions** about course organization, lesson lists, or course overviews
- **Multi-round tool usage allowed**: You can use tools up to 2 times per query to gather comprehensive information
- **Sequential reasoning**: Use results from previous tool calls to inform subsequent searches
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Multi-Round Strategy:
- **Round 1**: Gather initial information or course structure
- **Round 2**: Perform targeted searches based on Round 1 results
- Examples:
  - Round 1: Get course outline to find lesson titles
  - Round 2: Search specific lessons based on titles found
  - Round 1: Search broad topic
  - Round 2: Search related or comparative information

When using the course outline tool, always include:
- Course title
- Course link (if available)
- Complete lesson list with numbers and titles
- Lesson links (if available)

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tool usage
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

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
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def _make_api_call_with_retry(self, **api_params) -> Any:
        """
        Make API call with exponential backoff retry logic.

        Handles rate limiting (429), overloaded errors (529), and other transient failures.
        """
        max_retries = 3
        base_delay = 1  # Start with 1 second delay

        for attempt in range(max_retries + 1):
            try:
                return self.client.messages.create(**api_params)

            except anthropic.RateLimitError as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        "API rate limit exceeded after multiple retries. Please try again later."
                    ) from e
                delay = base_delay * (2**attempt) + (time.time() % 1)  # Add jitter
                print(
                    f"Rate limit hit, retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(delay)

            except anthropic.APIStatusError as e:
                if e.status_code in [
                    529,
                    502,
                    503,
                    504,
                ]:  # Server errors including overloaded
                    if attempt == max_retries:
                        if e.status_code == 529:
                            raise RuntimeError(
                                "Anthropic API is currently overloaded. Please try again in a few minutes."
                            ) from e
                        else:
                            raise RuntimeError(
                                f"Anthropic API is temporarily unavailable (status {e.status_code}). Please try again later."
                            ) from e
                    delay = base_delay * (2**attempt) + (time.time() % 1)  # Add jitter
                    print(
                        f"API temporarily unavailable (status {e.status_code}), retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    time.sleep(delay)
                else:
                    # For other status codes, don't retry
                    raise RuntimeError(f"API error: {str(e)}") from e

            except anthropic.APIConnectionError as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        "Unable to connect to Anthropic API. Please check your internet connection."
                    ) from e
                delay = base_delay * (2**attempt)
                print(
                    f"Connection error, retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(delay)

            except Exception as e:
                # For unexpected errors, don't retry
                raise RuntimeError(f"Unexpected error during API call: {str(e)}") from e

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential tool calling rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # If no tools or tool manager, use single round logic
        if not tools or not tool_manager:
            return self._generate_single_round_response(
                query, conversation_history, tools
            )

        # Use sequential rounds for tool-enabled queries
        return self._execute_sequential_rounds(
            query, conversation_history, tools, tool_manager
        )

    def _generate_single_round_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
    ) -> str:
        """Generate a single round response without tools (backward compatibility)"""
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available (for cases where tools exist but no tool_manager)
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self._make_api_call_with_retry(**api_params)

        # If tools were used but no tool manager, fall back to legacy behavior
        if response.stop_reason == "tool_use":
            return (
                "I have tools available but cannot execute them without a tool manager."
            )

        return response.content[0].text

    def _execute_sequential_rounds(
        self, query: str, conversation_history: Optional[str], tools: List, tool_manager
    ) -> str:
        """Execute up to 2 sequential rounds of tool calling"""
        context = RoundContext(query, conversation_history)
        max_rounds = 2

        while context.round_number <= max_rounds:
            # Execute current round
            round_result = self._execute_single_round(context, tools)

            # Execute tools if present
            if round_result.has_tool_use and tool_manager:
                success = self._execute_tools_for_round(
                    round_result, context, tool_manager
                )
                if not success:
                    return "I encountered an error while searching for information."

            # Check termination conditions after tool execution
            if not self._should_continue_rounds(
                context.round_number, round_result, max_rounds
            ):
                return round_result.get_text_content()

            context.increment_round()

        # Return final response after max rounds
        return round_result.get_text_content()

    def _execute_single_round(
        self, context: RoundContext, tools: Optional[List] = None
    ) -> RoundResult:
        """Execute a single round of API call"""
        try:
            # Build system prompt with conversation history if available
            system_content = self._build_system_prompt(context)

            # Prepare API parameters
            api_params = {
                **self.base_params,
                "messages": context.messages,
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            response = self._make_api_call_with_retry(**api_params)
            has_tool_use = response.stop_reason == "tool_use"

            return RoundResult(response, has_tool_use, execution_success=True)

        except Exception as e:
            # Create fallback response for API errors
            class MockResponse:
                def __init__(self, text):
                    self.content = [type("obj", (object,), {"text": text})()]

            fallback_response = MockResponse(
                f"I apologize, but I encountered an error: {str(e)}"
            )
            return RoundResult(
                fallback_response,
                has_tool_use=False,
                execution_success=False,
                error=str(e),
            )

    def _should_continue_rounds(
        self, round_num: int, round_result: RoundResult, max_rounds: int
    ) -> bool:
        """Check if we should continue to next round"""
        # Condition 1: Maximum rounds reached
        if round_num >= max_rounds:
            return False

        # Condition 2: No tool use in response
        if not round_result.has_tool_use:
            return False

        # Condition 3: Tool execution failed
        if not round_result.execution_success:
            return False

        return True

    def _execute_tools_for_round(
        self, round_result: RoundResult, context: RoundContext, tool_manager
    ) -> bool:
        """Execute tools for current round and update context"""
        try:
            # Add assistant's response to context
            context.add_assistant_message(round_result.response.content)

            # Execute all tool calls
            tool_results = []
            for content_block in round_result.response.content:
                if content_block.type == "tool_use":
                    result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        }
                    )

            # Add tool results to context
            if tool_results:
                context.add_user_message(tool_results)

            return True

        except Exception as e:
            # Add error message to context for graceful degradation
            error_result = [
                {
                    "type": "tool_result",
                    "tool_use_id": "error",
                    "content": f"Tool execution failed: {str(e)}",
                }
            ]
            context.add_user_message(error_result)
            return False

    def _build_system_prompt(self, context: RoundContext) -> str:
        """Build system prompt with conversation history and round context"""
        base_prompt = self.SYSTEM_PROMPT

        # Add conversation history if available
        if context.conversation_history:
            base_prompt += f"\n\nPrevious conversation:\n{context.conversation_history}"

        # Add round-specific guidance for round 2
        if context.round_number == 2:
            base_prompt += "\n\nROUND 2: You have previous tool results available. Use them to make informed decisions about additional searches or provide a comprehensive final answer."

        return base_prompt

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """Legacy method for backward compatibility.
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
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self._make_api_call_with_retry(**final_params)
        return final_response.content[0].text
