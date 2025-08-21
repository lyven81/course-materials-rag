from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from models import SourceObject
from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track SourceObject instances from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track SourceObject instances for the UI
        relevance_scores = results.get_relevance_scores()

        for i, (doc, meta) in enumerate(zip(results.documents, results.metadata)):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Create snippet (first 150 characters of content)
            content_snippet = doc[:150] + "..." if len(doc) > 150 else doc

            # Get additional metadata
            course_link = self.store.get_course_link(course_title)
            lesson_link = None
            lesson_title = None
            if lesson_num is not None:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                lesson_title = self.store.get_lesson_title(course_title, lesson_num)

            # Get relevance score
            relevance_score = relevance_scores[i] if i < len(relevance_scores) else 0.5

            # Create SourceObject
            source_obj = SourceObject(
                course_title=course_title,
                lesson_number=lesson_num,
                lesson_title=lesson_title,
                content_snippet=content_snippet,
                course_link=course_link,
                lesson_link=lesson_link,
                relevance_score=relevance_score,
                citation_id=i + 1,  # 1-based citation numbering
            )
            sources.append(source_obj)

            formatted.append(f"{header}\n{doc}")

        # Store SourceObject instances for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for getting course outlines with lesson structure"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the complete outline of a course including title, link, and all lessons",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["course_name"],
            },
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the outline tool to get course structure.

        Args:
            course_name: Course name/title to get outline for

        Returns:
            Formatted course outline or error message
        """
        # First resolve the course name to exact title
        exact_title = self.store._resolve_course_name(course_name)
        if not exact_title:
            return f"No course found matching '{course_name}'"

        # Get the course metadata using the exact title
        try:
            results = self.store.course_catalog.get(ids=[exact_title])
            if not results or not results.get("metadatas"):
                return f"No metadata found for course '{exact_title}'"

            metadata = results["metadatas"][0]
            course_title = metadata.get("title", exact_title)
            course_link = metadata.get("course_link", "No link available")
            instructor = metadata.get("instructor", "Unknown")

            # Parse lessons from JSON
            import json

            lessons_json = metadata.get("lessons_json", "[]")
            lessons = json.loads(lessons_json)

            # Format the course outline
            outline = [f"**{course_title}**"]
            outline.append(f"Instructor: {instructor}")
            outline.append(f"Course Link: {course_link}")
            outline.append("")
            outline.append("**Lessons:**")

            if not lessons:
                outline.append("No lessons found")
            else:
                for lesson in lessons:
                    lesson_num = lesson.get("lesson_number")
                    lesson_title = lesson.get("lesson_title", "Untitled")
                    lesson_link = lesson.get("lesson_link")

                    lesson_line = f"Lesson {lesson_num}: {lesson_title}"
                    if lesson_link:
                        lesson_line += f" ({lesson_link})"
                    outline.append(lesson_line)

            return "\n".join(outline)

        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> List[SourceObject]:
        """Get SourceObject instances from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
