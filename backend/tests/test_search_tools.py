import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SourceObject
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test suite for CourseSearchTool"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_get_tool_definition(self):
        """Test that tool definition is properly structured"""
        definition = self.search_tool.get_tool_definition()

        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)

        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])
        self.assertIn("course_name", schema["properties"])
        self.assertIn("lesson_number", schema["properties"])
        self.assertEqual(schema["required"], ["query"])

    def test_execute_successful_search(self):
        """Test successful search execution with results"""
        # Mock search results
        mock_results = SearchResults(
            documents=["This is course content about Python", "More Python content"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2},
            ],
            distances=[0.2, 0.3],
            error=None,
        )

        # Mock vector store methods
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = (
            "http://example.com/course"
        )
        self.mock_vector_store.get_lesson_link.return_value = (
            "http://example.com/lesson1"
        )
        self.mock_vector_store.get_lesson_title.return_value = "Introduction"

        # Execute search
        result = self.search_tool.execute("Python functions")

        # Verify vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="Python functions", course_name=None, lesson_number=None
        )

        # Verify result format
        self.assertIn("Python Basics", result)
        self.assertIn("This is course content about Python", result)
        self.assertIn("More Python content", result)

        # Verify sources were tracked
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertIsInstance(self.search_tool.last_sources[0], SourceObject)

    def test_execute_with_course_filter(self):
        """Test search execution with course name filter"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = None
        self.mock_vector_store.get_lesson_link.return_value = None
        self.mock_vector_store.get_lesson_title.return_value = None

        result = self.search_tool.execute("decorators", course_name="Advanced Python")

        # Verify correct parameters passed
        self.mock_vector_store.search.assert_called_once_with(
            query="decorators", course_name="Advanced Python", lesson_number=None
        )

        self.assertIn("Advanced Python", result)

    def test_execute_with_lesson_filter(self):
        """Test search execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 3}],
            distances=[0.15],
            error=None,
        )

        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = None
        self.mock_vector_store.get_lesson_link.return_value = None
        self.mock_vector_store.get_lesson_title.return_value = "Functions"

        result = self.search_tool.execute("variables", lesson_number=3)

        self.mock_vector_store.search.assert_called_once_with(
            query="variables", course_name=None, lesson_number=3
        )

        self.assertIn("Lesson 3", result)

    def test_execute_no_results(self):
        """Test handling when search returns no results"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("nonexistent topic")

        self.assertEqual(result, "No relevant content found.")
        self.assertEqual(len(self.search_tool.last_sources), 0)

    def test_execute_no_results_with_filters(self):
        """Test no results message includes filter information"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute(
            "topic", course_name="Missing Course", lesson_number=5
        )

        self.assertIn("Missing Course", result)
        self.assertIn("lesson 5", result)

    def test_execute_search_error(self):
        """Test handling of search errors"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("any query")

        self.assertEqual(result, "Database connection failed")

    def test_source_object_creation(self):
        """Test that SourceObject instances are created correctly"""
        mock_results = SearchResults(
            documents=["Test content for source tracking"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = "http://test-course.com"
        self.mock_vector_store.get_lesson_link.return_value = "http://test-lesson.com"
        self.mock_vector_store.get_lesson_title.return_value = "Test Lesson"

        self.search_tool.execute("test query")

        source = self.search_tool.last_sources[0]
        self.assertEqual(source.course_title, "Test Course")
        self.assertEqual(source.lesson_number, 1)
        self.assertEqual(source.lesson_title, "Test Lesson")
        self.assertEqual(source.course_link, "http://test-course.com")
        self.assertEqual(source.lesson_link, "http://test-lesson.com")
        self.assertEqual(source.citation_id, 1)
        self.assertGreater(source.relevance_score, 0)


class TestCourseOutlineTool(unittest.TestCase):
    """Test suite for CourseOutlineTool"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.outline_tool = CourseOutlineTool(self.mock_vector_store)

    def test_get_tool_definition(self):
        """Test that outline tool definition is properly structured"""
        definition = self.outline_tool.get_tool_definition()

        self.assertEqual(definition["name"], "get_course_outline")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)

        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("course_name", schema["properties"])
        self.assertEqual(schema["required"], ["course_name"])

    def test_execute_successful_outline_retrieval(self):
        """Test successful course outline retrieval"""
        # Mock course resolution
        self.mock_vector_store._resolve_course_name.return_value = "Python Programming"

        # Mock course catalog data
        mock_catalog_result = {
            "metadatas": [
                {
                    "title": "Python Programming",
                    "instructor": "John Doe",
                    "course_link": "http://example.com/python",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "http://example.com/lesson1"}, {"lesson_number": 2, "lesson_title": "Variables", "lesson_link": null}]',
                }
            ]
        }

        self.mock_vector_store.course_catalog.get.return_value = mock_catalog_result

        result = self.outline_tool.execute("Python")

        # Verify course name resolution
        self.mock_vector_store._resolve_course_name.assert_called_once_with("Python")

        # Verify catalog query
        self.mock_vector_store.course_catalog.get.assert_called_once_with(
            ids=["Python Programming"]
        )

        # Verify result format
        self.assertIn("**Python Programming**", result)
        self.assertIn("Instructor: John Doe", result)
        self.assertIn("Course Link: http://example.com/python", result)
        self.assertIn("Lesson 1: Introduction", result)
        self.assertIn("Lesson 2: Variables", result)
        self.assertIn("http://example.com/lesson1", result)

    def test_execute_course_not_found(self):
        """Test handling when course name cannot be resolved"""
        self.mock_vector_store._resolve_course_name.return_value = None

        result = self.outline_tool.execute("Nonexistent Course")

        self.assertEqual(result, "No course found matching 'Nonexistent Course'")

    def test_execute_no_metadata(self):
        """Test handling when course has no metadata"""
        self.mock_vector_store._resolve_course_name.return_value = "Test Course"
        self.mock_vector_store.course_catalog.get.return_value = {"metadatas": []}

        result = self.outline_tool.execute("Test")

        self.assertIn("No metadata found for course 'Test Course'", result)

    def test_execute_no_lessons(self):
        """Test handling course with no lessons"""
        self.mock_vector_store._resolve_course_name.return_value = "Empty Course"
        mock_catalog_result = {
            "metadatas": [
                {
                    "title": "Empty Course",
                    "instructor": "Jane Smith",
                    "course_link": "http://example.com/empty",
                    "lessons_json": "[]",
                }
            ]
        }

        self.mock_vector_store.course_catalog.get.return_value = mock_catalog_result

        result = self.outline_tool.execute("Empty")

        self.assertIn("**Empty Course**", result)
        self.assertIn("No lessons found", result)


class TestToolManager(unittest.TestCase):
    """Test suite for ToolManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_vector_store = Mock()

    def test_register_tool(self):
        """Test tool registration"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(search_tool)

        self.assertIn("search_course_content", self.tool_manager.tools)
        self.assertEqual(self.tool_manager.tools["search_course_content"], search_tool)

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        outline_tool = CourseOutlineTool(self.mock_vector_store)

        self.tool_manager.register_tool(search_tool)
        self.tool_manager.register_tool(outline_tool)

        definitions = self.tool_manager.get_tool_definitions()

        self.assertEqual(len(definitions), 2)
        tool_names = [d["name"] for d in definitions]
        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)

    def test_execute_tool(self):
        """Test tool execution through manager"""
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        mock_tool.execute.return_value = "Test result"

        self.tool_manager.register_tool(mock_tool)

        result = self.tool_manager.execute_tool("test_tool", param1="value1")

        mock_tool.execute.assert_called_once_with(param1="value1")
        self.assertEqual(result, "Test result")

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        result = self.tool_manager.execute_tool("nonexistent_tool")

        self.assertEqual(result, "Tool 'nonexistent_tool' not found")

    def test_get_last_sources(self):
        """Test retrieving last sources from tools"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_sources = [Mock(spec=SourceObject)]

        self.tool_manager.register_tool(search_tool)

        sources = self.tool_manager.get_last_sources()

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources, search_tool.last_sources)

    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_sources = [Mock(spec=SourceObject)]

        self.tool_manager.register_tool(search_tool)
        self.tool_manager.reset_sources()

        self.assertEqual(len(search_tool.last_sources), 0)


if __name__ == "__main__":
    unittest.main()
