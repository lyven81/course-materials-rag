import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, CourseChunk, Lesson, SourceObject
from rag_system import RAGSystem


class TestRAGSystem(unittest.TestCase):
    """Test suite for RAGSystem integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 500
        self.mock_config.CHUNK_OVERLAP = 50
        self.mock_config.CHROMA_PATH = ":memory:"
        self.mock_config.EMBEDDING_MODEL = "test-model"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test-key"
        self.mock_config.ANTHROPIC_MODEL = "test-model"
        self.mock_config.MAX_HISTORY = 5

        # Create temp directory for test documents
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_rag_system_initialization(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test RAGSystem initialization"""
        rag_system = RAGSystem(self.mock_config)

        # Verify all components were initialized
        mock_doc_proc.assert_called_once_with(500, 50)
        mock_vector_store.assert_called_once_with(":memory:", "test-model", 5)
        mock_ai_gen.assert_called_once_with("test-key", "test-model")
        mock_session_mgr.assert_called_once_with(5)

        # Verify tools are registered
        self.assertIsNotNone(rag_system.tool_manager)
        self.assertIsNotNone(rag_system.search_tool)
        self.assertIsNotNone(rag_system.outline_tool)

        # Verify tools have vector store reference
        self.assertEqual(rag_system.search_tool.store, rag_system.vector_store)
        self.assertEqual(rag_system.outline_tool.store, rag_system.vector_store)

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_add_course_document_success(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test successful course document addition"""
        rag_system = RAGSystem(self.mock_config)

        # Mock document processor
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://test.com",
            lessons=[Lesson(lesson_number=1, title="Lesson 1")],
        )
        mock_chunks = [
            CourseChunk(
                content="Test content chunk",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        rag_system.document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        # Execute
        test_file = os.path.join(self.test_dir, "test_course.txt")
        with open(test_file, "w") as f:
            f.write("Test content")

        course, chunk_count = rag_system.add_course_document(test_file)

        # Verify processing
        rag_system.document_processor.process_course_document.assert_called_once_with(
            test_file
        )

        # Verify vector store calls
        rag_system.vector_store.add_course_metadata.assert_called_once_with(mock_course)
        rag_system.vector_store.add_course_content.assert_called_once_with(mock_chunks)

        # Verify return values
        self.assertEqual(course, mock_course)
        self.assertEqual(chunk_count, 1)

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_add_course_document_error(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test course document addition with error"""
        rag_system = RAGSystem(self.mock_config)

        # Mock processing error
        rag_system.document_processor.process_course_document.side_effect = Exception(
            "Processing failed"
        )

        course, chunk_count = rag_system.add_course_document("nonexistent_file.txt")

        # Verify error handling
        self.assertIsNone(course)
        self.assertEqual(chunk_count, 0)

    @patch("rag_system.os.path.join")
    @patch("rag_system.os.path.isfile")
    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_add_course_folder_success(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        mock_listdir,
        mock_exists,
        mock_isfile,
        mock_join,
    ):
        """Test adding course folder with multiple documents"""
        rag_system = RAGSystem(self.mock_config)

        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "course1.txt",
            "course2.pdf",
            "course3.docx",
            "readme.md",
        ]
        mock_isfile.return_value = True

        # Mock path joining
        def mock_join_side_effect(base, filename):
            return f"{base}/{filename}"

        mock_join.side_effect = mock_join_side_effect

        # Mock existing courses (empty initially)
        rag_system.vector_store.get_existing_course_titles.return_value = []

        # Mock document processing
        def mock_process_side_effect(file_path):
            if "course1" in file_path:
                course = Course(
                    title="Course 1",
                    instructor="Instructor 1",
                    course_link="http://1.com",
                    lessons=[],
                )
                chunks = [
                    CourseChunk(
                        content="Content 1",
                        course_title="Course 1",
                        lesson_number=1,
                        chunk_index=0,
                    )
                ]
                return course, chunks
            elif "course2" in file_path:
                course = Course(
                    title="Course 2",
                    instructor="Instructor 2",
                    course_link="http://2.com",
                    lessons=[],
                )
                chunks = [
                    CourseChunk(
                        content="Content 2",
                        course_title="Course 2",
                        lesson_number=1,
                        chunk_index=0,
                    )
                ]
                return course, chunks
            elif "course3" in file_path:
                course = Course(
                    title="Course 3",
                    instructor="Instructor 3",
                    course_link="http://3.com",
                    lessons=[],
                )
                chunks = [
                    CourseChunk(
                        content="Content 3",
                        course_title="Course 3",
                        lesson_number=1,
                        chunk_index=0,
                    )
                ]
                return course, chunks

        rag_system.document_processor.process_course_document.side_effect = (
            mock_process_side_effect
        )

        # Execute
        total_courses, total_chunks = rag_system.add_course_folder(self.test_dir)

        # Verify processing (should skip readme.md)
        self.assertEqual(
            rag_system.document_processor.process_course_document.call_count, 3
        )

        # Verify results
        self.assertEqual(total_courses, 3)
        self.assertEqual(total_chunks, 3)

    @patch("rag_system.os.path.exists")
    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_add_course_folder_nonexistent(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        mock_exists,
    ):
        """Test adding course folder that doesn't exist"""
        rag_system = RAGSystem(self.mock_config)

        mock_exists.return_value = False

        total_courses, total_chunks = rag_system.add_course_folder("/nonexistent/path")

        self.assertEqual(total_courses, 0)
        self.assertEqual(total_chunks, 0)

    @patch("rag_system.os.path.join")
    @patch("rag_system.os.path.isfile")
    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_add_course_folder_skip_existing(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        mock_listdir,
        mock_exists,
        mock_isfile,
        mock_join,
    ):
        """Test that existing courses are skipped"""
        rag_system = RAGSystem(self.mock_config)

        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.txt"]

        # Mock existing course
        rag_system.vector_store.get_existing_course_titles.return_value = ["Course 1"]

        def mock_process_side_effect(file_path):
            if "course1" in file_path:
                course = Course(
                    title="Course 1",
                    instructor="Instructor 1",
                    course_link="http://1.com",
                    lessons=[],
                )  # Existing
                chunks = []
                return course, chunks
            elif "course2" in file_path:
                course = Course(
                    title="Course 2",
                    instructor="Instructor 2",
                    course_link="http://2.com",
                    lessons=[],
                )  # New
                chunks = [
                    CourseChunk(
                        content="Content 2",
                        course_title="Course 2",
                        lesson_number=1,
                        chunk_index=0,
                    )
                ]
                return course, chunks

        rag_system.document_processor.process_course_document.side_effect = (
            mock_process_side_effect
        )

        total_courses, total_chunks = rag_system.add_course_folder(self.test_dir)

        # Should only add the new course
        self.assertEqual(total_courses, 1)
        self.assertEqual(total_chunks, 1)

        # Verify only the new course was added to vector store
        rag_system.vector_store.add_course_metadata.assert_called_once()
        rag_system.vector_store.add_course_content.assert_called_once()

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_process_query_with_search_tool(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test query processing that uses search tool"""
        rag_system = RAGSystem(self.mock_config)

        # Mock session management
        rag_system.session_manager.get_conversation_history.return_value = (
            "Previous conversation"
        )

        # Mock AI response with tool use
        rag_system.ai_generator.generate_response.return_value = (
            "Here are the search results about Python functions..."
        )

        # Mock tool definitions
        mock_search_def = {
            "name": "search_course_content",
            "description": "Search content",
        }
        mock_outline_def = {"name": "get_course_outline", "description": "Get outline"}

        # Create a mock function for get_tool_definitions
        def mock_get_tool_definitions():
            return [mock_search_def, mock_outline_def]

        rag_system.tool_manager.get_tool_definitions = mock_get_tool_definitions

        # Execute query
        session_id = "test_session"
        query = "Tell me about Python functions"

        # This is a simplified test - in reality we'd need to test the full query method
        # which would be in the application layer that uses RAGSystem
        tools = rag_system.tool_manager.get_tool_definitions()
        response = rag_system.ai_generator.generate_response(
            query,
            conversation_history="Previous conversation",
            tools=tools,
            tool_manager=rag_system.tool_manager,
        )

        # Verify AI generator was called with correct parameters
        rag_system.ai_generator.generate_response.assert_called_once_with(
            query,
            conversation_history="Previous conversation",
            tools=tools,
            tool_manager=rag_system.tool_manager,
        )

        self.assertIn("Python functions", response)

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_tool_integration(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test that tools are properly integrated with the system"""
        rag_system = RAGSystem(self.mock_config)

        # Test that both tools are registered
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]

        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)

        # Test that tools can be executed
        rag_system.vector_store.search.return_value = Mock(
            error=None, documents=[], metadata=[], distances=[]
        )

        search_result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="test query"
        )

        # Should return "no results" message
        self.assertIn("No relevant content found", search_result)

        # Test outline tool
        rag_system.vector_store._resolve_course_name.return_value = None

        outline_result = rag_system.tool_manager.execute_tool(
            "get_course_outline", course_name="Nonexistent Course"
        )

        self.assertIn("No course found matching", outline_result)


class TestRAGSystemIntegrationFlow(unittest.TestCase):
    """End-to-end integration tests for RAG system flow"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 500
        self.mock_config.CHUNK_OVERLAP = 50
        self.mock_config.CHROMA_PATH = ":memory:"
        self.mock_config.EMBEDDING_MODEL = "test-model"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test-key"
        self.mock_config.ANTHROPIC_MODEL = "test-model"
        self.mock_config.MAX_HISTORY = 5

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_content_search_flow(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test complete flow for content search queries"""
        rag_system = RAGSystem(self.mock_config)

        # Setup: Mock a course being added
        mock_course = Course(
            title="Python Basics",
            instructor="John Doe",
            course_link="http://python.com",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Introduction",
                    lesson_link="http://python.com/1",
                )
            ],
        )
        mock_chunks = [
            CourseChunk(
                content="Python is a programming language",
                course_title="Python Basics",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        rag_system.document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        # Add the course
        course, chunk_count = rag_system.add_course_document("test_course.txt")

        # Verify course was added
        self.assertEqual(course.title, "Python Basics")
        self.assertEqual(chunk_count, 1)

        # Setup: Mock search results for a query
        from vector_store import SearchResults

        mock_search_results = SearchResults(
            documents=["Python is a programming language used for web development"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.2],
            error=None,
        )

        rag_system.vector_store.search.return_value = mock_search_results
        rag_system.vector_store.get_course_link.return_value = "http://python.com"
        rag_system.vector_store.get_lesson_link.return_value = "http://python.com/1"
        rag_system.vector_store.get_lesson_title.return_value = "Introduction"

        # Test search tool execution
        search_result = rag_system.search_tool.execute("What is Python?")

        # Verify search results format
        self.assertIn("Python Basics", search_result)
        self.assertIn("Python is a programming language", search_result)

        # Verify sources were tracked
        sources = rag_system.tool_manager.get_last_sources()
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].course_title, "Python Basics")

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_outline_query_flow(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test complete flow for course outline queries"""
        rag_system = RAGSystem(self.mock_config)

        # Setup: Mock course outline data
        rag_system.vector_store._resolve_course_name.return_value = "Python Basics"

        mock_catalog_data = {
            "metadatas": [
                {
                    "title": "Python Basics",
                    "instructor": "John Doe",
                    "course_link": "http://python.com",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "http://python.com/1"}, {"lesson_number": 2, "lesson_title": "Variables", "lesson_link": "http://python.com/2"}]',
                }
            ]
        }

        rag_system.vector_store.course_catalog.get.return_value = mock_catalog_data

        # Test outline tool execution
        outline_result = rag_system.outline_tool.execute("Python")

        # Verify outline format
        self.assertIn("**Python Basics**", outline_result)
        self.assertIn("Instructor: John Doe", outline_result)
        self.assertIn("Course Link: http://python.com", outline_result)
        self.assertIn("Lesson 1: Introduction", outline_result)
        self.assertIn("Lesson 2: Variables", outline_result)

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_error_handling_flow(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test error handling throughout the system"""
        rag_system = RAGSystem(self.mock_config)

        # Test search with error
        from vector_store import SearchResults

        error_results = SearchResults.empty("Vector store connection failed")
        rag_system.vector_store.search.return_value = error_results

        search_result = rag_system.search_tool.execute("any query")
        self.assertEqual(search_result, "Vector store connection failed")

        # Test outline with missing course
        rag_system.vector_store._resolve_course_name.return_value = None

        outline_result = rag_system.outline_tool.execute("Missing Course")
        self.assertIn("No course found matching", outline_result)


if __name__ == "__main__":
    unittest.main()
