import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os
import sys

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk, SourceObject
from vector_store import SearchResults


@pytest.fixture
def mock_config():
    """Mock configuration object with default values"""
    config = Mock()
    config.CHUNK_SIZE = 500
    config.CHUNK_OVERLAP = 50
    config.CHROMA_PATH = ":memory:"
    config.EMBEDDING_MODEL = "test-model"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_HISTORY = 5
    return config


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_course():
    """Sample course object for testing"""
    return Course(
        title="Python Programming",
        instructor="Dr. Jane Smith",
        course_link="https://example.com/python-course",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction to Python",
                lesson_link="https://example.com/python-course/lesson-1"
            ),
            Lesson(
                lesson_number=2,
                title="Variables and Data Types",
                lesson_link="https://example.com/python-course/lesson-2"
            )
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a high-level programming language known for its simplicity.",
            course_title="Python Programming",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables in Python are used to store data values.",
            course_title="Python Programming",
            lesson_number=2,
            chunk_index=0
        )
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "Python is a programming language used for web development",
            "Variables store data values in Python programs"
        ],
        metadata=[
            {"course_title": "Python Programming", "lesson_number": 1},
            {"course_title": "Python Programming", "lesson_number": 2}
        ],
        distances=[0.1, 0.2],
        error=None
    )


@pytest.fixture
def sample_source_objects():
    """Sample source objects for testing"""
    return [
        SourceObject(
            course_title="Python Programming",
            lesson_number=1,
            lesson_title="Introduction to Python",
            course_link="https://example.com/python-course",
            lesson_link="https://example.com/python-course/lesson-1",
            citation_id=1,
            relevance_score=0.9
        ),
        SourceObject(
            course_title="Python Programming",
            lesson_number=2,
            lesson_title="Variables and Data Types",
            course_link="https://example.com/python-course",
            lesson_link="https://example.com/python-course/lesson-2",
            citation_id=2,
            relevance_score=0.8
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Mock vector store with common methods"""
    store = Mock()
    store.search.return_value = SearchResults.empty()
    store.get_existing_course_titles.return_value = []
    store.add_course_metadata.return_value = None
    store.add_course_content.return_value = None
    store.get_course_link.return_value = None
    store.get_lesson_link.return_value = None
    store.get_lesson_title.return_value = None
    store._resolve_course_name.return_value = None
    store.course_catalog.get.return_value = {"metadatas": []}
    return store


@pytest.fixture
def mock_document_processor():
    """Mock document processor"""
    processor = Mock()
    processor.process_course_document.return_value = (None, [])
    return processor


@pytest.fixture
def mock_ai_generator():
    """Mock AI generator"""
    generator = Mock()
    generator.generate_response.return_value = "Test response"
    return generator


@pytest.fixture
def mock_session_manager():
    """Mock session manager"""
    manager = Mock()
    manager.create_session.return_value = "test-session-123"
    manager.get_conversation_history.return_value = ""
    manager.add_exchange.return_value = None
    return manager


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager"""
    manager = Mock()
    manager.get_tool_definitions.return_value = []
    manager.execute_tool.return_value = "Tool execution result"
    manager.get_last_sources.return_value = []
    manager.reset_sources.return_value = None
    return manager


@pytest.fixture
def mock_rag_system(mock_config):
    """Mock RAG system with all dependencies mocked"""
    with patch('rag_system.SessionManager') as mock_session_mgr, \
         patch('rag_system.AIGenerator') as mock_ai_gen, \
         patch('rag_system.VectorStore') as mock_vector_store, \
         patch('rag_system.DocumentProcessor') as mock_doc_proc:
        
        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)
        
        # Configure mocks
        rag.session_manager.create_session.return_value = "test-session-123"
        rag.ai_generator.generate_response.return_value = "Test response"
        rag.vector_store.search.return_value = SearchResults.empty()
        rag.tool_manager.get_last_sources.return_value = []
        
        return rag


@pytest.fixture
def course_document_content():
    """Sample course document content for file-based testing"""
    return """Course Title: Python Programming Basics
Course Link: https://example.com/python-basics
Course Instructor: Dr. John Doe

Lesson 0: Introduction
Lesson Link: https://example.com/python-basics/intro
This lesson introduces Python programming and its applications.

Lesson 1: Variables and Data Types
Lesson Link: https://example.com/python-basics/variables
Learn about variables, strings, integers, and other data types in Python.

Lesson 2: Control Structures
Learn about if statements, loops, and conditional logic.
"""


@pytest.fixture
def create_test_course_file(temp_directory, course_document_content):
    """Create a test course file in temp directory"""
    def _create_file(filename="test_course.txt", content=None):
        if content is None:
            content = course_document_content
        file_path = os.path.join(temp_directory, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    return _create_file


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for AI generator testing"""
    with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Mock AI response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        yield mock_client


@pytest.fixture
def empty_search_results():
    """Empty search results for testing no-results scenarios"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def error_search_results():
    """Search results with error for testing error scenarios"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Database connection failed"
    )


@pytest.fixture
def course_catalog_data():
    """Sample course catalog data for outline testing"""
    return {
        'metadatas': [{
            'title': 'Python Programming',
            'instructor': 'Dr. Jane Smith',
            'course_link': 'https://example.com/python-course',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction to Python", "lesson_link": "https://example.com/python-course/lesson-1"}, {"lesson_number": 2, "lesson_title": "Variables and Data Types", "lesson_link": "https://example.com/python-course/lesson-2"}]'
        }]
    }