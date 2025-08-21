import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import json
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SourceObject, Course, Lesson


# Create test app without static file mounting to avoid filesystem dependencies
def create_test_app():
    """Create FastAPI app for testing without static file dependencies"""
    from pydantic import BaseModel
    from typing import List, Optional
    from fastapi import HTTPException
    
    app = FastAPI(title="Course Materials RAG System Test", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceObject]
        session_id: str
        source_summary: Optional[str] = None
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources, source_summary = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
                source_summary=source_summary
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}
    
    # Store mock for access in tests
    app.state.mock_rag_system = mock_rag_system
    
    return app


@pytest.fixture
def test_app():
    """Create test app fixture"""
    return create_test_app()


@pytest.fixture
def test_client(test_app):
    """Create test client fixture"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request for testing"""
    return {
        "query": "What is Python?",
        "session_id": None
    }


@pytest.fixture
def sample_query_response_data(sample_source_objects):
    """Sample query response data"""
    return {
        "answer": "Python is a high-level programming language.",
        "sources": sample_source_objects,
        "session_id": "test-session-123",
        "source_summary": "Based on 1 course, 2 lessons"
    }


class TestQueryEndpoint:
    """Test suite for /api/query endpoint"""
    
    def test_query_endpoint_success(self, test_client, test_app, sample_source_objects):
        """Test successful query with response"""
        # Configure mock RAG system
        mock_rag = test_app.state.mock_rag_system
        mock_rag.session_manager.create_session.return_value = "test-session-123"
        mock_rag.query.return_value = (
            "Python is a high-level programming language.",
            sample_source_objects,
            "Based on 1 course, 2 lessons"
        )
        
        # Make request
        response = test_client.post("/api/query", json={
            "query": "What is Python?",
            "session_id": None
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Python is a high-level programming language."
        assert data["session_id"] == "test-session-123"
        assert data["source_summary"] == "Based on 1 course, 2 lessons"
        assert len(data["sources"]) == 2
        
        # Verify sources structure
        source = data["sources"][0]
        assert "course_title" in source
        assert "lesson_number" in source
        assert "lesson_title" in source
        assert "citation_id" in source
        
        # Verify RAG system was called correctly
        mock_rag.query.assert_called_once_with("What is Python?", "test-session-123")
    
    def test_query_endpoint_with_existing_session(self, test_client, test_app, sample_source_objects):
        """Test query with existing session ID"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.query.return_value = (
            "Continuing the conversation about Python.",
            sample_source_objects,
            "Based on 1 course, 1 lesson"
        )
        
        response = test_client.post("/api/query", json={
            "query": "Tell me more about functions",
            "session_id": "existing-session-456"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing-session-456"
        assert data["answer"] == "Continuing the conversation about Python."
        
        # Verify session wasn't created since we provided one
        mock_rag.session_manager.create_session.assert_not_called()
        mock_rag.query.assert_called_once_with("Tell me more about functions", "existing-session-456")
    
    def test_query_endpoint_empty_query(self, test_client, test_app):
        """Test query with empty string"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.session_manager.create_session.return_value = "test-session"
        mock_rag.query.return_value = ("Please provide a query.", [], None)
        
        response = test_client.post("/api/query", json={
            "query": "",
            "session_id": None
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Please provide a query."
        assert data["sources"] == []
    
    def test_query_endpoint_missing_query_field(self, test_client):
        """Test request with missing query field"""
        response = test_client.post("/api/query", json={
            "session_id": "test-session"
        })
        
        assert response.status_code == 422  # Validation error
        assert "query" in response.json()["detail"][0]["loc"]
    
    def test_query_endpoint_invalid_json(self, test_client):
        """Test request with invalid JSON"""
        response = test_client.post("/api/query", data="invalid json")
        
        assert response.status_code == 422
    
    def test_query_endpoint_rag_system_error(self, test_client, test_app):
        """Test handling of RAG system errors"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.session_manager.create_session.return_value = "test-session"
        mock_rag.query.side_effect = Exception("Database connection failed")
        
        response = test_client.post("/api/query", json={
            "query": "What is Python?",
            "session_id": None
        })
        
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]
    
    def test_query_endpoint_session_creation_error(self, test_client, test_app):
        """Test handling of session creation errors"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.session_manager.create_session.side_effect = Exception("Session creation failed")
        
        response = test_client.post("/api/query", json={
            "query": "What is Python?",
            "session_id": None
        })
        
        assert response.status_code == 500
        assert "Session creation failed" in response.json()["detail"]
    
    def test_query_endpoint_response_format(self, test_client, test_app, sample_source_objects):
        """Test that response follows expected format"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.session_manager.create_session.return_value = "session-123"
        mock_rag.query.return_value = (
            "Test answer",
            sample_source_objects,
            "Test summary"
        )
        
        response = test_client.post("/api/query", json={
            "query": "test query"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        required_fields = ["answer", "sources", "session_id", "source_summary"]
        for field in required_fields:
            assert field in data
        
        # Verify data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert isinstance(data["source_summary"], str) or data["source_summary"] is None


class TestCoursesEndpoint:
    """Test suite for /api/courses endpoint"""
    
    def test_courses_endpoint_success(self, test_client, test_app):
        """Test successful courses statistics retrieval"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Python Basics", "Advanced Python", "Web Development"]
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Basics" in data["course_titles"]
        assert "Advanced Python" in data["course_titles"]
        assert "Web Development" in data["course_titles"]
        
        mock_rag.get_course_analytics.assert_called_once()
    
    def test_courses_endpoint_empty_database(self, test_client, test_app):
        """Test courses endpoint with no courses"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_courses_endpoint_error(self, test_client, test_app):
        """Test courses endpoint with system error"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.get_course_analytics.side_effect = Exception("Vector store unavailable")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Vector store unavailable" in response.json()["detail"]
    
    def test_courses_endpoint_response_format(self, test_client, test_app):
        """Test that courses response follows expected format"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course 1", "Course 2"]
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Verify all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)


class TestRootEndpoint:
    """Test suite for / endpoint"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns basic info"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Course Materials RAG System API" in data["message"]
    
    def test_root_endpoint_method_not_allowed(self, test_client):
        """Test root endpoint with wrong HTTP method"""
        response = test_client.post("/")
        
        assert response.status_code == 405  # Method Not Allowed


class TestMiddleware:
    """Test suite for middleware functionality"""
    
    def test_cors_headers(self, test_client):
        """Test that CORS headers are properly set"""
        response = test_client.options("/api/query", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        
        # Should not be forbidden due to CORS
        assert response.status_code != 403
    
    def test_trusted_host_middleware(self, test_client):
        """Test that trusted host middleware allows requests"""
        response = test_client.get("/", headers={"Host": "example.com"})
        
        # Should not be blocked by trusted host middleware
        assert response.status_code == 200


class TestApiIntegration:
    """Integration tests for API endpoints"""
    
    def test_query_to_courses_flow(self, test_client, test_app, sample_source_objects):
        """Test typical user flow: query -> check courses"""
        mock_rag = test_app.state.mock_rag_system
        
        # Setup mock responses
        mock_rag.session_manager.create_session.return_value = "flow-session"
        mock_rag.query.return_value = (
            "Python is great for beginners.",
            sample_source_objects,
            "Based on 1 course"
        )
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Python Programming"]
        }
        
        # First query
        query_response = test_client.post("/api/query", json={
            "query": "What is Python good for?"
        })
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # Then check courses
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        
        # Verify consistency
        assert len(courses_data["course_titles"]) == courses_data["total_courses"]
        assert "Python Programming" in courses_data["course_titles"]
    
    def test_session_persistence_across_queries(self, test_client, test_app):
        """Test that session ID persists across multiple queries"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.session_manager.create_session.return_value = "persistent-session"
        mock_rag.query.return_value = ("Response", [], None)
        
        # First query (creates session)
        response1 = test_client.post("/api/query", json={"query": "First query"})
        session_id = response1.json()["session_id"]
        
        # Second query (uses existing session)
        response2 = test_client.post("/api/query", json={
            "query": "Second query",
            "session_id": session_id
        })
        
        # Verify same session ID
        assert response1.json()["session_id"] == response2.json()["session_id"]
        
        # Verify session was created only once
        mock_rag.session_manager.create_session.assert_called_once()


class TestErrorScenarios:
    """Test various error scenarios"""
    
    def test_malformed_requests(self, test_client):
        """Test handling of malformed requests"""
        # Missing content-type
        response = test_client.post("/api/query", data="not json")
        assert response.status_code == 422
        
        # Invalid JSON structure for query
        response = test_client.post("/api/query", json={"wrong_field": "value"})
        assert response.status_code == 422
    
    def test_large_query_handling(self, test_client, test_app):
        """Test handling of very large queries"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.session_manager.create_session.return_value = "large-query-session"
        mock_rag.query.return_value = ("Handled large query", [], None)
        
        # Create a large query string
        large_query = "What is Python? " * 1000  # ~14KB query
        
        response = test_client.post("/api/query", json={"query": large_query})
        
        # Should still be processed successfully
        assert response.status_code == 200
        mock_rag.query.assert_called_once_with(large_query, "large-query-session")
    
    def test_concurrent_requests_different_sessions(self, test_client, test_app):
        """Test handling of concurrent requests with different sessions"""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.session_manager.create_session.side_effect = ["session-1", "session-2"]
        mock_rag.query.return_value = ("Concurrent response", [], None)
        
        # Simulate concurrent requests
        response1 = test_client.post("/api/query", json={"query": "Query 1"})
        response2 = test_client.post("/api/query", json={"query": "Query 2"})
        
        # Should handle both successfully
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Should have different session IDs
        assert response1.json()["session_id"] != response2.json()["session_id"]