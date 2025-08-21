# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Course Materials RAG (Retrieval-Augmented Generation) System** - a full-stack web application that enables semantic search and AI-powered Q&A over educational course materials. The system uses ChromaDB for vector storage, Anthropic's Claude for response generation, and provides a clean web interface for user interactions.

## Core Architecture

The application follows a layered architecture with clear separation of concerns:

### Backend Components (`/backend/`)
- **`rag_system.py`**: Main orchestrator that coordinates all components
- **`app.py`**: FastAPI server with CORS-enabled endpoints (`/api/query`, `/api/courses`)
- **`document_processor.py`**: Parses structured course files and creates text chunks
- **`vector_store.py`**: ChromaDB wrapper handling embeddings and semantic search
- **`ai_generator.py`**: Anthropic Claude API integration with tool-calling support
- **`search_tools.py`**: Tool system allowing Claude to search course content autonomously
- **`session_manager.py`**: Manages conversation history for contextual responses
- **`config.py`**: Centralized configuration using environment variables

### Data Flow Architecture
1. **Document Processing**: Course files → Structured parsing → Text chunking → Vector embeddings
2. **Query Processing**: User query → Claude tool decision → Vector search → Context assembly → AI response
3. **Tool-Based Search**: Claude autonomously decides when to search and formats results

### Course Document Structure
Course files must follow this format:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional lesson url]
[lesson content]

Lesson 1: [title]
[lesson content]
```

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Create .env file from template
cp .env.example .env
# Then add your ANTHROPIC_API_KEY to .env
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

The application serves the web interface at `http://localhost:8000` and API docs at `http://localhost:8000/docs`.

### Adding Course Materials
Place course text files in the `/docs/` folder. The system automatically loads them on startup and avoids reprocessing existing courses.

## Key Configuration Parameters

Located in `backend/config.py`:
- **CHUNK_SIZE**: 800 characters per text chunk (affects search granularity)
- **CHUNK_OVERLAP**: 100 characters overlap (preserves context across chunks)  
- **MAX_RESULTS**: 5 search results returned per query
- **MAX_HISTORY**: 2 conversation exchanges kept in session memory
- **ANTHROPIC_MODEL**: Currently uses `claude-sonnet-4-20250514`
- **EMBEDDING_MODEL**: Uses `all-MiniLM-L6-v2` for vector embeddings

## RAG System Flow

1. **Document Ingestion**: Text files parsed into Course/Lesson objects, chunked with overlap, embedded via sentence transformers
2. **Query Processing**: FastAPI receives query → RAG system manages session → AI Generator calls Claude with search tool available
3. **Tool Execution**: Claude decides to search → CourseSearchTool performs semantic search → Results formatted with course/lesson context
4. **Response Generation**: Claude synthesizes search results into final answer → Sources tracked and returned to frontend

## Session Management

The system maintains conversation context through:
- Unique session IDs generated per conversation
- Message history limited to `MAX_HISTORY * 2` messages
- Context passed to Claude for coherent multi-turn conversations

## Vector Search Capabilities

- **Semantic similarity** search using sentence transformers
- **Course filtering** by partial title matching
- **Lesson filtering** by specific lesson numbers
- **Smart chunking** that preserves sentence boundaries
- **Context enrichment** with course/lesson metadata in chunks

## Frontend Integration

- Vanilla HTML/CSS/JavaScript with no framework dependencies
- Real-time query processing with loading states
- Collapsible source citations displaying which courses/lessons provided answers
- Course statistics sidebar showing available materials

## Error Handling

The system includes robust error handling at each layer:
- Document processing continues despite individual file failures
- Vector search gracefully handles empty results
- AI generation includes fallbacks for tool execution failures
- Frontend displays user-friendly error messages
- make sure to use uv to manage all dependencies