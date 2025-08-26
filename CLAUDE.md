# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name
```

### Environment Setup
- Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Application runs on `http://localhost:8000`
- API docs available at `http://localhost:8000/docs`

## Architecture Overview

### High-Level Architecture
This is a **Retrieval-Augmented Generation (RAG) system** for course materials built with FastAPI backend and vanilla JavaScript frontend. The system uses **tool-based search** where the AI agent can call search functions to retrieve relevant information.

### Core Components Flow
1. **RAGSystem** (`backend/rag_system.py`) - Main orchestrator that coordinates all components
2. **DocumentProcessor** - Processes course documents into chunks
3. **VectorStore** (ChromaDB) - Stores and searches document embeddings
4. **AIGenerator** (Claude) - Generates responses using retrieved context
5. **ToolManager** - Manages search tools that the AI can call
6. **SessionManager** - Handles conversation history

### Key Architecture Patterns

#### Tool-Based Search Architecture
- AI agent uses **function calling** to search for information
- `ToolManager` registers available tools (currently `CourseSearchTool`)
- Tools return sources that are tracked separately from the AI response
- Located in `backend/search_tools.py`

#### Vector Storage Strategy
- **Dual storage**: Course metadata + content chunks stored separately
- Course metadata includes title, instructor, description for high-level queries
- Content chunks are text segments for detailed information retrieval
- ChromaDB with sentence-transformers embeddings (`all-MiniLM-L6-v2`)

#### Session Management
- Each conversation gets a unique session ID
- Conversation history limited by `MAX_HISTORY` config
- Sessions track user query + AI response pairs

### Data Models (`backend/models.py`)
- `Course`: Metadata (title, instructor, description, lessons)
- `Lesson`: Individual lesson within a course
- `CourseChunk`: Text chunk with course reference and metadata

### Configuration (`backend/config.py`)
Key settings:
- `CHUNK_SIZE: 800` - Size of text chunks for vector storage
- `CHUNK_OVERLAP: 100` - Overlap between chunks
- `MAX_RESULTS: 5` - Maximum search results
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"`

### Frontend Architecture
- Single-page vanilla JavaScript application
- Real-time chat interface with markdown support
- Course statistics sidebar with collapsible sections
- Uses Marked.js for markdown rendering

### Document Processing Pipeline
1. Documents loaded from `docs/` folder on startup
2. `DocumentProcessor` extracts course structure and creates chunks
3. Metadata and chunks stored in ChromaDB
4. Duplicate detection prevents re-processing existing courses

### API Endpoints
- `POST /api/query` - Main query endpoint with session support
- `GET /api/courses` - Course statistics and analytics
- `GET /` - Serves frontend static files

## Development Notes

### Working with the RAG System
- Documents are automatically loaded from `docs/` folder on startup
- Add new course files (PDF, DOCX, TXT) to `docs/` folder
- System automatically detects and processes new courses
- ChromaDB data persists in `./chroma_db` directory

### Tool System Extension
To add new search tools:
1. Implement tool class in `backend/search_tools.py`
2. Register with `ToolManager` in `RAGSystem.__init__`
3. Tool must return sources for proper attribution

### Configuration Changes
- Edit `backend/config.py` for system parameters
- Restart application after config changes
- Environment variables override dataclass defaults
- always us uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies