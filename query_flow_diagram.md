# RAG System Query Flow Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant TM as Tool Manager<br/>(search_tools.py)
    participant ST as Search Tool<br/>(CourseSearchTool)
    participant VS as Vector Store<br/>(vector_store.py)
    participant SM as Session Manager<br/>(session_manager.py)
    participant Claude as Anthropic<br/>Claude API

    Note over U,Claude: User Query Processing Flow

    %% Frontend Processing
    U->>F: Types query & clicks send
    F->>F: Disable input, show loading
    F->>API: POST /api/query<br/>{query, session_id}

    %% Backend Entry Point
    API->>API: Validate QueryRequest
    API->>RAG: rag_system.query(query, session_id)

    %% Session & History Management
    RAG->>SM: get_conversation_history(session_id)
    SM-->>RAG: Previous messages or None

    %% AI Generation with Tools
    RAG->>AI: generate_response(query, history, tools, tool_manager)
    AI->>AI: Build system prompt + context
    AI->>Claude: Create message with tools enabled

    %% Tool Decision & Execution
    Claude-->>AI: Response with tool_use
    AI->>TM: execute_tool("search_course_content", params)
    TM->>ST: execute(query, course_name, lesson_number)

    %% Vector Search Process
    ST->>VS: search(query, filters)
    VS->>VS: ChromaDB semantic search
    VS-->>ST: SearchResults{documents, metadata, distances}
    ST->>ST: Format results with course/lesson context
    ST-->>TM: Formatted search results
    TM-->>AI: Tool execution results

    %% Final Response Generation
    AI->>Claude: Send tool results for synthesis
    Claude-->>AI: Final natural language response
    AI-->>RAG: Generated response text

    %% Response Assembly
    RAG->>TM: get_last_sources()
    TM-->>RAG: Source list for UI
    RAG->>SM: add_exchange(session_id, query, response)
    RAG-->>API: (response, sources)

    %% API Response
    API->>API: Create QueryResponse object
    API-->>F: JSON{answer, sources, session_id}

    %% Frontend Display
    F->>F: Remove loading animation
    F->>F: Display response + sources
    F->>F: Re-enable input
    F-->>U: Show AI response with sources

    %% Data Flow Annotations
    Note over VS: ChromaDB Vector Database<br/>- Course metadata<br/>- Content chunks<br/>- Embeddings
    Note over Claude: Anthropic Claude API<br/>- Tool-aware LLM<br/>- Decides when to search<br/>- Synthesizes results
    Note over SM: Session Management<br/>- Conversation history<br/>- Context preservation<br/>- Max 10 messages
```

## Flow Components

### **Frontend Layer**
- **User Interface**: HTML form with chat interface
- **JavaScript**: Handles events, API calls, DOM updates
- **Loading States**: Visual feedback during processing

### **API Layer** 
- **FastAPI**: REST endpoints with Pydantic validation
- **CORS**: Cross-origin request handling
- **Static Files**: Serves frontend assets

### **RAG Orchestration**
- **RAG System**: Main coordinator between components
- **Query Processing**: Formats prompts and manages flow
- **Response Assembly**: Combines AI output with sources

### **AI & Tools**
- **AI Generator**: Anthropic Claude integration
- **Tool Manager**: Registers and executes search tools
- **Search Tool**: Course-aware semantic search
- **Tool Execution**: Two-phase Claude API calls

### **Data Layer**
- **Vector Store**: ChromaDB with sentence-transformers
- **Session Manager**: Conversation history tracking
- **Document Processing**: Chunked course content

### **Key Decision Points**
1. **Tool Usage**: Claude decides when to search vs. use knowledge
2. **Search Strategy**: Course/lesson filtering based on query
3. **Context Assembly**: Combines search results with conversation history
4. **Source Tracking**: Maintains provenance for user transparency

The diagram shows how the system intelligently routes queries through semantic search when needed while maintaining conversation context and providing transparent source attribution.