import pytest
import tempfile
import shutil
import os
from config import Config
from rag_system import RAGSystem


class TestRealSystemIntegration:
    """Test the real system with actual components to identify issues"""

    @pytest.fixture
    def real_test_config(self):
        """Create config for real system testing with temp directory"""
        temp_dir = tempfile.mkdtemp()
        config = Config()
        config.CHROMA_PATH = temp_dir
        config.ANTHROPIC_API_KEY = "test-key"  # This will fail but let's see where
        yield config
        shutil.rmtree(temp_dir)

    def test_system_initialization(self, real_test_config):
        """Test if the system can be initialized with real components"""
        try:
            rag_system = RAGSystem(real_test_config)
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.tool_manager is not None
            assert rag_system.search_tool is not None
            print("✓ System initialization successful")
        except Exception as e:
            print(f"✗ System initialization failed: {e}")
            raise

    def test_course_loading(self, real_test_config):
        """Test loading actual course documents"""
        try:
            rag_system = RAGSystem(real_test_config)

            docs_path = "../docs"
            if os.path.exists(docs_path):
                courses, chunks = rag_system.add_course_folder(docs_path)
                print(f"✓ Loaded {courses} courses with {chunks} chunks")
                assert courses > 0, "No courses were loaded"
                assert chunks > 0, "No chunks were created"
            else:
                print("✗ Docs folder not found")
                pytest.skip("Docs folder not available")

        except Exception as e:
            print(f"✗ Course loading failed: {e}")
            raise

    def test_vector_store_search(self, real_test_config):
        """Test vector store search functionality directly"""
        try:
            rag_system = RAGSystem(real_test_config)

            # Add some test content first
            docs_path = "../docs"
            if os.path.exists(docs_path):
                courses, chunks = rag_system.add_course_folder(docs_path)
                print(f"Loaded {courses} courses for search test")

                if chunks > 0:
                    # Test direct vector store search
                    results = rag_system.vector_store.search("programming")

                    if results.error:
                        print(f"✗ Vector store search error: {results.error}")
                        raise Exception(f"Vector store search failed: {results.error}")
                    else:
                        print(f"✓ Vector store search returned {len(results.documents)} results")
                        print(f"Sample result: {results.documents[0][:100] if results.documents else 'No documents'}")
                else:
                    print("✗ No chunks loaded for search test")
            else:
                pytest.skip("Docs folder not available")

        except Exception as e:
            print(f"✗ Vector store search test failed: {e}")
            raise

    def test_search_tool_execution(self, real_test_config):
        """Test CourseSearchTool execution with real data"""
        try:
            rag_system = RAGSystem(real_test_config)

            # Load courses
            docs_path = "../docs"
            if os.path.exists(docs_path):
                courses, chunks = rag_system.add_course_folder(docs_path)
                print(f"Loaded {courses} courses for search tool test")

                if chunks > 0:
                    # Test search tool execution
                    result = rag_system.search_tool.execute("programming")
                    print(f"✓ Search tool result: {result[:200]}...")

                    # Check if it returned an error message
                    if "error" in result.lower() or "failed" in result.lower():
                        print(f"✗ Search tool returned error: {result}")
                        raise Exception(f"Search tool execution failed: {result}")
                    else:
                        print("✓ Search tool execution successful")
                else:
                    print("✗ No chunks loaded for search tool test")
            else:
                pytest.skip("Docs folder not available")

        except Exception as e:
            print(f"✗ Search tool execution test failed: {e}")
            raise

    def test_tool_manager_execution(self, real_test_config):
        """Test ToolManager execution with real tools"""
        try:
            rag_system = RAGSystem(real_test_config)

            # Load courses
            docs_path = "../docs"
            if os.path.exists(docs_path):
                courses, chunks = rag_system.add_course_folder(docs_path)
                print(f"Loaded {courses} courses for tool manager test")

                if chunks > 0:
                    # Test tool manager execution
                    result = rag_system.tool_manager.execute_tool(
                        "search_course_content",
                        query="programming"
                    )
                    print(f"✓ Tool manager result: {result[:200]}...")

                    # Check for error patterns
                    if "error" in result.lower() or "failed" in result.lower():
                        print(f"✗ Tool manager returned error: {result}")
                        raise Exception(f"Tool manager execution failed: {result}")
                    else:
                        print("✓ Tool manager execution successful")

                    # Test sources
                    sources = rag_system.tool_manager.get_last_sources()
                    print(f"✓ Sources retrieved: {len(sources)} sources")
                    for i, source in enumerate(sources[:3]):  # Show first 3
                        print(f"  Source {i+1}: {source}")
                else:
                    print("✗ No chunks loaded for tool manager test")
            else:
                pytest.skip("Docs folder not available")

        except Exception as e:
            print(f"✗ Tool manager execution test failed: {e}")
            raise

    def test_ai_generator_without_api_key(self, real_test_config):
        """Test AI generator behavior when API key is invalid"""
        try:
            rag_system = RAGSystem(real_test_config)

            # This should fail because we don't have a real API key
            # But let's see what error we get
            try:
                response = rag_system.ai_generator.generate_response(
                    "What is Python?",
                    tools=rag_system.tool_manager.get_tool_definitions(),
                    tool_manager=rag_system.tool_manager
                )
                print(f"✗ Unexpected success: {response}")
            except Exception as ai_error:
                print(f"✓ Expected AI error (no valid API key): {type(ai_error).__name__}: {ai_error}")
                # This is expected - we don't have a real API key

        except Exception as e:
            print(f"✗ AI generator test setup failed: {e}")
            raise

    def test_check_dependencies(self):
        """Test that all required dependencies are available"""
        try:
            import chromadb
            print("✓ chromadb available")
        except ImportError as e:
            print(f"✗ chromadb missing: {e}")

        try:
            import anthropic
            print("✓ anthropic available")
        except ImportError as e:
            print(f"✗ anthropic missing: {e}")

        try:
            import sentence_transformers
            print("✓ sentence_transformers available")
        except ImportError as e:
            print(f"✗ sentence_transformers missing: {e}")

    def test_environment_setup(self):
        """Test environment configuration"""
        config = Config()
        print(f"API Key set: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
        print(f"Model: {config.ANTHROPIC_MODEL}")
        print(f"Embedding model: {config.EMBEDDING_MODEL}")
        print(f"Chunk size: {config.CHUNK_SIZE}")
        print(f"Max results: {config.MAX_RESULTS}")

    def test_document_processor_directly(self, real_test_config):
        """Test document processor with real files"""
        try:
            rag_system = RAGSystem(real_test_config)

            docs_path = "../docs"
            if os.path.exists(docs_path):
                files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
                if files:
                    test_file = os.path.join(docs_path, files[0])
                    print(f"Testing document processor with: {test_file}")

                    course, chunks = rag_system.document_processor.process_course_document(test_file)

                    if course:
                        print(f"✓ Course processed: {course.title}")
                        print(f"✓ Chunks created: {len(chunks)}")
                        print(f"✓ Sample chunk: {chunks[0].content[:100] if chunks else 'No chunks'}...")
                    else:
                        print("✗ No course data returned from document processor")
                else:
                    print("✗ No text files found in docs folder")
            else:
                pytest.skip("Docs folder not available")

        except Exception as e:
            print(f"✗ Document processor test failed: {e}")
            raise