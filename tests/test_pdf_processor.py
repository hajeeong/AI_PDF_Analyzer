import unittest
import os
import tempfile
import shutil
from src.pdf_processor.processor import PDFProcessor

class TestPDFProcessor(unittest.TestCase):
    """Test cases for the PDF Processor module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Sample PDF path - update this to point to a test PDF
        # In a real test environment, we would have a small test PDF
        self.sample_pdf_path = os.path.join('tests', 'data', 'sample_annual_report.pdf')
        
        # Skip tests if the sample PDF doesn't exist
        if not os.path.exists(self.sample_pdf_path):
            self.skipTest("Sample PDF not found. Place a test PDF at tests/data/sample_annual_report.pdf")
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test initialization of PDFProcessor."""
        processor = PDFProcessor(self.sample_pdf_path, self.test_dir)
        
        # Check attributes
        self.assertEqual(processor.report_path, self.sample_pdf_path)
        self.assertTrue(os.path.exists(processor.output_dir))
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        processor = PDFProcessor(self.sample_pdf_path, self.test_dir)
        metadata = processor.extract_metadata_with_pypdf2()
        
        # Check that metadata has been extracted
        self.assertIsInstance(metadata, dict)
        self.assertIn('total_pages', metadata)
        self.assertGreater(metadata['total_pages'], 0)
    
    def test_extract_text(self):
        """Test text extraction."""
        processor = PDFProcessor(self.sample_pdf_path, self.test_dir)
        text_content = processor.extract_text_with_pdfplumber()
        
        # Check that text has been extracted
        self.assertIsInstance(text_content, dict)
        self.assertGreater(len(text_content), 0)
        
        # Check content of first page
        first_page = text_content.get(1, '')
        self.assertGreater(len(first_page), 0)
    
    def test_identify_sections(self):
        """Test section identification."""
        processor = PDFProcessor(self.sample_pdf_path, self.test_dir)
        
        # Need to extract text first
        processor.extract_text_with_pdfplumber()
        
        # Identify sections
        sections = processor.identify_sections()
        
        # Check sections
        self.assertIsInstance(sections, dict)
        # We can't guarantee any specific sections in a generic test
        # but the function should return something
    
    def test_extract_tables(self):
        """Test table extraction."""
        processor = PDFProcessor(self.sample_pdf_path, self.test_dir)
        tables = processor.extract_tables()
        
        # Check tables
        self.assertIsInstance(tables, dict)
        # Even if no tables are found, it should return an empty dict
    
    def test_process_report(self):
        """Test full report processing."""
        processor = PDFProcessor(self.sample_pdf_path, self.test_dir)
        results = processor.process_report()
        
        # Check results
        self.assertIsInstance(results, dict)
        self.assertIn('metadata', results)
        self.assertIn('sections', results)
        self.assertIn('tables', results)
        self.assertIn('output_dir', results)
        
        # Check output files
        self.assertTrue(os.path.exists(os.path.join(processor.output_dir, "full_text.txt")))


# tests/test_text_processor.py
import unittest
import os
import tempfile
import shutil
import numpy as np
import json
from src.retrieval.text_processor import TextProcessor

class TestTextProcessor(unittest.TestCase):
    """Test cases for the Text Processor module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test processed directory with sample data
        self.processed_dir = os.path.join(self.test_dir, "processed")
        os.makedirs(os.path.join(self.processed_dir, "sections"), exist_ok=True)
        
        # Create sample section content
        with open(os.path.join(self.processed_dir, "sections", "sample_section.txt"), 'w') as f:
            f.write("This is a sample section content for testing purposes. " * 10)
        
        # Create a sample full text file
        with open(os.path.join(self.processed_dir, "full_text.txt"), 'w') as f:
            f.write("--- Page 1 ---\n")
            f.write("This is page 1 content for testing purposes. " * 10)
            f.write("\n\n--- Page 2 ---\n")
            f.write("This is page 2 content for testing purposes. " * 10)
        
        # Create a sample table CSV
        with open(os.path.join(self.processed_dir, "table_1_1.csv"), 'w') as f:
            f.write("Column1,Column2,Column3\n")
            f.write("Value1,Value2,Value3\n")
            f.write("Value4,Value5,Value6\n")
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test initialization of TextProcessor."""
        processor = TextProcessor(self.processed_dir)
        
        # Check attributes
        self.assertEqual(processor.processed_dir, self.processed_dir)
        self.assertIsNotNone(processor.nlp)
        self.assertIsNotNone(processor.embedding_model)
    
    def test_load_processed_text(self):
        """Test loading processed text."""
        processor = TextProcessor(self.processed_dir)
        sections_content = processor.load_processed_text()
        
        # Check loaded content
        self.assertIsInstance(sections_content, dict)
        self.assertGreater(len(sections_content), 0)
        self.assertIn("sample section", sections_content)
    
    def test_load_tables(self):
        """Test loading tables."""
        processor = TextProcessor(self.processed_dir)
        tables = processor.load_tables()
        
        # Check loaded tables
        self.assertIsInstance(tables, dict)
        self.assertGreater(len(tables), 0)
        self.assertIn("table_1_1", tables)
    
    def test_clean_text(self):
        """Test text cleaning."""
        processor = TextProcessor(self.processed_dir)
        sample_text = "--- Page 1 ---\nThis is a test.    Extra spaces should be removed.\n\n"
        cleaned_text = processor.clean_text(sample_text)
        
        # Check cleaned text
        self.assertNotIn("--- Page 1 ---", cleaned_text)
        self.assertNotIn("    ", cleaned_text)  # Extra spaces should be gone
        self.assertIn("This is a test.", cleaned_text)
    
    def test_create_chunks(self):
        """Test chunk creation."""
        processor = TextProcessor(self.processed_dir)
        sections_content = processor.load_processed_text()
        chunks = processor.create_chunks(sections_content)
        
        # Check chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # Check chunk structure
        first_chunk = chunks[0]
        self.assertIn("id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("metadata", first_chunk)
    
    def test_process_tables_to_text(self):
        """Test table to text conversion."""
        processor = TextProcessor(self.processed_dir)
        tables = processor.load_tables()
        table_chunks = processor.process_tables_to_text(tables)
        
        # Check table chunks
        self.assertIsInstance(table_chunks, list)
        self.assertGreater(len(table_chunks), 0)
        
        # Check chunk content
        first_chunk = table_chunks[0]
        self.assertIn("Table from page", first_chunk["text"])
        self.assertEqual(first_chunk["metadata"]["source"], "table")
    
    def test_generate_embeddings(self):
        """Test embedding generation."""
        processor = TextProcessor(self.processed_dir)
        
        # Create chunks first
        sections_content = processor.load_processed_text()
        processor.create_chunks(sections_content)
        
        # Generate embeddings
        embeddings = processor.generate_embeddings()
        
        # Check embeddings
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], len(processor.chunks))
    
    def test_save_processed_data(self):
        """Test saving processed data."""
        processor = TextProcessor(self.processed_dir)
        
        # Process the data
        sections_content = processor.load_processed_text()
        processor.create_chunks(sections_content)
        processor.generate_embeddings()
        
        # Save the data
        output_dir = processor.save_processed_data()
        
        # Check saved files
        self.assertTrue(os.path.exists(os.path.join(output_dir, "chunks.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "embeddings.npy")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "section_map.json")))
    
    def test_process(self):
        """Test full processing pipeline."""
        processor = TextProcessor(self.processed_dir)
        result = processor.process()
        
        # Check result
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["chunks_count"], 0)
        self.assertTrue(os.path.exists(result["output_dir"]))


# tests/test_vector_store.py
import unittest
import os
import tempfile
import shutil
import numpy as np
import json
from src.retrieval.vector_store import LocalVectorStore

class TestLocalVectorStore(unittest.TestCase):
    """Test cases for the Local Vector Store module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create vector store directory structure
        self.vector_store_dir = os.path.join(self.test_dir, "vector_store")
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        # Create sample chunks
        self.sample_chunks = [
            {
                "id": "chunk1",
                "text": "This is a test chunk about financial performance.",
                "metadata": {
                    "section": "financial_overview",
                    "source": "text",
                    "chunk_index": 0
                }
            },
            {
                "id": "chunk2",
                "text": "This chunk discusses risk factors in detail.",
                "metadata": {
                    "section": "risk_factors",
                    "source": "text",
                    "chunk_index": 0
                }
            },
            {
                "id": "chunk3",
                "text": "Revenue increased by 15% compared to last year.",
                "metadata": {
                    "section": "financial_overview",
                    "source": "text",
                    "chunk_index": 1
                }
            }
        ]
        
        # Create sample embeddings (random for testing)
        self.sample_embeddings = np.random.rand(3, 384)  # 3 chunks, 384-dim embeddings
        
        # Create sample section map
        self.sample_section_map = {
            "financial_overview": ["chunk1", "chunk3"],
            "risk_factors": ["chunk2"]
        }
        
        # Save the sample data
        with open(os.path.join(self.vector_store_dir, "chunks.json"), 'w') as f:
            json.dump(self.sample_chunks, f)
        
        np.save(os.path.join(self.vector_store_dir, "embeddings.npy"), self.sample_embeddings)
        
        with open(os.path.join(self.vector_store_dir, "section_map.json"), 'w') as f:
            json.dump(self.sample_section_map, f)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test initialization of LocalVectorStore."""
        store = LocalVectorStore(self.vector_store_dir)
        
        # Check attributes
        self.assertEqual(store.vector_store_dir, self.vector_store_dir)
        self.assertIsNotNone(store.embedding_model)
        self.assertIsNotNone(store.embeddings)
        self.assertEqual(len(store.chunks), 3)
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        store = LocalVectorStore(self.vector_store_dir)
        results = store.similarity_search("financial performance", top_k=2)
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)  # Could be less if threshold is applied
        
        # Check result structure
        if results:
            first_result = results[0]
            self.assertIn("id", first_result)
            self.assertIn("text", first_result)
            self.assertIn("metadata", first_result)
            self.assertIn("similarity", first_result)
    
    def test_search_by_section(self):
        """Test search within a specific section."""
        store = LocalVectorStore(self.vector_store_dir)
        results = store.search_by_section("financial_overview", "revenue")
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # All results should be from the financial_overview section
        for result in results:
            self.assertEqual(result["metadata"]["section"], "financial_overview")
    
    def test_list_sections(self):
        """Test listing available sections."""
        store = LocalVectorStore(self.vector_store_dir)
        sections = store.list_sections()
        
        # Check sections
        self.assertIsInstance(sections, list)
        self.assertEqual(set(sections), {"financial_overview", "risk_factors"})
    
    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        store = LocalVectorStore(self.vector_store_dir)
        results = store.hybrid_search("financial revenue", top_k=2)
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        # Check result structure
        if results:
            first_result = results[0]
            self.assertIn("similarity", first_result)


# tests/test_query_processor.py
import unittest
import os
import json
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch
from src.retrieval.vector_store import LocalVectorStore
from src.ai_interface.query_processor import QueryProcessor

class TestQueryProcessor(unittest.TestCase):
    """Test cases for the Query Processor module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock vector store
        self.vector_store = MagicMock(spec=LocalVectorStore)
        
        # Set up mock search results
        self.mock_search_results = [
            {
                "id": "chunk1",
                "text": "The company's revenue increased by 15% to $1.2 billion.",
                "metadata": {
                    "section": "financial_highlights",
                    "source": "text",
                    "chunk_index": 0
                },
                "similarity": 0.92
            },
            {
                "id": "chunk2",
                "text": "Key risk factors include market volatility and regulatory changes.",
                "metadata": {
                    "section": "risk_factors",
                    "source": "text",
                    "chunk_index": 0
                },
                "similarity": 0.85
            }
        ]
        
        # Configure vector store mock
        self.vector_store.similarity_search.return_value = self.mock_search_results
        self.vector_store.hybrid_search.return_value = self.mock_search_results
        self.vector_store.list_sections.return_value = ["financial_highlights", "risk_factors"]
        self.vector_store.search_by_section.return_value = [self.mock_search_results[0]]
        
        # Mock OpenAI API responses
        self.openai_patcher = patch('openai.ChatCompletion.create')
        self.mock_openai = self.openai_patcher.start()
        
        # Configure OpenAI mock
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a mock AI response."
        self.mock_openai.return_value = mock_response
    
    def tearDown(self):
        """Clean up test environment."""
        # Stop patches
        self.openai_patcher.stop()
    
    def test_init(self):
        """Test initialization of QueryProcessor."""
        processor = QueryProcessor(self.vector_store)
        
        # Check attributes
        self.assertEqual(processor.vector_store, self.vector_store)
        self.assertIsNotNone(processor.prompts)
    
    def test_classify_query(self):
        """Test query classification."""
        processor = QueryProcessor(self.vector_store)
        
        # Test different query types
        self.assertEqual(processor.classify_query("What was the company's revenue last year?"), "financial")
        self.assertEqual(processor.classify_query("What are the main risk factors?"), "risk")
        self.assertEqual(processor.classify_query("Summarize the annual report."), "summary")
        self.assertEqual(processor.classify_query("Compare this year's performance to last year."), "comparison")
        self.assertEqual(processor.classify_query("Who is the CEO?"), "general")
    
    def test_retrieve_relevant_chunks(self):
        """Test retrieving relevant chunks for a query."""
        processor = QueryProcessor(self.vector_store)
        
        # Test general query retrieval
        result = processor.retrieve_relevant_chunks("What was the revenue?", "general")
        self.assertIn("context", result)
        self.assertIn("chunks", result)
        
        # Test financial query retrieval
        result = processor.retrieve_relevant_chunks("What was the profit margin?", "financial")
        self.assertIn("financial_data", result)
        self.assertIn("context", result)
        
        # Vector store methods should have been called
        self.vector_store.hybrid_search.assert_called()
    
    def test_format_context(self):
        """Test context formatting."""
        processor = QueryProcessor(self.vector_store)
        formatted = processor._format_context(self.mock_search_results)
        
        # Check that each chunk is included in the formatted context
        for chunk in self.mock_search_results:
            self.assertIn(chunk["text"], formatted)
    
    def test_generate_response(self):
        """Test response generation."""
        processor = QueryProcessor(self.vector_store)
        retrieved_info = {
            "context": "Sample context information.",
            "chunks": self.mock_search_results
        }
        
        response = processor.generate_response("What was the revenue?", retrieved_info)
        
        # Check response
        self.assertEqual(response, "This is a mock AI response.")
        
        # OpenAI API should have been called
        self.mock_openai.assert_called_once()
    
    def test_process_query(self):
        """Test full query processing."""
        processor = QueryProcessor(self.vector_store)
        result = processor.process_query("What was the company's revenue?")
        
        # Check result structure
        self.assertIn("query", result)
        self.assertIn("query_type", result)
        self.assertIn("response", result)
        self.assertIn("supporting_info", result)
        
        # Query should be classified
        self.assertEqual(result["query_type"], "financial")
        
        # Response should be generated
        self.assertEqual(result["response"], "This is a mock AI response.")


# Run the tests
if __name__ == '__main__':
    unittest.main()