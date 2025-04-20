import os
import json
import logging
from typing import List, Dict, Any, Optional
import re
import hashlib

import spacy
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Process extracted text from annual reports: chunking, cleaning, and embedding.
    """
    
    def __init__(self, 
                 processed_dir: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the text processor.
        
        Args:
            processed_dir: Directory containing processed report data
            embedding_model: Name of the sentence-transformers model to use
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.processed_dir = processed_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load NLP models
        logger.info(f"Loading NLP models: spaCy and {embedding_model}")
        try:
            self.nlp = spacy.load("en_core_web_md")
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.error(f"Error loading NLP models: {str(e)}")
            raise
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Chunk storage
        self.chunks = []
        self.embeddings = []
        self.section_to_chunks = {}
        
        logger.info(f"Initialized text processor for {processed_dir}")
    
    def load_processed_text(self) -> Dict[str, str]:
        """
        Load processed text from sections directory.
        
        Returns:
            Dictionary mapping section names to text content
        """
        sections_dir = os.path.join(self.processed_dir, "sections")
        sections_content = {}
        
        if not os.path.exists(sections_dir):
            logger.warning(f"Sections directory not found: {sections_dir}")
            
            # Fall back to full text file
            full_text_path = os.path.join(self.processed_dir, "full_text.txt")
            if os.path.exists(full_text_path):
                with open(full_text_path, 'r', encoding='utf-8') as f:
                    sections_content["full_text"] = f.read()
                logger.info(f"Loaded full text file: {full_text_path}")
            else:
                logger.error(f"No text content found in {self.processed_dir}")
                return {}
                
            return sections_content
        
        # Load each section file
        for filename in os.listdir(sections_dir):
            if filename.endswith(".txt"):
                section_name = os.path.splitext(filename)[0].replace('_', ' ')
                file_path = os.path.join(sections_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    sections_content[section_name] = f.read()
                
                logger.info(f"Loaded section: {section_name} ({len(sections_content[section_name])} chars)")
        
        return sections_content
    
    def load_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load extracted tables from CSV files.
        
        Returns:
            Dictionary mapping table names to pandas DataFrames
        """
        tables = {}
        
        # Look for CSV files in the processed directory
        for filename in os.listdir(self.processed_dir):
            if filename.startswith("table_") and filename.endswith(".csv"):
                table_name = os.path.splitext(filename)[0]
                file_path = os.path.join(self.processed_dir, filename)
                
                try:
                    df = pd.read_csv(file_path)
                    tables[table_name] = df
                    logger.info(f"Loaded table: {table_name} ({df.shape})")
                except Exception as e:
                    logger.error(f"Error loading table {table_name}: {str(e)}")
        
        return tables
    
    def clean_text(self, text: str) -> str:
        """
        Clean text before chunking and embedding.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove page markers
        text = re.sub(r'---\s*Page\s+\d+\s*---', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove header/footer patterns (common in annual reports)
        text = re.sub(r'\d{1,3}\s*[|]\s*[A-Za-z\s]+Annual Report \d{4}', '', text)
        text = re.sub(r'Annual Report \d{4}\s*[|]\s*[A-Za-z\s]+', '', text)
        
        # Clean up special characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        return text.strip()
    
    def create_chunks(self, sections_content: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Create chunks from section content with metadata.
        
        Args:
            sections_content: Dictionary mapping section names to text content
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        all_chunks = []
        section_to_chunks = {}
        
        for section_name, content in sections_content.items():
            # Clean the text
            cleaned_text = self.clean_text(content)
            
            # Split into chunks
            text_chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create chunk documents with metadata
            section_chunks = []
            for i, chunk_text in enumerate(text_chunks):
                # Create a unique ID for the chunk
                chunk_id = hashlib.md5(f"{section_name}_{i}_{chunk_text[:50]}".encode()).hexdigest()
                
                chunk = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "section": section_name,
                        "chunk_index": i,
                        "source": "section_text"
                    }
                }
                
                section_chunks.append(chunk)
                all_chunks.append(chunk)
            
            # Map section to its chunks
            section_to_chunks[section_name] = [chunk["id"] for chunk in section_chunks]
            logger.info(f"Created {len(section_chunks)} chunks for section: {section_name}")
        
        self.chunks = all_chunks
        self.section_to_chunks = section_to_chunks
        logger.info(f"Created {len(all_chunks)} total chunks from {len(sections_content)} sections")
        
        return all_chunks
    
    def process_tables_to_text(self, tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Convert tables to textual representations and create chunks.
        
        Args:
            tables: Dictionary mapping table names to pandas DataFrames
            
        Returns:
            List of table chunk dictionaries
        """
        table_chunks = []
        
        for table_name, df in tables.items():
            # Extract page number from table name (format: table_PAGE_INDEX)
            page_match = re.search(r'table_(\d+)_', table_name)
            page_num = int(page_match.group(1)) if page_match else 0
            
            # Create a textual representation of the table
            table_text = f"Table from page {page_num}:\n"
            
            # Add column headers
            headers = df.columns.tolist()
            table_text += " | ".join(str(h) for h in headers) + "\n"
            
            # Add rows (limit to 20 rows max to avoid too much text)
            for i, row in df.iloc[:20].iterrows():
                table_text += " | ".join(str(val) for val in row.tolist()) + "\n"
            
            if len(df) > 20:
                table_text += f"... and {len(df) - 20} more rows\n"
            
            # Add summary statistics for numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                table_text += "\nNumerical summary:\n"
                for col in num_cols:
                    if df[col].notna().any():
                        table_text += f"{col}: Min={df[col].min():.2f}, Max={df[col].max():.2f}, Mean={df[col].mean():.2f}\n"
            
            # Create chunk for the table
            chunk_id = hashlib.md5(f"{table_name}_{table_text[:50]}".encode()).hexdigest()
            
            chunk = {
                "id": chunk_id,
                "text": table_text,
                "metadata": {
                    "section": "financial_tables",
                    "table_name": table_name,
                    "page": page_num,
                    "source": "table"
                }
            }
            
            table_chunks.append(chunk)
            logger.info(f"Created text representation for table: {table_name}")
        
        # Append table chunks to main chunks list
        self.chunks.extend(table_chunks)
        logger.info(f"Created {len(table_chunks)} chunks from tables")
        
        return table_chunks
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all text chunks.
        
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(self.chunks)} chunks")
        
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in self.chunks]
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # Log progress
            if (i + batch_size) % 100 == 0 or i + batch_size >= len(texts):
                logger.info(f"Generated embeddings for {min(i+batch_size, len(texts))}/{len(texts)} chunks")
        
        # Combine all batches
        self.embeddings = np.vstack(all_embeddings)
        logger.info(f"Completed embedding generation, shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def save_processed_data(self, output_dir: Optional[str] = None) -> str:
        """
        Save processed chunks and embeddings to disk.
        
        Args:
            output_dir: Directory to save the processed data (default: vector_store subdirectory)
            
        Returns:
            Path to the directory where data was saved
        """
        output_dir = output_dir or os.path.join(self.processed_dir, "vector_store")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunks as JSON
        chunks_path = os.path.join(output_dir, "chunks.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        # Save embeddings as numpy array
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, self.embeddings)
        
        # Save section to chunks mapping
        section_map_path = os.path.join(output_dir, "section_map.json")
        with open(section_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.section_to_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")
        return output_dir
    
    def process(self) -> Dict[str, Any]:
        """
        Run the full text processing pipeline.
        
        Returns:
            Dictionary with processing summary
        """
        logger.info(f"Starting full text processing pipeline for {self.processed_dir}")
        
        # Load section text
        sections_content = self.load_processed_text()
        if not sections_content:
            logger.error("No text content found, aborting processing")
            return {"status": "error", "message": "No text content found"}
        
        # Create chunks from section text
        self.create_chunks(sections_content)
        
        # Load and process tables
        tables = self.load_tables()
        if tables:
            self.process_tables_to_text(tables)
        
        # Generate embeddings
        self.generate_embeddings()
        
        # Save processed data
        output_dir = self.save_processed_data()
        
        # Return processing summary
        summary = {
            "status": "success",
            "chunks_count": len(self.chunks),
            "embeddings_shape": self.embeddings.shape,
            "sections_count": len(sections_content),
            "tables_count": len(tables),
            "output_dir": output_dir
        }
        
        logger.info(f"Completed text processing pipeline: {summary}")
        return summary


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python text_processor.py <path_to_processed_report_dir>")
        sys.exit(1)
    
    processed_dir = sys.argv[1]
    processor = TextProcessor(processed_dir)
    results = processor.process()
    
    print(f"Processing complete: {results}")