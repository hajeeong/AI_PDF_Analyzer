import os
import re
from typing import Dict, List, Tuple, Any, Optional
import logging

import PyPDF2
import pdfplumber
import camelot
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Class for processing PDF annual reports, extracting text content and tables,
    and structuring the information for further analysis.
    """
    
    def __init__(self, report_path: str, output_dir: str = None):
        """
        Initialize the PDF processor with the path to the annual report.
        
        Args:
            report_path: Path to the PDF annual report file
            output_dir: Directory to save processed outputs (default: None)
        """
        self.report_path = report_path
        self.report_name = os.path.basename(report_path).split('.')[0]
        self.output_dir = output_dir or os.path.join('data', 'processed', self.report_name)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Document metadata
        self.metadata = {}
        self.total_pages = 0
        
        # Extracted content
        self.text_content = {}  # Dictionary mapping page numbers to text content
        self.section_map = {}   # Dictionary mapping section names to page ranges
        self.tables = {}        # Dictionary mapping table identifiers to pandas DataFrames
        
        logger.info(f"Initialized PDF processor for {self.report_name}")
    
    def extract_text_with_pdfplumber(self) -> Dict[int, str]:
        """
        Extract text content from the PDF using pdfplumber, which preserves more formatting.
        
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        logger.info(f"Extracting text with pdfplumber from {self.report_path}")
        text_content = {}
        
        try:
            with pdfplumber.open(self.report_path) as pdf:
                self.total_pages = len(pdf.pages)
                self.metadata['total_pages'] = self.total_pages
                
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    text_content[i+1] = page_text  # 1-indexed page numbers
                    
                    # Log progress every 10 pages
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i+1}/{self.total_pages} pages")
        
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {str(e)}")
            raise
        
        self.text_content = text_content
        logger.info(f"Completed text extraction, processed {self.total_pages} pages")
        
        return text_content
    
    def extract_metadata_with_pypdf2(self) -> Dict[str, Any]:
        """
        Extract PDF metadata using PyPDF2.
        
        Returns:
            Dictionary containing PDF metadata
        """
        logger.info(f"Extracting metadata from {self.report_path}")
        
        try:
            with open(self.report_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Basic metadata
                info = reader.metadata
                if info:
                    self.metadata.update({
                        'title': info.get('/Title', ''),
                        'author': info.get('/Author', ''),
                        'creator': info.get('/Creator', ''),
                        'producer': info.get('/Producer', ''),
                        'subject': info.get('/Subject', ''),
                        'creation_date': info.get('/CreationDate', ''),
                        'modification_date': info.get('/ModDate', '')
                    })
                
                # Page count
                self.total_pages = len(reader.pages)
                self.metadata['total_pages'] = self.total_pages
                
        except Exception as e:
            logger.error(f"Error extracting metadata with PyPDF2: {str(e)}")
            raise
        
        logger.info(f"Completed metadata extraction")
        return self.metadata
    
    def identify_sections(self) -> Dict[str, Tuple[int, int]]:
        """
        Identify document sections using common section titles in annual reports.
        
        Returns:
            Dictionary mapping section names to page ranges (start_page, end_page)
        """
        logger.info(f"Identifying sections in the report")
        
        # Common section titles in annual reports
        section_patterns = [
            r"(Management['']s Discussion and Analysis|MD&A)",
            r"(Risk Factors|Principal Risks|Key Risks)",
            r"(Financial Statements|Consolidated Financial Statements)",
            r"(Notes to .*Financial Statements)",
            r"(Independent Auditor['']s Report|Auditor['']s Report)",
            r"(Chairman['']s Statement|Letter to Shareholders)",
            r"(Corporate Governance|Governance Report)",
            r"(Directors[''] Report)",
            r"(Business Review|Operational Review)",
            r"(Executive Compensation|Remuneration Report)",
            r"(Sustainability Report|Corporate Responsibility|ESG)"
        ]
        
        section_locations = {}
        toc_page = self._find_table_of_contents()
        
        # If we found a table of contents, use it to identify section page numbers
        if toc_page:
            logger.info(f"Using table of contents on page {toc_page} to identify sections")
            section_locations = self._extract_sections_from_toc(toc_page)
        
        # If TOC extraction failed or was incomplete, fall back to pattern matching
        if not section_locations:
            logger.info("Identifying sections through pattern matching")
            
            # First pass: identify section start pages
            section_starts = {}
            for page_num, content in self.text_content.items():
                # Look for section headers at the beginning of a page or after blank lines
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if the line matches any section pattern
                    for pattern in section_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            section_name = re.search(pattern, line, re.IGNORECASE).group(0)
                            section_starts[section_name] = page_num
                            logger.info(f"Found section '{section_name}' starting on page {page_num}")
                            break
            
            # Second pass: create section ranges
            sorted_sections = sorted(section_starts.items(), key=lambda x: x[1])
            for i, (section, start_page) in enumerate(sorted_sections):
                if i < len(sorted_sections) - 1:
                    end_page = sorted_sections[i+1][1] - 1
                else:
                    end_page = self.total_pages
                
                section_locations[section] = (start_page, end_page)
        
        self.section_map = section_locations
        logger.info(f"Identified {len(section_locations)} sections")
        
        return section_locations
    
    def _find_table_of_contents(self) -> Optional[int]:
        """
        Find the table of contents page in the document.
        
        Returns:
            Page number of table of contents, or None if not found
        """
        toc_patterns = [
            r"table\s+of\s+contents",
            r"contents",
            r"index",
        ]
        
        # Check the first 15 pages for a TOC
        for page_num in range(1, min(16, self.total_pages + 1)):
            if page_num not in self.text_content:
                continue
                
            content = self.text_content[page_num].lower()
            
            for pattern in toc_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Verify it's actually a TOC by checking for page numbers
                    if re.search(r'\b\d+\b', content) and len(content.split('\n')) > 5:
                        logger.info(f"Found table of contents on page {page_num}")
                        return page_num
        
        logger.info("No table of contents found")
        return None
    
    def _extract_sections_from_toc(self, toc_page: int) -> Dict[str, Tuple[int, int]]:
        """
        Extract section page numbers from the table of contents.
        
        Args:
            toc_page: Page number containing the table of contents
            
        Returns:
            Dictionary mapping section names to page ranges
        """
        section_locations = {}
        toc_content = self.text_content[toc_page]
        
        # Extract lines that contain a section title and page number
        # Typical format: "Section Title..............42"
        toc_lines = toc_content.split('\n')
        for line in toc_lines:
            # Look for patterns where there's text followed by numbers at the end
            match = re.search(r'(.*[a-zA-Z].*)\s+(\d+)$', line.strip())
            if match:
                section_title = match.group(1).strip()
                page_num = int(match.group(2))
                
                # Only include if the page number is valid
                if 1 <= page_num <= self.total_pages:
                    section_locations[section_title] = (page_num, None)  # End page will be set later
        
        # Fill in end pages
        sorted_sections = sorted(section_locations.items(), key=lambda x: x[1][0])
        for i, (section, (start_page, _)) in enumerate(sorted_sections):
            if i < len(sorted_sections) - 1:
                end_page = sorted_sections[i+1][1][0] - 1
            else:
                end_page = self.total_pages
            
            section_locations[section] = (start_page, end_page)
        
        return section_locations
    
    def extract_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Extract tables from the PDF using Camelot.
        
        Returns:
            Dictionary mapping table identifiers to pandas DataFrames
        """
        logger.info(f"Extracting tables from {self.report_path}")
        tables_dict = {}
        
        # Focus on financial statement sections if identified
        financial_pages = []
        for section, (start, end) in self.section_map.items():
            if any(financial_term in section.lower() for financial_term in 
                   ['financial statement', 'balance sheet', 'income statement', 'cash flow']):
                financial_pages.extend(range(start, end + 1))
        
        # If no financial sections identified, use the entire document
        pages_to_check = financial_pages or range(1, self.total_pages + 1)
        
        try:
            # First try to extract tables from specified pages
            page_string = ','.join(str(p) for p in pages_to_check)
            tables = camelot.read_pdf(self.report_path, pages=page_string, flavor='lattice')
            
            logger.info(f"Found {len(tables)} tables using lattice mode")
            
            # If lattice mode didn't find many tables, try stream mode for text-based tables
            if len(tables) < 5:
                logger.info("Few tables found with lattice mode, trying stream mode")
                tables_stream = camelot.read_pdf(self.report_path, pages=page_string, flavor='stream')
                logger.info(f"Found {len(tables_stream)} additional tables using stream mode")
                
                # Merge the results
                tables.extend(tables_stream)
            
            # Process and store each table
            for i, table in enumerate(tables):
                # Generate a descriptive key for the table
                page_num = table.parsing_report['page']
                table_key = f"table_{page_num}_{i+1}"
                
                # Convert to pandas DataFrame and clean it
                df = table.df
                
                # Basic cleaning: remove empty rows and columns
                df = df.replace('', np.nan)
                df = df.dropna(how='all', axis=0)  # Drop rows that are all NaN
                df = df.dropna(how='all', axis=1)  # Drop columns that are all NaN
                
                # Use the first row as header if it doesn't look like data
                if not df.empty and not df.iloc[0].str.contains(r'\d+').any():
                    df.columns = df.iloc[0]
                    df = df.iloc[1:].reset_index(drop=True)
                
                # Store the processed table
                tables_dict[table_key] = df
                
                # Save table to CSV
                csv_path = os.path.join(self.output_dir, f"{table_key}.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved table {table_key} to {csv_path}")
                
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            # Continue with other processing even if table extraction fails
        
        self.tables = tables_dict
        logger.info(f"Completed table extraction, found {len(tables_dict)} tables")
        
        return tables_dict
    
    def process_report(self) -> Dict[str, Any]:
        """
        Process the entire report, extracting text, metadata, sections, and tables.
        
        Returns:
            Dictionary containing all processed information
        """
        logger.info(f"Starting full processing of {self.report_path}")
        
        # Extract metadata
        self.extract_metadata_with_pypdf2()
        
        # Extract text content
        self.extract_text_with_pdfplumber()
        
        # Identify document sections
        self.identify_sections()
        
        # Extract tables
        self.extract_tables()
        
        # Save the text content to files
        self._save_processed_text()
        
        # Create a summary of the processing results
        processing_summary = {
            'metadata': self.metadata,
            'sections': self.section_map,
            'tables': {k: v.shape for k, v in self.tables.items()},
            'output_dir': self.output_dir
        }
        
        logger.info(f"Completed processing of {self.report_path}")
        return processing_summary
    
    def _save_processed_text(self):
        """Save the processed text content to files."""
        # Save full text content
        full_text_path = os.path.join(self.output_dir, "full_text.txt")
        with open(full_text_path, 'w', encoding='utf-8') as f:
            for page_num in sorted(self.text_content.keys()):
                f.write(f"--- Page {page_num} ---\n")
                f.write(self.text_content[page_num])
                f.write("\n\n")
        
        # Save text by sections
        sections_dir = os.path.join(self.output_dir, "sections")
        os.makedirs(sections_dir, exist_ok=True)
        
        for section_name, (start_page, end_page) in self.section_map.items():
            # Create a valid filename
            section_filename = re.sub(r'[^\w\s-]', '', section_name).strip().replace(' ', '_')
            section_path = os.path.join(sections_dir, f"{section_filename}.txt")
            
            with open(section_path, 'w', encoding='utf-8') as f:
                for page_num in range(start_page, end_page + 1):
                    if page_num in self.text_content:
                        f.write(self.text_content[page_num])
                        f.write("\n\n")
        
        logger.info(f"Saved processed text to {self.output_dir}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python processor.py <path_to_annual_report.pdf>")
        sys.exit(1)
    
    report_path = sys.argv[1]
    processor = PDFProcessor(report_path)
    results = processor.process_report()
    
    print(f"Processing complete. Results saved to {results['output_dir']}")
    print(f"Found {len(results['sections'])} sections and {len(results['tables'])} tables.")