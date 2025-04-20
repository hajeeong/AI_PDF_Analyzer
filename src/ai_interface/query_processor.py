import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import re

import openai
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Process user queries and generate responses using LLM and retrieved document chunks.
    """
    
    def __init__(self, vector_store, 
                 model_name: str = "gpt-3.5-turbo", 
                 temperature: float = 0.3,
                 api_key: Optional[str] = None):
        """
        Initialize the query processor.
        
        Args:
            vector_store: Vector store instance for retrieving relevant chunks
            model_name: LLM model name
            temperature: Temperature parameter for LLM generation
            api_key: OpenAI API key (if not set in environment)
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        
        # Set up OpenAI API
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            if not openai.api_key:
                logger.warning("OpenAI API key not found in environment variables or constructor")
        
        # Load prompt templates
        self.prompts = self._load_prompts()
        
        logger.info(f"Initialized query processor with model: {model_name}")
    
    def _load_prompts(self) -> Dict[str, PromptTemplate]:
        """
        Load prompt templates for different query types.
        
        Returns:
            Dictionary mapping prompt names to PromptTemplate objects
        """
        prompts = {}
        
        # General query prompt
        general_query_template = """
        You are an AI assistant specialized in analyzing annual reports. 
        You are given context information extracted from an annual report, and a question.
        Answer the question based ONLY on the information provided in the context.
        If the answer cannot be determined from the context, say so.
        
        CONTEXT INFORMATION:
        {context}
        
        QUESTION: {query}
        
        YOUR ANSWER:
        """
        
        # Financial analysis prompt
        financial_analysis_template = """
        You are an AI financial analyst specialized in analyzing annual reports.
        You are given financial data and context information extracted from an annual report, and a question.
        Provide a detailed financial analysis based ONLY on the information provided.
        Include relevant numerical data and trends in your answer.
        If the answer cannot be determined from the context, say so.
        
        FINANCIAL DATA:
        {financial_data}
        
        CONTEXT INFORMATION:
        {context}
        
        QUESTION: {query}
        
        YOUR FINANCIAL ANALYSIS:
        """
        
        # Risk assessment prompt
        risk_assessment_template = """
        You are an AI risk analyst specialized in analyzing annual reports.
        You are given risk information extracted from an annual report, and a question.
        Provide a detailed risk assessment based ONLY on the information provided.
        Be objective and factual in your assessment.
        If the answer cannot be determined from the context, say so.
        
        RISK INFORMATION:
        {risk_data}
        
        CONTEXT INFORMATION:
        {context}
        
        QUESTION: {query}
        
        YOUR RISK ASSESSMENT:
        """
        
        # Summary prompt
        summary_template = """
        You are an AI analyst specialized in summarizing annual reports.
        You are given context information extracted from an annual report.
        Provide a concise summary of the information provided.
        Be factual and highlight the most important points.
        
        CONTEXT INFORMATION:
        {context}
        
        YOUR SUMMARY:
        """
        
        # Comparison prompt
        comparison_template = """
        You are an AI analyst specialized in comparing annual reports.
        You are given information from two different reports or sections, and a question about the comparison.
        Provide a detailed comparison based ONLY on the information provided.
        Highlight similarities, differences, and trends.
        If the answer cannot be determined from the context, say so.
        
        INFORMATION FROM FIRST REPORT/SECTION:
        {context_1}
        
        INFORMATION FROM SECOND REPORT/SECTION:
        {context_2}
        
        QUESTION: {query}
        
        YOUR COMPARISON:
        """
        
        # Create prompt templates
        prompts["general"] = PromptTemplate(
            input_variables=["context", "query"],
            template=general_query_template
        )
        
        prompts["financial"] = PromptTemplate(
            input_variables=["financial_data", "context", "query"],
            template=financial_analysis_template
        )
        
        prompts["risk"] = PromptTemplate(
            input_variables=["risk_data", "context", "query"],
            template=risk_assessment_template
        )
        
        prompts["summary"] = PromptTemplate(
            input_variables=["context"],
            template=summary_template
        )
        
        prompts["comparison"] = PromptTemplate(
            input_variables=["context_1", "context_2", "query"],
            template=comparison_template
        )
        
        return prompts
    
    def classify_query(self, query: str) -> str:
        """
        Classify the query type to determine the appropriate prompt template.
        
        Args:
            query: User query
            
        Returns:
            Query type (general, financial, risk, summary, comparison)
        """
        query_lower = query.lower()
        
        # Check for financial query
        financial_terms = ['revenue', 'profit', 'earnings', 'income', 'margin', 'eps', 'dividend', 
                          'balance sheet', 'cash flow', 'financial', 'fiscal', 'quarter', 'growth']
        
        # Check for risk query
        risk_terms = ['risk', 'threat', 'uncertainty', 'challenge', 'liability', 'compliance', 
                     'regulatory', 'litigation', 'hazard', 'exposure', 'vulnerability']
        
        # Check for summary request
        summary_terms = ['summarize', 'summary', 'summarise', 'overview', 'brief', 'snapshot']
        
        # Check for comparison request
        comparison_terms = ['compare', 'comparison', 'versus', 'vs', 'difference', 'different', 
                          'similarities', 'changed', 'change', 'growth', 'decline']
        
        # Count term occurrences
        financial_count = sum(1 for term in financial_terms if term in query_lower)
        risk_count = sum(1 for term in risk_terms if term in query_lower)
        summary_count = sum(1 for term in summary_terms if term in query_lower)
        comparison_count = sum(1 for term in comparison_terms if term in query_lower)
        
        # Determine query type
        max_count = max(financial_count, risk_count, summary_count, comparison_count)
        
        if max_count == 0:
            return "general"
        
        if financial_count == max_count:
            return "financial"
        elif risk_count == max_count:
            return "risk"
        elif summary_count == max_count:
            return "summary"
        elif comparison_count == max_count:
            return "comparison"
        else:
            return "general"
    
    def retrieve_relevant_chunks(self, query: str, query_type: str) -> Dict[str, Any]:
        """
        Retrieve chunks relevant to the query.
        
        Args:
            query: User query
            query_type: Type of query (general, financial, risk, summary, comparison)
            
        Returns:
            Dictionary with relevant chunks and context
        """
        # For general queries, do a hybrid search
        if query_type == "general":
            chunks = self.vector_store.hybrid_search(query, top_k=5)
            return {
                "context": self._format_context(chunks),
                "chunks": chunks
            }
        
        # For financial queries, focus on financial tables and related text
        elif query_type == "financial":
            # First, look for financial tables
            financial_chunks = self.vector_store.similarity_search(
                "financial statements balance sheet income statement cash flow", 
                top_k=3
            )
            
            # Then get query-specific chunks
            query_chunks = self.vector_store.similarity_search(query, top_k=3)
            
            # Combine and deduplicate
            all_chunks = financial_chunks + query_chunks
            unique_chunks = []
            chunk_ids = set()
            
            for chunk in all_chunks:
                if chunk["id"] not in chunk_ids:
                    unique_chunks.append(chunk)
                    chunk_ids.add(chunk["id"])
            
            return {
                "financial_data": self._format_context([c for c in unique_chunks 
                                                       if c["metadata"].get("source") == "table"]),
                "context": self._format_context([c for c in unique_chunks 
                                               if c["metadata"].get("source") != "table"]),
                "chunks": unique_chunks
            }
        
        # For risk queries, focus on risk sections
        elif query_type == "risk":
            # Try to find risk-related sections
            risk_chunks = []
            for section_name in self.vector_store.list_sections():
                if any(risk_term in section_name.lower() for risk_term in 
                       ["risk", "challenge", "uncertainty", "threat"]):
                    section_chunks = self.vector_store.search_by_section(section_name, query, top_k=3)
                    risk_chunks.extend(section_chunks)
            
            # If no risk sections found, do a general search for risk-related content
            if not risk_chunks:
                risk_chunks = self.vector_store.similarity_search("company risks and challenges", top_k=3)
                
            # Get query-specific chunks
            query_chunks = self.vector_store.similarity_search(query, top_k=3)
            
            # Combine and deduplicate
            all_chunks = risk_chunks + query_chunks
            unique_chunks = []
            chunk_ids = set()
            
            for chunk in all_chunks:
                if chunk["id"] not in chunk_ids:
                    unique_chunks.append(chunk)
                    chunk_ids.add(chunk["id"])
            
            return {
                "risk_data": self._format_context(risk_chunks),
                "context": self._format_context(query_chunks),
                "chunks": unique_chunks
            }
        
        # For summary queries, get a broad sampling of sections
        elif query_type == "summary":
            sections = self.vector_store.list_sections()
            summary_chunks = []
            
            # Get chunks from important sections
            important_sections = ["Management Discussion", "Business Overview", "Financial Highlights", 
                                 "Chairman Statement", "CEO Message", "Operations Review"]
            
            for section_name in sections:
                if any(important_term in section_name for important_term in important_sections):
                    section_chunks = self.vector_store.search_by_section(section_name, top_k=1)
                    summary_chunks.extend(section_chunks)
            
            # If not enough chunks, add more from general search
            if len(summary_chunks) < 5:
                additional_chunks = self.vector_store.similarity_search(
                    "company overview performance highlights", 
                    top_k=5-len(summary_chunks)
                )
                summary_chunks.extend(additional_chunks)
            
            return {
                "context": self._format_context(summary_chunks),
                "chunks": summary_chunks
            }
        
        # For comparison queries, try to identify what's being compared
        elif query_type == "comparison":
            # Extract potential comparison entities
            comparison_entities = self._extract_comparison_entities(query)
            
            # Get chunks for each entity
            entity_chunks = []
            for entity in comparison_entities:
                entity_chunks.append(self.vector_store.similarity_search(entity, top_k=3))
            
            # If comparison entities found, format accordingly
            if len(entity_chunks) >= 2:
                return {
                    "context_1": self._format_context(entity_chunks[0]),
                    "context_2": self._format_context(entity_chunks[1]),
                    "chunks": entity_chunks[0] + entity_chunks[1]
                }
            else:
                # Fall back to general search
                chunks = self.vector_store.similarity_search(query, top_k=6)
                
                # Split chunks into two groups for comparison
                mid = len(chunks) // 2
                return {
                    "context_1": self._format_context(chunks[:mid]),
                    "context_2": self._format_context(chunks[mid:]),
                    "chunks": chunks
                }
        
        # Default case
        chunks = self.vector_store.similarity_search(query, top_k=5)
        return {
            "context": self._format_context(chunks),
            "chunks": chunks
        }
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context text.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context text
        """
        if not chunks:
            return "No relevant information found."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            if "similarity" in chunk:
                relevance = f" (Relevance: {chunk['similarity']:.2f})"
            else:
                relevance = ""
                
            header = f"[EXCERPT {i+1}{relevance}]"
            if "metadata" in chunk and "section" in chunk["metadata"]:
                header += f" From section: {chunk['metadata']['section']}"
                
            context_parts.append(f"{header}\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """
        Extract entities being compared in a comparison query.
        
        Args:
            query: Comparison query
            
        Returns:
            List of entities to compare
        """
        # Look for comparison patterns
        comparison_patterns = [
            r"compare\s+(.*?)\s+(?:and|with|to|versus|vs\.?)\s+(.*?)(?:\?|$|\.|,)",
            r"(.*?)\s+(?:versus|vs\.?)\s+(.*?)(?:\?|$|\.|,)",
            r"difference\s+between\s+(.*?)\s+and\s+(.*?)(?:\?|$|\.|,)",
            r"how\s+(?:does|did|do)\s+(.*?)\s+compare\s+(?:to|with)\s+(.*?)(?:\?|$|\.|,)"
        ]
        
        for pattern in comparison_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        # Check for year comparisons
        year_pattern = r"(\d{4})\s+(?:and|versus|vs\.?|compared to|compared with)\s+(\d{4})"
        year_match = re.search(year_pattern, query, re.IGNORECASE)
        if year_match:
            return [year_match.group(1), year_match.group(2)]
        
        # If no specific comparison found, return empty list
        return []
    
    def generate_response(self, query: str, retrieved_info: Dict[str, Any]) -> str:
        """
        Generate a response using the LLM based on retrieved information.
        
        Args:
            query: User query
            retrieved_info: Information retrieved from vector store
            
        Returns:
            Generated response
        """
        # Classify query type
        query_type = self.classify_query(query)
        logger.info(f"Classified query as: {query_type}")
        
        # Get appropriate prompt template
        prompt_template = self.prompts.get(query_type, self.prompts["general"])
        
        # Prepare prompt variables
        prompt_variables = {"query": query}
        prompt_variables.update({k: v for k, v in retrieved_info.items() if k != "chunks"})
        
        # Generate the prompt
        prompt = prompt_template.format(**prompt_variables)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{
                    "role": "system",
                    "content": "You are a helpful assistant specialized in analyzing annual reports."
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                temperature=self.temperature,
                max_tokens=1200
            )
            
            # Extract the response text
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query from start to finish.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with response and supporting information
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        query_type = self.classify_query(query)
        retrieved_info = self.retrieve_relevant_chunks(query, query_type)
        
        # Generate response
        response = self.generate_response(query, retrieved_info)
        
        # Prepare result
        result = {
            "query": query,
            "query_type": query_type,
            "response": response,
            "supporting_info": retrieved_info["chunks"] if "chunks" in retrieved_info else []
        }
        
        logger.info(f"Generated response for query (length: {len(response)})")
        
        return result


# Example usage
if __name__ == "__main__":
    import sys
    from vector_store import LocalVectorStore
    
    if len(sys.argv) < 3:
        print("Usage: python query_processor.py <vector_store_dir> <query>")
        sys.exit(1)
    
    vector_store_dir = sys.argv[1]
    query = sys.argv[2]
    
    vector_store = LocalVectorStore(vector_store_dir)
    processor = QueryProcessor(vector_store)
    
    result = processor.process_query(query)
    
    print(f"\nQuery: {result['query']}")
    print(f"Query type: {result['query_type']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nBased on {len(result['supporting_info'])} relevant chunks")