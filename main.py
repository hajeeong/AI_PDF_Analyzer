import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional
import json

from src.pdf_processor.processor import PDFProcessor
from src.retrieval.text_processor import TextProcessor
from src.retrieval.vector_store import create_vector_store
from src.ai_interface.query_processor import QueryProcessor
from src.report_generator.generator import ReportGenerator
from src.scheduler.report_scheduler import ReportScheduler, ReportAnalysisTask, email_alert_handler
from src.web_app.app import start_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('annual_report_ai.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load application configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.info("Using default configuration")
        return {
            "vector_store_type": "local",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.3,
            "web_app": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            }
        }


def process_report(report_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Process an annual report PDF.
    
    Args:
        report_path: Path to the annual report PDF
        output_dir: Directory to save processed data (default: auto-generated)
        
    Returns:
        Processing results dictionary
    """
    logger.info(f"Processing report: {report_path}")
    
    # Step 1: Process PDF
    pdf_processor = PDFProcessor(report_path, output_dir)
    pdf_result = pdf_processor.process_report()
    
    processed_dir = pdf_result["output_dir"]
    logger.info(f"PDF processed, output in {processed_dir}")
    
    # Step 2: Process text
    text_processor = TextProcessor(processed_dir)
    text_result = text_processor.process()
    
    logger.info(f"Text processed: {text_result['chunks_count']} chunks created")
    
    return {
        "pdf_result": pdf_result,
        "text_result": text_result,
        "processed_dir": processed_dir,
        "vector_store_dir": os.path.join(processed_dir, "vector_store")
    }


def interactive_query_mode(vector_store_dir: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
    """
    Run interactive query mode for a processed report.
    
    Args:
        vector_store_dir: Path to the vector store directory
        model_name: LLM model name
        temperature: Temperature parameter for LLM generation
    """
    logger.info(f"Starting interactive query mode with {vector_store_dir}")
    
    # Initialize vector store
    vector_store = create_vector_store("local", vector_store_dir=vector_store_dir)
    
    # Initialize query processor
    query_processor = QueryProcessor(vector_store, model_name=model_name, temperature=temperature)
    
    print("\n===== Annual Report AI Agent =====")
    print("Enter your questions about the report (type 'exit' to quit):")
    
    while True:
        # Get user input
        query = input("\nQuestion: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Exiting interactive mode.")
            break
        
        if not query:
            continue
        
        try:
            # Process query
            result = query_processor.process_query(query)
            
            # Print response
            print("\nResponse:")
            print(result["response"])
            
            # Print supporting info
            print("\nBased on:")
            for i, chunk in enumerate(result["supporting_info"][:2]):  # Show top 2 chunks
                if "similarity" in chunk:
                    print(f"[Chunk {i+1}, Similarity: {chunk['similarity']:.2f}]")
                else:
                    print(f"[Chunk {i+1}]")
                print(f"Section: {chunk['metadata'].get('section', 'Unknown')}")
                print(f"Text: {chunk['text'][:150]}...")
                print()
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Error: {str(e)}")


def generate_report_outputs(vector_store_dir: str, output_format: str, output_dir: Optional[str] = None, 
                           model_name: str = "gpt-3.5-turbo", company_name: Optional[str] = None):
    """
    Generate report outputs (presentation or text summary).
    
    Args:
        vector_store_dir: Path to the vector store directory
        output_format: Output format ('presentation' or 'text')
        output_dir: Directory to save generated outputs
        model_name: LLM model name
        company_name: Company name (optional)
    """
    # Initialize vector store
    vector_store = create_vector_store("local", vector_store_dir=vector_store_dir)
    
    # Initialize report generator
    generator = ReportGenerator(
        vector_store,
        model_name=model_name,
        company_name=company_name,
        output_dir=output_dir
    )
    
    try:
        if output_format == 'presentation':
            logger.info("Generating presentation")
            output_path = generator.generate_presentation()
            print(f"Generated presentation: {output_path}")
            
        elif output_format == 'text':
            logger.info("Generating text report")
            output_path = generator.generate_text_report()
            print(f"Generated text report: {output_path}")
            
        else:
            logger.error(f"Unsupported output format: {output_format}")
            print(f"Unsupported output format: {output_format}")
    
    except Exception as e:
        logger.error(f"Error generating {output_format}: {str(e)}")
        print(f"Error generating {output_format}: {str(e)}")


def schedule_analysis(report_path: str, schedule_type: str = "now", 
                     comparison_report_path: Optional[str] = None, 
                     config_path: Optional[str] = None):
    """
    Schedule automated analysis for a report.
    
    Args:
        report_path: Path to the annual report PDF
        schedule_type: Type of schedule ('now', 'daily', 'weekly')
        comparison_report_path: Path to a report to compare with (optional)
        config_path: Path to scheduler configuration file (optional)
    """
    # Initialize scheduler
    scheduler = ReportScheduler(config_path)
    
    # Register alert handler
    scheduler.register_alert_handler(email_alert_handler)
    
    # Create task
    task = ReportAnalysisTask(
        report_path=report_path,
        comparison_report_path=comparison_report_path
    )
    
    # Determine schedule parameters
    schedule_params = {}
    
    if schedule_type == "daily":
        schedule_params = {
            "hours": 9,
            "minutes": 0,
            "seconds": 0
        }
    elif schedule_type == "weekly":
        schedule_params = {
            "day_of_week": "mon",
            "hours": 9,
            "minutes": 0,
            "seconds": 0
        }
    
    # Schedule task
    logger.info(f"Scheduling analysis with {schedule_type} schedule")
    task_id = scheduler.add_report_task(task, 
                                      "interval" if schedule_type in ["daily", "weekly"] else "now", 
                                      schedule_params)
    
    # Start scheduler
    scheduler.start()
    
    print(f"Scheduled analysis task: {task_id}")
    print(f"Schedule type: {schedule_type}")
    
    if schedule_type == "now":
        print("Processing will start immediately.")
        
        try:
            # Wait for task to complete
            import time
            print("Waiting for task to complete...")
            
            while True:
                status = scheduler.get_task_status(task_id)
                if status and status["status"] in ["completed", "failed"]:
                    print(f"\nTask completed with status: {status['status']}")
                    if status["status"] == "failed":
                        print(f"Error: {status['error']}")
                    break
                
                time.sleep(5)
                print(".", end="", flush=True)
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            scheduler.stop()
    else:
        print(f"Task will run according to {schedule_type} schedule.")
        print("Press Ctrl+C to stop the scheduler.")
        
        try:
            # Keep program running
            import time
            while True:
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
        finally:
            scheduler.stop()


def create_config_file(output_path: str):
    """
    Create a sample configuration file.
    
    Args:
        output_path: Path to save the configuration file
    """
    config = {
        "vector_store_type": "local",  # Options: "local", "pinecone"
        "model_name": "gpt-3.5-turbo",  # OpenAI model name
        "temperature": 0.3,  # Model temperature
        "openai_api_key": "YOUR_OPENAI_API_KEY",  # Set your API key here or in environment
        
        # Pinecone configuration (if using Pinecone)
        "pinecone": {
            "api_key": "YOUR_PINECONE_API_KEY",
            "environment": "YOUR_PINECONE_ENVIRONMENT",
            "index_name": "annual-reports"
        },
        
        # Web app configuration
        "web_app": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False
        },
        
        # Scheduler configuration
        "scheduler": {
            "tasks_dir": "data/tasks",
            "polling_interval": 60,  # seconds
            
            # Email notification settings (for alerts)
            "email": {
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "smtp_user": "alerts@example.com",
                "smtp_password": "your_password",
                "recipient": "analyst@example.com"
            },
            
            # Scheduled tasks
            "scheduled_tasks": [
                # Example of a scheduled task
                {
                    "report_path": "data/reports/example_report.pdf",
                    "schedule_type": "cron",
                    "schedule_params": {
                        "day_of_week": "mon",
                        "hour": 9,
                        "minute": 0
                    },
                    "analysis_queries": [
                        "What are the key financial highlights?",
                        "What are the main risk factors?",
                        "What is the company's strategy and outlook?",
                        "How has the company performed compared to previous periods?",
                        "What are the significant accounting policies?"
                    ]
                }
            ]
        }
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created sample configuration file: {output_path}")
    print("Edit this file to customize your application settings.")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Annual Report AI Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process report command
    process_parser = subparsers.add_parser("process", help="Process an annual report PDF")
    process_parser.add_argument("report_path", help="Path to the annual report PDF")
    process_parser.add_argument("--output", "-o", help="Directory to save processed data")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a processed report")
    query_parser.add_argument("vector_store", help="Path to the vector store directory")
    query_parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="LLM model name")
    query_parser.add_argument("--temperature", "-t", type=float, default=0.3, help="LLM temperature")
    query_parser.add_argument("--config", "-c", help="Path to configuration file")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate report outputs")
    generate_parser.add_argument("vector_store", help="Path to the vector store directory")
    generate_parser.add_argument("--format", "-f", choices=["presentation", "text"], required=True,
                               help="Output format (presentation or text)")
    generate_parser.add_argument("--output", "-o", help="Directory to save generated outputs")
    generate_parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="LLM model name")
    generate_parser.add_argument("--company", help="Company name (optional)")
    generate_parser.add_argument("--config", "-c", help="Path to configuration file")
    
    # Schedule command
    schedule_parser = subparsers.add_parser("schedule", help="Schedule automated analysis")
    schedule_parser.add_argument("report_path", help="Path to the annual report PDF")
    schedule_parser.add_argument("--type", "-t", choices=["now", "daily", "weekly"], default="now",
                               help="Schedule type")
    schedule_parser.add_argument("--compare", help="Path to a report to compare with (optional)")
    schedule_parser.add_argument("--config", "-c", help="Path to scheduler configuration file")
    
    # Web app command
    webapp_parser = subparsers.add_parser("webapp", help="Start the web application")
    webapp_parser.add_argument("--host", default="0.0.0.0", help="Host to run the web app on")
    webapp_parser.add_argument("--port", "-p", type=int, default=5000, help="Port to run the web app on")
    webapp_parser.add_argument("--debug", "-d", action="store_true", help="Run in debug mode")
    webapp_parser.add_argument("--config", "-c", help="Path to configuration file")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create a sample configuration file")
    config_parser.add_argument("output_path", help="Path to save the configuration file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "process":
        # Process report
        process_report(args.report_path, args.output)
        
    elif args.command == "query":
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
            
        # Interactive query mode
        model_name = args.model or config.get("model_name", "gpt-3.5-turbo")
        temperature = args.temperature or config.get("temperature", 0.3)
        
        interactive_query_mode(args.vector_store, model_name, temperature)
        
    elif args.command == "generate":
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
            
        # Generate report outputs
        model_name = args.model or config.get("model_name", "gpt-3.5-turbo")
        
        generate_report_outputs(
            args.vector_store,
            args.format,
            args.output,
            model_name,
            args.company
        )
        
    elif args.command == "schedule":
        # Schedule analysis
        schedule_analysis(
            args.report_path,
            args.type,
            args.compare,
            args.config
        )
        
    elif args.command == "webapp":
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
            
        # Get web app configuration
        web_config = config.get("web_app", {})
        host = args.host or web_config.get("host", "0.0.0.0")
        port = args.port or web_config.get("port", 5000)
        debug = args.debug or web_config.get("debug", False)
        
        # Start web app
        logger.info(f"Starting web app on {host}:{port}")
        start_app(host=host, port=port, debug=debug)
        
    elif args.command == "create-config":
        # Create sample configuration file
        create_config_file(args.output_path)
        
    else:
        # No command specified, show help
        parser.print_help()


if __name__ == "__main__":
    # Check for environment variables
    if 'OPENAI_API_KEY' not in os.environ:
        logger.warning("OPENAI_API_KEY environment variable not set.")
        logger.warning("Set this variable or provide it in configuration.")
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)