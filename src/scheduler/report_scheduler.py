import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import time
import threading
import queue
import hashlib

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ..pdf_processor.processor import PDFProcessor
from ..retrieval.text_processor import TextProcessor
from ..retrieval.vector_store import LocalVectorStore, create_vector_store
from ..ai_interface.query_processor import QueryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportAnalysisTask:
    """
    Task definition for scheduled report analysis.
    """
    
    def __init__(self, 
                 report_path: str,
                 task_id: Optional[str] = None,
                 analysis_queries: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
                 comparison_report_path: Optional[str] = None):
        """
        Initialize a report analysis task.
        
        Args:
            report_path: Path to the annual report PDF
            task_id: Unique identifier for the task (default: auto-generated)
            analysis_queries: List of analysis queries to run on the report
            output_dir: Directory to store processing results
            comparison_report_path: Path to another report for comparison (optional)
        """
        self.report_path = report_path
        self.report_name = os.path.basename(report_path).split('.')[0]
        
        # Auto-generate task ID if not provided
        if task_id:
            self.task_id = task_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            hash_part = hashlib.md5(f"{report_path}_{timestamp}".encode()).hexdigest()[:8]
            self.task_id = f"task_{self.report_name}_{hash_part}"
        
        # Default analysis queries if none provided
        self.analysis_queries = analysis_queries or [
            "What are the company's key financial highlights?",
            "What are the main risk factors mentioned in the report?",
            "What is the company's strategy and outlook for the future?",
            "How has the company performed compared to the previous year?",
            "What are the significant accounting policies?"
        ]
        
        # Set output directory
        self.output_dir = output_dir or os.path.join("data", "processed", self.report_name)
        
        # Optional comparison report
        self.comparison_report_path = comparison_report_path
        
        # Task status and results
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary representation.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "report_path": self.report_path,
            "report_name": self.report_name,
            "analysis_queries": self.analysis_queries,
            "output_dir": self.output_dir,
            "comparison_report_path": self.comparison_report_path,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportAnalysisTask':
        """
        Create task from dictionary representation.
        
        Args:
            data: Dictionary representation of the task
            
        Returns:
            ReportAnalysisTask instance
        """
        task = cls(
            report_path=data["report_path"],
            task_id=data["task_id"],
            analysis_queries=data["analysis_queries"],
            output_dir=data["output_dir"],
            comparison_report_path=data.get("comparison_report_path")
        )
        
        task.status = data["status"]
        
        if data.get("start_time"):
            task.start_time = datetime.fromisoformat(data["start_time"])
            
        if data.get("end_time"):
            task.end_time = datetime.fromisoformat(data["end_time"])
            
        task.error = data.get("error")
        
        return task


class ReportScheduler:
    """
    Scheduler for automated report analysis and alerts.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the report scheduler.
        
        Args:
            config_path: Path to the scheduler configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        
        # Task queues and processing thread
        self.task_queue = queue.Queue()
        self.results = {}
        self.tasks = {}
        self.processing_thread = None
        self.thread_running = False
        
        # Alert handlers
        self.alert_handlers = []
        
        # Directory for tasks and results
        self.tasks_dir = self.config.get("tasks_dir", "data/tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
        
        logger.info("Initialized report scheduler")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load scheduler configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "tasks_dir": "data/tasks",
            "polling_interval": 60,  # seconds
            "vector_store_type": "local",
            "model_name": "gpt-3.5-turbo",
            "scheduled_tasks": []
        }
        
        if not config_path:
            logger.info("No configuration file provided, using defaults")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                    
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.warning(f"Error loading configuration: {str(e)}")
            logger.info("Using default configuration")
            return default_config
    
    def start(self):
        """Start the scheduler and processing thread."""
        # Start the scheduler
        self.scheduler.start()
        
        # Start the processing thread
        self.thread_running = True
        self.processing_thread = threading.Thread(target=self._process_task_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Load and schedule existing tasks
        self._load_scheduled_tasks()
        
        logger.info("Report scheduler started")
    
    def stop(self):
        """Stop the scheduler and processing thread."""
        # Stop the scheduler
        self.scheduler.shutdown()
        
        # Stop the processing thread
        self.thread_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("Report scheduler stopped")
    
    def add_report_task(self, task: ReportAnalysisTask, schedule_type: str = "now",
                       schedule_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a report analysis task to the scheduler.
        
        Args:
            task: ReportAnalysisTask instance
            schedule_type: When to schedule the task ("now", "cron", "interval", "date")
            schedule_params: Parameters for the schedule (depending on schedule_type)
            
        Returns:
            Task ID
        """
        # Validate schedule type
        if schedule_type not in ["now", "cron", "interval", "date"]:
            raise ValueError(f"Invalid schedule type: {schedule_type}")
        
        # Store task
        self.tasks[task.task_id] = task
        
        # Set up the schedule
        if schedule_type == "now":
            # Add directly to the queue
            self.task_queue.put(task)
            logger.info(f"Added task {task.task_id} to immediate execution queue")
            
        else:
            # Create the appropriate trigger
            if schedule_type == "cron":
                trigger = CronTrigger(**schedule_params)
            elif schedule_type == "interval":
                trigger = IntervalTrigger(**schedule_params)
            elif schedule_type == "date":
                from apscheduler.triggers.date import DateTrigger
                trigger = DateTrigger(**schedule_params)
            
            # Add job to scheduler
            self.scheduler.add_job(
                self._queue_task,
                trigger=trigger,
                id=task.task_id,
                args=[task.task_id],
                replace_existing=True
            )
            
            logger.info(f"Scheduled task {task.task_id} with {schedule_type} schedule")
            
            # Save the scheduled task
            self._save_scheduled_task(task, schedule_type, schedule_params)
        
        return task.task_id
    
    def _queue_task(self, task_id: str):
        """
        Queue a task for execution.
        
        Args:
            task_id: ID of the task to queue
        """
        task = self.tasks.get(task_id)
        if task:
            self.task_queue.put(task)
            logger.info(f"Queued task {task_id} for execution")
        else:
            logger.warning(f"Task {task_id} not found")
    
    def _process_task_queue(self):
        """Process tasks from the queue."""
        while self.thread_running:
            try:
                # Get task from queue with timeout
                try:
                    task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process the task
                logger.info(f"Processing task {task.task_id}")
                task.status = "running"
                task.start_time = datetime.now()
                
                try:
                    # Execute the task
                    result = self._execute_task(task)
                    
                    # Update task status
                    task.status = "completed"
                    task.results = result
                    
                except Exception as e:
                    logger.error(f"Error executing task {task.task_id}: {str(e)}")
                    task.status = "failed"
                    task.error = str(e)
                
                finally:
                    task.end_time = datetime.now()
                    self.task_queue.task_done()
                    
                    # Save task result
                    self._save_task_result(task)
                    
                    # Check for alerts
                    if task.status == "completed":
                        self._check_for_alerts(task)
                
            except Exception as e:
                logger.error(f"Error in task processing thread: {str(e)}")
                time.sleep(5)  # Prevent tight loop in case of persistent errors
    
    def _execute_task(self, task: ReportAnalysisTask) -> Dict[str, Any]:
        """
        Execute a report analysis task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution results
        """
        result = {"analyses": {}}
        
        # Step 1: Process the PDF
        logger.info(f"Processing PDF: {task.report_path}")
        pdf_processor = PDFProcessor(task.report_path, task.output_dir)
        pdf_result = pdf_processor.process_report()
        result["pdf_processing"] = pdf_result
        
        # Step 2: Process the extracted text
        logger.info(f"Processing text from {task.output_dir}")
        text_processor = TextProcessor(task.output_dir)
        text_result = text_processor.process()
        result["text_processing"] = text_result
        
        # Step 3: Create vector store
        vector_store_dir = os.path.join(task.output_dir, "vector_store")
        vector_store = create_vector_store(
            self.config.get("vector_store_type", "local"),
            vector_store_dir=vector_store_dir
        )
        
        # Step 4: Create query processor
        query_processor = QueryProcessor(
            vector_store,
            model_name=self.config.get("model_name", "gpt-3.5-turbo")
        )
        
        # Step 5: Run analysis queries
        logger.info(f"Running {len(task.analysis_queries)} analysis queries")
        for i, query in enumerate(task.analysis_queries):
            logger.info(f"Running analysis query {i+1}/{len(task.analysis_queries)}: {query}")
            query_result = query_processor.process_query(query)
            result["analyses"][query] = query_result
        
        # Step 6: If comparison report specified, process it too
        if task.comparison_report_path and os.path.exists(task.comparison_report_path):
            logger.info(f"Processing comparison report: {task.comparison_report_path}")
            comparison_name = os.path.basename(task.comparison_report_path).split('.')[0]
            comparison_dir = os.path.join("data", "processed", comparison_name)
            
            # Check if already processed
            if not os.path.exists(os.path.join(comparison_dir, "vector_store")):
                # Process comparison report
                comp_processor = PDFProcessor(task.comparison_report_path, comparison_dir)
                comp_processor.process_report()
                
                comp_text_processor = TextProcessor(comparison_dir)
                comp_text_processor.process()
            
            # Run comparison analyses
            comparison_store = create_vector_store(
                self.config.get("vector_store_type", "local"),
                vector_store_dir=os.path.join(comparison_dir, "vector_store")
            )
            
            # Generate comparison report
            comparison_results = self._generate_comparison_report(
                vector_store,
                comparison_store,
                query_processor
            )
            
            result["comparison"] = comparison_results
        
        logger.info(f"Task {task.task_id} execution completed")
        return result
    
    def _generate_comparison_report(self, 
                                   current_store: Any, 
                                   comparison_store: Any,
                                   query_processor: QueryProcessor) -> Dict[str, Any]:
        """
        Generate a comparison report between two annual reports.
        
        Args:
            current_store: Vector store for current report
            comparison_store: Vector store for comparison report
            query_processor: Query processor for generating responses
            
        Returns:
            Comparison report results
        """
        comparison_queries = [
            "Compare the financial performance between these two reports.",
            "How have the risk factors changed between these reports?",
            "Compare the business outlook and strategy between these reports.",
            "What significant changes in operations are mentioned between these reports?",
            "Compare the dividend policies between these reports."
        ]
        
        results = {}
        
        for query in comparison_queries:
            # Get chunks from both reports
            current_chunks = current_store.similarity_search(query, top_k=3)
            comparison_chunks = comparison_store.similarity_search(query, top_k=3)
            
            # Format contexts
            context_1 = query_processor._format_context(current_chunks)
            context_2 = query_processor._format_context(comparison_chunks)
            
            # Generate response using comparison prompt
            prompt_template = query_processor.prompts["comparison"]
            prompt = prompt_template.format(
                context_1=context_1,
                context_2=context_2,
                query=query
            )
            
            try:
                import openai
                response = openai.ChatCompletion.create(
                    model=query_processor.model_name,
                    messages=[{
                        "role": "system",
                        "content": "You are a helpful assistant specialized in comparing annual reports."
                    }, {
                        "role": "user", 
                        "content": prompt
                    }],
                    temperature=0.3,
                    max_tokens=1200
                )
                
                answer = response.choices[0].message.content.strip()
                results[query] = answer
                
            except Exception as e:
                logger.error(f"Error generating comparison response: {str(e)}")
                results[query] = f"Error generating comparison: {str(e)}"
        
        return results
    
    def _check_for_alerts(self, task: ReportAnalysisTask):
        """
        Check for alert conditions in the task results.
        
        Args:
            task: Completed task with results
        """
        # Skip if no alerts are registered
        if not self.alert_handlers:
            return
        
        # Basic alerts for risk factors
        risk_alerts = []
        
        # Check analysis results for risk-related information
        for query, result in task.results.get("analyses", {}).items():
            if "risk" in query.lower():
                response = result.get("response", "")
                
                # Look for concerning phrases
                concerning_phrases = [
                    "significant risk", "material risk", "critical risk", 
                    "major concern", "substantial challenge", "serious threat"
                ]
                
                for phrase in concerning_phrases:
                    if phrase in response.lower():
                        risk_alerts.append({
                            "type": "risk_alert",
                            "severity": "high",
                            "description": f"Potential significant risk identified: '{phrase}'",
                            "query": query,
                            "context": response[:200] + "..."  # Excerpt from response
                        })
        
        # Check for comparison alerts if comparison was performed
        if "comparison" in task.results:
            for query, response in task.results["comparison"].items():
                # Look for phrases indicating negative changes
                negative_phrases = [
                    "significant decrease", "substantial decline", "major reduction",
                    "negative trend", "concerning development", "deteriorated", "worsened"
                ]
                
                for phrase in negative_phrases:
                    if phrase in response.lower():
                        risk_alerts.append({
                            "type": "comparison_alert",
                            "severity": "medium",
                            "description": f"Negative trend identified: '{phrase}'",
                            "query": query,
                            "context": response[:200] + "..."  # Excerpt from response
                        })
        
        # Trigger alerts if any found
        if risk_alerts:
            alert_data = {
                "task_id": task.task_id,
                "report_name": task.report_name,
                "alerts": risk_alerts,
                "timestamp": datetime.now().isoformat()
            }
            
            # Call all registered alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert_data)
                except Exception as e:
                    logger.error(f"Error in alert handler: {str(e)}")
    
    def register_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Register a handler function for alerts.
        
        Args:
            handler: Function to call when alerts are triggered
        """
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")
    
    def _save_scheduled_task(self, task: ReportAnalysisTask, 
                            schedule_type: str, 
                            schedule_params: Dict[str, Any]):
        """
        Save a scheduled task to disk.
        
        Args:
            task: Task to save
            schedule_type: Type of schedule
            schedule_params: Schedule parameters
        """
        task_data = task.to_dict()
        task_data.update({
            "schedule_type": schedule_type,
            "schedule_params": schedule_params
        })
        
        file_path = os.path.join(self.tasks_dir, f"{task.task_id}_schedule.json")
        
        with open(file_path, 'w') as f:
            json.dump(task_data, f, indent=2)
            
        logger.info(f"Saved scheduled task to {file_path}")
    
    def _save_task_result(self, task: ReportAnalysisTask):
        """
        Save task result to disk.
        
        Args:
            task: Task with results
        """
        result_data = task.to_dict()
        
        # Add results, but don't include large chunks in the supporting info
        if hasattr(task, 'results') and task.results:
            # Deep copy results but simplify supporting_info to save space
            result_data["results"] = {}
            
            for key, value in task.results.items():
                if key == "analyses":
                    result_data["results"][key] = {}
                    for query, query_result in value.items():
                        # Simplify supporting_info to just metadata
                        simplified_result = query_result.copy()
                        if "supporting_info" in simplified_result:
                            simplified_result["supporting_info"] = [
                                {
                                    "id": chunk["id"],
                                    "metadata": chunk["metadata"],
                                    "similarity": chunk.get("similarity", 0)
                                }
                                for chunk in simplified_result["supporting_info"]
                            ]
                        result_data["results"][key][query] = simplified_result
                else:
                    result_data["results"][key] = value
        
        file_path = os.path.join(self.tasks_dir, f"{task.task_id}_result.json")
        
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        logger.info(f"Saved task result to {file_path}")
    
    def _load_scheduled_tasks(self):
        """Load scheduled tasks from disk and schedule them."""
        # Look for task schedule files
        for filename in os.listdir(self.tasks_dir):
            if filename.endswith("_schedule.json"):
                file_path = os.path.join(self.tasks_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)
                    
                    # Recreate task
                    task = ReportAnalysisTask.from_dict(task_data)
                    
                    # Reschedule if not completed
                    if task.status not in ["completed", "failed"]:
                        schedule_type = task_data.get("schedule_type")
                        schedule_params = task_data.get("schedule_params", {})
                        
                        if schedule_type and schedule_type != "now":
                            # Add to scheduler
                            self.tasks[task.task_id] = task
                            
                            if schedule_type == "cron":
                                trigger = CronTrigger(**schedule_params)
                            elif schedule_type == "interval":
                                trigger = IntervalTrigger(**schedule_params)
                            elif schedule_type == "date":
                                from apscheduler.triggers.date import DateTrigger
                                trigger = DateTrigger(**schedule_params)
                            
                            self.scheduler.add_job(
                                self._queue_task,
                                trigger=trigger,
                                id=task.task_id,
                                args=[task.task_id],
                                replace_existing=True
                            )
                            
                            logger.info(f"Loaded and scheduled task {task.task_id}")
                    
                except Exception as e:
                    logger.error(f"Error loading scheduled task {filename}: {str(e)}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status dictionary or None if not found
        """
        # Check in memory
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        
        # Check on disk
        result_path = os.path.join(self.tasks_dir, f"{task_id}_result.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading task result {task_id}: {str(e)}")
        
        # Not found
        return None


# Example alert handler
def email_alert_handler(alert_data: Dict[str, Any]):
    """
    Example alert handler that would send an email.
    
    Args:
        alert_data: Alert data dictionary
    """
    import smtplib
    from email.message import EmailMessage
    
    # In a real implementation, these would come from configuration
    smtp_server = "smtp.example.com"
    smtp_port = 587
    smtp_user = "alerts@example.com"
    smtp_password = "password"
    recipient = "analyst@example.com"
    
    # Create message
    msg = EmailMessage()
    msg['Subject'] = f"Annual Report Alert: {alert_data['report_name']}"
    msg['From'] = smtp_user
    msg['To'] = recipient
    
    # Build message body
    body = f"Alerts for annual report: {alert_data['report_name']}\n\n"
    
    for alert in alert_data['alerts']:
        body += f"[{alert['severity'].upper()}] {alert['description']}\n"
        body += f"Query: {alert['query']}\n"
        body += f"Context: {alert['context']}\n\n"
    
    body += f"\nGenerated at: {alert_data['timestamp']}\n"
    body += f"Task ID: {alert_data['task_id']}"
    
    msg.set_content(body)
    
    # For demonstration, just log instead of actually sending
    logger.info(f"Would send email alert:\n{body}")
    
    # In a real implementation, uncomment this to send the email
    """
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        logger.info(f"Sent email alert to {recipient}")
    except Exception as e:
        logger.error(f"Error sending email alert: {str(e)}")
    """


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python report_scheduler.py <path_to_annual_report.pdf>")
        sys.exit(1)
    
    report_path = sys.argv[1]
    
    # Create scheduler
    scheduler = ReportScheduler()
    
    # Register alert handler
    scheduler.register_alert_handler(email_alert_handler)
    
    # Create task
    task = ReportAnalysisTask(report_path)
    
    # Add task for immediate execution
    task_id = scheduler.add_report_task(task, "now")
    
    # Start scheduler
    scheduler.start()
    
    try:
        print(f"Processing report {report_path}")
        print(f"Task ID: {task_id}")
        print("Press Ctrl+C to stop...")
        
        # Wait for task to complete
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
        print("\nStopping scheduler...")
    finally:
        scheduler.stop()