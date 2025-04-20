import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import queue
import uuid
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, redirect

from ..pdf_processor.processor import PDFProcessor
from ..retrieval.text_processor import TextProcessor
from ..retrieval.vector_store import create_vector_store
from ..ai_interface.query_processor import QueryProcessor
from ..report_generator.generator import ReportGenerator
from ..scheduler.report_scheduler import ReportScheduler, ReportAnalysisTask

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'data', 'reports')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.config['PROCESSED_FOLDER'] = os.path.join(os.getcwd(), 'data', 'processed')
app.config['GENERATED_FOLDER'] = os.path.join(os.getcwd(), 'data', 'generated')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

# Global variables for session management
active_reports = {}
processing_tasks = {}
report_locks = {}
scheduler = None

# Valid file extensions
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def background_process_report(report_id, file_path):
    """
    Process a report in the background.
    
    Args:
        report_id: Unique identifier for the report
        file_path: Path to the PDF file
    """
    try:
        logger.info(f"Starting background processing for report {report_id}: {file_path}")
        
        # Update task status
        processing_tasks[report_id]["status"] = "processing"
        processing_tasks[report_id]["progress"] = 10
        processing_tasks[report_id]["message"] = "Processing PDF and extracting text..."
        
        # Step 1: Process PDF
        output_dir = os.path.join(app.config['PROCESSED_FOLDER'], report_id)
        pdf_processor = PDFProcessor(file_path, output_dir)
        pdf_result = pdf_processor.process_report()
        
        processing_tasks[report_id]["progress"] = 40
        processing_tasks[report_id]["message"] = "Creating text chunks and embeddings..."
        
        # Step 2: Process text and create embeddings
        text_processor = TextProcessor(output_dir)
        text_result = text_processor.process()
        
        processing_tasks[report_id]["progress"] = 70
        processing_tasks[report_id]["message"] = "Building vector store..."
        
        # Step 3: Create vector store
        vector_store = create_vector_store("local", vector_store_dir=os.path.join(output_dir, "vector_store"))
        
        # Step 4: Create query processor
        query_processor = QueryProcessor(vector_store)
        
        processing_tasks[report_id]["progress"] = 90
        processing_tasks[report_id]["message"] = "Finalizing processing..."
        
        # Step 5: Extract company name
        report_generator = ReportGenerator(vector_store, query_processor)
        company_name = report_generator.company_name
        
        # Save report details
        active_reports[report_id] = {
            "id": report_id,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "processed_dir": output_dir,
            "vector_store_dir": os.path.join(output_dir, "vector_store"),
            "company_name": company_name,
            "upload_time": datetime.now().isoformat(),
            "processing_complete": True
        }
        
        # Update task status
        processing_tasks[report_id]["status"] = "completed"
        processing_tasks[report_id]["progress"] = 100
        processing_tasks[report_id]["message"] = "Processing complete."
        
        logger.info(f"Completed processing for report {report_id}")
        
    except Exception as e:
        logger.error(f"Error processing report {report_id}: {str(e)}")
        processing_tasks[report_id]["status"] = "failed"
        processing_tasks[report_id]["message"] = f"Error: {str(e)}"


def save_active_reports():
    """Save active reports to disk."""
    reports_file = os.path.join(app.config['PROCESSED_FOLDER'], 'active_reports.json')
    with open(reports_file, 'w') as f:
        json.dump(active_reports, f, indent=2)


def load_active_reports():
    """Load active reports from disk."""
    global active_reports
    reports_file = os.path.join(app.config['PROCESSED_FOLDER'], 'active_reports.json')
    if os.path.exists(reports_file):
        with open(reports_file, 'r') as f:
            active_reports = json.load(f)
            
            # Initialize locks for all reports
            for report_id in active_reports:
                report_locks[report_id] = threading.Lock()


def initialize_app():
    """Initialize the application state."""
    global scheduler
    
    # Load active reports
    load_active_reports()
    
    # Initialize scheduler
    scheduler = ReportScheduler()
    scheduler.start()
    
    # Register alert handler
    def web_alert_handler(alert_data):
        alert_id = str(uuid.uuid4())
        alerts_dir = os.path.join(app.config['PROCESSED_FOLDER'], 'alerts')
        os.makedirs(alerts_dir, exist_ok=True)
        
        # Save alert to file
        alert_file = os.path.join(alerts_dir, f"{alert_id}.json")
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        logger.info(f"Saved alert {alert_id} to {alert_file}")
    
    scheduler.register_alert_handler(web_alert_handler)


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', reports=active_reports)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for the report
        report_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{report_id}_{filename}")
        file.save(file_path)
        
        # Create a lock for this report
        report_locks[report_id] = threading.Lock()
        
        # Initialize processing task
        processing_tasks[report_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued for processing..."
        }
        
        # Start background processing
        thread = threading.Thread(
            target=background_process_report, 
            args=(report_id, file_path)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "report_id": report_id,
            "message": "File uploaded and processing started.",
            "status_url": url_for('check_processing_status', report_id=report_id)
        })
    
    return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400


@app.route('/status/<report_id>')
def check_processing_status(report_id):
    """Check the processing status of a report."""
    if report_id not in processing_tasks:
        return jsonify({"error": "Report not found"}), 404
    
    return jsonify(processing_tasks[report_id])


@app.route('/report/<report_id>')
def view_report(report_id):
    """View a processed report."""
    if report_id not in active_reports:
        return render_template('error.html', message="Report not found")
    
    report = active_reports[report_id]
    
    # Check if processing is complete
    if not report.get("processing_complete", False):
        return render_template('processing.html', report_id=report_id)
    
    return render_template('report.html', report=report)


@app.route('/api/query/<report_id>', methods=['POST'])
def query_report(report_id):
    """Process a query against a report."""
    if report_id not in active_reports:
        return jsonify({"error": "Report not found"}), 404
    
    report = active_reports[report_id]
    
    # Check if processing is complete
    if not report.get("processing_complete", False):
        return jsonify({"error": "Report is still processing"}), 400
    
    # Get query from request
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data['query']
    
    try:
        # Acquire lock for this report
        with report_locks[report_id]:
            # Initialize vector store and query processor
            vector_store = create_vector_store("local", vector_store_dir=report["vector_store_dir"])
            query_processor = QueryProcessor(vector_store)
            
            # Process query
            result = query_processor.process_query(query)
            
            return jsonify({
                "query": query,
                "response": result["response"],
                "query_type": result["query_type"],
                "supporting_info": [
                    {
                        "text": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                        "metadata": chunk["metadata"],
                        "similarity": chunk.get("similarity", 0)
                    }
                    for chunk in result["supporting_info"][:3]  # Limit to top 3 for display
                ]
            })
            
    except Exception as e:
        logger.error(f"Error processing query for report {report_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate/<report_id>/<report_type>')
def generate_report(report_id, report_type):
    """Generate a report presentation or document."""
    if report_id not in active_reports:
        return jsonify({"error": "Report not found"}), 404
    
    report = active_reports[report_id]
    
    # Check if processing is complete
    if not report.get("processing_complete", False):
        return jsonify({"error": "Report is still processing"}), 400
    
    # Validate report type
    if report_type not in ["presentation", "text"]:
        return jsonify({"error": "Invalid report type"}), 400
    
    try:
        # Acquire lock for this report
        with report_locks[report_id]:
            # Initialize vector store and query processor
            vector_store = create_vector_store("local", vector_store_dir=report["vector_store_dir"])
            query_processor = QueryProcessor(vector_store)
            
            # Initialize report generator
            generator = ReportGenerator(
                vector_store, 
                query_processor, 
                company_name=report.get("company_name"),
                output_dir=app.config['GENERATED_FOLDER']
            )
            
            # Generate report
            if report_type == "presentation":
                output_path = generator.generate_presentation()
            else:
                output_path = generator.generate_text_report()
            
            # Return download link
            file_name = os.path.basename(output_path)
            return jsonify({
                "message": f"{report_type.capitalize()} report generated successfully",
                "download_url": url_for('download_file', filename=file_name)
            })
            
    except Exception as e:
        logger.error(f"Error generating {report_type} for report {report_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/schedule/<report_id>', methods=['POST'])
def schedule_analysis(report_id):
    """Schedule automated analysis for a report."""
    if report_id not in active_reports:
        return jsonify({"error": "Report not found"}), 404
    
    report = active_reports[report_id]
    
    # Check if processing is complete
    if not report.get("processing_complete", False):
        return jsonify({"error": "Report is still processing"}), 400
    
    # Get schedule parameters from request
    data = request.json
    if not data:
        return jsonify({"error": "No schedule parameters provided"}), 400
    
    schedule_type = data.get("schedule_type", "now")
    schedule_params = data.get("schedule_params", {})
    analysis_queries = data.get("queries", [
        "What are the key financial highlights?",
        "What are the main risk factors?",
        "What is the company's strategy and outlook?",
        "How has the company performed compared to previous periods?",
        "What are the significant accounting policies?"
    ])
    
    try:
        # Create task
        task = ReportAnalysisTask(
            report_path=report["file_path"],
            analysis_queries=analysis_queries,
            output_dir=report["processed_dir"]
        )
        
        # Schedule task
        task_id = scheduler.add_report_task(task, schedule_type, schedule_params)
        
        return jsonify({
            "message": f"Analysis scheduled successfully",
            "task_id": task_id,
            "schedule_type": schedule_type
        })
        
    except Exception as e:
        logger.error(f"Error scheduling analysis for report {report_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/tasks')
def list_tasks():
    """List all scheduled tasks."""
    if not scheduler:
        return jsonify({"error": "Scheduler not initialized"}), 500
    
    # Get tasks from scheduler
    tasks_dir = scheduler.tasks_dir
    tasks = []
    
    for filename in os.listdir(tasks_dir):
        if filename.endswith("_result.json"):
            file_path = os.path.join(tasks_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    tasks.append({
                        "task_id": task_data.get("task_id"),
                        "report_name": task_data.get("report_name"),
                        "status": task_data.get("status"),
                        "start_time": task_data.get("start_time"),
                        "end_time": task_data.get("end_time")
                    })
            except Exception as e:
                logger.error(f"Error loading task result {filename}: {str(e)}")
    
    return jsonify({"tasks": tasks})


@app.route('/api/task/<task_id>')
def get_task(task_id):
    """Get details of a specific task."""
    if not scheduler:
        return jsonify({"error": "Scheduler not initialized"}), 500
    
    # Get task from scheduler
    task_status = scheduler.get_task_status(task_id)
    
    if not task_status:
        return jsonify({"error": "Task not found"}), 404
    
    return jsonify(task_status)


@app.route('/api/alerts')
def list_alerts():
    """List all alerts."""
    alerts_dir = os.path.join(app.config['PROCESSED_FOLDER'], 'alerts')
    
    if not os.path.exists(alerts_dir):
        return jsonify({"alerts": []})
    
    alerts = []
    for filename in os.listdir(alerts_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(alerts_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    alert_data = json.load(f)
                    alert_id = filename.split('.')[0]
                    alerts.append({
                        "alert_id": alert_id,
                        "report_name": alert_data.get("report_name"),
                        "timestamp": alert_data.get("timestamp"),
                        "alert_count": len(alert_data.get("alerts", []))
                    })
            except Exception as e:
                logger.error(f"Error loading alert {filename}: {str(e)}")
    
    return jsonify({"alerts": alerts})


@app.route('/api/alert/<alert_id>')
def get_alert(alert_id):
    """Get details of a specific alert."""
    alert_file = os.path.join(app.config['PROCESSED_FOLDER'], 'alerts', f"{alert_id}.json")
    
    if not os.path.exists(alert_file):
        return jsonify({"error": "Alert not found"}), 404
    
    try:
        with open(alert_file, 'r') as f:
            alert_data = json.load(f)
            return jsonify(alert_data)
    except Exception as e:
        logger.error(f"Error loading alert {alert_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download a generated file."""
    return send_from_directory(app.config['GENERATED_FOLDER'], filename, as_attachment=True)


@app.route('/api/compare', methods=['POST'])
def compare_reports():
    """Compare two reports."""
    data = request.json
    if not data or 'report1_id' not in data or 'report2_id' not in data:
        return jsonify({"error": "Missing report IDs"}), 400
    
    report1_id = data['report1_id']
    report2_id = data['report2_id']
    
    if report1_id not in active_reports:
        return jsonify({"error": f"Report {report1_id} not found"}), 404
    
    if report2_id not in active_reports:
        return jsonify({"error": f"Report {report2_id} not found"}), 404
    
    report1 = active_reports[report1_id]
    report2 = active_reports[report2_id]
    
    # Check if processing is complete for both reports
    if not report1.get("processing_complete", False) or not report2.get("processing_complete", False):
        return jsonify({"error": "Both reports must be fully processed"}), 400
    
    # Get comparison queries from request or use defaults
    comparison_queries = data.get("queries", [
        "Compare the financial performance between these two reports.",
        "How have the risk factors changed between these reports?",
        "Compare the business outlook and strategy between these reports.",
        "What significant changes in operations are mentioned between these reports?"
    ])
    
    comparison_results = {}
    
    try:
        # Initialize vector stores for both reports
        vector_store1 = create_vector_store("local", vector_store_dir=report1["vector_store_dir"])
        vector_store2 = create_vector_store("local", vector_store_dir=report2["vector_store_dir"])
        
        # Initialize query processor (we'll use the one from report1)
        query_processor = QueryProcessor(vector_store1)
        
        # Process each comparison query
        for query in comparison_queries:
            # Get chunks from both reports
            chunks1 = vector_store1.similarity_search(query, top_k=3)
            chunks2 = vector_store2.similarity_search(query, top_k=3)
            
            # Format contexts
            context1 = query_processor._format_context(chunks1)
            context2 = query_processor._format_context(chunks2)
            
            # Generate comparison response
            prompt_template = query_processor.prompts["comparison"]
            prompt = prompt_template.format(
                context_1=context1,
                context_2=context2,
                query=query
            )
            
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
            comparison_results[query] = answer
        
        return jsonify({
            "report1": {
                "id": report1_id,
                "name": report1.get("file_name"),
                "company": report1.get("company_name")
            },
            "report2": {
                "id": report2_id,
                "name": report2.get("file_name"),
                "company": report2.get("company_name")
            },
            "comparisons": comparison_results
        })
        
    except Exception as e:
        logger.error(f"Error comparing reports {report1_id} and {report2_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/sections/<report_id>')
def get_report_sections(report_id):
    """Get the sections of a report."""
    if report_id not in active_reports:
        return jsonify({"error": "Report not found"}), 404
    
    report = active_reports[report_id]
    
    # Check if processing is complete
    if not report.get("processing_complete", False):
        return jsonify({"error": "Report is still processing"}), 400
    
    try:
        # Initialize vector store
        vector_store = create_vector_store("local", vector_store_dir=report["vector_store_dir"])
        
        # Get sections
        sections = vector_store.list_sections()
        
        return jsonify({"sections": sections})
        
    except Exception as e:
        logger.error(f"Error retrieving sections for report {report_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

initialize_app()
# Start the Flask application
def start_app(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask application."""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_app(debug=True)