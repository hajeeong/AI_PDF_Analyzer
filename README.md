# Annual Report AI Agent

An intelligent application that enables users to ask natural language questions and receive meaningful answers from annual report PDFs. The agent can process, analyze, and generate insights from complex business documents.

## Features

- **PDF Processing**: Extract text and tables from annual reports
- **Natural Language Querying**: Ask questions about the report content in plain English
- **Automated Analysis**: Schedule automated report analysis with alerts for significant findings
- **Report Generation**: Generate executive summaries and presentations automatically
- **Web Interface**: User-friendly web application for interacting with reports
- **Comparison Analysis**: Compare different annual reports to identify changes and trends

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/annual-report-ai-agent.git
   cd annual-report-ai-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Linux/Mac
   export OPENAI_API_KEY=your_openai_api_key
   
   # Windows
   set OPENAI_API_KEY=your_openai_api_key
   ```

5. Create a configuration file (optional):
   ```bash
   python main.py create-config config/config.json
   ```

6. Edit the configuration file to customize settings

## Usage

### Web Application

The easiest way to use the application is through the web interface:

```bash
python main.py webapp --port 5000
```

Then open a web browser and navigate to `http://localhost:5000`

### Command Line Interface

The application provides various commands through a command line interface:

#### Process a Report

```bash
python main.py process path/to/annual_report.pdf --output data/processed/company_name
```

#### Interactive Query Mode

```bash
python main.py query data/processed/company_name/vector_store
```

#### Generate Reports

```bash
# Generate a text summary
python main.py generate data/processed/company_name/vector_store --format text --output data/generated

# Generate a presentation
python main.py generate data/processed/company_name/vector_store --format presentation --output data/generated
```

#### Schedule Automated Analysis

```bash
# Run once immediately
python main.py schedule path/to/annual_report.pdf --type now

# Schedule daily analysis
python main.py schedule path/to/annual_report.pdf --type daily

# Schedule with comparison
python main.py schedule path/to/annual_report.pdf --type now --compare path/to/previous_report.pdf
```

## Project Structure

```
annual-report-ai-agent/
├── config/               # Configuration files
├── data/                 # Data directory
│   ├── generated/        # Generated reports and presentations
│   ├── processed/        # Processed report data
│   ├── reports/          # Raw report PDF files
│   ├── tasks/            # Task data for scheduled analysis
│   └── vector_store/     # Vector embeddings and indexes
├── src/                  # Source code
│   ├── ai_interface/     # AI query processing modules
│   ├── pdf_processor/    # PDF extraction modules
│   ├── report_generator/ # Report generation modules
│   ├── retrieval/        # Vector retrieval modules
│   ├── scheduler/        # Task scheduling modules
│   └── web_app/          # Web application interface
├── tests/                # Test files
├── main.py               # Application entry point
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Requirements

The main dependencies for this project are:

- **PDF Processing**: PyPDF2, pdfplumber, camelot-py, opencv-python
- **Data Processing**: pandas, numpy, matplotlib
- **NLP & AI**: spaCy, sentence-transformers, langchain, openai
- **Vector Storage**: pinecone-client (optional)
- **Web Interface**: Flask
- **Task Scheduling**: APScheduler
- **Report Generation**: python-pptx

See `requirements.txt` for the complete list of dependencies.

## Configuration

The application can be configured through a JSON configuration file. Create a sample configuration file with:

```bash
python main.py create-config config/config.json
```

Key configuration options include:

- **vector_store_type**: Type of vector store to use ("local" or "pinecone")
- **model_name**: Language model to use (e.g., "gpt-3.5-turbo")
- **temperature**: Temperature parameter for model generation
- **web_app**: Configuration for the web application
- **scheduler**: Configuration for the task scheduler

## Advanced Usage

### Using a Different Vector Database

By default, the application uses a local vector store. To use Pinecone:

1. Create a Pinecone account and get an API key
2. Set up the config file with your Pinecone credentials:
   ```json
   {
     "vector_store_type": "pinecone",
     "pinecone": {
       "api_key": "YOUR_PINECONE_API_KEY",
       "environment": "YOUR_PINECONE_ENVIRONMENT",
       "index_name": "annual-reports"
     }
   }
   ```

3. Use the configuration file when running the application:
   ```bash
   python main.py query data/processed/company_name/vector_store --config config/config.json
   ```

### Customizing Alerts

The application can send alerts for significant findings. Configure email notifications in the config file:

```json
{
  "scheduler": {
    "email": {
      "smtp_server": "smtp.example.com",
      "smtp_port": 587,
      "smtp_user": "alerts@example.com",
      "smtp_password": "your_password",
      "recipient": "analyst@example.com"
    }
  }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses various open-source libraries and tools
- Special thanks to the developers of LangChain, Sentence Transformers, and other key dependencies