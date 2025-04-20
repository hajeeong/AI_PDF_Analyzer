document.addEventListener('DOMContentLoaded', function() {
    // Handle file upload area if present
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        const fileInput = document.getElementById('fileInput');
        const browseButton = document.getElementById('browseButton');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('bg-light');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('bg-light');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('bg-light');
            
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        if (browseButton && fileInput) {
            browseButton.addEventListener('click', () => {
                fileInput.click();
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length) {
                    handleFile(e.target.files[0]);
                }
            });
        }
    }
    
    // Handle example questions if present
    const exampleQuestions = document.querySelectorAll('.example-question');
    const queryInput = document.getElementById('queryInput');
    const sendButton = document.getElementById('sendButton');
    
    if (exampleQuestions.length && queryInput && sendButton) {
        exampleQuestions.forEach(button => {
            button.addEventListener('click', function() {
                queryInput.value = this.textContent;
                sendButton.click();
            });
        });
    }
    
    // Generic function to handle file upload (to be implemented in page-specific scripts)
    function handleFile(file) {
        console.log('File selected:', file.name);
        // Implementation will be in page-specific scripts
    }
});