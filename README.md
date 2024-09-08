# Invoice Data Extractor

## Overview

The Invoice Data Extractor is a Streamlit-based web application that uses Google's Generative AI to extract structured data from invoice documents. It supports various file formats including PDF, DOCX, PNG, JPG, and JPEG. The extracted data is presented in a user-friendly table format.

## Features

- Upload invoice documents in multiple formats (PDF, DOCX, PNG, JPG, JPEG)
- Extract structured data from invoices using Google's Generative AI
- Display extracted data in a comprehensive, easy-to-read table
- Handle nested JSON structures and flatten them for tabular display

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository or download the source code.

2. Navigate to the project directory:
   ```
   cd path/to/invoice-extractor
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: If the `requirements.txt` file is not provided, you need to install the following packages:
   ```
   pip install streamlit google-generativeai pandas PyPDF2 python-docx
   ```

4. Set up your Google API key:
   - Obtain an API key from the Google Cloud Console
   - Replace the placeholder API key in the code with your actual key:
     ```python
     API_KEY = "YOUR_ACTUAL_API_KEY"
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run invoice_extractor_frontend.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Use the file uploader to select an invoice document.

4. Click the "Process Invoice" button to extract data from the uploaded document.

5. View the extracted data in the table displayed on the page.

## File Structure

- `invoice_extractor.py`: Main application file containing the Streamlit interface and data processing logic.
- `requirements.txt`: List of Python packages required for the project (if provided).

## Troubleshooting

If you encounter any issues:

1. Ensure all required packages are installed correctly.
2. Check that your Google API key is valid and has the necessary permissions.
3. Verify that the input file is in one of the supported formats and is not corrupted.