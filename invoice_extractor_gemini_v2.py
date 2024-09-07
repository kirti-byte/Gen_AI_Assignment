import google.generativeai as genai
from pathlib import Path
import pandas as pd
import json
import PyPDF2
from PIL import Image
import io
import docx2txt

# Reset display options to default
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')

# Securely configure the API key (replace with your actual API key)
API_KEY = "AIzaSyCNoFHCAJjBh4y_V8Kj0BjNhjmiNruM3sQ"

# Configure the Generative AI client with the provided API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """Extracts text content from a Word document."""
    try:
        text = docx2txt.process(docx_path)
        return text.strip()
    except Exception as e:
        print(f"Error reading Word document: {e}")
        return ""

def load_image(image_path: str) -> list:
    """Loads and validates image data from the specified path."""
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"Could not find the image: {img}")
    
    mime_type = f"image/{img.suffix[1:]}"  # Get mime type from file extension
    return [{"mime_type": mime_type, "data": img.read_bytes()}]

def extract_invoice_data(file_path: str, system_prompt: str, user_prompt: str) -> str:
    """
    Extracts invoice data from various file formats using AI.
    
    Supports PDF, Word documents, and image formats (PNG, JPG).
    """
    try:
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()

        if file_extension == '.pdf':
            content = extract_text_from_pdf(file_path)
            input_prompt = [system_prompt, content, user_prompt]
        elif file_extension == '.docx':
            content = extract_text_from_docx(file_path)
            input_prompt = [system_prompt, content, user_prompt]
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            image_info = load_image(file_path)
            input_prompt = [system_prompt, image_info[0], user_prompt]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        response = model.generate_content(input_prompt)

        if not response or not response.text:
            raise ValueError("No response received from the model.")

        return response.text

    except Exception as e:
        print(f"Error during data extraction: {e}")
        return ""

# System prompt explaining the role of AI in extracting invoice data
system_prompt = """
You are a highly experienced AI specialized in extracting and understanding data from invoices.
Your goal is to extract structured data from invoices, which may vary in format, language, and structure.
You will:
1. Identify and extract key details such as invoice number, date, supplier name, item details, amounts, and addresses.
2. Convert all information into English, especially if the invoice is in another language.
3. Ensure extracted addresses are split into street, city, state, and zip code.
4. Clearly separate line items, identifying products/services with quantities and unit prices.
5. Identify responsible parties like buyers, suppliers, and authorized signatories.
6. If text is in a foreign language, detect and translate all relevant names and terms into English.
7. invoice could be empty with no line items in thi scase extract whaterver information is available.
8. invoices can be of different tyoer including medical that will contain opd information etc. so generate columns accordingly.
Output the data in a well-structured JSON format with appropriate tags for clarity and consistency.
"""

# User prompt for converting invoice data into JSON format
user_prompt = """
Extract all invoice information and convert it into a single, valid JSON object. Ensure:
- The entire response is a single JSON object, with no text before or after.
- All numbers are formatted consistently as strings (e.g., "252.00" instead of 252.00)
- No trailing commas in objects or arrays
- All property names and string values are in double quotes
Include:
- Invoice number, date, supplier name, and buyer details.
- Extract itemized products/services with quantities, unit prices, and total amounts.
- Split addresses into components (street, city, state, zip code).
- Identify any responsible parties mentioned (e.g., buyer, supplier, signatory).
- If the invoice original language is not in English, detect the language, translate supplier name and line items in English, and specify the original language.
- Ensure that each piece of information is tagged correctly to ensure clarity, such as separating amounts, taxes, and any discount information.
Do not include any explanations or text outside of the JSON object.
"""

def process_invoice(file_path: str):
    """Process an invoice file and save the extracted data to a CSV file."""
    response_text = extract_invoice_data(file_path, system_prompt, user_prompt)
    cleaned_response_text = response_text.strip('```json').strip('```').strip()

    try:
        json_data = json.loads(cleaned_response_text)
        main_df = pd.json_normalize(json_data)
        output_file = f"{Path(file_path).stem}_analysis.csv"
        main_df.to_csv(output_file, index=False)
        print(f"Invoice data saved to {output_file}")
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print("Ensure the input string is a valid JSON format without extra markdown formatting.")
        print("Response Text:", cleaned_response_text)

# Example usage
file_path = "K:\Fractal_Gen_AI_Assignments\Gen_AI_Assignment\images\invoice1.pdf"  # Can be PDF, DOCX, PNG, or JPG
process_invoice(file_path)