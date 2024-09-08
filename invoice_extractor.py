import streamlit as st
import google.generativeai as genai
from pathlib import Path
import pandas as pd
import json
import PyPDF2
import docx2txt
import logging
from typing import Optional
import tempfile
import os
from typing import Dict, Any


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Generative AI configuration
API_KEY = "your google-api-key"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Comprehensive system prompt
SYSTEM_PROMPT = """
You are a highly experienced AI specialized in extracting and understanding data from invoices. 
Your goal is to extract structured data from invoices, which may vary in format, language, and structure.

**Task:** Extract all relevant information from the provided invoice and structure it into a well-formatted JSON object.

**Points to Remember:**
1. **Accuracy is crucial**. Double-check all extracted information.
2. **Be adaptable**. Invoices can vary greatly in format and content.
3. **Maintain consistency** in data representation across different invoices.
4. **Handle edge cases** gracefully (e.g., missing information, unusual formats).

**Key Information to Extract:**
- Invoice number and date
- Supplier and buyer details (including addresses)
- Line items (products/services, quantities, unit prices, total amounts)
- Tax information (e.g., GST, VAT, sales tax)
- Total amount (before and after tax)
- Payment terms and methods
- Any additional fees or discounts

**Specific Instructions:**
1. **Identify and extract key details** such as invoice number, date, supplier name, item details, amounts, and addresses.
2. **Convert all information into English**, especially if the invoice is in another language. Specify the original language if translated.
3. **Split addresses** into components: street, city, state, and zip code.
4. **Clearly separate line items**, identifying products/services with quantities and unit prices.
5. **Identify responsible parties** like buyers, suppliers, and authorized signatories.
6. **Handle empty invoices** gracefully. If there are no line items, extract whatever information is available.
7. **Adapt to different invoice types**, including but not limited to:
   - Standard commercial invoices
   - Medical invoices (e.g., including OPD information)
   - Service invoices
   - Utility bills

**Examples:**
1. Standard Invoice:
   ```json
   {
     "invoice_number": "INV-2024-001",
     "invoice_date": "2024-03-15",
     "supplier": {
       "name": "ABC Corp",
       "address": {
         "street": "123 Business St",
         "city": "Metropolis",
         "state": "State",
         "zip_code": "12345"
       }
     },
     "buyer": {
       "name": "XYZ Ltd",
       "address": {
         "street": "456 Commerce Ave",
         "city": "Businessville",
         "state": "State",
         "zip_code": "67890"
       }
     },
     "line_items": [
       {
         "description": "Widget A",
         "quantity": "10",
         "unit_price": "15.00",
         "total": "150.00"
       },
       {
         "description": "Service B",
         "quantity": "5",
         "unit_price": "30.00",
         "total": "150.00"
       }
     ],
     "subtotal": "300.00",
     "tax": {
       "rate": "10%",
       "amount": "30.00"
     },
     "total": "330.00",
     "payment_terms": "Net 30",
     "payment_method": "Bank Transfer"
   }
   ```

2. Medical Invoice:
   ```json
   {
     "invoice_number": "MED-2024-001",
     "invoice_date": "2024-03-20",
     "hospital_name": "City General Hospital",
     "patient": {
       "name": "John Doe",
       "id": "PATIENT123"
     },
     "services": [
       {
         "description": "OPD Consultation",
         "date": "2024-03-20",
         "doctor": "Dr. Smith",
         "charge": "100.00"
       },
       {
         "description": "Blood Test",
         "date": "2024-03-20",
         "charge": "50.00"
       }
     ],
     "total_charge": "150.00",
     "insurance_coverage": "100.00",
     "patient_responsibility": "50.00"
   }
   ```

**Output Format:**
- Provide a single, valid JSON object.
- Format all numerical values consistently as strings (e.g., "252.00" instead of 252.00).
- Ensure all property names and string values are in double quotes.
- Do not include any text or explanations outside the JSON object.
- Do not use trailing commas in objects or arrays.

Remember, the goal is to create a structured, consistent representation of the invoice data that can be easily processed and analyzed.
"""

def extract_text(file_path: Path) -> str:
    """Extract text from PDF or DOCX file."""
    if file_path.suffix.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_path.suffix.lower() == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF file."""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in reader.pages).strip()
    except Exception as e:
        logging.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX file."""
    try:
        return docx2txt.process(file_path).strip()
    except Exception as e:
        logging.error(f"Error reading Word document: {e}")
        return ""

def load_image(file_path: Path) -> list:
    """Load image data from file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find the image: {file_path}")
    
    mime_type = f"image/{file_path.suffix[1:]}"
    return [{"mime_type": mime_type, "data": file_path.read_bytes()}]

def extract_invoice_data(file_path: Path, user_prompt: Optional[str] = None) -> str:
    """Extract invoice data from file using AI model."""
    try:
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.pdf', '.docx']:
            content = extract_text(file_path)
            input_prompt = [SYSTEM_PROMPT, content]
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            image_info = load_image(file_path)
            input_prompt = [SYSTEM_PROMPT, image_info[0]]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        if user_prompt:
            input_prompt.append(user_prompt)

        response = model.generate_content(input_prompt)
        
        if not response or not response.text:
            raise ValueError("No response received from the model.")
        
        return response.text

    except Exception as e:
        logging.error(f"Error during data extraction: {e}")
        return ""

def clean_json_string(json_string: str) -> str:
    """Clean and validate JSON string."""
    json_string = json_string.strip().strip('```json').strip('```')
    
    try:
        return json.dumps(json.loads(json_string), indent=2)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON: {e}")
        return json_string
    
def flatten_json(data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten a nested JSON structure."""
    result = {}
    for key, value in data.items():
        new_key = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(flatten_json(value, f"{new_key}_"))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    result.update(flatten_json(item, f"{new_key}_{i}_"))
                else:
                    result[f"{new_key}_{i}"] = item
        else:
            result[new_key] = value
    return result

def process_invoice(file_path: str, user_prompt: Optional[str] = None) -> pd.DataFrame:
    """Process invoice file and return extracted data as a flattened DataFrame."""
    file_path = Path(file_path)
    response_text = extract_invoice_data(file_path, user_prompt)
    cleaned_json = clean_json_string(response_text)

    try:
        json_data = json.loads(cleaned_json)
        flattened_data = flatten_json(json_data)
        return pd.DataFrame([flattened_data])
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {e}")
        logging.debug("Cleaned JSON string:")
        logging.debug(cleaned_json)
        return pd.DataFrame()

def main():
    st.title("Invoice Data Extractor")

    uploaded_file = st.file_uploader("Choose an invoice file", type=["pdf", "docx", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.success("File uploaded successfully!")

        if st.button("Process Invoice"):
            with st.spinner("Processing invoice..."):
                df = process_invoice(tmp_file_path)
            
            if not df.empty:
                st.success("Invoice processed successfully!")
                st.subheader("Extracted Data")
                
                # Transpose the DataFrame for better readability
                df_transposed = df.T.reset_index()
                df_transposed.columns = ['Field', 'Value']
                
                # Display the transposed DataFrame
                st.table(df_transposed)
            else:
                st.error("Failed to process the invoice. Please check the logs for more information.")

        # Clean up the temporary file
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
