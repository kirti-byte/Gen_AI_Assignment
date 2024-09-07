import google.generativeai as genai
from pathlib import Path
import pandas as pd
import json
import PyPDF2  # Library to read PDFs

# Reset display options to default.
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')


# Securely configure the API key (replace with your actual API key)
API_KEY = "AIzaSyCNoFHCAJjBh4y_V8Kj0BjNhjmiNruM3sQ"

# Configure the Generative AI client with the provided API key.
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: Extracted text content from the PDF.
    """
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



def load_image(image_path: str) -> list:
    """
    Validates and loads the image data from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        list: A list containing the image parts with the mime type and binary data.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
    """
    img = Path(image_path)

    # Check if the image file exists at the specified path.
    if not img.exists():
        raise FileNotFoundError(f"Could not find the image: {img}")

    # Return the image data in the required format.
    return [{"mime_type": "image/png", "data": img.read_bytes()}]

def extract_invoice_data(image_path: str, system_prompt: str, user_prompt: str) -> str:
    """
    Extracts invoice data by generating content using AI based on the provided image and prompts.

    Args:
        image_path (str): The path to the invoice image file.
        system_prompt (str): The system prompt defining the AI's task.
        user_prompt (str): The prompt requesting JSON conversion of the invoice data.
        user_prompt2 (str): The prompt requesting conversion of JSON data to a DataFrame.

    Returns:
        str: The generated AI response text containing extracted invoice data.
    """
    try:
        image_info = load_image(image_path)
        input_prompt = [system_prompt, image_info[0], user_prompt]
        response = model.generate_content(input_prompt)

        # Validate response.
        if not response or not response.text:
            raise ValueError("No response received from the model.")

        return response.text

    except Exception as e:
        print(f"Error during data extraction: {e}")
        return ""


# System prompt explaining the role of AI in extracting invoice data.
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
Output the data in a well-structured JSON format with appropriate tags for clarity and consistency.
"""

# Define the path to the invoice image.
image_path = "K:\\Fractal_Gen_AI_Assignments\\Gen_AI_Assignment\\images\\chinese_invoice_image.png"
pdf_path = "K:\\Fractal_Gen_AI_Assignments\\Gen_AI_Assignment\\images\\invoice3.pdf"


# User prompt for converting invoice data into JSON format.
user_prompt = """
Extract all invoice information and convert it into a structured JSON format. Include:
- Invoice number, date, supplier name, and buyer details.
- Extract itemized products/services with quantities, unit prices, and total amounts.
- Split addresses into components (street, city, state, zip code).
- Identify any responsible parties mentioned (e.g., buyer, supplier, signatory).
- If the invoice original language is not in English, detect the language, translate suuplier name and line items in English, and specify the original language.
- Ensure that each piece of information is tagged correctly to ensure clarity, such as separating amounts, taxes, and any discount information.
"""

# user_prompt = """give supplier name in english"""


# Extract text from the PDF.
extracted_text = extract_text_from_pdf(pdf_path)

# Extract invoice data using the AI model.
response_text = extract_invoice_data(extracted_text, system_prompt, user_prompt)
cleaned_response_text = response_text.strip('```json').strip('```').strip()

# Extract invoice data using the AI model.
# response_text = extract_invoice_data(image_path, system_prompt, user_prompt)
# cleaned_response_text = response_text.strip('```json').strip('```').strip()

try:
    # Attempt to load JSON data after cleaning
    json_data = json.loads(cleaned_response_text)
    # print("Parsed JSON Data:", json_data)
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    print("Ensure the input string is a valid JSON format without extra markdown formatting.")
    print("Response Text:", cleaned_response_text)


# Convert main data to DataFrame
main_df = pd.json_normalize(json_data)
# print(main_df)
main_df.to_csv("invoice_analysis.csv")