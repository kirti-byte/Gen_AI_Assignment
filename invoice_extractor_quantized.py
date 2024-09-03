import pytesseract
from PIL import Image
import pandas as pd
import re
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Set up Tesseract OCR path (update this path according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # for Windows
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # for macOS/Linux

def extract_text_from_image(image_path1):
    
    image = Image.open(image_path1)
    text = pytesseract.image_to_string(image)
    return text

def extract_invoice_data_with_rules(text):
    patterns = {
        'Invoice Number': r'Invoice\s*(?:#|Number|No|Num)?\s*[:.]?\s*(\w+)',
        'Date': r'Date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        'Total Amount': r'Total\s*(?:Amount)?\s*[:.]?\s*[$£€]?\s*([\d,.]+)',
        'Vendor Name': r'(.*?)\n'  # Assumes vendor name is the first line
    }
    
    data = {} 
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()
    
    return data

def extract_invoice_data_with_model(text):
    # Load a smaller, quantized model
    model_name = "deepset/tinyroberta-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
    
    questions = {
       "prompt": "what are the different insights can be drived from this invoice data, make seperate column for those attributs along with its value."
    }
    
    data = {}
    for key, question in questions.items():
        result = qa_pipeline(question=question, context=text)
        data[key] = result['answer']
    
    return data

def main(image_path1, use_model=False):
    # Extract text from image
    text = extract_text_from_image(image_path1)
    
    # Extract invoice data
    if use_model:
        invoice_data = extract_invoice_data_with_model(text)
    else:
        invoice_data = extract_invoice_data_with_rules(text)
    
    # Create DataFrame
    df = pd.DataFrame([invoice_data])
    
    return df
    # return 0

# Example usage
image_path1 = "K:\Fractal_Gen_AI_Assignments\GenAI_Assignment2\images\invoice_image.png"
result_df = main(image_path1, use_model=True)  # Set to True to use the model-based approach
print(result_df)

# To save the DataFrame to a CSV file
csv_path = "invoice_analysis.csv"
result_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")
