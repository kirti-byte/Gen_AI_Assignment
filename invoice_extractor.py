import pytesseract
from PIL import Image
import pandas as pd
from transformers import GPTJForCausalLM, AutoTokenizer
import torch

# Set up Tesseract OCR path (update this path according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # for Windows
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # for macOS/Linux

# Load GPT-J model and tokenizer
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def extract_invoice_data(text):
    prompt = f"Extract the following information from this invoice text:\nInvoice Number:\nDate:\nTotal Amount:\nVendor Name:\n\nInvoice text:\n{text}\n\nExtracted Information:"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=300, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(output[0])
    
    return response

def parse_extracted_data(extracted_text):
    lines = extracted_text.split('\n')
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    return data

def main(image_path):
    # Extract text from image
    text = extract_text_from_image(image_path)
    
    # Extract invoice data using LLM
    extracted_text = extract_invoice_data(text)
    
    # Parse extracted data
    invoice_data = parse_extracted_data(extracted_text)
    
    # Create DataFrame
    df = pd.DataFrame([invoice_data])
    
    return df

# Example usage
image_path = "GenAI_Assignment2/images/invoice_image.png"
result_df = main(image_path)
print(result_df)