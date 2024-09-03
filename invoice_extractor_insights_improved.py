import os
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import pytesseract
from PIL import Image

# Set up Tesseract OCR path (update this path according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # for Windows
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # for macOS/Linux

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        print(f"Extracted text (first 1000 characters): {text[:1000]}...")
        return text
    except Exception as e:
        print(f"Error in extract_text_from_image: {str(e)}")
        return None

def setup_model():
    try:
        # Set custom cache directory
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load a more capable seq2seq model
        model_name = "google/flan-t5-small"  # Using a smaller model for faster loading
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
        
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"Error in setup_model: {str(e)}")
        return None

def extract_invoice_attributes(text, nlp_pipeline):
    try:
        prompt = f"""
        Extract key attributes and their values from the following invoice text. Provide the output as 'Attribute: Value' pairs, one per line. The invoice text includes typical information such as invoice number, date, company name, items purchased, quantities, and prices.

        Example invoice text:
        Invoice Number: INV12345
        Date: 2024-08-01
        Company: Example Corp
        Items:
        - Product A: 10 units @ $5.00 each
        - Product B: 2 units @ $20.00 each
        Total: $70.00

        Invoice text:
        {text[:1500]}  # Limiting text to 1500 characters
        
        Attributes:
        """
        
        print("Generating attributes...")
        result = nlp_pipeline(prompt, max_length=1500, num_return_sequences=1)[0]['generated_text']
        print(f"Generated attributes: {result}")
        
        attributes = {}
        for line in result.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                attributes[key.strip()] = value.strip()
        
        return attributes
    except Exception as e:
        print(f"Error in extract_invoice_attributes: {str(e)}")
        return {}

def generate_insights(text, nlp_pipeline):
    try:
        prompt = f"""
        Analyze the following invoice text and provide 3 key insights or observations. Each insight should be formatted as: Insight: [Description]. Examples of insights include observations about the total amount, unusual item prices, or discrepancies.

        Example invoice text:
        Invoice Number: INV12345
        Date: 2024-08-01
        Company: Example Corp
        Items:
        - Product A: 10 units @ $5.00 each
        - Product B: 2 units @ $20.00 each
        Total: $70.00

        Invoice text:
        {text[:1500]}  # Limiting text to 1500 characters
        
        Insights:
        """
        
        print("Generating insights...")
        result = nlp_pipeline(prompt, max_length=1500, num_return_sequences=1)[0]['generated_text']
        print(f"Generated insights: {result}")
        
        insights = [line.strip() for line in result.split('\n') if line.strip().startswith("Insight:")]
        return insights
    except Exception as e:
        print(f"Error in generate_insights: {str(e)}")
        return []

def main(image_path):
    # Extract text from image
    text = extract_text_from_image(image_path)
    if text is None:
        return None
    
    # Setup NLP pipeline
    nlp_pipeline = setup_model()
    if nlp_pipeline is None:
        return None
    
    # Extract invoice attributes
    invoice_data = extract_invoice_attributes(text, nlp_pipeline)
    print(f"Extracted invoice data: {invoice_data}")
    
    # Generate insights
    insights = generate_insights(text, nlp_pipeline)
    print(f"Generated insights: {insights}")
    
    # Create DataFrame
    df = pd.DataFrame([invoice_data])
    
    # Add insights as new columns
    for i, insight in enumerate(insights, 1):
        df[f'Insight_{i}'] = [insight]  # Ensure insights are added as a list to maintain DataFrame structure
    
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the invoice image as a command-line argument.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"Processing image: {image_path}")
    
    result_df = main(image_path)
    
    if result_df is not None and not result_df.empty:
        print("\nExtracted Data:")
        print(result_df)
        
        # To save the DataFrame to a CSV file
        csv_path = "invoice_analysis.csv"
        result_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    else:
        print("Failed to extract data from the invoice.")
