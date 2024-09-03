import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import pytesseract
from PIL import Image

# Set up Tesseract OCR path (update this path according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # for Windows
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # for macOS/Linux

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def setup_model():
    # Set custom cache directory
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load a more capable seq2seq model
    model_name = "google/flan-t5-base"  # You can try larger versions like "google/flan-t5-large" for better performance
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

def extract_invoice_attributes(text, nlp_pipeline):
    prompt = f"""
    Given the following invoice text, identify all relevant attributes and their values. 
    Format the output as 'Attribute: Value'. for example
    Invoice Number :   12210,   
    Date : 26/02/2019 , 
    Total Amount  : $154.06  , 
    Vendor Name : John Smith
0   
find out other attributes.
    """
    
    result = nlp_pipeline(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']
    
    attributes = {}
    for line in result.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            attributes[key.strip()] = value.strip()
    
    return attributes

def generate_insights(text, nlp_pipeline):
    prompt = f"""
    Analyze the following invoice text and provide 5 key insights or observations.
    Each insight should be on a new line and start with "Insight: ".
    
    Invoice text:
    {text}
    
    Insights:
    """
    
    result = nlp_pipeline(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']
    
    insights = [line.strip() for line in result.split('\n') if line.strip().startswith("Insight:")]
    return insights

def main(image_path):
    # Extract text from image
    text = extract_text_from_image(image_path)
    # print(text)
    # Setup NLP pipeline
    nlp_pipeline = setup_model()
    
    # Extract invoice attributes
    invoice_data = extract_invoice_attributes(text, nlp_pipeline)
    print(invoice_data)
    # # Generate insights
    # insights = generate_insights(text, nlp_pipeline)
    
    # # Create DataFrame
    # df = pd.DataFrame([invoice_data])
    
    # # Add insights as new columns
    # for i, insight in enumerate(insights, 1):
    #     df[f'Insight_{i}'] = insight
    
    # return df
    return 0

# Example usage
image_path = "K:\Fractal_Gen_AI_Assignments\GenAI_Assignment2\images\invoice_image.png"
result_df = main(image_path)
print(result_df)

# To save the DataFrame to a CSV file
# result_df.to_csv("images\invoice_analysis.csv", index=False)
# print("Results saved to invoice_analysis.csv")