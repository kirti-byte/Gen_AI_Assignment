# GenAI_Assignments
This repository contains assignment given by fractal's pilot army to enhance learning and hands on practice

A quantized model is a compressed version of a larger model. Quantization reduces the precision of the model's weights (e.g., from 32-bit floating-point to 8-bit integer), which significantly reduces the model's size and increases inference speed, often with minimal loss in accuracy.
The model we're using, "deepset/tinyroberta-squad2", is a small, efficient model fine-tuned for question answering tasks. While it's not explicitly mentioned as quantized in the code, it's designed to be lightweight and fast, suitable for deployment in environments with limited resources.
Key benefits of using this approach:

Faster inference: The smaller model size allows for quicker processing of invoices.
Lower memory usage: Requires less RAM, making it suitable for deployment on various devices.
Easier deployment: Smaller model size means easier distribution and updates.
Maintained accuracy: Despite being smaller, it's still capable of accurately extracting information from structured documents like invoices.

This approach strikes a balance between the speed of rule-based systems and the flexibility of large language models, making it well-suited for tasks like invoice information extraction where the input format is semi-structured and the required information is well-defined.

os: Used for file and directory operations.
transformers: A library by Hugging Face for state-of-the-art Natural Language Processing.
torch: PyTorch library for tensor computations and deep learning.

AutoTokenizer: Automatically selects the appropriate tokenizer for the model.
AutoModelForQuestionAnswering: Loads a pre-trained model for question answering tasks.
from_pretrained(): Downloads and caches the model and tokenizer.

torch.device(): Selects GPU if available, otherwise uses CPU.
model.to(device): Moves the model to the selected device.

pipeline(): Creates a high-level pipeline for easy inference.

