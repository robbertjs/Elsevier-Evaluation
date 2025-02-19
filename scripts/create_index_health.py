from datasets import load_dataset, concatenate_datasets
from PIL import Image
import pandas as pd
import csv
import os
import tqdm
import io 
import json
import base64

def decode_base64_to_pil_image(encoded_str: str) -> Image.Image:
    """Convert a base64 string to a PIL Image."""
    image_bytes = base64.b64decode(encoded_str)
    image_file = io.BytesIO(image_bytes)
    return Image.open(image_file)

def pil_image_to_base64(image: Image.Image, format='JPEG') -> str:
    """Convert a PIL Image object to a base64 encoded string."""
    buffered = io.BytesIO()
    # Convert the image to RGB format if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def pil_image_to_bytes(image: Image.Image, format='JPEG') -> bytes:
    """Convert a PIL Image object to bytes."""
    buffered = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format=format)
    image_byte_data = buffered.getvalue()
    return image_byte_data

def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """Convert bytes data to a PIL Image object."""
    image_file = io.BytesIO(image_bytes)
    image = Image.open(image_file)
    return image

base_path = "/ivi/ilps/personal/jqiao/colpali/index_data/"
pdfvqa_train = load_dataset("gigant/pdfvqa", split='train')
pdfvqa_val = load_dataset("gigant/pdfvqa", split='validation')
pdfvqa_test = load_dataset("gigant/pdfvqa", split='test')
pdfvqa_combined = concatenate_datasets([pdfvqa_test, pdfvqa_val, pdfvqa_train])

image_filenames = [f"pdfvqa_{i}" for i in range(len(pdfvqa_combined))]

# Add the new column
pdfvqa_combined = pdfvqa_combined.add_column("image_filename", image_filenames)
test_set_500 = load_dataset('vidore/syntheticDocQA_healthcare_industry_test', split='test')
test_set_500_ocr = load_dataset('vidore/syntheticDocQA_healthcare_industry_test_tesseract', split='test')

data_records = []
for entry, entry2 in tqdm.tqdm(zip(test_set_500, test_set_500_ocr)):
    encoded_image = pil_image_to_base64(entry['image'])
    # encoded_image = Image.open(entry['image'])
    data_records.append({
                "image": encoded_image,
                "image_filename": entry['image_filename'],
                "query": entry['query'],
                "answer": entry['answer'],
                "source": entry['source'],
                'page': entry['page'],
                'model': entry['model'],
                'prompt': entry['prompt'],
                'text_description': entry2['text_description']
                })

for entry in tqdm.tqdm(pdfvqa_combined):
    # Load the image and convert it to a base64 string
    encoded_image = pil_image_to_base64(entry['page'])
    # encoded_image = Image.open(entry['image'])
    data_records.append({
        "image": encoded_image,
        "image_filename": entry['image_filename'],
        "query": entry['questions'],
        "answer": entry['answers'],
        "source": "pdfvqa",
        'page': "",
        'model': "",
        'prompt': "",
        "text_description": entry['texts']
    })

for size, label in [(1000, "1k"), (2500, "2.5k"), (5000, "5k"), (7500, "7.5k"), (10000, "10k")]:
    output_path = base_path + f"ai_test_index_{label}.jsonl"
    with open(output_path, 'w') as file:
        for record in data_records[:size]:
            json.dump(record, file)
            file.write('\n')
    print("Data has been saved to JSONL format.")
