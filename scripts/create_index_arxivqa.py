from datasets import load_dataset
from PIL import Image
import pandas as pd
import csv
import os
import tqdm
import io 
import json
import base64
import easyocr

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
    # Save the image data to BytesIO object using the specified format
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def pil_image_to_bytes(image: Image.Image, format='JPEG') -> bytes:
    """Convert a PIL Image object to bytes."""
    # Create a BytesIO object to hold the image data
    buffered = io.BytesIO()
    # Convert the image to RGB format if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Save the image data to BytesIO object using the specified format
    image.save(buffered, format=format)
    # Get the byte data from the BytesIO object
    image_byte_data = buffered.getvalue()
    return image_byte_data

def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """Convert bytes data to a PIL Image object."""
    # Create a BytesIO object from the byte data
    image_file = io.BytesIO(image_bytes)
    # Use PIL to open the image from the BytesIO object
    image = Image.open(image_file)
    # You may need to convert the image to a specific mode or perform additional processing here
    return image

def extract_text(image):
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image)
    text_output = []
    for (bbox, text, prob) in results:
        text_output.append((text))

    return " ".join(text_output)

# load ArxivQ original test set 
base_path = "/ivi/ilps/personal/jqiao/colpali/index_data/"
file_path = base_path + "arxivqa.jsonl"

with open(file_path, 'r') as fr:
  arxiv_qa = [json.loads(line.strip()) for line in tqdm.tqdm(fr)]

colpali_train_path = "/ivi/ilps/personal/jqiao/colpali/data_dir/colpali_train_set"
train_set = load_dataset(colpali_train_path, split='train')


train_filenames = set(
    entry['image_filename'] for entry in tqdm.tqdm(train_set) if entry['source'] == 'arxiv_qa'
)

# Load the subsampled ArxivQ test set
test_set_500 = load_dataset('vidore/arxivqa_test_subsampled', split='test')
test_500_filenames = set(entry['image_filename'] for entry in test_set_500)
test_set_500_ocr = load_dataset('vidore/arxivqa_test_subsampled_tesseract', split='test')

# Prepare DataFrame
data_records = []

for entry, entry2 in tqdm.tqdm(zip(test_set_500, test_set_500_ocr)):
    encoded_image = pil_image_to_base64(entry['image'])
    data_records.append({
                "image": encoded_image,
                "image_filename": entry['image_filename'],
                "query": entry['query'],
                "answer": entry['answer'],
                "source": entry['source'],
                "options": entry['options'],
                'page': entry['page'],
                'model': entry['model'],
                'prompt': entry['prompt'],
                'text_description': entry2['text_description']
            })  


def save_data_to_jsonl(base_path, data_records):

    output_path = f"{base_path}arxivqa_test_index_{len(data_records)/1000}k.jsonl"
    with open(output_path, 'w') as file:
        for record in data_records[:len(data_records)]:
            json.dump(record, file)
            file.write('\n')
    print(f"Data has been saved to JSONL format with {len(data_records)/1000}k records at: {output_path}")


# Iterate through entries
for entry in tqdm.tqdm(arxiv_qa):

    if len(data_records) in [500, 2500, 5000, 7500, 10000, 50000]:
        save_data_to_jsonl(base_path, data_records)

    if entry['image'] not in train_filenames and entry['image'] not in test_500_filenames:
        # Load the image and convert it to a base64 string
        image_path = base_path + entry['image']
        image = Image.open(image_path)
        encoded_image = pil_image_to_base64(image)
        text_description = extract_text(image_path)

        data_records.append({
            "image": encoded_image,
            "image_filename": entry['image'],
            "query": entry['question'],
            "answer": entry['label'],
            "source": "arxiv_qa",
            "options": entry['options'],
            'page': "",
            'model': "",
            'prompt': "", 
            'text_description': text_description
        })

# for size, label in [(1000, "1k"), (2500, "2.5k"), (7500, "7.5k"), (10000, "10k"), (50000, "50k")]:
#     output_path = base_path + f"arxivqa_test_index_{label}.jsonl"
#     with open(output_path, 'w') as file:
#         for record in data_records[:size]:
#             json.dump(record, file)
#             file.write('\n')
#     print("Data has been saved to JSONL format.")
