from datasets import load_dataset
import easyocr
from PIL import Image
import numpy as np
import argparse
import json

class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder that converts NumPy data types to Python types. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def calculate_text_coverage(image, reader):
    # Perform OCR
    results = reader.readtext(np.array(image))
    # Calculate the total area of the image
    total_area = image.size[0] * image.size[1]

    # Initialize a variable to store the sum of areas covered by text
    text_area = 0

    # Iterate through the results to calculate the text coverage
    for (bbox, text, prob) in results:
        top_left, top_right, bottom_right, bottom_left = bbox
        # Calculate the width and height of the bounding box
        width = abs(top_right[0] - top_left[0])
        height = abs(bottom_left[1] - top_left[1])
        # Calculate the area of the bounding box
        bounding_box_area = width * height
        # Add to the total text area
        text_area += bounding_box_area

    # Calculate text coverage percentage
    text_coverage = (text_area / total_area) * 100
    return text_coverage, text_area, total_area, results

def calculate_text_coverage_and_density(image, reader):
    # Perform OCR
    results = reader.readtext(np.array(image))
    # Calculate the total area of the image
    total_area = image.size[0] * image.size[1]

    # Initialize variables to store the sum of areas covered by text and token count
    text_area = 0
    total_tokens = 0

    # Iterate through the results to calculate the text coverage and count tokens
    for (bbox, text, prob) in results:
        top_left, top_right, bottom_right, bottom_left = bbox
        width = abs(top_right[0] - top_left[0])
        height = abs(bottom_left[1] - top_left[1])
        bounding_box_area = width * height
        text_area += bounding_box_area
        total_tokens += len(text.split())  # Count tokens by splitting text by spaces

    # Calculate text coverage percentage and text density
    text_coverage = (text_area / total_area) * 100
    text_density = (total_tokens / text_area) * 100 if text_area > 0 else 0  # Avoid division by zero
    text_salience = text_coverage * text_density

    return text_coverage, text_area, total_area, results, text_density, text_salience

def serialize_ocr_results(ocr_results):
    """Convert OCR results to a list of dictionaries."""
    serialized_results = []
    for result in ocr_results:
        bbox, text, confidence = result
        # Convert bounding box coordinates to a simple list
        bbox = [list(point) for point in bbox]
        serialized_results.append({
            "bbox": bbox,
            "text": text,
            "confidence": confidence
        })
    return serialized_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files.')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset, split="test")
    # reader = easyocr.Reader(['en'])
    reader = easyocr.Reader(['en'], gpu=True)
    # Open the output file correctly
    with open(args.output, 'w') as f:
      for data in dataset:
        query = data['query']
        image = data['image']  # Assume this is a PIL image directly
        # text_coverage, text_area, total_area, results = calculate_text_coverage(image, reader)
        text_coverage, text_area, total_area, results, text_density, text_salience = calculate_text_coverage_and_density(image, reader)
        
        try: 
            json_output = json.dumps({
                'query': query,
                'text_area': text_area,
                'total_area': total_area,
                'text_coverage': text_coverage,
                'text_density': text_density,
                'text_salience': text_salience,
                'results': results
            }, cls=NumpyEncoder)
            f.write(json_output + '\n')

        except Exception as e:
            print(f"Error processing image for query '{query}': {e}")