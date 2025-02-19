import argparse
import json
import numpy as np
import cv2
from PIL import Image
import easyocr
from datasets import load_dataset
import tqdm

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy data types in JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def count_background_in_bbox(np_image, bbox, target_color=(255, 255, 255), threshold=10):
    """
    Count background pixels within a specified bounding box in an image.
    Ensures the image is in RGB format for processing.
    """
    x_min, y_min = int(min(bbox, key=lambda x: x[0])[0]), int(min(bbox, key=lambda x: x[1])[1])
    x_max, y_max = int(max(bbox, key=lambda x: x[0])[0]), int(max(bbox, key=lambda x: x[1])[1])

    # Check if coordinates result in a valid region
    if x_min >= x_max or y_min >= y_max:
        return 0  # No valid region to process

    # Crop the image to the bounding box
    cropped_img = np_image[y_min:y_max, x_min:x_max]

    # Check if cropped image is empty
    if cropped_img.size == 0:
        return 0  # No data in the cropped region

    # Crop the image to the bounding box
    cropped_img = np_image[y_min:y_max, x_min:x_max]

    # Ensure cropped image is in RGB
    if cropped_img.ndim == 2 or (cropped_img.shape[2] == 1):
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

    # Convert target color to a numpy array and calculate the distance
    target_color = np.array(target_color, np.uint8)
    dist = np.linalg.norm(cropped_img.astype("float32") - target_color.astype("float32"), axis=2)
    background_mask = (dist < threshold).astype(np.uint8)
    background_pixels = np.sum(background_mask)
    
    return background_pixels

def estimate_background_area(np_image, target_color=(255, 255, 255), threshold=10):
    """
    Estimate the total number of background pixels in the image.
    
    Parameters:
        np_image (np.ndarray): The image as a NumPy array.
        target_color (tuple): The background color to match (default is white).
        threshold (int): Distance threshold for considering a pixel as background.
    
    Returns:
        int: The total number of background pixels in the image.
    """
    # Ensure the image is in RGB format (i.e. three channels)
    if len(np_image.shape) == 2:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
    elif np_image.shape[2] == 1:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
    
    image_float = np.float32(np_image)
    target_color_arr = np.array(target_color, dtype=np.float32)
    dist = np.sqrt(np.sum((image_float - target_color_arr) ** 2, axis=2))
    background_mask = (dist < threshold).astype(np.uint8)
    background_pixels = int(np.sum(background_mask))
    
    return background_pixels


def calculate_coverage(np_image, reader):
    """
    Calculate coverage statistics from the image.
    
    Uses OCR to detect text, calculates the area of bounding boxes for text,
    subtracts background pixels within those boxes, and returns various metrics.
    
    Parameters:
        np_image (np.ndarray): The image as a NumPy array.
        reader (easyocr.Reader): An EasyOCR reader instance.
    
    Returns:
        dict: A dictionary containing metrics such as token count, text area,
              overall image area, total bounding box area, background area within
              text bounding boxes, and overall background pixel count.
    """
    # Run OCR on the image
    results = reader.readtext(np_image)
    total_area = np_image.shape[0] * np_image.shape[1]
    overall_background_pixels = estimate_background_area(np_image)

    text_bbox_area = 0
    total_tokens = 0
    total_bbox_background_area = 0

    for (bbox, text, prob) in results:
        # Count background pixels within the current text bounding box
        bbox_background_pixels = count_background_in_bbox(np_image, bbox)
        total_bbox_background_area += bbox_background_pixels

        # Calculate the bounding box area assuming rectangular shape
        top_left, top_right, bottom_right, bottom_left = bbox
        width = abs(top_right[0] - top_left[0])
        height = abs(bottom_left[1] - top_left[1])
        bounding_box_area = width * height
        text_bbox_area += bounding_box_area

        # Token count based on splitting detected text
        total_tokens += len(text.split())

    # Calculate the effective text area after subtracting background within text boxes
    total_text_area = text_bbox_area - total_bbox_background_area

    return {
        "total_tokens": total_tokens,
        "total_text_area": total_text_area,
        "total_area": total_area,
        "text_bbox_area": text_bbox_area,
        "background_area_within_bboxes": total_bbox_background_area,
        "background_pixels": overall_background_pixels
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from a dataset and compute coverage metrics.")
    parser.add_argument("--dataset", type=str, required=True, help="Name or path of the dataset to load.")
    parser.add_argument("--output", type=str, required=True, help="Output file path to write the results.")
    args = parser.parse_args()

    # Load the dataset split (using "test" split here; adjust if needed)
    dataset = load_dataset(args.dataset, split="test")
    
    # Initialize EasyOCR; set gpu=True if a GPU is available
    reader = easyocr.Reader(['en'], gpu=True)

    with open(args.output, 'w') as f:
        for data in tqdm.tqdm(dataset) :
            query = data.get('query', 'N/A')
                # Ensure image is in RGB mode using PIL
            image_np = np.array(data['image'])
            # Calculate coverage metrics using OCR and image processing
            background_info = calculate_coverage(image_np, reader)
            # Merge query with the background_info dictionary
            json_output = json.dumps({
                'query': query,
                **background_info
            }, cls=NumpyEncoder)
            
            f.write(json_output + '\n')