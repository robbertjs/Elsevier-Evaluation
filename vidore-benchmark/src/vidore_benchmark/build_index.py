import argparse
import logging
import os
from pathlib import Path
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from vidore_benchmark.compression.token_pooling import HierarchicalEmbeddingPooler
from vidore_benchmark.evaluation.indexing import indexing
from vidore_benchmark.retrievers.registry_utils import load_vision_retriever_from_registry
from vidore_benchmark.utils.logging_utils import setup_logging
import huggingface_hub
import json
import csv
import os
import base64
import tqdm
from PIL import Image
from datasets import Dataset
import re
import io
import numpy as np

logger = logging.getLogger(__name__)
load_dotenv(override=True)
OUTPUT_DIR = Path("outputs")

def decode_base64_to_pil_image(encoded_str: str) -> Image.Image:
    """Convert a base64 string to a PIL Image."""
    image_bytes = base64.b64decode(encoded_str)
    image_file = io.BytesIO(image_bytes)
    return Image.open(image_file)

def decode_base64_to_numpy_array(encoded_str: str):

    image_bytes = base64.b64decode(encoded_str)
    image_file = io.BytesIO(image_bytes)
    image = Image.open(image_file)
    return np.array(image)

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


def list_to_dict(data_list):
    keys = data_list[0].keys()
    data_dict = {key: [dic[key] for dic in data_list] for key in keys}
    return data_dict

def process_batch(dataset_dict):
    """Process the current batch of data."""
    # Convert the current batch dictionary to a Dataset
    dataset = Dataset.from_dict(dataset_dict)
    print("Processed a batch of size:", len(dataset_dict['query']))
    return dataset

def build_index(args):
    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(
        args.model_class,
        pretrained_model_name_or_path=args.model_name,
    )

    # Get the pooling strategy
    embedding_pooler = HierarchicalEmbeddingPooler(args.pool_factor) if args.use_token_pooling else None
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collection_name = args.collection_name
    savedir = OUTPUT_DIR / "indexing"
    savedir.mkdir(parents=True, exist_ok=True)

    if collection_name.endswith('.jsonl'):
        # dataset = []
        dataset_dict = {'query': [], 'image': [], 'image_filename': [], 'text_description': []}

        emb_passages = []
        with open(collection_name, 'r') as file:
            for line in tqdm.tqdm(file):
                data = json.loads(line)
                if "arxivqa" in collection_name:
                    image = Image.open("/ivi/ilps/personal/jqiao/colpali/index_data/" + str(data['image_filename']))
                    dataset_dict['image'].append(image)
                else:
                    # image = Image.open(data['image'])
                    if "image" not in data or data["image"] is None:
                        logger.warning(f"Skipping passage due to missing image: {data.get('image_filename', 'unknown')}")
                        continue  # Skip this sample entirely
                    try:
                        image = decode_base64_to_pil_image(data["image"])
                        dataset_dict["image"].append(image)
                    except Exception as e:
                        logger.warning(f"Failed to decode image for {data.get('image_filename', 'unknown')}: {e}")
                        continue  # Skip broken image
                    # image = decode_base64_to_pil_image(data['image'])
                    # dataset_dict['image'].append(image)
                dataset_dict['query'].append(str(data['query']))
                dataset_dict['image_filename'].append(str(data['image_filename']))
                dataset_dict['text_description'].append(str(data['text_description']))

                # Check if we've reached the batch size limit
                if len(dataset_dict['query']) == 500:
                    dataset = process_batch(dataset_dict)
                    batch_emb_passages = indexing(retriever,
                                    dataset,
                                    batch_passage=args.batch_passage)

                    if isinstance(batch_emb_passages, torch.Tensor):
                        batch_emb_passages = list(torch.unbind(batch_emb_passages))
                        emb_passages.extend(batch_emb_passages)
                    else:
                        emb_passages.extend(batch_emb_passages)

                    # emb_passages.extend(embs)
                    # Clear the dictionary for the next batch
                    dataset_dict = {'query': [], 'image': [], 'image_filename': [], 'text_description': []}
         
            if dataset_dict['query']:
                dataset = process_batch(dataset_dict)
                batch_emb_passages = indexing(
                                retriever,
                                dataset,
                                batch_passage=args.batch_passage)
                # emb_passages.extend(embs)

                if isinstance(batch_emb_passages, torch.Tensor):
                    batch_emb_passages = list(torch.unbind(batch_emb_passages))
                    emb_passages.extend(batch_emb_passages)
                else:
                    emb_passages.extend(batch_emb_passages)

        if embedding_pooler:
            for idx, emb_document in enumerate(emb_passages):
                emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
                emb_passages[idx] = emb_document
        
        lc = collection_name.lower()
        data_name = "custom"
        if "health" in lc:
            data_name = "health"
        elif "ai" in lc:
            data_name = "ai"
        elif "arxivqa" in lc:
            data_name = "arxivqa"
        
        print("start saving", len(emb_passages))
        save_path = savedir / f"{args.model_class}_{data_name}_indexing_results_{args.output_name}.pt"
        torch.save({"embeddings": emb_passages}, save_path)
        print("Embeddings saved in ", save_path)

    else:
        if os.path.isdir(collection_name):
            print(f"Loading datasets from local directory: `{collection_name}`")
            dataset_names = os.listdir(collection_name)
            dataset_names = [os.path.join(collection_name, dataset) for dataset in dataset_names]
        else:
            print(f"Loading datasets from the Hf Hub collection: {collection_name}")
            collection = huggingface_hub.get_collection(collection_name)
            dataset_names = [dataset_item.item_id for dataset_item in collection.items]

        emb_passages = []
        for dataset_name in dataset_names:
            print(f"\n ---------------------------\nProcessing {dataset_name}")
            dataset = load_dataset(dataset_name, split=args.split)
            embeddings = indexing(
                retriever,
                dataset,
                batch_passage=args.batch_passage,
            )
            emb_passages.extend(embeddings)

        if embedding_pooler:
            for idx, emb_document in enumerate(emb_passages):
                emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
                emb_passages[idx] = emb_document

        print("start saving")
        save_path = savedir / f"{args.model_class}_indexing_results_num_{number}.pt"
        torch.save({"embeddings": emb_passages}, save_path)
        print("Embeddings saved in ", save_path)

def main():
    parser = argparse.ArgumentParser(description="Build Index for Vision Retriever")
    parser.add_argument("--model-class", type=str, help="Model class")
    parser.add_argument("--model-name", type=str, help="Pretrained model name or path")
    parser.add_argument("--dataset-name", type=str, help="HuggingFace Hub dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--batch-query", type=int, default=8, help="Batch size for query embedding inference")
    parser.add_argument("--batch-passage", type=int, default=8, help="Batch size for passages embedding inference")
    parser.add_argument("--batch-score", type=int, default=16, help="Batch size for score computation")
    parser.add_argument("--collection-name", type=str, help="Dataset collection to use for evaluation")
    parser.add_argument("--use-token-pooling", action="store_true", help="Whether to use token pooling for text embeddings")
    parser.add_argument("--pool-factor", type=int, default=3, help="Pooling factor for hierarchical token pooling")
    parser.add_argument("--output-name", type=str, help="HuggingFace Hub dataset name")

    args = parser.parse_args()

    # Log each argument
    print("Script parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")


    build_index(args)

if __name__ == "__main__":
    main()