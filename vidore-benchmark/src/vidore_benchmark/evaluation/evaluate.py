from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
import torch
from datasets import Dataset
from tqdm import tqdm
from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from PIL import Image
# import pandas as pd
# import csv
# import os
# import tqdm
import io 
import base64


def keep_top_100_scores(data):
    # Iterate through each nested dictionary in the main dictionary
    for key, nested_dict in data.items():
        # Sort the items in the nested dictionary by score in descending order and keep the top 100
        top_100 = dict(sorted(nested_dict.items(), key=lambda item: item[1], reverse=True)[:100])
        data[key] = top_100
    return data


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


# def evaluate_dataset_matching(
#     vision_retriever: VisionRetriever,
#     ds: Dataset,
#     matching_type: str,
#     batch_query: int,
#     batch_passage: int,
#     batch_score: Optional[int] = None,
#     embedding_pooler: Optional[BaseEmbeddingPooler] = None,
# ) -> Dict[str, Optional[float]]:
#     """
#     Evaluate the model on a given dataset using the MTEB metrics.

#     NOTE: The dataset should contain the following columns:
#     - query: the query text
#     - image_filename: the filename of the image
#     - image: the image (PIL.Image) if `use_visual_embedding` is True
#     - text_description: the text description (i.e. the page caption or the text chunks) if
#         `use_visual_embedding` is False
#     """

#     # Dataset: sanity check
#     passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
#     required_columns = ["query", passage_column_name, "image_filename"]

#     if not all(col in ds.column_names for col in required_columns):
#         raise ValueError(f"Dataset should contain the following columns: {required_columns}")

#     seen_queries = set()
#     queries = []
#     for query in ds["query"]:
#         if query is not None and query not in seen_queries:
#             queries.append(query)
#             seen_queries.add(query)

#     if len(queries) == 0:
#         raise ValueError("All queries are None")

#     # Edge case: using the BM25Retriever
#     if isinstance(vision_retriever, BM25Retriever):
#         passages = ds[passage_column_name]
#         scores = vision_retriever.get_scores_bm25(queries=queries, passages=passages)
#         relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
#         metrics = vision_retriever.compute_metrics(relevant_docs, results)
#         return metrics

#     emb_queries = vision_retriever.forward_queries(queries, batch_size=batch_query)

#     # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
#     # that will be fed to the model in batches (this should be fine for queries as their memory footprint
#     # is negligible. This optimization is about efficient data loading, and is not related to the model's
#     # forward pass which is also batched.
#     emb_passages: List[torch.Tensor] = []

#     dataloader_prebatch_size = 10 * batch_passage
#     for passage_batch in tqdm(
#         batched(ds, n=dataloader_prebatch_size),
#         desc="Dataloader pre-batching",
#         total=math.ceil(len(ds) / (dataloader_prebatch_size)),
#     ):
#         passages: List[Any] = [db[passage_column_name] for db in passage_batch]
        
#         batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)
#         if isinstance(batch_emb_passages, torch.Tensor):
#             batch_emb_passages = list(torch.unbind(batch_emb_passages))
#             emb_passages.extend(batch_emb_passages)
#         else:
#             emb_passages.extend(batch_emb_passages)

#     if embedding_pooler is not None:
#         for idx, emb_document in tqdm(enumerate(emb_passages), total=len(emb_passages), desc="Pooling embeddings..."):
#             emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
#             emb_passages[idx] = emb_document
    
#     # For text lexical/semantic matching, compute token-level matching indices.
#     semantic_matching_indices: Optional[List[List[Tuple[List[int], List[int]]]]] = None
#     suffix = "<|endoftext|>" * 10
#     if matching_type in ("text_lexical"):
#         if not hasattr(vision_retriever, "processor") or not hasattr(vision_retriever.processor, "tokenizer"):
#             print("Tokenizer not found in vision_retriever.processor; skipping semantic matching indices computation.")
#         else:
#             tokenizer = vision_retriever.processor.tokenizer
#             # Tokenize each query into a list of tokens.
#             query_tokens: List[List[str]] = [tokenizer.tokenize("Query: " + query + suffix) for query in queries]
            
#             # Tokenize each passage text.
#             passage_texts = ds[passage_column_name]
#             passage_tokens: List[List[str]] = [tokenizer.tokenize(text) for text in passage_texts]
#             # Compute matching indices for every query–passage pair.
#             semantic_matching_indices = []
#             for qt in query_tokens:
#                 query_matching = []
#                 for pt in passage_tokens:
#                     common_tokens = set(qt).intersection(pt)
#                     # Record positions in the query that contain a common token.
#                     q_indices = [i for i, token in enumerate(qt) if token in common_tokens]
#                     # Record positions in the passage that contain a common token.
#                     p_indices = [i for i, token in enumerate(pt) if token in common_tokens]
#                     query_matching.append((q_indices, p_indices))
#                 semantic_matching_indices.append(query_matching)
#             print("Computed semantic matching indices for queries and passages.")
#     elif matching_type in ("text_semantic"):
#         if not hasattr(vision_retriever, "processor") or not hasattr(vision_retriever.processor, "tokenizer"):
#             print("Tokenizer not found in vision_retriever.processor; skipping semantic matching indices computation.")
#         else:
#             tokenizer = vision_retriever.processor.tokenizer
#             # Tokenize each query into a list of tokens.
#             query_tokens: List[List[str]] = [tokenizer.tokenize("Query: " + query + suffix) for query in queries]
#             special_tokens: List[List[str]] = [tokenizer.tokenize("Query: " + suffix)] 
#             # Tokenize each passage text.
#             passage_texts = ds[passage_column_name]
#             passage_tokens: List[List[str]] = [tokenizer.tokenize(text) for text in passage_texts]
#             # Compute matching indices for every query–passage pair.
#             semantic_matching_indices = []
#             for qt in query_tokens:
#                 query_matching = []
#                 for pt in passage_tokens:
#                     common_tokens = set(qt).intersection(pt)
#                     # Record positions in the query that contain a common token.
#                     # q_indices = [i for i, token in enumerate(qt) if token in unique_tokens]
#                     q_indices = [i for i, token in enumerate(qt) if token not in common_tokens and special_tokens]
#                     # Record positions in the passage that contain a common token.
#                     p_indices = [i for i, token in enumerate(pt) if token not in common_tokens and special_tokens]
#                     query_matching.append((q_indices, p_indices))
#                 semantic_matching_indices.append(query_matching)
#             print("Computed semantic matching indices for queries and passages.")
#     else:
#         semantic_matching_indices = None

#     start_time = time.time()
#     print("start to search ", start_time, "number of queries ", len(emb_queries), "number of passages ", len(ds), len(emb_passages))
#     # Get the similarity scores
#     scores = vision_retriever.get_matching_scores(query_embeddings=emb_queries, passage_embeddings=emb_passages, batch_size=batch_score, matching_type=matching_type, semantic_matching_indices=semantic_matching_indices)
#     # scores = vision_retriever.get_matching_scores(query_embeddings=emb_queries, passage_embeddings=emb_passages, batch_size=batch_score, matching_type=matching_type)

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Search took {elapsed_time} seconds to complete.")

#     # Get the relevant passages and results
#     relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
#     top_100_results = keep_top_100_scores(results)

#     # Compute the MTEB metrics
#     metrics, _ = vision_retriever.compute_metrics(relevant_docs, top_100_results)

#     return metrics


def evaluate_dataset_matching(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    matching_type: str,
    batch_query: int,
    batch_passage: int,
    batch_score: Optional[int] = None,
    embedding_pooler: Optional[BaseEmbeddingPooler] = None,
) -> Dict[str, Optional[float]]:
    """
    Evaluate the model on a given dataset using the MTEB metrics.

    NOTE: The dataset should contain the following columns:
    - query: the query text
    - image_filename: the filename of the image
    - image: the image (PIL.Image) if `use_visual_embedding` is True
    - text_description: the text description (i.e. the page caption or the text chunks) if
        `use_visual_embedding` is False
    """

    # Dataset: sanity check
    passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
    required_columns = ["query", passage_column_name, "image_filename"]

    if not all(col in ds.column_names for col in required_columns):
        raise ValueError(f"Dataset should contain the following columns: {required_columns}")

    seen_queries = set()
    queries = []
    for query in ds["query"]:
        if query is not None and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    if len(queries) == 0:
        raise ValueError("All queries are None")

    # Edge case: using the BM25Retriever
    if isinstance(vision_retriever, BM25Retriever):
        passages = ds[passage_column_name]
        scores = vision_retriever.get_scores_bm25(queries=queries, passages=passages)
        relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
        metrics = vision_retriever.compute_metrics(relevant_docs, results)
        return metrics

    emb_queries = vision_retriever.forward_queries(queries, batch_size=batch_query)

    # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
    # that will be fed to the model in batches (this should be fine for queries as their memory footprint
    # is negligible. This optimization is about efficient data loading, and is not related to the model's
    # forward pass which is also batched.
    emb_passages: List[torch.Tensor] = []

    dataloader_prebatch_size = 10 * batch_passage
    for passage_batch in tqdm(
        batched(ds, n=dataloader_prebatch_size),
        desc="Dataloader pre-batching",
        total=math.ceil(len(ds) / (dataloader_prebatch_size)),
    ):
        passages: List[Any] = [db[passage_column_name] for db in passage_batch]
        
        batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)
        if isinstance(batch_emb_passages, torch.Tensor):
            batch_emb_passages = list(torch.unbind(batch_emb_passages))
            emb_passages.extend(batch_emb_passages)
        else:
            emb_passages.extend(batch_emb_passages)

    if embedding_pooler is not None:
        for idx, emb_document in tqdm(enumerate(emb_passages), total=len(emb_passages), desc="Pooling embeddings..."):
            emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
            emb_passages[idx] = emb_document
    
    # For text lexical/semantic matching, compute token-level matching indices.
    semantic_matching_indices: Optional[List[List[Tuple[List[int], List[int]]]]] = None
    suffix = "<|endoftext|>" * 10

    if matching_type in ("text_lexical", "text_semantic"):
        if not hasattr(vision_retriever, "processor") or not hasattr(vision_retriever.processor, "tokenizer"):
            print("Tokenizer not found in vision_retriever.processor; skipping lexical matching indices computation.")
        else:
            tokenizer = vision_retriever.processor.tokenizer
            suffix = "<|endoftext|>" * 10
            # Tokenize each query.
            query_tokens: List[List[str]] = [
                tokenizer.tokenize("Query: " + query + suffix) for query in queries
            ]
            # Tokenize each passage.
            passage_texts = ds[passage_column_name]
            passage_tokens: List[List[str]] = [
                tokenizer.tokenize(text) for text in passage_texts
            ]

        # Build the semantic_matching_indices dictionary.
        semantic_matching_indices = {
            "query": {},
            "passage": {}
        }

        # Process queries: For each query, add an entry mapping the query index to its token list.
        for idx, query in enumerate(queries):
            # Optionally, adjust the query string (e.g. add a prefix/suffix) before tokenization.
            query_with_prefix = "Query: " + query + suffix
            tokens = tokenizer.tokenize(query_with_prefix)
            semantic_matching_indices["query"][idx] = tokens
        
        passages = ds[passage_column_name]
        # Process passages: For each passage, add an entry mapping the passage index to its token list.
        for idx, passage in enumerate(passages):
            tokens = tokenizer.tokenize(passage)
            semantic_matching_indices["passage"][idx] = tokens
    else:
        semantic_matching_indices = None


    start_time = time.time()
    print("start to search ", start_time, "number of queries ", len(emb_queries), "number of passages ", len(ds), len(emb_passages))
    # Get the similarity scores
    scores = vision_retriever.get_matching_scores(query_embeddings=emb_queries, passage_embeddings=emb_passages, batch_size=batch_score, matching_type=matching_type, semantic_matching_indices=semantic_matching_indices)
    # scores = vision_retriever.get_matching_scores(query_embeddings=emb_queries, passage_embeddings=emb_passages, batch_size=batch_score, matching_type=matching_type)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Search took {elapsed_time} seconds to complete.")

    # Get the relevant passages and results
    relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
    top_100_results = keep_top_100_scores(results)

    # Compute the MTEB metrics
    metrics, _ = vision_retriever.compute_metrics(relevant_docs, top_100_results)

    return metrics


def evaluate_dataset(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    batch_query: int,
    batch_passage: int,
    batch_score: Optional[int] = None,
    embedding_pooler: Optional[BaseEmbeddingPooler] = None,
) -> Dict[str, Optional[float]]:
    """
    Evaluate the model on a given dataset using the MTEB metrics.

    NOTE: The dataset should contain the following columns:
    - query: the query text
    - image_filename: the filename of the image
    - image: the image (PIL.Image) if `use_visual_embedding` is True
    - text_description: the text description (i.e. the page caption or the text chunks) if
        `use_visual_embedding` is False
    """


    # Dataset: sanity check
    passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
    required_columns = ["query", passage_column_name, "image_filename"]    

    if not all(col in ds.column_names for col in required_columns):
        raise ValueError(f"Dataset should contain the following columns: {required_columns}")

    # Remove `None` queries (i.e. pages for which no question was generated) and duplicates
    # queries = list(set(ds["query"]))
    # --> old buggy behavior - this differs from colpali-engine implementation where duplicates are NOT removed
    # for fairness with externally evaluated retrievers since bug, we maintain this behavior and remove duplicates
    # This slightly boosts scores on docvqa typically
    seen_queries = set()
    queries = []
    for query in ds["query"]:
        if query is not None and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    if len(queries) == 0:
        raise ValueError("All queries are None")

    # Edge case: using the BM25Retriever
    if isinstance(vision_retriever, BM25Retriever):
        passages = ds[passage_column_name]
        scores = vision_retriever.get_scores_bm25(queries=queries, passages=passages)
        relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
        metrics = vision_retriever.compute_metrics(relevant_docs, results)
        return metrics

    # Get the embeddings for the queries and passages
    emb_queries = vision_retriever.forward_queries(queries, batch_size=batch_query)

    # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
    # that will be fed to the model in batches (this should be fine for queries as their memory footprint
    # is negligible. This optimization is about efficient data loading, and is not related to the model's
    # forward pass which is also batched.
    emb_passages: List[torch.Tensor] = []

    dataloader_prebatch_size = 10 * batch_passage

    for passage_batch in tqdm(
        batched(ds, n=dataloader_prebatch_size),
        desc="Dataloader pre-batching",
        total=math.ceil(len(ds) / (dataloader_prebatch_size)),
    ):
        passages: List[Any] = [db[passage_column_name] for db in passage_batch]


        batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)

        if isinstance(batch_emb_passages, torch.Tensor):
            batch_emb_passages = list(torch.unbind(batch_emb_passages))
            emb_passages.extend(batch_emb_passages)
        else:
            emb_passages.extend(batch_emb_passages)

    if embedding_pooler is not None:
        for idx, emb_document in tqdm(enumerate(emb_passages), total=len(emb_passages), desc="Pooling embeddings..."):
            emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
            emb_passages[idx] = emb_document
    
    start_time = time.time()
    print("start to search ", start_time, "number of queries ", len(emb_queries), "number of passages ", len(ds), len(emb_passages))
    # Get the similarity scores
    scores = vision_retriever.get_scores(emb_queries, emb_passages, batch_size=batch_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Search took {elapsed_time} seconds to complete.")

    # Get the relevant passages and results
    relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
    top_100_results = keep_top_100_scores(results)

    # Compute the MTEB metrics
    metrics, query_metrics = vision_retriever.compute_metrics(relevant_docs, top_100_results)

    return metrics, query_metrics, top_100_results

def evaluate_dataset_from_imagetexts(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    batch_query: int,
    batch_passage: int,
    batch_score: Optional[int] = None,
    embedding_pooler: Optional[BaseEmbeddingPooler] = None,
) -> Dict[str, Optional[float]]:
    """
    Evaluate the model on a given dataset using the MTEB metrics.

    NOTE: The dataset should contain the following columns:
    - query: the query text
    - image_filename: the filename of the image
    - image: the image (PIL.Image) if `use_visual_embedding` is True
    - text_description: the text description (i.e. the page caption or the text chunks) if
        `use_visual_embedding` is False
    """
    seen_queries = set()
    queries = []
    for query in ds["query"]:
        if query is not None and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    if len(queries) == 0:
        raise ValueError("All queries are None")
    # Get the embeddings for the queries and passages
    emb_queries = vision_retriever.forward_queries(queries, batch_size=batch_query)

    emb_passages: List[torch.Tensor] = []

    dataloader_prebatch_size = 10 * batch_passage

    for passage_batch in tqdm(
        batched(ds, n=dataloader_prebatch_size),
        desc="Dataloader pre-batching",
        total=math.ceil(len(ds) / (dataloader_prebatch_size)),
    ):
        # passages: List[Any] = [db['text_description'] for db in passage_batch]

        # images: List[Any] = [db["image"] for db in passage_batch]
        # texts: List[Any] = [db["text_description"] for db in passage_batch]
        # passages = [(i,t) for i, t in zip(images, texts)]
        passages: List[Any] = [(db["image"], db["text_description"]) for db in passage_batch]
        batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)

        if isinstance(batch_emb_passages, torch.Tensor):
            batch_emb_passages = list(torch.unbind(batch_emb_passages))
            emb_passages.extend(batch_emb_passages)
        else:
            emb_passages.extend(batch_emb_passages)

    if embedding_pooler is not None:
        for idx, emb_document in tqdm(enumerate(emb_passages), total=len(emb_passages), desc="Pooling embeddings..."):
            emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
            emb_passages[idx] = emb_document
    
    start_time = time.time()
    print("start to search ", start_time, "number of queries ", len(emb_queries), "number of passages ", len(ds), len(emb_passages))
    # Get the similarity scores
    scores = vision_retriever.get_scores(emb_queries, emb_passages, batch_size=batch_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Search took {elapsed_time} seconds to complete.")

    # Get the relevant passages and results
    relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
    top_100_results = keep_top_100_scores(results)

    # Compute the MTEB metrics
    metrics, query_metrics = vision_retriever.compute_metrics(relevant_docs, top_100_results)

    return metrics, query_metrics, top_100_results



# def evaluate_dataset(
#     vision_retriever: VisionRetriever,
#     ds: Dataset,
#     batch_query: int,
#     batch_passage: int,
#     batch_score: Optional[int] = None,
#     embedding_pooler: Optional[BaseEmbeddingPooler] = None,
# ) -> Dict[str, Optional[float]]:
#     """
#     Evaluate the model on a given dataset using the MTEB metrics.

#     NOTE: The dataset should contain the following columns:
#     - query: the query text
#     - image_filename: the filename of the image
#     - image: the image (PIL.Image) if `use_visual_embedding` is True
#     - text_description: the text description (i.e. the page caption or the text chunks) if
#         `use_visual_embedding` is False
#     """

#     # Dataset: sanity check
#     passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
#     required_columns = ["query", passage_column_name, "image_filename"]

#     if not all(col in ds.column_names for col in required_columns):
#         raise ValueError(f"Dataset should contain the following columns: {required_columns}")

#     # new_images = []
#     # if passage_column_name == "image":
#     #     for entry in ds:
#     #         new_image = decode_base64_to_pil_image(pil_image_to_base64(entry[passage_column_name]))
#     #         new_images.append(new_image)
#     #     ds[passage_column_name] = new_images

#     # if passage_column_name == "image":
#     #     # Convert all images in the dataset from base64 back to PIL Images
#     #     ds = ds.map(lambda x: {"image": decode_base64_to_pil_image(pil_image_to_base64(x['image']))})

#     # Remove `None` queries (i.e. pages for which no question was generated) and duplicates
#     # queries = list(set(ds["query"]))
#     # --> old buggy behavior - this differs from colpali-engine implementation where duplicates are NOT removed
#     # for fairness with externally evaluated retrievers since bug, we maintain this behavior and remove duplicates
#     # This slightly boosts scores on docvqa typically
#     seen_queries = set()
#     queries = []
#     for query in ds["query"]:
#         if query is not None and query not in seen_queries:
#             queries.append(query)
#             seen_queries.add(query)

#     if len(queries) == 0:
#         raise ValueError("All queries are None")
#     # Edge case: using the BM25Retriever
#     if isinstance(vision_retriever, BM25Retriever):
#         passages = ds[passage_column_name]
#         scores = vision_retriever.get_scores_bm25(queries=queries, passages=passages)
#         relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
#         metrics = vision_retriever.compute_metrics(relevant_docs, results)
#         return metrics

#     # Get the embeddings for the queries and passages
#     emb_queries = vision_retriever.forward_queries(queries, batch_size=batch_query)

#     # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
#     # that will be fed to the model in batches (this should be fine for queries as their memory footprint
#     # is negligible. This optimization is about efficient data loading, and is not related to the model's
#     # forward pass which is also batched.
#     emb_passages: List[torch.Tensor] = []

#     dataloader_prebatch_size = 10 * batch_passage

#     for passage_batch in tqdm(
#         batched(ds, n=dataloader_prebatch_size),
#         desc="Dataloader pre-batching",
#         total=math.ceil(len(ds) / (dataloader_prebatch_size)),
#     ):
#         # passages: List[Any] = [db[passage_column_name] for db in passage_batch]
#         passages: List[Any] = [db[passage_column_name] for db in passage_batch]
        
#         batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)
#         if isinstance(batch_emb_passages, torch.Tensor):
#             batch_emb_passages = list(torch.unbind(batch_emb_passages))
#             emb_passages.extend(batch_emb_passages)
#         else:
#             emb_passages.extend(batch_emb_passages)

    # if embedding_pooler is not None:
    #     for idx, emb_document in tqdm(enumerate(emb_passages), total=len(emb_passages), desc="Pooling embeddings..."):
    #         emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
    #         emb_passages[idx] = emb_document
    
    # start_time = time.time()
    # print("start to search ", start_time, "number of queries ", len(emb_queries), "number of passages ", len(ds), len(emb_passages))
    # # Get the similarity scores
    # scores = vision_retriever.get_scores(emb_queries, emb_passages, batch_size=batch_score)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Search took {elapsed_time} seconds to complete.")

    # # Get the relevant passages and results
    # relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
    # top_100_results = keep_top_100_scores(results)

    # # Compute the MTEB metrics
    # metrics, query_metrics = vision_retriever.compute_metrics(relevant_docs, top_100_results)

    # return metrics, query_metrics, top_100_results

def evaluate_dataset_from_indexing(
    vision_retriever: VisionRetriever,
    query_ds: Dataset,
    passages_ds: Dataset,
    batch_query: int,
    emb_passages: list,
    batch_score: Optional[int] = None,
    ) -> Dict[str, Optional[float]]:

    # Dataset: sanity check
    passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
    required_columns = ["query", passage_column_name, "image_filename"]

    # if not all(col in ds.column_names for col in required_columns):
    #     raise ValueError(f"Dataset should contain the following columns: {required_columns}")
    print(query_ds)
    seen_queries = set()
    queries = []
    for query in query_ds["query"]:
        if query is not None and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    if len(queries) == 0:
        raise ValueError("All queries are None")

    # Edge case: using the BM25Retriever
    # if isinstance(vision_retriever, BM25Retriever):
    #     passages = ds[passage_column_name]
    #     scores = vision_retriever.get_scores_bm25(queries=queries, passages=passages)
    #     relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
    #     metrics = vision_retriever.compute_metrics(relevant_docs, results)
    #     return metrics

    # Get the embeddings for the queries and passages
    emb_queries = vision_retriever.forward_queries(queries, batch_size=batch_query)

    start_time = time.time()
    print("start to search ", start_time, "number of queries ", len(emb_queries), "number of passages ", len(emb_passages), len(emb_passages))
    # Get the similarity scores
    scores = vision_retriever.get_scores(emb_queries, emb_passages, batch_size=batch_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Search took {elapsed_time} seconds to complete.")

    # Get the relevant passages and results
    print("Get the relevant passages and results")
    relevant_docs, results = vision_retriever.get_relevant_docs_results(passages_ds, queries, scores)

    top_100_results = keep_top_100_scores(results)

    # Compute the MTEB metrics
    metrics, query_metrics = vision_retriever.compute_metrics(relevant_docs, top_100_results)

    return metrics, query_metrics, top_100_results