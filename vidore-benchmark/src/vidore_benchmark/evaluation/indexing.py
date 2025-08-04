from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
import torch
from datasets import Dataset
from tqdm import tqdm
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched

def indexing(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    batch_passage: int,
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
        # Does this retriever support text+image batching?
    use_text_image = hasattr(vision_retriever, "process_images_texts")
    cols = set(ds.column_names)

    # Decide validation and how to form `passages`
    if use_text_image and {"image", "text_description"} <= cols:
        required_columns = ["query", "image", "text_description", "image_filename"]
        passage_mode = "text_image"
        print("passage_mode: image + text_description")
    else:
        passage_mode = "image" if vision_retriever.use_visual_embedding else "text_description"
        required_columns = ["query", passage_mode, "image_filename"]
        print("passage_column_name ", passage_mode)

    # Validate columns
    if not all(col in ds.column_names for col in required_columns):
        raise ValueError(f"Dataset should contain the following columns: {required_columns}")

    emb_passages: List[torch.Tensor] = []
    dataloader_prebatch_size = 10 * batch_passage

    for passage_batch in tqdm(
        batched(ds, n=dataloader_prebatch_size),
        desc="Dataloader pre-batching",
        total=math.ceil(len(ds) / (dataloader_prebatch_size)),
    ):
        # Build passages per mode
        if passage_mode == "text_image":
            # Each element is (PIL.Image, str)
            passages: List[Any] = [(db["image"], db["text_description"]) for db in passage_batch]
        else:
            passages = [db[passage_mode] for db in passage_batch]

        batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)

        if isinstance(batch_emb_passages, torch.Tensor):
            batch_emb_passages = list(torch.unbind(batch_emb_passages))
            emb_passages.extend(batch_emb_passages)
        else:
            emb_passages.extend(batch_emb_passages)

    return emb_passages

    # # Dataset: sanity check
    # passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
    # print("passage_column_name ", passage_column_name)
    # required_columns = ["query", passage_column_name, "image_filename"]

    # if not all(col in ds.column_names for col in required_columns):
    #     raise ValueError(f"Dataset should contain the following columns: {required_columns}")

    # emb_passages: List[torch.Tensor] = []

    # dataloader_prebatch_size = 10 * batch_passage

    # for passage_batch in tqdm(
    #     batched(ds, n=dataloader_prebatch_size),
    #     desc="Dataloader pre-batching",
    #     total=math.ceil(len(ds) / (dataloader_prebatch_size)),
    # ):
    #     passages: List[Any] = [db[passage_column_name] for db in passage_batch]
    #     batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)
        
    #     if isinstance(batch_emb_passages, torch.Tensor):
    #         batch_emb_passages = list(torch.unbind(batch_emb_passages))
    #         emb_passages.extend(batch_emb_passages)
    #     else:
    #         emb_passages.extend(batch_emb_passages)

    # return emb_passages