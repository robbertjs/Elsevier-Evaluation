from __future__ import annotations

import logging
from typing import List, Optional, Union, cast, Any, Dict, List, Optional
import os, io,base64
import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.data_utils import ListDataset
from vidore_benchmark.utils.torch_utils import get_torch_device
from torch.utils.data import Dataset
from typing import List, TypeVar
from typing import List, Tuple, Any
from torch.utils.data import Dataset as TorchDataset
from vidore_benchmark.utils.iter_utils import batched

T = TypeVar("T")

logger = logging.getLogger(__name__)

load_dotenv(override=True)

# # at top if needed
# import os, io, base64
# from PIL import Image

# def _to_pil(img):
#     if isinstance(img, Image.Image):
#         return img
#     if isinstance(img, str):
#         # filepath first
#         if os.path.exists(img):
#             try:
#                 return Image.open(img).convert("RGB")
#             except Exception:
#                 pass
#         # then assume base64
#         try:
#             return Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB")
#         except Exception:
#             pass
#     # last resort dummy
#     return Image.new("RGB", (224, 224), (127, 127, 127))



class ImagesTextDataset(TorchDataset):
    def __init__(self, elements: List[Tuple[Image.Image, Any]]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Any]:
        return self.elements[idx]

# class ListDataset(TorchDataset[T]):
#     def __init__(self, elements: List[T]):
#         self.elements = elements

#     def __len__(self) -> int:
#         return len(self.elements)

#     def __getitem__(self, idx: int) -> T:
#         return self.elements[idx]


@register_vision_retriever("colqwen2TextImage")
class ColQwen2RetrieverTextImage(VisionRetriever):
    """
    ColQwen2 retriever that implements the model from "ColPali: Efficient Document Retrieval
    with Vision Language Models".
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "vidore/colqwen2-v0.1",
        device: str = "auto",
        use_visual: bool = False,
    ):
        super().__init__()

        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[colpali-engine]"` '
                "to use ColQwen2Retriever."
            )

        self.device = get_torch_device(device)
        logger.info(f"Using device: {self.device}")

        # Load the model and LORA adapter
        self.model = cast(
            ColQwen2,
            ColQwen2.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="sdpa",
                #attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            ).eval(),
        )

        # Load the processor
        self.processor = cast(
            ColQwen2Processor,
            ColQwen2Processor.from_pretrained(
                pretrained_model_name_or_path,
                use_fast=True,
                image_processor_kwargs={"size": {"shortest_edge": 384}},  # was ~448
                ),
        )
        print("Loaded custom processor.\n")
        self._use_visual = use_visual

    @property
    def use_visual_embedding(self) -> bool:
        return self._use_visual

    def process_images_texts(self, passages: List, **kwargs):
        # norm = []
        # for elem in passages:
        #     # Accept dicts or (img, text, *extra)
        #     if isinstance(elem, dict):
        #         img = elem.get("image") or elem.get("image_filename")
        #         text = (
        #             elem.get("text_description")
        #             or elem.get("caption")
        #             or elem.get("text")
        #             or elem.get("passage")
        #             or ""
        #         )
        #     elif isinstance(elem, (list, tuple)):
        #         img = elem[0]
        #         text = elem[1] if len(elem) > 1 else ""
        #     else:
        #         raise TypeError(f"Unsupported passage type: {type(elem)}")

        #     norm.append((_to_pil(img), text))
        return self.processor.process_images_texts(passages=passages).to(self.device)

    def process_images(self, images: List[Image.Image], **kwargs):
        return self.processor.process_images(images=images).to(self.device)

    def process_queries(self, queries: List[str], **kwargs):
        return self.processor.process_queries(queries=queries).to(self.device)

    def process_passages(self, passages: List[str], **kwargs):
        return self.processor.process_passages(passages=passages).to(self.device)

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_queries,
        )

        query_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_query in tqdm(dataloader, desc="Forward pass queries...", leave=False):
                embeddings_query = self.model(**batch_query).to("cpu")
                query_embeddings.extend(list(torch.unbind(embeddings_query)))

        return query_embeddings

    def forward_passages(self, passages: List[Any], batch_size: int, **kwargs) -> List[torch.Tensor]:
        passage_embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(batched(passages, n=batch_size), desc="Forward pass documents...", leave=False):
                # batch is a list of (PIL.Image, str)
                inputs = self.processor.process_images_texts(passages=batch).to(self.device)
                emb = self.model(**inputs).to("cpu")
                passage_embeddings.extend(list(torch.unbind(emb)))
        return passage_embeddings


    # def forward_passages(self, passages: List[Any], batch_size: int, **kwargs) -> List[torch.Tensor]:
    #     dataloader = DataLoader(
    #         dataset=ImagesTextDataset(passages),
    #         batch_size=batch_size,
    #         shuffle=False,
    #         collate_fn=self.process_images_texts,
    #     )

    #     passage_embeddings: List[torch.Tensor] = []

    #     with torch.no_grad():
    #         for batch_doc in tqdm(dataloader, desc="Forward pass documents...", leave=False):
    #             embeddings_doc = self.model(**batch_doc).to("cpu")
    #             passage_embeddings.extend(list(torch.unbind(embeddings_doc)))

    #     return passage_embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("`batch_size` must be provided for ColQwenRetriever's scoring")
        scores = self.processor.score(
            query_embeddings,
            passage_embeddings,
            batch_size=batch_size,
            device="cpu",
        )
        return scores

    def get_matching_scores(
        self,
        matching_type: str,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        semantic_matching_indices: Optional[List[List[Tuple[List[int], List[int]]]]] = None,
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("`batch_size` must be provided for ColQwenRetriever's scoring")
        scores = self.processor.matching_score(
            qs=query_embeddings,
            ps=passage_embeddings,
            batch_size=batch_size,
            device="cpu",
            matching_type=matching_type,
            semantic_matching_indices=semantic_matching_indices
        )
        return scores