import math
from typing import List, Optional, Union, cast
import torch
from tqdm import tqdm
from vidore_benchmark.evaluation.scoring import score_multi_vector
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("jina-colbert")
class JinaaiColbertRetriever(VisionRetriever):
    """
    BGEM3Retriever class to retrieve embeddings the BGE-M3 model (multi-vector embeddings + ColBERT scoring).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "jinaai/jina-colbert-v2",
        device: str = "auto",
    ):
        super().__init__()

        try:
            from pylate import indexes, models, retrieve
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[bge-m3]"` '
                "to use BGEM3ColbertRetriever."
            )

        self.device = get_torch_device(device)

        self.model = models.ColBERT(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            # use_fp16=True,
            device=self.device,
        )

    @property
    def use_visual_embedding(self) -> bool:
        return False

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []

        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            # query_batch = list(query_batch)
            with torch.no_grad():
                output = self.model.encode(
                    query_batch,
                    is_query=True,
                    batch_size = batch_size
                    # max_length=512,
                    # return_dense=False,
                    # return_sparse=False,
                    # return_colbert_vecs=True,
                )
            list_emb_queries.extend([torch.Tensor(elt) for elt in output])

        return list_emb_queries

    def forward_passages(self, passages: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_passages: List[torch.Tensor] = []

        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[str], passage_batch)
            with torch.no_grad():
                output = self.model.encode(
                    passage_batch,
                    is_query=False,
                    batch_size = batch_size
                    # max_length=512,
                    # return_dense=False,
                    # return_sparse=False,
                    # return_colbert_vecs=True,
                )
            list_emb_passages.extend([torch.Tensor(elt) for elt in output])

        return list_emb_passages

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 4,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("The batch size must be specified for the ColBERT scoring.")
        scores = score_multi_vector(query_embeddings, passage_embeddings, batch_size=batch_size)
        return scores
