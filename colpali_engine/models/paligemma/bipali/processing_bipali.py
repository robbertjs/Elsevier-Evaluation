from typing import List, Optional, Union
import torch
from transformers import BatchFeature, PaliGemmaProcessor
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from typing import ClassVar, List, Optional, Tuple, Union

class BiPaliProcessor(ColPaliProcessor):
    """
    Processor for BiPali. Mirrors the `ColPaliProcessor` class.
    """

    visual_prompt_prefix: ClassVar[str] = "<image><bos>Describe the image."
    query_prefix: ClassVar[str] = "Query: "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        return self.score_single_vector(qs, ps, device=device)


    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColPali.
        """

        if suffix is None:
            suffix = self.query_augmentation_token * 1
        texts_query: List[str] = []

        for query in queries:
            query = self.tokenizer.bos_token + self.query_prefix + query
            query += suffix  # add suffix (pad tokens)

            # NOTE: Make input ISO to PaliGemma's processor
            query += "\n"

            texts_query.append(query)

        batch_query = self.tokenizer(
            texts_query,
            text_pair=None,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
        )

        return batch_query

