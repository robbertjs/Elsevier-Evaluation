from typing import List, Optional, Union
import math
from typing import ClassVar, List, Optional, Tuple, Union
import torch
import torch
from transformers import BatchFeature

from colpali_engine.models.qwen2.colqwen2 import ColQwen2Processor

class BiQwen2Processor(ColQwen2Processor):
    """
    Processor for ColQwen2.
    """
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColQwen2.
        """
        if suffix is None:
            suffix = self.query_augmentation_token # we remove buffer tokens
        texts_query: List[str] = []

        for query in queries:
            query = self.query_prefix + query + suffix
            texts_query.append(query)

        batch_query = self(
            text=texts_query,
            return_tensors="pt",
            padding="longest",
        )
        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_single_vector(qs, ps, device=device)

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id


    def matching_score(
        self,
        matching_type: str,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        # semantic_matching_indices: List,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        print("matching type ", matching_type)
        if matching_type == "all_type":
            score = self.score_single_vector(qs, ps, device=device, **kwargs)
        elif matching_type == "image_semantic":
            score = self.score_single_vector_image_semantic(qs, ps, device=device, **kwargs)
        elif matching_type == "image_special_token":
            score = self.score_single_vector_image_special(qs, ps, device=device, **kwargs)
        # elif matching_type == "text_special_token":
        #     score = self.score_multi_vector_text_special(qs, ps, device=device, **kwargs)
        # elif matching_type == "text_semantic":
        #     score = self.score_multi_vector_text_lexical(qs, ps, device=device, semantic_matching_indices=semantic_matching_indices, **kwargs)
        # elif matching_type == "text_lexical":
        #     score = self.score_multi_vector_text_lexical(qs, ps, device=device, semantic_matching_indices=semantic_matching_indices, **kwargs)
        return score
