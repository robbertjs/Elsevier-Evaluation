from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

from colpali_engine.utils.torch_utils import get_torch_device


class BaseVisualRetrieverProcessor(ABC):
    """
    Base class for visual retriever processors.
    """

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_single_vector_image_semantic(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        
        # Select the first 2 and last 1 tokens from each query
        # qs_semantic = [q[2:-1] for q in qs]
        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)

        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores


    @staticmethod
    def score_single_vector_image_special(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        
        # qs_special = [torch.cat([q[:2], q[-1:]], dim=0) for q in qs]

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        # Ensure that the feature dimension (d) matches
        if qs_stacked.shape[1] != ps_stacked.shape[1]:
            raise ValueError(f"Feature dimension mismatch between queries and passages: {qs_stacked.shape[1]} vs {ps_stacked.shape[1]}")

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)

        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores


    @staticmethod
    def score_single_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)

        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    def ensure_bfloat16_tensor(item, device=None):
        """
        Ensures that `item` is a PyTorch tensor of dtype bfloat16 on the given device.
        If `item` is already a tensor, move/convert it directly; 
        if not, create a new tensor from it.
        """
        if isinstance(item, torch.Tensor):
            # Convert existing tensor to bfloat16 and move to device if provided
            return item.to(dtype=torch.bfloat16, device=device)
        else:
            # Convert Python list/tuple/numpy array, etc. to bfloat16 tensor
            return torch.tensor(item, dtype=torch.bfloat16, device=device)

    @staticmethod
    def score_multi_vector_text_nonlexical(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        semantic_matching_indices: Optional[Dict[str, Dict[int, List[str]]]] = None,
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        A simpler lexical matching implementation.
        
        For each query–passage pair, the function computes a similarity matrix (via dot‐products)
        over token embeddings (with shape [B, C, n, s]). Then, rather than using the best‐matching
        passage token index, we simply build a binary mask per query token by checking whether the
        token (from the query’s token list) appears anywhere in the passage’s token list.
        
        The similarity scores for tokens that do not match lexically are zeroed out before being
        summed over the query tokens.
        """
        device = device or get_torch_device("auto")
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        
        scores_list: List[torch.Tensor] = []
        
        # Process queries in batches.
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size],
                batch_first=True,
                padding_value=0
            ).to(device)
            qs_batch_special = qs_batch[:, 2:-10, :]
        
            for j in range(0, len(ps), batch_size):
                try:
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                        ps[j : j + batch_size],
                        batch_first=True,
                        padding_value=0
                    ).to(device)
                except TypeError:
                    embedding_dim = qs_batch_special.size(-1)
                    default_shape = (batch_size, 1, embedding_dim)
                    ps_batch = torch.zeros(default_shape, dtype=qs_batch_special.dtype, device=device)
        
                # Compute the similarity matrix between query and passage tokens.
                # Shape: [B (queries), C (passages), n (query tokens), s (passage tokens)]
                similarity_matrix = torch.einsum("bnd,csd->bcns", qs_batch_special, ps_batch)
                # For each query token, take the maximum similarity over the passage tokens.
                max_similarity, _ = similarity_matrix.max(dim=3)
        
                if semantic_matching_indices is not None:
                    semantic_mask = torch.zeros_like(max_similarity, dtype=torch.float32)
                    B, C, n = max_similarity.shape
                    for b in range(B):
                        abs_query_index = i + b
                        # Get the token list for this query.
                        query_tokens = semantic_matching_indices["query"].get(abs_query_index, [])
                        # Normalize tokens (if desired)
                        # query_tokens = [tok.lower() for tok in query_tokens]
                        for c in range(C):
                            abs_passage_index = j + c
                            passage_tokens = semantic_matching_indices["passage"].get(abs_passage_index, [])

                            for t in range(n):
                                # Be sure that the token index t exists in your token list.
                                if t < len(query_tokens):
                                    if query_tokens[t] not in passage_tokens:
                                        semantic_mask[b, c, t] = 1.0
                    # Apply the mask: only tokens that appear in the passage contribute.
                    max_similarity = max_similarity * semantic_mask
        
                # Sum over query tokens to get one score per query–passage pair.
                query_wise_score = max_similarity.sum(dim=2)
                scores_batch.append(query_wise_score)
        
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)
        
        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        return scores.to(torch.float32)

    @staticmethod
    def score_multi_vector_text_lexical(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        semantic_matching_indices: Optional[Dict[str, Dict[int, List[str]]]] = None,
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        A simpler lexical matching implementation.
        
        For each query–passage pair, the function computes a similarity matrix (via dot‐products)
        over token embeddings (with shape [B, C, n, s]). Then, rather than using the best‐matching
        passage token index, we simply build a binary mask per query token by checking whether the
        token (from the query’s token list) appears anywhere in the passage’s token list.
        
        The similarity scores for tokens that do not match lexically are zeroed out before being
        summed over the query tokens.
        """
        device = device or get_torch_device("auto")
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        
        scores_list: List[torch.Tensor] = []
        
        # Process queries in batches.
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size],
                batch_first=True,
                padding_value=0
            ).to(device)
            # For example, you might remove special tokens via slicing.
            # (Make sure that you have built your token lists accordingly.)
            qs_batch_special = qs_batch[:, 2:-10, :]
        
            for j in range(0, len(ps), batch_size):
                try:
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                        ps[j : j + batch_size],
                        batch_first=True,
                        padding_value=0
                    ).to(device)
                except TypeError:
                    embedding_dim = qs_batch_special.size(-1)
                    default_shape = (batch_size, 1, embedding_dim)
                    ps_batch = torch.zeros(default_shape, dtype=qs_batch_special.dtype, device=device)
        
                # Compute the similarity matrix between query and passage tokens.
                # Shape: [B (queries), C (passages), n (query tokens), s (passage tokens)]
                similarity_matrix = torch.einsum("bnd,csd->bcns", qs_batch_special, ps_batch)
                # For each query token, take the maximum similarity over the passage tokens.
                # We end up with max_similarity of shape [B, C, n].
                max_similarity, _ = similarity_matrix.max(dim=3)
        
                if semantic_matching_indices is not None:
                    # Instead of using the best-matching token index, we simply build a binary mask.
                    # For each query token (by absolute index) we check if that token appears in the passage.
                    semantic_mask = torch.zeros_like(max_similarity, dtype=torch.float32)
                    B, C, n = max_similarity.shape
                    for b in range(B):
                        abs_query_index = i + b
                        # Get the token list for this query.
                        query_tokens = semantic_matching_indices["query"].get(abs_query_index, [])
                        # Normalize tokens (if desired)
                        # query_tokens = [tok.lower() for tok in query_tokens]
                        for c in range(C):
                            abs_passage_index = j + c
                            passage_tokens = semantic_matching_indices["passage"].get(abs_passage_index, [])
                            # passage_tokens = [tok.lower() for tok in passage_tokens]
                            # For each token position in the (sliced) query embeddings,
                            # check if the corresponding query token is in the passage token list.
                            for t in range(n):
                                # Be sure that the token index t exists in your token list.
                                if t < len(query_tokens):
                                    if query_tokens[t] in passage_tokens:
                                        semantic_mask[b, c, t] = 1.0
                    # Apply the mask: only tokens that appear in the passage contribute.
                    max_similarity = max_similarity * semantic_mask
        
                # Sum over query tokens to get one score per query–passage pair.
                query_wise_score = max_similarity.sum(dim=2)
                scores_batch.append(query_wise_score)
        
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)
        
        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        return scores.to(torch.float32)

    @staticmethod
    def score_multi_vector_text_special(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        def to_bfloat16_tensor(item, device):
            if isinstance(item, torch.Tensor):
                return item.to(dtype=torch.bfloat16, device=device)
            else:
                return torch.tensor(item, dtype=torch.bfloat16, device=device)

        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            # Select special query tokens:
            # Use first 2 and last 10 tokens if available; otherwise, use all tokens.
            if qs_batch.shape[1] < 12:
                qs_batch_special = qs_batch
            else:
                qs_batch_special = torch.cat([qs_batch[:, :2, :], qs_batch[:, -10:, :]], dim=1)

            for j in range(0, len(ps), batch_size):
                try:
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0).to(device)
                except TypeError as e:
                    embedding_dim = 128  # Example embedding dimension
                    default_shape = (batch_size, 1, embedding_dim)  # Modify based on expected tensor sizes
                    ps_batch = torch.zeros(default_shape, dtype=torch.bfloat16, device=device)

                query_wise_max_score_special = (torch.einsum("bnd,csd->bcns", qs_batch_special, ps_batch).max(dim=3)[0].sum(dim=2))

                scores_batch.append(query_wise_max_score_special)
            
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores


    @staticmethod
    def score_multi_vector_text_tmp(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        semantic_matching_indices:  Optional[List[List[Tuple[List[int], List[int]]]]] = None,
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        
        def to_bfloat16_tensor(item, device):
            if isinstance(item, torch.Tensor):
                return item.to(dtype=torch.bfloat16, device=device)
            else:
                return torch.tensor(item, dtype=torch.bfloat16, device=device)

        device = device or get_torch_device("auto")

        # We will accumulate scores in blocks.
        scores_list: List[torch.Tensor] = []

        # Process queries in batches.
        for i in range(0, len(qs), batch_size):
            # Determine the query indices for this batch.
            batch_q_indices = list(range(i, min(i + batch_size, len(qs))))
            current_batch_size_q = len(batch_q_indices)
            # Prepare a list to accumulate score rows for this query batch.
            batch_scores_rows = []

            # Process passages in batches.
            for j in range(0, len(qs), batch_size):
                batch_p_indices = list(range(j, min(j + batch_size, len(ps))))
                current_batch_size_p = len(batch_p_indices)

                # Prepare a tensor to hold the scores for this block
                block_scores = torch.empty((current_batch_size_q, current_batch_size_p), device=device, dtype=torch.float32)

                # Now, for each query in the current query batch...
                for qi, global_q_idx in enumerate(batch_q_indices):
                    q_emb = qs[global_q_idx].to(device)  # shape: [n_q_tokens, D]
                    # For each passage in the current passage batch...
                    for pi, global_p_idx in enumerate(batch_p_indices):
                        p_emb = ps[global_p_idx].to(device)  # shape: [n_p_tokens, D]
                        # Retrieve the matching indices computed earlier.
                        # semantic_matching_indices is assumed to be a nested list such that:
                        # semantic_matching_indices[global_q_idx][global_p_idx] = (q_indices, p_indices)
                        q_indices, p_indices = semantic_matching_indices[global_q_idx][global_p_idx]
                        if len(q_indices) == 0 or len(p_indices) == 0:
                            # If there are no overlapping token indices, assign a score of zero.
                            block_scores[qi, pi] = 0.0
                        else:
                            # Select only the embeddings corresponding to the matching indices.
                            q_emb_lex = q_emb[q_indices, :]  # shape: [n_q_selected, D]
                            p_emb_lex = p_emb[p_indices, :]  # shape: [n_p_selected, D]
                            # Compute dot products: result shape [n_q_selected, n_p_selected]
                            dot = torch.matmul(q_emb_lex, p_emb_lex.t())
                            # For each query token (row), take the maximum over the passage tokens.
                            max_per_q = dot.max(dim=1)[0]  # shape: [n_q_selected]
                            # Sum these maximum values to obtain a single score.
                            block_scores[qi, pi] = max_per_q.sum()
                # Append this block's scores along the passage dimension.
                batch_scores_rows.append(block_scores)
            # Concatenate all passage-blocks (columns) for this query batch.
            batch_scores = torch.cat(batch_scores_rows, dim=1)
            scores_list.append(batch_scores)

        # Concatenate all query batches (rows).
        scores = torch.cat(scores_list, dim=0)
        if scores.shape[0] != len(qs):
            raise ValueError(f"Expected {len(qs)} scores, got {scores.shape[0]}")

        return scores.to(torch.float32)

    @staticmethod
    def score_multi_vector_image_qtm(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        def to_bfloat16_tensor(item, device):
            if isinstance(item, torch.Tensor):
                return item.to(dtype=torch.bfloat16, device=device)
            else:
                return torch.tensor(item, dtype=torch.bfloat16, device=device)

        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            # Select special query tokens:
            # Use first 2 and last 10 tokens if available; otherwise, use all tokens.
            qs_batch_special = qs_batch[:, 2:-10, :]

            for j in range(0, len(ps), batch_size):
                try:
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0).to(device)
                except TypeError as e:
                    embedding_dim = 128  # Example embedding dimension
                    default_shape = (batch_size, 1, embedding_dim)  # Modify based on expected tensor sizes
                    ps_batch = torch.zeros(default_shape, dtype=torch.bfloat16, device=device)

                query_wise_max_score_special = (torch.einsum("bnd,csd->bcns", qs_batch_special, ps_batch).max(dim=3)[0].sum(dim=2))

                scores_batch.append(query_wise_max_score_special)
            
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector_text_qtm(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        def to_bfloat16_tensor(item, device):
            if isinstance(item, torch.Tensor):
                return item.to(dtype=torch.bfloat16, device=device)
            else:
                return torch.tensor(item, dtype=torch.bfloat16, device=device)

        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            # Select special query tokens:
            # Use first 2 and last 10 tokens if available; otherwise, use all tokens.
            qs_batch_special = qs_batch[:, 2:-10, :]

            for j in range(0, len(ps), batch_size):
                try:
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0).to(device)
                except TypeError as e:
                    embedding_dim = 128  # Example embedding dimension
                    default_shape = (batch_size, 1, embedding_dim)  # Modify based on expected tensor sizes
                    ps_batch = torch.zeros(default_shape, dtype=torch.bfloat16, device=device)

                query_wise_max_score_special = (torch.einsum("bnd,csd->bcns", qs_batch_special, ps_batch).max(dim=3)[0].sum(dim=2))

                scores_batch.append(query_wise_max_score_special)
            
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores


    @staticmethod
    def score_multi_vector_image_special(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        def to_bfloat16_tensor(item, device):
            if isinstance(item, torch.Tensor):
                return item.to(dtype=torch.bfloat16, device=device)
            else:
                return torch.tensor(item, dtype=torch.bfloat16, device=device)

        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )

            qs_batch_special = torch.cat([qs_batch[:, :2, :], qs_batch[:, -10:, :]], dim=1)

            for j in range(0, len(ps), batch_size):
                try:
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0).to(device)
                except TypeError as e:
                    embedding_dim = 128  # Example embedding dimension
                    default_shape = (batch_size, 1, embedding_dim)  # Modify based on expected tensor sizes
                    ps_batch = torch.zeros(default_shape, dtype=torch.bfloat16, device=device)

                query_wise_max_score_special = (torch.einsum("bnd,csd->bcns", qs_batch_special, ps_batch).max(dim=3)[0].sum(dim=2))

                scores_batch.append(query_wise_max_score_special)
            
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        def to_bfloat16_tensor(item, device):
            if isinstance(item, torch.Tensor):
                return item.to(dtype=torch.bfloat16, device=device)
            else:
                return torch.tensor(item, dtype=torch.bfloat16, device=device)

        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            for j in range(0, len(ps), batch_size):
                try:
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0).to(device)
                except TypeError as e:
                    embedding_dim = 128  # Example embedding dimension
                    default_shape = (batch_size, 1, embedding_dim)  # Modify based on expected tensor sizes
                    ps_batch = torch.zeros(default_shape, dtype=torch.bfloat16, device=device)
                                
                
                query_wise_max_score = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2)
                # print("qs_batch ", qs_batch.shape)
                # print("ps_batch ", ps_batch.shape)
                # print("query_wise_max_score ", query_wise_max_score.shape)

                # query_wise_sum_score = tmp_score.sum(dim=3).sum(dim=2)
                # print("query_wise_sum_score ", query_wise_sum_score.shape, "\n")
                
                # attention_weights = torch.softmax(tmp_score, dim=3)
                # weighted_query_wise_sum_score = (tmp_score * attention_weights).sum(dim=3).sum(dim=2)

                # attention_weights = torch.softmax(tmp_score, dim=3)
                # weighted_query_wise_sum_score = (tmp_score * attention_weights).max(dim=3)[0].sum(dim=2)
                # max_over_query = tmp_score.max(dim=2)[0].sum(dim=2)
                scores_batch.append(query_wise_max_score)
            
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @abstractmethod
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 14,
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
        image of size (height, width) with the given patch size.
        """
        pass
