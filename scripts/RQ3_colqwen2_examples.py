from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange
import torch
from PIL import Image
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from colpali_engine.models import ColQwen2, ColQwen2Processor
import matplotlib.pyplot as plt
import seaborn as sns
import easyocr

EPSILON = 1e-10

def plot_confusion_matrix_tokens_vs_passages(
    query_embeddings: torch.Tensor,
    passages_embeddings: torch.Tensor,
    query_tokens: List[str],
    figsize: Tuple[int, int] = (300, 25),  # Increase the figure size as needed
    annot: bool = False
):
    """
    Plots a confusion matrix–style heatmap showing similarity scores between each query token and every passage token.
    
    The similarity matrix is computed using an einsum operation (dot product) between the query embeddings and passage embeddings.
    It assumes that:
      - query_embeddings is of shape (B, num_query_tokens, embedding_dim)
      - passages_embeddings is of shape (C, num_passage_tokens, embedding_dim)
    
    For visualization, we assume that both batch dimensions are 1 so that the similarity matrix becomes of shape 
    (num_query_tokens, num_passage_tokens).
    
    Args:
        query_embeddings: torch.Tensor of shape (B, num_query_tokens, embedding_dim)
        passages_embeddings: torch.Tensor of shape (C, num_passage_tokens, embedding_dim)
        query_tokens: List[str] representing the query tokens (used as row labels)
        figsize: Tuple indicating the figure size (width, height)
        annot: bool indicating whether to annotate each cell with its numeric value.
    """
    # Compute the similarity matrix using the dot product between each query token and each passage token.
    # Here, "bnd, csd -> bcns" means:
    #   - b: batch size for queries
    #   - n: number of query tokens
    #   - c: batch size for passages
    #   - s: number of passage tokens
    #   - d: embedding dimension
    similarity_tensor = torch.einsum("bnd,csd->bcns", query_embeddings, passages_embeddings)
    # For visualization, we take the first element from each batch, yielding a matrix of shape (num_query_tokens, num_passage_tokens)
    sim_map = similarity_tensor[0, 0]
    
    # Convert to a NumPy array
    sim_map_np = sim_map.float().cpu().detach().numpy()
    
    # Create labels for passage tokens (using their indices)
    passage_labels = [str(i) for i in range(sim_map_np.shape[1])]
    
    # Create a mask that is True for the row-wise maximum values.
    # For each row (query token), the maximum value (or values, if there are ties) will be True.
    max_val_mask = (sim_map_np == np.max(sim_map_np, axis=1, keepdims=True))
    # Create the figure
    plt.figure(figsize=figsize)
    # Plot the full similarity heatmap with a blue colormap.
    ax = sns.heatmap(
        sim_map_np,
        annot=annot,
        fmt=".2f",
        cmap="Blues",
        xticklabels=passage_labels,
        yticklabels=query_tokens,
        linewidths=1,
        linecolor="black",
        cbar=True
    )
    
    # Overlay the maximum values using a different colormap (autumn) with transparency.
    # The mask here is inverted (i.e. cells that are not the maximum are masked out).
    ax = sns.heatmap(
        sim_map_np,
        annot=annot,
        fmt=".2f",
        cmap="Oranges",
        xticklabels=passage_labels,
        yticklabels=query_tokens,
        alpha=0.5,
        linewidths=1,
        linecolor="black",
        mask=~max_val_mask,
        cbar=True,
        ax=ax
    )

    num_rows, num_cols = sim_map_np.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if max_val_mask[i, j]:
                # Place an "x" at the center of the cell (j, i)
                ax.text(j + 0.5, i + 0.5, "x", ha='center', va='center', color='black', fontsize=30)

    # Set larger font size for tick labels
    ax.tick_params(axis='x', labelsize=25)  # Increase x-axis tick label size
    ax.tick_params(axis='y', labelsize=25)  # Increase y-axis tick label size
    plt.xlabel("Passage Token Index", fontsize=20)
    plt.ylabel("Query Token", fontsize=20)
    plt.title("Confusion Matrix: Similarity Between Query Tokens and Passage Tokens", fontsize=20)
    plt.tight_layout()
    plt.savefig("/ivi/ilps/personal/jqiao/colpali/RQ3/query_wise_confusion_matrix_text.pdf")
    plt.show()


def plot_confusion_matrix_tokens_vs_passages_filtered(
    query_embeddings: torch.Tensor,
    passages_embeddings: torch.Tensor,
    query_tokens: List[str],
    passage_tokens: List[str],
    figsize: Tuple[int, int] = (50, 35),  # Increase the figure size as needed
    annot: bool = True
):
    """
    Plots a confusion matrix–style heatmap showing similarity scores between each query token and every passage token.
    
    The similarity matrix is computed using an einsum operation (dot product) between the query embeddings and passage embeddings.
    It assumes that:
      - query_embeddings is of shape (B, num_query_tokens, embedding_dim)
      - passages_embeddings is of shape (C, num_passage_tokens, embedding_dim)
    
    For visualization, we assume that both batch dimensions are 1 so that the similarity matrix becomes of shape 
    (num_query_tokens, num_passage_tokens).
    
    This function removes columns (passage tokens) that do not have any row-wise highest similarity score and overlays an "x"
    on cells containing the row‑wise maximum.
    
    Args:
        query_embeddings: torch.Tensor of shape (B, num_query_tokens, embedding_dim)
        passages_embeddings: torch.Tensor of shape (C, num_passage_tokens, embedding_dim)
        query_tokens: List[str] representing the query tokens (used as row labels)
        figsize: Tuple indicating the figure size (width, height)
        annot: bool indicating whether to annotate each cell with its numeric value.
    """
    # Compute the similarity matrix between each query token and each passage token.
    # "bnd, csd -> bcns": 
    #   b: query batch, n: number of query tokens, d: embedding dim,
    #   c: passage batch, s: number of passage tokens.
    similarity_tensor = torch.einsum("bnd,csd->bcns", query_embeddings, passages_embeddings)
    # For visualization, we take the first element from each batch, yielding a matrix of shape (num_query_tokens, num_passage_tokens)
    sim_map = similarity_tensor[0, 0]
    
    # Convert to a NumPy array.
    sim_map_np = sim_map.float().cpu().detach().numpy()
    passage_labels = [str(i) for i in range(sim_map_np.shape[1])]
    # Create a mask that is True for the row-wise maximum values.
    # For each row (query token), the maximum value (or values, if there are ties) will be True.
    max_val_mask = (sim_map_np == np.max(sim_map_np, axis=1, keepdims=True))
    
    # Determine which columns (passage tokens) have at least one row-wise maximum.
    valid_cols = np.any(max_val_mask, axis=0)  # shape: (num_passage_tokens,)
    
    # Filter the similarity matrix, maximum mask, and passage labels to only keep these columns.
    sim_map_np_filtered = sim_map_np[:, valid_cols]
    max_val_mask_filtered = max_val_mask[:, valid_cols]
    passage_labels_filtered = [passage_tokens[i] for i in range(len(passage_labels)) if valid_cols[i]]
    
    # Create the figure.
    plt.figure(figsize=figsize)
    
    # Plot the filtered similarity heatmap with a blue colormap.
    ax = sns.heatmap(
        sim_map_np_filtered,
        annot=annot,
        fmt=".2f",
        cmap="Blues",
        xticklabels=passage_labels_filtered,
        yticklabels=query_tokens,
        linewidths=1,
        linecolor="black",
        cbar=True
    )
    
    # Overlay the maximum values with an "x" symbol.
    num_rows, num_cols = sim_map_np_filtered.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if max_val_mask_filtered[i, j]:
                # Place an "x" at the center of the cell (j, i)
                ax.text(j + 0.5, i + 0.5, "x", ha='center', va='center', color='black', fontsize=60)
    
    # Set larger font size for tick labels.
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    plt.xlabel("Passage Token Index", fontsize=40)
    plt.ylabel("Query Token", fontsize=40)
    plt.title("Confusion Matrix: Similarity Between Query Tokens and Passage Tokens", fontsize=40)
    plt.tight_layout()
    plt.savefig("/ivi/ilps/personal/jqiao/colpali/RQ3/query_wise_confusion_matrix_text_filtered.jpg")
    plt.show()


def plot_confusion_matrix_tokens_vs_patches(
    similarity_map: torch.Tensor,
    query_tokens: List[str],
    figsize: Tuple[int, int] = (300, 25),  # Increased the figure size for better visualization
    annot: bool = True
):
    """
    Plots a confusion matrix–style heatmap showing similarity scores between each query token
    and every image patch. The similarity_map is expected to be of shape 
    (num_tokens, n_patches_x, n_patches_y). The image patch dimensions are flattened so that:
      - Rows correspond to query tokens.
      - Columns correspond to a flattened image patch index (from 0 to n_patches_x * n_patches_y - 1).

    Args:
        similarity_map: torch.Tensor of shape (num_tokens, n_patches_x, n_patches_y).
        query_tokens: List of strings representing the query tokens (used as row labels).
        figsize: Size of the matplotlib figure.
        annot: Whether to annotate the cells with their numeric values.
    """
    num_tokens, n_patches_x, n_patches_y = similarity_map.shape
    num_patches = n_patches_x * n_patches_y

    # Flatten the spatial dimensions into one dimension: shape becomes (num_tokens, num_patches)
    flattened = similarity_map.view(num_tokens, num_patches)

    flattened_np = flattened.float().cpu().detach().numpy()
    patch_labels = [str(i) for i in range(num_patches)]

    # max_val_mask = (flattened_np == flattened_np.max(axis=0))  # Find the max in each column
    max_val_mask = (flattened_np == np.max(flattened_np, axis=1, keepdims=True))  # Find the max in each row

    plt.figure(figsize=figsize)

    ax = sns.heatmap(flattened_np, annot=annot, fmt=".2f", cmap="Oranges",
                    xticklabels=patch_labels, yticklabels=query_tokens, linewidths=1, linecolor='black', mask=~max_val_mask, cbar=True)

    # Highlighting max values with a different color
    ax = sns.heatmap(flattened_np, annot=annot, fmt=".2f", cmap="Blues",
                    xticklabels=patch_labels, yticklabels=query_tokens, alpha=0.5, linewidths=1, linecolor='black', ax=ax, mask=max_val_mask, cbar=True)

    num_rows, num_cols = flattened_np.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if max_val_mask[i, j]:
                # Place an "x" at the center of the cell (j, i)
                ax.text(j + 0.5, i + 0.5, "x", ha='center', va='center', color='black', fontsize=30)

    # Set larger font size for tick labels
    ax.tick_params(axis='x', labelsize=25)  # Increase x-axis tick label size
    ax.tick_params(axis='y', labelsize=25)  # Increase y-axis tick label size
    plt.xlabel("Image Patch Index", fontsize=30)
    plt.ylabel("Query Token", fontsize=30)
    plt.title("Confusion Matrix: Similarity Scores Between Query Tokens and Image Patches", fontsize=30)
    plt.tight_layout()  # Adjust layout to ensure everything fits nicely
    plt.savefig("/ivi/ilps/personal/jqiao/colpali/RQ3/query_wise_confusion_matrix_image.pdf")
    plt.show()


def plot_confusion_matrix_tokens_vs_patches_filtered(
    similarity_map: torch.Tensor,
    query_tokens: List[str],
    figsize: Tuple[int, int] = (50, 35),  # Increased the figure size for better visualization
    annot: bool = True
):
    """
    Plots a confusion matrix–style heatmap showing similarity scores between each query token
    and every image patch. The similarity_map is expected to be of shape 
    (num_tokens, n_patches_x, n_patches_y). The image patch dimensions are flattened so that:
      - Rows correspond to query tokens.
      - Columns correspond to a flattened image patch index (from 0 to n_patches_x * n_patches_y - 1).

    Args:
        similarity_map: torch.Tensor of shape (num_tokens, n_patches_x, n_patches_y).
        query_tokens: List of strings representing the query tokens (used as row labels).
        figsize: Size of the matplotlib figure.
        annot: Whether to annotate the cells with their numeric values.
    """
    num_tokens, n_patches_x, n_patches_y = similarity_map.shape
    num_patches = n_patches_x * n_patches_y

    # Flatten the spatial dimensions into one dimension: shape becomes (num_tokens, num_patches)
    flattened = similarity_map.view(num_tokens, num_patches)

    flattened_np = flattened.float().cpu().detach().numpy()
    patch_labels = [str(i) for i in range(num_patches)]

    # max_val_mask = (flattened_np == flattened_np.max(axis=0))  # Find the max in each column
    max_val_mask = (flattened_np == np.max(flattened_np, axis=1, keepdims=True))  # Find the max in each row

    # Determine which columns (passage tokens) have at least one row-wise maximum.
    valid_cols = np.any(max_val_mask, axis=0)  # shape: (num_passage_tokens,)
    # Filter the similarity matrix, maximum mask, and passage labels to only keep these columns.
    sim_map_np_filtered = flattened_np[:, valid_cols]
    max_val_mask_filtered = max_val_mask[:, valid_cols]
    passage_labels_filtered = [patch_labels[i] for i in range(len(patch_labels)) if valid_cols[i]]
    
    # Create the figure.
    plt.figure(figsize=figsize)
    
    # Plot the filtered similarity heatmap with a blue colormap.
    ax = sns.heatmap(
        sim_map_np_filtered,
        annot=annot,
        fmt=".2f",
        cmap="Blues",
        xticklabels=passage_labels_filtered,
        yticklabels=query_tokens,
        linewidths=1,
        linecolor="black",
        cbar=True
    )
    
    # Overlay the maximum values with an "x" symbol.
    num_rows, num_cols = sim_map_np_filtered.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if max_val_mask_filtered[i, j]:
                # Place an "x" at the center of the cell (j, i)
                ax.text(j + 0.5, i + 0.5, "x", ha='center', va='center', color='black', fontsize=60)

    # Set larger font size for tick labels.
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    plt.xlabel("Patch Token Index", fontsize=40)
    plt.ylabel("Query Token", fontsize=40)
    plt.title("Confusion Matrix: Similarity Between Query Tokens and Patch Tokens", fontsize=40)
    plt.tight_layout()
    plt.savefig("/ivi/ilps/personal/jqiao/colpali/RQ3/query_wise_confusion_matrix_patch_filtered.jpg")
    plt.show()


def extract_text(image):
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image)
    text_output = []
    for (bbox, text, prob) in results:
        text_output.append((text))

    return " ".join(text_output)

model_name = "vidore/colqwen2-v0.1"
device = get_torch_device("auto")
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Load the image and query
image = Image.open("/ivi/ilps/personal/jqiao/colpali/scripts/image.jpg")
query = "What services does HealthTeamWorks provide?"
passage = extract_text(image)

# Preprocess inputs
batch_images = processor.process_images([image]).to(device)
batch_queries = processor.process_queries([query]).to(device)
batch_passages = processor.process_passages([passage]).to(device)

# Forward passes
with torch.no_grad():
    image_embeddings = model.forward(**batch_images)
    query_embeddings = model.forward(**batch_queries)
    passages_embeddings = model.forward(**batch_passages)

n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size, spatial_merge_size=2)

# Get the tensor mask to filter out the embeddings that are not related to the image
image_mask = processor.get_image_mask(batch_images)
batched_similarity_maps = get_similarity_maps_from_embeddings(
    image_embeddings=image_embeddings,
    query_embeddings=query_embeddings,
    n_patches=n_patches,
    image_mask=image_mask,
)
# Get the similarity map for our (only) input image
similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)
# Tokenize the query
suffix = "<|endoftext|>" * 10
query = "Query: What services does HealthTeamWorks provide?" + suffix
query_tokens = processor.tokenizer.tokenize(query)
passage_tokens = processor.tokenizer.tokenize(passage)
print("number of passage_tokens ", len(passage_tokens))
print("number of query_tokens ", len(query_tokens))
print("passages_embeddings ", passages_embeddings.shape)
# plot_confusion_matrix_tokens_vs_patches(similarity_maps, query_tokens, annot=False)
# plot_confusion_matrix_tokens_vs_passages(query_embeddings, passages_embeddings, query_tokens, annot=False)
plot_confusion_matrix_tokens_vs_patches_filtered(similarity_maps, query_tokens, annot=True)
plot_confusion_matrix_tokens_vs_passages_filtered(query_embeddings, passages_embeddings, query_tokens,passage_tokens, annot=False)
