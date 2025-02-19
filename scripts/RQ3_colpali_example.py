from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
import easyocr
from einops import rearrange
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def add_custom_annotations(
    ax: plt.Axes,
    data: np.ndarray,
    max_mask: np.ndarray,
    fontsize: int = 30,
    fmt: str = ".2f"
) -> None:
    """
    Annotate each cell of the heatmap with its similarity value.
    If a cell is a row-wise maximum (as indicated in max_mask), append an "x" marker.

    Args:
        ax: Matplotlib Axes on which to draw annotations.
        data: 2D NumPy array of similarity values.
        max_mask: Boolean mask of the same shape as data indicating the row‑wise max cells.
        fontsize: Font size for the text.
        fmt: Format specifier for the similarity value.
    """
    num_rows, num_cols = data.shape
    for i in range(num_rows):
        for j in range(num_cols):
            # Format the similarity value.
            text = f"{data[i, j]:{fmt}}"
            if max_mask[i, j]:
                # Append an "x" for maximum cells; you can change the marker or style if desired.
                # text += "\n x"
                fontweight = "bold"
                color="yellow"
            else:
                fontweight = "normal"
                color = "black"
            ax.text(j + 0.5, i + 0.5, text,
                    ha="center", va="center",
                    color=color, fontsize=fontsize,
                    fontweight=fontweight)

def add_custom_annotations(
    ax: plt.Axes,
    data: np.ndarray,
    max_mask: np.ndarray,
    fontsize: int = 30,
    fmt: str = ".2f"
) -> None:
    """
    Annotate each cell of the heatmap with its similarity value only if it is a row-wise maximum.
    Cells with the maximum value are marked, while others are not annotated.

    Args:
        ax: Matplotlib Axes on which to draw annotations.
        data: 2D NumPy array of similarity values.
        max_mask: Boolean mask of the same shape as data indicating the row‑wise max cells.
        fontsize: Font size for the text.
        fmt: Format specifier for the similarity value.
    """
    num_rows, num_cols = data.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if max_mask[i, j]:  # Only annotate if this cell is the maximum in its row
                # Format the similarity value and prepare to annotate
                text = f"{data[i, j]:{fmt}}"
                ax.text(j + 0.5, i + 0.5, text,  # Position the text in the center of the cell
                        ha="center", va="center",
                        color="yellow", fontsize=fontsize,  # Use yellow for maximum values
                        fontweight="bold")  # Bold to highlight maximum values



def plot_confusion_matrix_tokens_vs_passages_filtered(
    query_embeddings: torch.Tensor,
    passages_embeddings: torch.Tensor,
    query_tokens: List[str],
    passage_tokens: List[str],
    figsize: Tuple[int, int] = (55, 50),
    output_file: Optional[str] = None
) -> None:
    """
    Plots a filtered confusion matrix heatmap between query tokens and passage tokens.
    Filters out columns that do not contain any row‑wise maximum similarity.
    Each cell is annotated with its similarity value (with an appended "x" for maximum cells).

    Args:
        query_embeddings: Tensor of shape (B, num_query_tokens, embedding_dim)
        passages_embeddings: Tensor of shape (C, num_passage_tokens, embedding_dim)
        query_tokens: List of query token strings (row labels)
        passage_tokens: List of passage token strings (column labels)
        figsize: Figure size (width, height)
        output_file: Optional file path to save the plot
    """
    similarity_tensor = torch.einsum("bnd,csd->bcns", query_embeddings, passages_embeddings)
    sim_map = similarity_tensor[0, 0]  # Assume single batch element for visualization
    sim_map_np = sim_map.float().cpu().detach().numpy()

    max_val_mask = sim_map_np == np.max(sim_map_np, axis=1, keepdims=True)
    # Only keep columns where at least one row has the maximum value.
    valid_cols = np.any(max_val_mask, axis=0)
    sim_map_np_filtered = sim_map_np[:, valid_cols]
    max_val_mask_filtered = max_val_mask[:, valid_cols]
    passage_labels_filtered = [passage_tokens[i] for i, valid in enumerate(valid_cols) if valid]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(sim_map_np_filtered,
                     annot=False,
                     cmap="Blues",
                     xticklabels=passage_labels_filtered,
                     yticklabels=query_tokens,
                     linewidths=1,
                     linecolor="black",
                     cbar=False)
    add_custom_annotations(ax, sim_map_np_filtered, max_val_mask_filtered, fontsize=100)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=50)  # Set the colorbar label font size
     
    ax.tick_params(axis="x", labelsize=80)
    ax.tick_params(axis="y", labelsize=80)
    plt.yticks(rotation=0)
    plt.xticks(rotation=60) 
    # plt.xlabel("Passage Token", fontsize=80)
    # plt.ylabel("Query Token", fontsize=80)
    # plt.title("Confusion Matrix: Similarity Between Query Tokens and Passage Tokens", fontsize=80)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()

        
def extract_text(image: Image.Image) -> str:
    """
    Extract text from an image using EasyOCR.

    Args:
        image: PIL Image object

    Returns:
        Extracted text as a single string.
    """
    # Convert PIL image to NumPy array (EasyOCR accepts NumPy arrays)
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(np.array(image))
    texts = [text for (_, text, _) in results]
    return " ".join(texts)

def add_index_for_patch(tokens):
    pad_index = 0
    new_tokens = []
    # Loop through the tokens and replace <|image_pad|> with the current index
    for token in tokens:
        if token == '<image>':
            new_tokens.append("Patch:"+str(pad_index))
            pad_index += 1
        else:
            new_tokens.append(token)
    return new_tokens


def main() -> None:
    # Set up model, processor, and device.
    # model_name = "/ivi/ilps/personal/jqiao/colpali/models/colqwen2-v1.3-PairwiseCELoss"
    # model_name = "vidore/colqwen2-v0.1"
    model_name = "/ivi/ilps/personal/jqiao/colpali/models/colpali-v1.3-PairwiseCELoss"

    device = get_torch_device("auto")
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()

    processor = ColPaliProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_name = model_name.split("/")[-1]

    # Load image, query, and extract passage text.
    image_path = "/ivi/ilps/personal/jqiao/colpali/scripts/image.jpg"
    image = Image.open(image_path)
    query = "What services does HealthTeamWorks provide?"
    passage = extract_text(image)

    # Preprocess inputs.
    batch_images = processor.process_images([image]).to(device)
    batch_queries = processor.process_queries([query]).to(device)
    batch_passages = processor.process_passages([passage]).to(device)

    # Forward passes with torch.no_grad() to avoid gradient calculations.
    with torch.no_grad():
        image_embeddings = model.forward(**batch_images)
        query_embeddings = model.forward(**batch_queries)
        passages_embeddings = model.forward(**batch_passages)

    n_patches = processor.get_n_patches(
        image_size=image.size,
        patch_size=model.patch_size,
    )
    image_mask = processor.get_image_mask(batch_images)
    batched_similarity_maps = get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )

    # Retrieve the similarity map for the (only) input image.
    similarity_maps = batched_similarity_maps[0]  # Shape: (query_length, n_patches_x, n_patches_y)

    # Tokenize query and passage (adding a suffix if needed).
    suffix = tokenizer.pad_token * 10
    #  + self.query_prefix + query
    tokenized_query = tokenizer.bos_token + "Query: " + query + suffix
    query_tokens = processor.tokenizer.tokenize(tokenized_query)
    passage_tokens = processor.tokenizer.tokenize(passage)
    patch_tokens = tokenizer.convert_ids_to_tokens(batch_images['input_ids'][0])
    patch_tokens = add_index_for_patch(patch_tokens)

    print(f"Number of passage tokens: {len(passage_tokens)}")
    print(f"Number of query tokens: {len(query_tokens)}, {query_tokens}")
    print(f"query embeddings shape: {query_embeddings.shape}")
    print(f"Passages embeddings shape: {passages_embeddings.shape}")
    print("image_embeddings ", image_embeddings.shape)
    print("patch_tokens ", len(patch_tokens))
    
    plot_confusion_matrix_tokens_vs_passages_filtered(
        query_embeddings=query_embeddings,
        passages_embeddings=passages_embeddings,
        query_tokens=query_tokens,
        passage_tokens=passage_tokens,
        output_file=f"/ivi/ilps/personal/jqiao/colpali/plots/{model_name}_confusion_matrix_text_filtered2.pdf"
    )

    plot_confusion_matrix_tokens_vs_passages_filtered(
        query_embeddings=query_embeddings,
        passages_embeddings=image_embeddings,
        query_tokens=query_tokens,
        passage_tokens=patch_tokens,
        output_file=f"/ivi/ilps/personal/jqiao/colpali/plots/{model_name}_confusion_matrix_patch_filtered2.pdf"
    )

if __name__ == "__main__":
    main()

# srun --gres=gpu:nvidia_rtx_a6000:1 --partition=gpu --mem=30G python /ivi/ilps/personal/jqiao/colpali/scripts/RQ3_v5.py