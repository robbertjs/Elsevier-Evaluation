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



# torch.set_printoptions(profile="full")

model_name = "vidore/colqwen2-v0.1"
device = get_torch_device("auto")

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Load the image and query
image = Image.open("/ivi/ilps/personal/jqiao/colpali/scripts/example_image.png")
query = "What is the estimated total savings for a PV system in Durham under the net metering (flat rate) billing option over the system useful life of 25 years?"
# Preprocess inputs
batch_images = processor.process_images([image]).to(device)
batch_queries = processor.process_queries([query]).to(device)

# Forward passes
with torch.no_grad():
    image_embeddings = model.forward(**batch_images)
    query_embeddings = model.forward(**batch_queries)

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
query_tokens = processor.tokenizer.tokenize(query)

def plot_confusion_matrix(data: torch.Tensor, labels, title: str = "Confusion Matrix"):
    """
    Plots a confusion matrix using seaborn heatmap functionality.
    
    Args:
        data: 2D array or tensor of shape (num_tokens, num_patches) with the scores.
        labels: List of labels for the query tokens.
        title: Title for the confusion matrix plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Query Tokens')
    plt.xlabel('Image Patches')
    plt.show()

    plt.savefig(f"confusion_matrix.png")

def prepare_confusion_data(similarity_maps: torch.Tensor) -> torch.Tensor:
    """
    Prepares a 'confusion' matrix where each row corresponds to a query token and each column corresponds to the patch index
    that token is most similar to the most often.
    
    Args:
        similarity_maps: Tensor of shape (num_tokens, height, width) representing similarity scores.
        
    Returns:
        2D tensor where each row is a histogram of maximum patch indices for each token.
    """
    num_tokens, height, width = similarity_maps.shape
    num_patches = height * width

    # Flatten the spatial dimensions into one dimension for easier indexing
    flat_similarity = similarity_maps.view(num_tokens, num_patches)

    # Find the indices of the max similarity scores
    max_indices = flat_similarity.argmax(dim=1)

    # Create a histogram for each token of max indices
    histograms = torch.zeros((num_tokens, num_patches), dtype=torch.int)
    for i in range(num_tokens):
        histograms[i, max_indices[i]] += 1  # Increment the patch index where max occurs

    return histograms

# Assuming `similarity_maps` is already computed and is a tensor
max_scores = prepare_confusion_data(similarity_maps)

# Assuming `query_tokens` is a list of strings representing token labels
plot_confusion_matrix(max_scores, query_tokens)



# with open("similarity.txt", "w") as outfn:
#     outfn.write(f"image_embeddings {image_embeddings.shape} {image_embeddings}\n")
#     outfn.write(f"query_embeddings {query_embeddings.shape} {query_embeddings}\n")
#     outfn.write(f"n_patches {n_patches}\n")
#     outfn.write(f"similarity_maps {similarity_maps.shape} {similarity_maps}\n")
#     outfn.write(f"query_tokens {len(query_tokens)} {query_tokens}\n")

# plots = plot_all_similarity_maps(
#     image=image,
#     query_tokens=query_tokens,
#     similarity_maps=similarity_maps,
# )
# for idx, (fig, ax) in enumerate(plots):
#     fig.savefig(f"similarity_map_{idx}.png")

# srun --gres=gpu:nvidia_rtx_a6000:1 --partition=gpu --mem=30G python /ivi/ilps/personal/jqiao/colpali/scripts/RQ3_v2.py