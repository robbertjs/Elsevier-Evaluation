import logging
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Dict, List, Optional, cast
import huggingface_hub
import typer
from datasets import Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from datasets import Dataset
import json
from vidore_benchmark.compression.token_pooling import HierarchicalEmbeddingPooler
from vidore_benchmark.evaluation.evaluate import evaluate_dataset, evaluate_dataset_from_indexing, evaluate_dataset_matching, evaluate_dataset_from_imagetexts
from vidore_benchmark.evaluation.interfaces import MetadataModel, ViDoReBenchmarkResults
from vidore_benchmark.retrievers.registry_utils import load_vision_retriever_from_registry
from vidore_benchmark.utils.logging_utils import setup_logging
import torch
import tqdm
import time
import pandas as pd
from PIL import Image
import io

# Function to convert byte arrays to PIL Image objects (if needed)
def convert_bytes_to_image(byte_data):
    if byte_data is not None:
        return Image.open(io.BytesIO(byte_data))
    return None


logger = logging.getLogger(__name__)

load_dotenv(override=True)

OUTPUT_DIR = Path("outputs")

app = typer.Typer(
    help="CLI for evaluating retrievers on the ViDoRe benchmark.",
    no_args_is_help=True,
)

def sanitize_model_id(model_class: str, pretrained_model_name_or_path: Optional[str] = None) -> str:
    """
    Return sanitized model ID for saving metrics.
    """
    model_id = pretrained_model_name_or_path if pretrained_model_name_or_path is not None else model_class
    model_id = model_id.replace("/", "_")
    return model_id

def add_column(examples):
    # Compute the length of each query in the batch
    examples["text_description"] = ['' for query in examples["query"]]
    return examples

@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    setup_logging(log_level)
    logger.info("Logging level set to `%s`", log_level)


@app.command()
def evaluate_retriever(
    model_class: Annotated[str, typer.Option(help="Model class")],
    pretrained_model_name_or_path: Annotated[
        Optional[str],
        typer.Option(
            "--model-name",
            help="If model class is a Hf model, this arg is passed to the `model.from_pretrained` method.",
        ),
    ] = None,
    dataset_name: Annotated[Optional[str], typer.Option(help="HuggingFace Hub dataset name")] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_query: Annotated[int, typer.Option(help="Batch size for query embedding inference")] = 8,
    batch_passage: Annotated[int, typer.Option(help="Batch size for passages embedding inference")] = 8,
    batch_score: Annotated[Optional[int], typer.Option(help="Batch size for score computation")] = 16,
    collection_name: Annotated[
        Optional[str],
        typer.Option(help="Dataset collection to use for evaluation. Can be a Hf collection id or a local dirpath."),
    ] = None,
    collection_name2: Annotated[
        Optional[str],
        typer.Option(help="Dataset collection to use for evaluation. Can be a Hf collection id or a local dirpath."),
    ] = None,
    use_token_pooling: Annotated[bool, typer.Option(help="Whether to use token pooling for text embeddings")] = False,
    pool_factor: Annotated[int, typer.Option(help="Pooling factor for hierarchical token pooling")] = 3,
    indexing_path: Annotated[str, typer.Option(help="INDEX")] = None,
    data_index_name: Annotated[str, typer.Option(help="INDEX")] = None,
    use_visual: Annotated[bool, typer.Option(help="x")] = False,
    matching_type: Annotated[str, typer.Option(help="matching type")] = "",
):
    """
    Evaluate the retriever on the given dataset or collection.
    The metrics are saved to a JSON file.
    """
    # Log all parameters
    logging.info(f"Starting evaluation with the following parameters:")
    logging.info(f"Model Class: {model_class}")
    if pretrained_model_name_or_path:
        logging.info(f"Pretrained Model Name/Path: {pretrained_model_name_or_path}")
    if dataset_name:
        logging.info(f"Dataset Name: {dataset_name}")
    logging.info(f"Split: {split}")
    logging.info(f"Batch Query Size: {batch_query}")
    logging.info(f"Batch Passage Size: {batch_passage}")
    if batch_score:
        logging.info(f"Batch Score Size: {batch_score}")
    if collection_name:
        logging.info(f"Collection Name: {collection_name}")
    logging.info(f"Use Token Pooling: {use_token_pooling}")
    logging.info(f"Pooling Factor: {pool_factor}")

    logging.info(f"Evaluating retriever `{model_class}`")
    print(f"Use Token Pooling: {use_token_pooling}")

    # Sanity check
    if dataset_name is None and collection_name is None:
        raise ValueError("Please provide a dataset name or collection name")
    elif dataset_name is not None and collection_name is not None:
        raise ValueError("Please provide only one of dataset name or collection name")

    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(
        model_class,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

    # Sanitize the model ID to use as a filename
    model_id = sanitize_model_id(model_class, pretrained_model_name_or_path)

    # Get the pooling strategy
    embedding_pooler = HierarchicalEmbeddingPooler(pool_factor) if use_token_pooling else None

    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataset(s) and evaluate
    if dataset_name is None and collection_name is None:
        raise ValueError("Please provide a dataset name or collection name.")

    elif dataset_name is not None:
        dataset = cast(Dataset, load_dataset(dataset_name, split=split))

        if "syntheticDocQA_healthcare_industry_test" or "arxivqa_test_subsampled" \
            "syntheticDocQA_healthcare_industry_test_tesseract" or "arxivqa_test_subsampled_tesseract" in dataset_name:
            metrics = {
                    dataset_name: evaluate_dataset_matching(
                        retriever,
                        dataset,
                        batch_query=batch_query,
                        batch_passage=batch_passage,
                        batch_score=batch_score,
                        embedding_pooler=embedding_pooler,
                        matching_type=matching_type
                    )
                }

            if use_token_pooling:
                savepath = OUTPUT_DIR / f"{model_id}_{matching_type}_metrics_pool_factor_{pool_factor}.json"
            else:
                savepath = OUTPUT_DIR / f"{model_id}_{matching_type}_metrics.json"

            print(f"nDCG@5 for {matching_type} {model_id} on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

            results = ViDoReBenchmarkResults(
                metadata=MetadataModel(
                    timestamp=datetime.now(),
                    vidore_benchmark_version=version("vidore_benchmark"),
                ),
                metrics={dataset_name: metrics[dataset_name]},
            )

            with open(str(savepath), "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=4))

            print(f"Benchmark results saved to `{savepath}`")
        
        else:
            metrics = {
                dataset_name: evaluate_dataset(
                    retriever,
                    dataset,
                    batch_query=batch_query,
                    batch_passage=batch_passage,
                    batch_score=batch_score,
                    embedding_pooler=embedding_pooler,
                )
            }

            if use_token_pooling:
                savepath = OUTPUT_DIR / f"{model_id}_metrics_pool_factor_{pool_factor}.json"
            else:
                savepath = OUTPUT_DIR / f"{model_id}_metrics.json"

            print(f"nDCG@5 for {model_id} on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

            results = ViDoReBenchmarkResults(
                metadata=MetadataModel(
                    timestamp=datetime.now(),
                    vidore_benchmark_version=version("vidore_benchmark"),
                ),
                metrics={dataset_name: metrics[dataset_name]},
            )

            with open(str(savepath), "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=4))

            print(f"Benchmark results saved to `{savepath}`")

    elif collection_name is not None and indexing_path is None and collection_name2 is None:
        if os.path.isdir(collection_name):
            print(f"Loading datasets from local directory: `{collection_name}`")
            dataset_names = os.listdir(collection_name)
            dataset_names = [os.path.join(collection_name, dataset) for dataset in dataset_names]
        else:
            print(f"Loading datasets from the Hf Hub collection: {collection_name}")
            collection = huggingface_hub.get_collection(collection_name)
            dataset_names = [dataset_item.item_id for dataset_item in collection.items]

        # Placeholder for all metrics
        metrics_all: Dict[str, Dict[str, float]] = {}
        results_all: List[ViDoReBenchmarkResults] = []

        savedir = OUTPUT_DIR / model_class / model_id.replace("/", "_")
        savedir.mkdir(parents=True, exist_ok=True)

        for dataset_name in dataset_names:
            print(f"\n ---------------------------\nEvaluating {dataset_name}")
            dataset = cast(Dataset, load_dataset(dataset_name, split=split))
            dataset_name = dataset_name.replace(collection_name + "/", "")
            agg_metrics, query_metrics, run_results = evaluate_dataset(
                    retriever,
                    dataset,
                    batch_query=batch_query,
                    batch_passage=batch_passage,
                    batch_score=batch_score,
                    embedding_pooler=embedding_pooler,
                )

            metrics = {
                dataset_name: agg_metrics
            }
            metrics_all.update(metrics)

            # Sanitize the dataset item to use as a filename
            dataset_item_id = dataset_name.replace("/", "_")

            if use_token_pooling:
                savepath = savedir / f"{dataset_item_id}_metrics_pool_factor_{pool_factor}.json"
            else:
                savepath = savedir / f"{dataset_item_id}_metrics.json"
                query_metrics_savepath = savedir / f"{dataset_item_id}_query_metrics.json"
                run_results_savepath = savedir / f"{dataset_item_id}_run_results.json"

            print(f"nDCG@5 for {model_id} on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

            results = ViDoReBenchmarkResults(
                metadata=MetadataModel(
                    timestamp=datetime.now(),
                    vidore_benchmark_version=version("vidore_benchmark"),
                ),
                metrics={dataset_name: metrics[dataset_name]},
            )
            results_all.append(results)

            with open(str(query_metrics_savepath), "w", encoding="utf-8") as f:
                f.write(json.dumps(query_metrics))
            print(f"query_metrics saved to `{query_metrics_savepath}`")

            with open(str(run_results_savepath), "w", encoding="utf-8") as f:
                f.write(json.dumps(run_results))
            print(f"run_results saved to `{run_results_savepath}`")

            with open(str(savepath), "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=4))
            print(f"Benchmark results saved to `{savepath}`")


        if use_token_pooling:
            savepath_all = OUTPUT_DIR / f"{model_id}_all_metrics_pool_factor_{pool_factor}.json"
        else:
            savepath_all = OUTPUT_DIR / f"{model_id}_all_metrics.json"

        results_merged = ViDoReBenchmarkResults.merge(results_all)

        with open(str(savepath_all), "w", encoding="utf-8") as f:
            f.write(results_merged.model_dump_json(indent=4))

        print(f"Concatenated metrics saved to `{savepath_all}`")

    elif indexing_path is not None:
        if collection_name.endswith(".jsonl"):
            # Placeholder for all metrics
            metrics_all: Dict[str, Dict[str, float]] = {}
            results_all: List[ViDoReBenchmarkResults] = []

            savedir = OUTPUT_DIR / model_id.replace("/", "_")
            savedir.mkdir(parents=True, exist_ok=True)

            print(f"\n ---------------------------\nLoading passages and index {indexing_path}")
            passages = []
            indexing = torch.load(indexing_path)["embeddings"]
            query_ds = {'query': []}
            
            passages_ds = {'query': [], 'image_filename': []}
            number_of_queries = 100 if "health" or "ai" in collection_name else 500 if "arxivqa" in collection_name else 0
      
            with open(collection_name, 'r') as file:
                for line in tqdm.tqdm(file, desc="Processing indexing path"):
                    data = json.loads(line)
                    if len(query_ds['query']) < number_of_queries:
                        query_ds['query'].append(str(data['query']))
                    
                    passages_ds['query'].append(str(data['query']))
                    passages_ds['image_filename'].append(str(data['image_filename']))

            query_ds = Dataset.from_dict(query_ds)
            passages_ds = Dataset.from_dict(passages_ds)
            # start_time = time.time()
            # print("start to search ", start_time, "number of queries ", len(query_ds), "number of passages ", len(passages_ds))
            agg_metrics, query_metrics, run_results = evaluate_dataset_from_indexing(
                    retriever,
                    query_ds,
                    passages_ds,
                    batch_query=batch_query,
                    emb_passages=indexing,
                    batch_score=batch_score,
                )
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f"Search took {elapsed_time} seconds to complete.")

            metrics = {
                data_index_name: agg_metrics,
                # "elapsed_time": elapsed_time
            }

            if use_token_pooling:
                savepath = savedir / f"{data_index_name}_metrics_pool_factor_{pool_factor}.json"
            else:
                savepath = savedir / f"{data_index_name}_metrics.json"
                query_metrics_savepath = savedir / f"{data_index_name}_query_metrics.json"
                run_results_savepath = savedir / f"{data_index_name}_run_results.json"

            print(f"nDCG@5 for on {data_index_name}: {metrics[data_index_name]['ndcg_at_5']}")

            with open(str(query_metrics_savepath), "w", encoding="utf-8") as f:
                f.write(json.dumps(query_metrics))
            print(f"query_metrics saved to `{query_metrics_savepath}`")

            with open(str(run_results_savepath), "w", encoding="utf-8") as f:
                f.write(json.dumps(run_results))
            print(f"run_results saved to `{run_results_savepath}`")

            with open(str(savepath), "w", encoding="utf-8") as f:
                json_line = json.dumps(metrics) + "\n"  # Convert dict to JSON string and add a newline
                f.write(json_line)
            print(f"Benchmark results saved to `{savepath}`")
            
        else:
            if os.path.isdir(collection_name):
                print(f"Loading datasets from local directory: `{collection_name}`")
                dataset_names = os.listdir(collection_name)
                dataset_names = [os.path.join(collection_name, dataset) for dataset in dataset_names]
            else:
                print(f"Loading datasets from the Hf Hub collection: {collection_name}")
                collection = huggingface_hub.get_collection(collection_name)
                dataset_names = [dataset_item.item_id for dataset_item in collection.items]

            # Placeholder for all metrics
            metrics_all: Dict[str, Dict[str, float]] = {}
            results_all: List[ViDoReBenchmarkResults] = []

            savedir = OUTPUT_DIR / model_id.replace("/", "_")
            savedir.mkdir(parents=True, exist_ok=True)

            print(f"\n ---------------------------\nLoading passages and index")
            passages = []
            for dataset_name in dataset_names:
                passages.append(load_dataset(dataset_name, split=split))
            passages_ds = concatenate_datasets(passages)
            indexing = torch.load(indexing_path)["embeddings"]

            for dataset_name in dataset_names:
                print(f"\n ---------------------------\nEvaluating {dataset_name}")
                query_ds = cast(Dataset, load_dataset(dataset_name, split=split))
                dataset_name = dataset_name.replace(collection_name + "/", "")
                
                agg_metrics, query_metrics, run_results = evaluate_dataset_from_indexing(
                        retriever,
                        query_ds,
                        passages_ds,
                        batch_query=batch_query,
                        emb_passages=indexing,
                        batch_score=batch_score,
                    )

                metrics = {
                    dataset_name: agg_metrics
                }
                metrics_all.update(metrics)

                # Sanitize the dataset item to use as a filename
                dataset_item_id = dataset_name.replace("/", "_")

                if use_token_pooling:
                    savepath = savedir / f"{dataset_item_id}_metrics_pool_factor_{pool_factor}.json"
                else:
                    savepath = savedir / f"{dataset_item_id}_metrics.json"
                    query_metrics_savepath = savedir / f"{dataset_item_id}_query_metrics.json"
                    run_results_savepath = savedir / f"{dataset_item_id}_run_results.json"

                print(f"nDCG@5 for {model_id} on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

                results = ViDoReBenchmarkResults(
                    metadata=MetadataModel(
                        timestamp=datetime.now(),
                        vidore_benchmark_version=version("vidore_benchmark"),
                    ),
                    metrics={dataset_name: metrics[dataset_name]},
                )
                results_all.append(results)

                with open(str(query_metrics_savepath), "w", encoding="utf-8") as f:
                    f.write(json.dumps(query_metrics))
                print(f"query_metrics saved to `{query_metrics_savepath}`")

                with open(str(run_results_savepath), "w", encoding="utf-8") as f:
                    f.write(json.dumps(run_results))
                print(f"run_results saved to `{run_results_savepath}`")

                with open(str(savepath), "w", encoding="utf-8") as f:
                    f.write(results.model_dump_json(indent=4))
                print(f"Benchmark results saved to `{savepath}`")


            if use_token_pooling:
                savepath_all = OUTPUT_DIR / f"{model_id}_all_metrics_pool_factor_{pool_factor}.json"
            else:
                savepath_all = OUTPUT_DIR / f"{model_id}_all_metrics.json"

            results_merged = ViDoReBenchmarkResults.merge(results_all)

            with open(str(savepath_all), "w", encoding="utf-8") as f:
                f.write(results_merged.model_dump_json(indent=4))

            print(f"Concatenated metrics saved to `{savepath_all}`")

    elif collection_name is not None and collection_name2 is not None:
        if os.path.isdir(collection_name):
            print(f"Loading datasets from local directory: `{collection_name}`")
            dataset_names1 = os.listdir(collection_name)
            dataset_names1 = [os.path.join(collection_name, dataset) for dataset in dataset_names]
        else:
            print(f"Loading datasets from the Hf Hub collection: {collection_name} {collection_name2}")
            collection1 = huggingface_hub.get_collection(collection_name)
            dataset_names1 = [dataset_item.item_id for dataset_item in collection1.items]

            collection2 = huggingface_hub.get_collection(collection_name2)
            dataset_names2 = [dataset_item.item_id for dataset_item in collection2.items]

        # Placeholder for all metrics
        metrics_all: Dict[str, Dict[str, float]] = {}
        results_all: List[ViDoReBenchmarkResults] = []

        savedir = OUTPUT_DIR / model_class / model_id.replace("/", "_")
        savedir.mkdir(parents=True, exist_ok=True)

        for dataset_name, dataset_name2 in zip(dataset_names1, dataset_names2):
            print(f"\n ---------------------------\nEvaluating {dataset_name}")
            dataset1 = load_dataset(dataset_name, split=split)
            dataset2 = load_dataset(dataset_name2, split=split)
            # Add a placeholder "text_description" column to dataset1 if it does not exist
            if "text_description" not in dataset1.column_names:
                dataset1 = dataset1.add_column("text_description", [""] * len(dataset1))

            dataset = concatenate_datasets([dataset2, dataset1])
            print("Dataset columns:", dataset.column_names)
            print(dataset["text_description"][:5])
            print(dataset["image"][:5])

            # if len(dataset1) != len(dataset2):
            #     print("Warning: Datasets do not have the same number of entries!")
            # df1 = dataset1.to_pandas()
            # df2 = dataset2.to_pandas()

            # print(df1.iloc[0])
            # merged_df = pd.merge(df1, df2[['query', 'text_description']], on='query', how='left')
            # print(merged_df.columns)
            # # print(merged_df.iloc[0])
            # dataset = Dataset.from_pandas(merged_df)
            # dataset = cast(Dataset, dataset)
            
            dataset_name = dataset_name.replace(collection_name + "/", "")

            agg_metrics, query_metrics, run_results = evaluate_dataset_from_imagetexts(
                    retriever,
                    dataset,
                    batch_query=batch_query,
                    batch_passage=batch_passage,
                    batch_score=batch_score,
                    embedding_pooler=embedding_pooler,
                )

            metrics = {
                dataset_name: agg_metrics
            }
            metrics_all.update(metrics)

            # Sanitize the dataset item to use as a filename
            dataset_item_id = dataset_name.replace("/", "_") + "image_text"

            if use_token_pooling:
                savepath = savedir / f"{dataset_item_id}_metrics_pool_factor_{pool_factor}.json"
            else:
                savepath = savedir / f"{dataset_item_id}_metrics.json"
                query_metrics_savepath = savedir / f"{dataset_item_id}_query_metrics.json"
                run_results_savepath = savedir / f"{dataset_item_id}_run_results.json"

            print(f"nDCG@5 for {model_id} on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

            results = ViDoReBenchmarkResults(
                metadata=MetadataModel(
                    timestamp=datetime.now(),
                    vidore_benchmark_version=version("vidore_benchmark"),
                ),
                metrics={dataset_name: metrics[dataset_name]},
            )
            results_all.append(results)

            with open(str(query_metrics_savepath), "w", encoding="utf-8") as f:
                f.write(json.dumps(query_metrics))
            print(f"query_metrics saved to `{query_metrics_savepath}`")

            with open(str(run_results_savepath), "w", encoding="utf-8") as f:
                f.write(json.dumps(run_results))
            print(f"run_results saved to `{run_results_savepath}`")

            with open(str(savepath), "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=4))
            print(f"Benchmark results saved to `{savepath}`")


        if use_token_pooling:
            savepath_all = OUTPUT_DIR / f"{model_id}_all_metrics_pool_factor_{pool_factor}.json"
        else:
            savepath_all = OUTPUT_DIR / f"{model_id}_all_metrics.json"

        results_merged = ViDoReBenchmarkResults.merge(results_all)

        with open(str(savepath_all), "w", encoding="utf-8") as f:
            f.write(results_merged.model_dump_json(indent=4))

        print(f"Concatenated metrics saved to `{savepath_all}`")


    print("Done.")

if __name__ == "__main__":
    app()
