"""
A comprehensive benchmark suite for OasisDB to compare HNSW and IVF_FLAT performance.
This script is adapted to the OasisDB SDK's "set_params" workflow.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Dict, Any

import h5py
import numpy as np
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure the client SDK is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from client import OasisDBClient, OasisDBError

# --- 1. Configuration ---

# Dataset Configuration
DATASET_URL = "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
DATASET_FILENAME = "benchmark/fashion-mnist-784-euclidean/dataset/fashion-mnist-784-euclidean.hdf5" # Using a simple filename
VECTOR_DIMENSION = 784
TOP_K = 100

# --- Benchmark Configurations ---
BENCHMARK_CONFIGS = [
    {
        "name": "HNSW (M=16, efc=200)",
        "collection_name": "fashion_mnist_hnsw",
        "index_params": {
            "index_type": "hnsw",
            "M": "16",               # Parameters must be strings for the SDK
            "ef_construction": "200"
        },
        "search_param_name": "efsearch",
        "search_param_values": [100, 120, 150, 200, 250, 300, 400]
    },
    {
        "name": "IVF_FLAT (nlist=1024)",
        "collection_name": "fashion_mnist_ivf",
        "index_params": {
            "index_type": "ivf_flat",
            "nlist": "1024"
        },
        "search_param_name": "nprobe",
        "search_param_values": [1, 5, 10, 20, 40, 60, 80, 100]
    }
]

# --- 2. Data Loading (Unchanged) ---
def download_file(url: str, fname: str):
    if os.path.exists(fname):
        print(f"Dataset '{fname}' already exists. Skipping download.")
        return
    print(f"Downloading {fname}...")
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total_size = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=fname) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk); pbar.update(len(chunk))
    except Exception as e:
        print(f"\nFailed to download file: {e}"); sys.exit(1)

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    download_file(DATASET_URL, DATASET_FILENAME)
    print(f"Loading data from '{DATASET_FILENAME}' into memory...")
    with h5py.File(DATASET_FILENAME, 'r') as f:
        return np.array(f['train']), np.array(f['test']), np.array(f['neighbors'])

# --- 3. Core Benchmark Functions ---

def run_single_search_test(
    client: OasisDBClient,
    collection_name: str,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float]:
    """Runs a benchmark for the currently set parameters and returns (QPS, Recall)."""
    num_queries = len(query_vectors)
    all_results_ids = []

    start_time = time.time()
    # <<< CHANGE 1: The search_vectors call is now simpler. >>>
    # It relies on the server using the parameters set by client.set_params().
    for vec in query_vectors:
        search_results = client.search_vectors(collection_name, vec.tolist(), limit=TOP_K)
        all_results_ids.append({int(doc_id) for doc_id in search_results['ids']})
    end_time = time.time()

    total_time = end_time - start_time
    qps = num_queries / total_time
    avg_recall = calculate_recall(all_results_ids, ground_truth)
    
    return qps, avg_recall

def calculate_recall(
    results_ids: List[set[int]],
    ground_truth: np.ndarray
) -> float:
    """Helper function to calculate average recall."""
    total_recall = 0
    num_queries = len(results_ids)
    for i, result_ids in enumerate(results_ids):
        ground_truth_ids = set(ground_truth[i][:TOP_K])
        intersection = len(result_ids.intersection(ground_truth_ids))
        total_recall += intersection / TOP_K
    return total_recall / num_queries

def run_benchmark_for_config(
    client: OasisDBClient,
    config: Dict[str, Any],
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray
) -> List[Dict[str, float]]:
    """Manages the lifecycle for a full benchmark run of a single index type."""
    collection_name = config["collection_name"]
    results = []

    try:
        # 1. Create collection with specific index parameters
        print(f"\n--- Starting Benchmark for: {config['name']} ---")
        print(f"Creating collection '{collection_name}'...")

        # <<< CHANGE 2: Adapt the create_collection call to match the SDK. >>>
        # We separate index_type from the other build-time parameters.
        build_params = config["index_params"].copy()
        index_type = build_params.pop("index_type")
        client.create_collection(
            collection_name, 
            dimension=VECTOR_DIMENSION, 
            index_type=index_type, 
            parameters=build_params
        )
        
        # 2. Insert data and build index
        docs_to_insert = [{"id": str(i), "vector": v.tolist()} for i, v in enumerate(base_vectors)]
        print("Inserting data and building index...")
        start_build = time.time()
        client.build_index(collection_name, docs_to_insert)
        print(f"Index build complete in {time.time() - start_build:.2f} seconds.")

        # 3. Iterate through search parameters and run tests
        print("Running search tests with varying parameters...")
        for param_value in config["search_param_values"]:
            # <<< CHANGE 3: Use client.set_params() before running the search batch. >>>
            search_params = {config["search_param_name"]: param_value}
            print(f"  Setting parameters: {search_params}")
            client.set_params(collection_name, search_params)

            qps, recall = run_single_search_test(client, collection_name, query_vectors, ground_truth)
            print(f"  -> Result: QPS: {qps:.2f}, Recall: {recall:.4f}")
            results.append({"qps": qps, "recall": recall, "param": param_value})

    finally:
        # 4. Clean up the collection
        print(f"Cleaning up collection '{collection_name}'...")
        try:
            client.delete_collection(collection_name)
        except OasisDBError as e:
            print(f"Could not delete collection (likely auto-cleaned): {e}")
    
    return results

# --- 4. Plotting Function (Unchanged) ---
def plot_results(all_results: Dict[str, List[Dict[str, float]]]):
    plt.figure(figsize=(12, 8))
    for name, results in all_results.items():
        if not results: continue
        results.sort(key=lambda x: x['recall'])
        qps_values = [r['qps'] for r in results]
        recall_values = [r['recall'] for r in results]
        plt.plot(qps_values, recall_values, marker='o', linestyle='-', label=name)
    plt.title('OasisDB Performance Benchmark (Fashion-MNIST)')
    plt.xlabel('QPS (Queries per Second)')
    plt.ylabel('Recall@100')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(0.0, 1.05)
    graph_filename = "oasisdb_performance_curves.png"
    plt.savefig(graph_filename)
    print(f"\nPerformance graph saved to: {graph_filename}")
    plt.show()

# --- 5. Main Orchestrator (Unchanged) ---
def main():
    client = OasisDBClient()
    all_benchmark_results = {}
    try:
        ok = client.health_check(); assert ok, "Health check failed"
        print("Health check: OK")
        base_vectors, query_vectors, ground_truth = load_data()
        for config in BENCHMARK_CONFIGS:
            try: client.delete_collection(config["collection_name"])
            except OasisDBError: pass
            results = run_benchmark_for_config(client, config, base_vectors, query_vectors, ground_truth)
            all_benchmark_results[config['name']] = results
    except (OasisDBError, Exception) as e:
        print(f"\nA critical error occurred during the benchmark suite: {e}")
    finally:
        client.close()
    if all_benchmark_results:
        plot_results(all_benchmark_results)
    else:
        print("\nNo benchmark results were collected to plot.")

if __name__ == "__main__":
    main()