import torch
import os
import shutil
import random
import time
import json
import sys
import numpy as np
from itertools import product
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, list_repo_files

def download_model_files(repo_id, target_dir, file_types=None):
    if file_types is None:
        file_types = [".safetensors"]

    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading files from {repo_id}...")

    files = list_repo_files(repo_id)
    matched_files = \
        [f for f in files if any(f.endswith(ft) for ft in file_types)]

    for filename in matched_files:
        print(f"Downloading: {filename}")
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename)
        target_path = os.path.join(target_dir, filename)

        # Ensure target subdirs exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(cached_file, target_path)

    print(f"\n✅ Downloaded {len(matched_files)} files to: {target_dir}")

# download_model_files(
#     repo_id="deepseek-ai/deepseek-v2-lite-chat",
#     target_dir="./DeepSeek-V2-Lite",
# )

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./DeepSeek-V2-Lite", 
                                          trust_remote_code=True)

num_cache_size_list = [1, 2, 4, 8, 16, 32, 64]
num_lookahead_size_list = [2, 4, 6, 8, 10, 12]
prefetch_size_list = [2, 4, 6, 8, 10, 12]
use_oracle_list = [True, False]
use_prefetch_list = [True, False]

settings = [
    [[False], [True], [1, 2, 4, 8, 16, 32, 64], [8], [6]],
    [[True], [True], [1, 2, 4, 8, 16, 32, 64], [8], [6]],
    [[True], [False], [1, 2, 4, 8, 16, 32, 64], [8], [6]],
    [[True], [True], [32], [2, 4, 6, 8, 10, 12], [6]],
    [[True], [False], [32], [2, 4, 6, 8, 10, 12], [6]],
    [[True], [True], [32], [8], [2, 4, 6, 8, 10, 12]],
    [[True], [False], [32], [8], [2, 4, 6, 8, 10, 12]],
]

out_csv_path = "test_results.csv"
# if os.path.exists(out_csv_path):
#     os.remove(out_csv_path)

# with open(out_csv_path, "w") as f:
#     f.write("use_prefetch, use_oracle, num_cache_size, num_lookahead_size, "
#             + "prefetch_size, time_taken, prefetch_hit_ratio, cache_hit_ratio\n")
    
for use_prefetch, use_oracle, num_cache_size, num_lookahead_size, \
    prefetch_size in product(
    use_prefetch_list,
    use_oracle_list,
    num_cache_size_list,
    num_lookahead_size_list,
    prefetch_size_list,
):
    print("//// Test Configs: "
        f"num_cache_size: {num_cache_size}, num_lookahead_size: " +
        f"{num_lookahead_size}, prefetch_size: {prefetch_size}, " +
        f"use_oracle: {use_oracle}, use_prefetch: {use_prefetch} ////")

    model = AutoModelForCausalLM.from_pretrained(
        "./DeepSeek-V2-Lite",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        num_cache_size=num_cache_size,
        num_lookahead_size=num_lookahead_size,
        prefetch_size=prefetch_size,
        use_oracle=use_oracle,
        use_prefetch=use_prefetch,
    )


    # Prepare input
    input_text = "Explain the concept of mixture-of-experts in machine learning."
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Warm up cache in the CPU
    for _ in range(1):
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False,)

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    time_start = time.time()
    num_iter = 2
    for i in range(num_iter):
        outputs = model.generate(**inputs, max_new_tokens=99, do_sample=True, temperature=0.7)
    time_taken = time.time() - time_start
    print(f"Time taken for {num_iter} iterations: ", time_taken)

    out_path = f"test_output"
    prefetch_reqs = 0
    prefetch_hits = 0
    cache_hits = 0
    num_experts = 0
    for filename in os.listdir(out_path):
        if filename.endswith(".json") and "vtensor" in filename:
            file_path = os.path.join(out_path, filename)
            data_i = json.load(open(file_path, "r"))
            step_idx = data_i["step_idx"]
            for i in range(1, 27):
                experts = data_i[f"vtensor_{i}"]["get_requested"]
                if experts == "None" or len(experts) > 6:
                    continue
                pref_reqs = data_i[f"vtensor_{i}"]["prefetch_requested"]
                if pref_reqs == "None":
                    pref_reqs = []
                pref_reqs = torch.tensor(pref_reqs).flatten()
                experts = torch.tensor(experts).flatten()
                pref_hits = torch.isin(pref_reqs, experts).sum()
                cache_hit = data_i[f"vtensor_{i}"]["cache_hit"]
                # Count the number of "True" values in the "cache_hit" list
                cache_hits += cache_hit.count("True")
                num_experts += len(experts)
                prefetch_reqs += len(pref_reqs)
                prefetch_hits += pref_hits

    print(f"{use_prefetch}, {use_oracle}, {num_cache_size}, {num_lookahead_size}, {prefetch_size}, " + 
        f"{time_taken}, {prefetch_hits / prefetch_reqs}, {cache_hits / num_experts}")
    with open(out_csv_path, "a") as f:
        f.write(f"{use_prefetch}, {use_oracle}, {num_cache_size}, {num_lookahead_size}, " +
                f"{prefetch_size}, {time_taken}, {prefetch_hits / prefetch_reqs}, " +
                f"{cache_hits / num_experts}\n")
    # Clean up
    # os.remove(out_path)
    # Remove out_path directory
    os.system(f"rm -rf {out_path}")