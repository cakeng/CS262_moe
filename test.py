import torch
import os
import shutil
import random
import time
import numpy as np
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
model = AutoModelForCausalLM.from_pretrained(
    "./DeepSeek-V2-Lite",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 
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

# Generate output
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

time_start = time.time()
num_iter = 2
for i in range(num_iter):
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"Time taken for {num_iter} iterations: ", time.time() - time_start)