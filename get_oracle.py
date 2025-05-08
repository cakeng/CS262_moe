import torch
import json
import os

# path = "run_664602"
path = "run_739942"
expert_idx = {}

# Iterate over all files in the directory
for filename in os.listdir(path):
    # Check if the file is a .safetensors file
    if filename.endswith(".json") and "vtensor" in filename:
        # Construct full file path
        file_path = os.path.join(path, filename)
        data_i = json.load(open(file_path, "r"))
        step_idx = data_i["step_idx"]
        for i in range(1, 27):
            experts = data_i[f"vtensor_{i}"]["get_requested"]
            expert_idx[f"{step_idx}_{i}"] = experts

torch.save(expert_idx, "oracle_expert_idx.pt")            
print(expert_idx)



        