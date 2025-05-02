import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", 
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V2-Lite",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # Use torch.float16 if bfloat16 is not supported
).cuda()

# Prepare input
input_text = "Explain the concept of mixture-of-experts in machine learning."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=200, 
                         do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
