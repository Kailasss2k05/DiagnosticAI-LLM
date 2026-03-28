from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "lora_adapter")

model.eval()

# Test prompt
prompt = """### Instruction:
Diagnose the issue

### Input:
Fan not working

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(output[0], skip_special_tokens=True))