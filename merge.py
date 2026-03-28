from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load base
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load LoRA
model = PeftModel.from_pretrained(model, "lora_adapter")

# Merge
model = model.merge_and_unload()

# Save
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")