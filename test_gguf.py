from llama_cpp import Llama

llm = Llama(
    model_path="model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048
)

prompt = prompt = """### Instruction:
Diagnose the issue

### Input:
Wire burning smell

### Response:
"""

output = llm(
    prompt,
    max_tokens=100,
    temperature=0.2
)

print(output["choices"][0]["text"])