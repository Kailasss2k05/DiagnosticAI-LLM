import json

input_file = "electric.txt"
output_file = "train.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

samples = content.split("### Instruction:")

data = []

for s in samples:
    if "### Input:" in s and "### Response:" in s:
        try:
            instruction = "Answer the electrician query"

            input_text = s.split("### Input:")[1].split("### Response:")[0].strip()
            output_text = s.split("### Response:")[1].strip()

            data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
        except:
            continue

with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

print("✅ JSONL created:", output_file)