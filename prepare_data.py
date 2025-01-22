from transformers import GPT2Tokenizer
import json
import os

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad_token to eos_token

# Load data from JSON file
with open("mall_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Combine all the input-output pairs from the JSON data
combined_data = []

# Process mall_info, miza_boumiel, and css_store input-output pairs into combined_data
for store_data in data:
    input_text = store_data.get("input")
    output_text = store_data.get("output")
    if input_text and output_text:  # Make sure both input and output exist
        # Format the input-output pair as: "Input: {input_text} Output: {output_text}"
        combined_data.append(f"Input: {input_text} Output: {output_text}")

# Tokenize the dataset
def tokenize_data(data, max_length=512):
    tokenized = []
    for text in data:
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        decoded_text = tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)
        tokenized.append(decoded_text)
    return tokenized

# Tokenize your dataset with a maximum length (you can adjust the max_length if needed)
tokenized_data = tokenize_data(combined_data, max_length=512)

# Save the tokenized dataset for later use
output_dir = './tokenized_data'
os.makedirs(output_dir, exist_ok=True)

# Save the tokenized data (just saving the decoded text)
with open(os.path.join(output_dir, "tokenized_data.txt"), "w", encoding="utf-8") as f:
    for item in tokenized_data:
        f.write(item + "\n")

print("Dataset tokenized and saved without tensor symbols.")
