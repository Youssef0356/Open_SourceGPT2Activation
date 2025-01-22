import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_text = input("write prompt:")
inputs = tokenizer(input_text, return_tensors="pt")
print("Tokenized Input:", inputs)

# Generate output with adjusted sampling parameters
model.eval()
with torch.no_grad():
    output = model.generate(
        inputs['input_ids'],  # Use input_ids from the tokenized input
        max_length=50,         # Limit output length to ensure short, direct answers
        num_return_sequences=1,  # Only one response at a time
        do_sample=True,        # Enable sampling for variability
        temperature=0.5,       # Lower temperature for more deterministic output
        top_k=30,              # Top-K sampling for diversity
        top_p=0.9,             # Nucleus sampling for more controlled diversity
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(inputs['input_ids']),  # Explicit attention mask
        eos_token_id=tokenizer.eos_token_id,  # Ensure EOS token ends the sequence
    )

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# Clean the response (remove any unnecessary [50256] or special tokens)
decoded_output = decoded_output.strip()

# Ensure the output is valid
if len(decoded_output) == 0 or decoded_output.startswith('[50256]'):
    print("The generated response is invalid. Try again with a different prompt.")
else:
    # Print the response in a direct format
    print(f"Me: {input_text}\nBot: {decoded_output}")
