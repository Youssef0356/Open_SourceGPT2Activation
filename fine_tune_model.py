import json
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize the dataset
data_path = 'mall_data.json'  # Change to your dataset path
with open(data_path, "r") as f:
    dataset = json.load(f)

# Tokenize the questions and answers
tokenized_data = []
max_length = 128  # Maximum length for padding

for item in dataset:
    question = item['input']
    answer = item['output']
    tokenized_data.append(tokenizer(question + " " + answer, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt"))

# Create a PyTorch Dataset and DataLoader
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        # Ensure all tensors have the same size by padding them to max_length
        self.input_ids = torch.stack([item['input_ids'].squeeze(0) for item in tokenized_data])
        self.attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in tokenized_data])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

dataset = TextDataset(tokenized_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Enable GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
model.train()
num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Fine-tuning complete and model saved.")
