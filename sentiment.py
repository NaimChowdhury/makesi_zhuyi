# The goal of this script will be to test some sentiment analysis tools.
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load pre-trained model and tokenizer for Chinese
MODEL_NAME = 'bert-base-chinese'
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Assuming binary sentiment classification
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Encode the text and obtain model predictions
text = "这部电影很好看"
inputs = tokenizer.encode_plus(text, return_tensors="pt", add_special_tokens=True, max_length=512, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Get model predictions
with torch.no_grad():
    logits = model(input_ids, attention_mask=attention_mask)[0]

# Calculate probabilities
probs = softmax(logits, dim=1)  # Shape: [batch_size, num_labels]

# Assuming label 0 is "negative" and label 1 is "positive"
prob_negative = probs[0][0].item()
prob_positive = probs[0][1].item()

print(f"Probability of negative sentiment: {prob_negative:.4f}")
print(f"Probability of positive sentiment: {prob_positive:.4f}")