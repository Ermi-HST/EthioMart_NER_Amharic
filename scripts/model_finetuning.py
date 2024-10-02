import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from evaluate import load  # Updated import for metrics
import shap

# Load the labeled dataset (CoNLL format)
data_files = {"train": "data/labels/ner_labels.txt"}
dataset = load_dataset("text", data_files=data_files, split='train')

# Tokenizer and model
model_name = "distilbert-base-multilingual-cased"  # You can use "bert-tiny-amharic" if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=4)  # Adjust `num_labels` based on your entities

# Tokenization function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['text'],
        padding='max_length',   # Ensure padding to max length
        truncation=True,        # Truncate sequences longer than max_length
        max_length=512,         # DistilBERT max length is 512
        return_tensors='pt'     # Return PyTorch tensors
    )
    return tokenized_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate after every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
# trainer.train()

# Save the fine-tuned model
trainer.save_model("models/fine-tuned/")

# Define label map based on the entities in your dataset
label_list = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]  # Update with your actual labels
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# Function to align predictions with actual labels
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=-1)
    
    # Convert predictions and labels from indices to actual entity labels
    pred_labels = [[id2label[p] for p in pred] for pred in preds]
    true_labels = [[id2label[l] for l in label] for label in label_ids]
    
    return pred_labels, true_labels

# Load evaluation metrics
metric = load("seqeval")  # Updated to use the `evaluate` library

# Evaluate the fine-tuned model
predictions, label_ids, metrics = trainer.predict(tokenized_dataset)
preds, labels = align_predictions(predictions, label_ids)

# Compute evaluation metrics (F1-score, precision, recall)
results = metric.compute(predictions=preds, references=labels)
print("Evaluation Results:", results)

# Save evaluation metrics
with open("models/fine-tuned/evaluation_results.txt", "w") as f:
    f.write(str(results))

# SHAP interpretability
explainer = shap.Explainer(model, tokenized_dataset)

# Compute SHAP values for a few examples
shap_values = explainer(tokenized_dataset[:10])

# Plot SHAP values
shap.plots.text(shap_values)
