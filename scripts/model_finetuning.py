import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer

# Load the labeled dataset in CoNLL format
data_files = {"train": "data/labels/ner_labels.txt"}
datasets = load_dataset('text', data_files=data_files, split='train')

# Tokenizer and model (XLM-Roberta as an example)
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=4)  # Adjust num_labels as needed

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], padding=True, truncation=True, is_split_into_words=True)
    return tokenized_inputs

tokenized_dataset = datasets.map(tokenize_and_align_labels, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("models/fine-tuned/")
