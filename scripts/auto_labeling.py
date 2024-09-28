import pandas as pd
import re

# Load the preprocessed data
df = pd.read_csv('data/processed/preprocessed_telegram_messages.csv')

# Example lists of known products and locations (you can extend these)
product_keywords = ['bottle', 'phone', 'laptop', 'shoe', 'shirt']
locations = ['Addis Ababa', 'Bole', 'Gonder', 'Mekelle', 'Hawassa']

# Helper function to identify entities in a sentence and create CoNLL-style labeling
def label_entities(text):
    labeled_data = []
    
    # Tokenize the text
    tokens = text.split()
    
    for token in tokens:
        label = 'O'  # Default label is "O" (outside any entity)
        
        # Label prices (numbers followed by "birr")
        if re.match(r'\d+', token) and 'birr' in text:
            label = 'B-PRICE'
        
        # Label known products
        elif token.lower() in product_keywords:
            label = 'B-Product'
        
        # Label known locations
        elif token in locations:
            label = 'B-LOC'
        
        labeled_data.append(f"{token}\t{label}")
    
    labeled_data.append("")  # Add a blank line between sentences/messages (CoNLL format)
    return labeled_data

# Create a CoNLL-style dataset
labeled_dataset = []
for text in df['cleaned_text']:
    labeled_dataset.extend(label_entities(text))

# Save the labeled dataset in CoNLL format
with open('data/labels/ner_labels.txt', 'w', encoding='utf-8') as f:
    for line in labeled_dataset:
        f.write(f"{line}\n")

print("NER labeled dataset saved to data/labels/ner_labels.txt")
