import pandas as pd
import re

# Load the raw data
df = pd.read_csv('data/raw/telegram_messages.csv')

# Basic text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()
    return text

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Save preprocessed data
df.to_csv('data/processed/preprocessed_telegram_messages.csv', index=False)
print("Preprocessed data saved to data/processed/preprocessed_telegram_messages.csv")
