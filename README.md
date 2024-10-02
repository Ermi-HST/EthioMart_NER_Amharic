# EthioMart Amharic Named Entity Recognition (NER) System

## Project Overview
The **EthioMart Amharic Named Entity Recognition (NER)** project aims to extract key business entities such as product names, prices, and locations from unstructured text data in Amharic language Telegram messages. With the increasing use of Telegram for e-commerce activities in Ethiopia, this project provides a centralized solution for processing and analyzing product-related information from multiple independent channels.

## Objectives
- **Real-time Data Extraction**: Develop a system that collects and processes messages from various Ethiopian-based e-commerce Telegram channels.
- **Entity Recognition**: Fine-tune a pre-trained language model to accurately identify and extract entities like product names, prices, and locations.
- **Interpretability**: Implement model interpretability techniques to ensure the predictions are transparent and understandable.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Model Fine-Tuning](#model-fine-tuning)
5. [Evaluation Metrics](#evaluation-metrics)
6. [License](#license)
7. [Contributing](#contributing)
8. [Acknowledgements](#acknowledgements)

## Installation
To run this project, you will need Python 3.6 or higher. Install the necessary libraries by executing the following command:

bash
pip install -r requirements.txt

## Usage
The project consists of several scripts for different stages of the pipeline:

Data Ingestion: Collect data from Telegram channels.

bash
python scripts/data_ingestion.py
Data Preprocessing: Clean and prepare the raw data for labeling.

bash
python scripts/preprocess.py
Automated Data Labeling: Generate labeled data in CoNLL format for NER.

bash
python scripts/auto_labeling.py
Model Fine-Tuning: Fine-tune a pre-trained model on the labeled dataset.

bash
python scripts/model_finetuning.py

## Project Structure
The project follows a structured organization to enhance readability and maintainability:

bash
/EthioMart_NER_Amharic
    /data
        /raw                  # Raw Telegram messages
        /processed             # Preprocessed text data
        /labels                # Labeled data for NER
    /models
        /fine-tuned            # Fine-tuned NER models
    /scripts
        data_ingestion.py      # Script for data collection from Telegram
        preprocess.py           # Script for preprocessing the data
        auto_labeling.py        # Script for automated data labeling
        model_finetuning.py     # Script for fine-tuning the NER model
    requirements.txt             # File listing required libraries
    README.md                    # Project overview and instructions

## Model Fine-Tuning
The fine-tuning process is performed using Hugging Face's transformers library. The project employs models like DistilBERT and XLM-Roberta, chosen for their performance in token classification tasks across multiple languages.

## Evaluation Metrics
The performance of the NER model is evaluated using key metrics such as:

F1-score
Precision
Recall
These metrics provide insights into the model's ability to correctly identify entities in the text.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

Contributing
Contributions are welcome! If you have suggestions for improvements or find issues, please submit a pull request or open an issue.

Acknowledgements
Hugging Face for providing pre-trained models and tools for NLP.
Telethon for enabling easy interaction with the Telegram API.
The open-source community for the various libraries and resources used in this project.
markdown
Copy code

### Notes on the README:
- **Sections**: It includes sections for installation, usage, project structure, model fine-tuning, evaluation metrics, license, contributing, and acknowledgments.
- **Code Blocks**: Usage examples are provided in code blocks for clarity.
- **Markdown Formatting**: The formatting uses markdown syntax for better readability on GitHub.

You can modify any sections as needed to fit your preferences or specific details about the project. Let me know if you need any changes or additional information!
