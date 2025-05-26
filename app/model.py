from transformers import pipeline

# Load the sentiment-analysis pipeline (DistilBERT pretrained model)
sentiment_pipeline = pipeline("sentiment-analysis")