from transformers import pipeline

# Load from Hugging Face OR local
classifier = pipeline(
    "sentiment-analysis",
    model="ashwini10521/finetuned_bert"
)

# Example predictions
texts = [
    "This movie was absolutely amazing!",
    "Worst movie I have ever seen"
]

for text in texts:
    result = classifier(text)
    print(f"\nText: {text}")
    print("Prediction:", result)