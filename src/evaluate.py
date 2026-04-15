import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("./finetuned_bert")
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_bert")

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = dataset.map(tokenize, batched=True)

# Preprocess
def preprocess(dataset):
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset

dataset = preprocess(dataset)

# Trainer
trainer = Trainer(model=model)

# Predictions
preds = trainer.predict(dataset["test"])

y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.savefig("../results/confusion_matrix.png")
plt.show()

# Metrics
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

accuracy = accuracy_score(y_true, y_pred)
print("\nAccuracy:", accuracy)