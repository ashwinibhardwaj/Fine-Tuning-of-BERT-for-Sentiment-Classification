import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Preprocessing
def preprocess(dataset):
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset

dataset = preprocess(tokenized_dataset)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Training config
training_args = TrainingArguments(
    output_dir="./finetuned_bert",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=2).select(range(5000)),
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
trainer.evaluate()

# Save model
trainer.save_model("./finetuned_bert")
tokenizer.save_pretrained("./finetuned_bert")