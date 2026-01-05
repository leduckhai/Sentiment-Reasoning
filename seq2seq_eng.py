import pandas as pd
import numpy as np
import random
from datasets import Dataset, load_metric
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import pandas as pd
import evaluate
import torch
import nltk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import argparse
import numpy as np


# Initialize argparse
parser = argparse.ArgumentParser(description='Configure training parameters.')

# Add arguments for training configuration
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_train_epochs', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer')
parser.add_argument('--model_checkpoint', type=str, default="luqh/ClinicalT5-base", help='Model checkpoint to use')

# Parse arguments
args = parser.parse_args()

# Assign variables from args
batch_size = args.batch_size
num_train_epochs = args.num_train_epochs
learning_rate = args.learning_rate
model_checkpoint = args.model_checkpoint

# Now you can use these variables in your training setup
print(f"Training setup:")
print(f"Batch size: {batch_size}")
print(f"Number of training epochs: {num_train_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Model checkpoint: {model_checkpoint}")

id2label = {'0': "negative", '1': "neutral", '2': "positive"}
label2id = {"negative": '0', "neutral": '1', 'positive': '2'}

# Load the Sentiment-Reasoning dataset from Hugging Face
ds = load_dataset("leduckhai/Sentiment-Reasoning")

# Get train and test splits
train_dataset = ds['train']
test_dataset = ds['test']

# Convert label to string
train_dataset = train_dataset.map(lambda x: {'label': str(x['label'])})
test_dataset = test_dataset.map(lambda x: {'label': str(x['label'])})

print("Train labels:", set(train_dataset['label']))
print("Test labels:", set(test_dataset['label']))
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")


# Output unique values to verify

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
id2label = {'0': "Negative", '1': "Neutral", '2': "Positive"}
label2id = {"Negative": '0', "Neutral": '1', 'Positive': '2'}
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=examples["label"], max_length=8, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

print('tokenized_train_dataset', tokenized_train_dataset)
print('tokenized_test_dataset', tokenized_test_dataset)

print(train_dataset['text'])
# Load the individual metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred if pred.isdigit() else -1 for pred in decoded_preds]  # Replace non-digit predictions with '-1'
    decoded_labels = [label if label.isdigit() else -1 for label in decoded_labels]  # Replace non-digit labels with '-1'
    predictions = decoded_preds
    labels = decoded_labels
    metrics_result = {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)['accuracy'],

    }
    
    return metrics_result

# This modified function should now work without the TypeError


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

tokenized_train_dataset=tokenized_train_dataset.remove_columns(['text', 'label'])
tokenized_test_dataset=tokenized_test_dataset.remove_columns(['text', 'label'])
model_name = model_checkpoint.split("/")[-1]

transformers.logging.set_verbosity_info()
training_args = Seq2SeqTrainingArguments(
    output_dir=f"results/{model_name}",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy='epoch',
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    bf16=True,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
)

# Setting up the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=3)]

)


trainer.train()
trainer.save_state()
trainer.save_model()
