from datasets import load_dataset, Dataset
import pandas as pd
import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import evaluate
import numpy as np
import transformers

# Initialize argparse
parser = argparse.ArgumentParser(description='Configure training parameters.')

# Add arguments for training configuration
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_train_epochs', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer')
parser.add_argument('--model_checkpoint', type=str, default="VietAI/vit5-base", help='Model checkpoint to use')

# Parse arguments
args = parser.parse_args()

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, 'positive': 2}


# Assign variables from args
batch_size = args.batch_size
num_train_epochs = args.num_train_epochs
learning_rate = args.learning_rate
model_checkpoint = args.model_checkpoint

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=3, id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]


train_df = pd.read_excel('train.xlsx')#pd.concat([df, df_dev]).reset_index(drop=True)
train_dataset = Dataset.from_pandas(train_df)

testset =  pd.read_excel('test.xlsx')
test_with_asr = pd.read_excel('test_asr.xlsx')
testset['text'] = test_with_asr['asr']

print(train_df['label'].unique())
print(testset['label'].unique())
test_dataset = Dataset.from_pandas(testset[['text', 'label']])




def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_dataset_train = train_dataset.map(preprocess_function, batched=True)
tokenized_dataset_test = test_dataset.map(preprocess_function, batched=True)


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# Load the individual metrics
import evaluate

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    neg,neu,pos = f1.compute(predictions=predictions, references=labels, average=None)['f1']

    # Compute each metric as needed
    metrics_result = {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        "macro_f1": f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
        "f1_neg": neg,
        "f1_neu": neu,
        "f1_pos": pos

    }
    
    return metrics_result

# This modified function should now work without the TypeError


## Train


training_args = TrainingArguments(
    output_dir=f"results/{model_name}",
    lr_scheduler_type='cosine',
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=2,
    bf16=True,
    warmup_ratio=0.05,
    metric_for_best_model='eval_macro_f1',
#     push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=3)]

)
print('model_checkpoint', model_checkpoint)
trainer.train()
trainer.save_model()
trainer.evaluate()