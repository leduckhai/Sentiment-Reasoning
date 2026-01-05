import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import random
from datasets import Dataset, load_dataset
import transformers
from transformers import TrainingArguments
import evaluate
import argparse
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

# Initialize argparse
parser = argparse.ArgumentParser(description='Configure training parameters.')

# Add arguments for training configuration
parser.add_argument('--model_name_or_path', type=str, default='vtrungnhan9/vmlu-llm', help='Model name or path')
parser.add_argument('--rationale_col', type=str, default='human_justification_en', help='Column for rationale (human_justification, human_justification_en, or empty string)')
parser.add_argument('--text_col', type=str, default='text_en', help='Column for text (text or text_en)')

# Parse arguments
args = parser.parse_args()
model_name_or_path = args.model_name_or_path
rationale_col = args.rationale_col
text_col = args.text_col

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    cache_dir='./models',
    load_in_8bit=True,
)

model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

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


def template(inp, out, rationale=''):
    conversation = [
        {"role": "user", "content": f"""sentiment analysis: '{inp.strip()}'"""},
    ]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    rationale_text = rationale.strip() if rationale else ''
    prompt = (prompt + str(out).strip() + '\n' + rationale_text).strip()
    return prompt


def add_train_text(example):
    inp = example[text_col] if example[text_col] else example['text']
    out = example['label']
    
    if rationale_col and rationale_col in example and example[rationale_col]:
        rationale = example[rationale_col]
    else:
        rationale = ''
    
    example['train_text'] = template(inp, out, rationale)
    return example


# Apply template to create train_text column
train_dataset = train_dataset.map(add_train_text)

# Print a sample to verify
print("\n--- Sample train_text ---")
print(train_dataset[0]['train_text'])
print("---")

# Load the individual metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.argmax(logits, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred if pred.isdigit() else -1 for pred in decoded_preds]
    decoded_labels = [label if label.isdigit() else -1 for label in decoded_labels]
    predictions = decoded_preds
    labels = decoded_labels
    neg, neu, pos = f1.compute(predictions=predictions, references=labels, average=None)['f1']
    metrics_result = {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        "macro_f1": f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
        "f1_neg": neg,
        "f1_neu": neu,
        "f1_pos": pos
    }
    
    return metrics_result


training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    report_to=[],
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    optim="adamw_bnb_8bit",
    bf16=True,
    output_dir=f"results/{model_name_or_path.split('/')[-1]}_{rationale_col}v2",
    logging_strategy="epoch",
    dataloader_num_workers=4,
    save_total_limit=3,
    save_strategy='epoch',
)

trainer = SFTTrainer(
    model,
    packing=True,
    max_seq_length=180,
    args=training_args,
    train_dataset=train_dataset.shuffle(),
    compute_metrics=compute_metrics,
    peft_config=peft_config,
    dataset_text_field='train_text',
)

trainer.train()
trainer.save_model()