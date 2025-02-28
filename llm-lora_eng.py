import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from trl import AutoModelForCausalLMWithValueHead
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
import evaluate

# model_name_or_path = "vtrungnhan9/vmlu-llm"
# rationale_col = 'cot_rationale'

# Initialize argparse
parser = argparse.ArgumentParser(description='Configure training parameters.')

# Add arguments for training configuration
parser.add_argument('--model_name_or_path', type=str, default='vtrungnhan9/vmlu-llm', help='vtrungnhan9/vmlu-llm')
parser.add_argument('--rationale_col', type=str, default='human_justification', help='cot_rationale')
# parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
# parser.add_argument('--model_checkpoint', type=str, default="VietAI/vit5-base", help='Model checkpoint to use')

# Parse arguments
args = parser.parse_args()
model_name_or_path = args.model_name_or_path
rationale_col = args.rationale_col

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token='hf_GxsYTZDZhHcQEzYEvWrus')
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
#     load_in_8bit=True,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
#     use_cache=True,
    cache_dir='./models',
    load_in_8bit=True,
    token='hf_GxsYTZDZhHcQEzYEvWrus'

)

model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


train_df = pd.read_excel('train_eng.xlsx')#pd.concat([df, df_dev]).reset_index(drop=True)
train_df['label'] = train_df['label'].astype(str)

train_dataset = Dataset.from_pandas(train_df)

testset =  pd.read_excel('test_eng.xlsx')

testset['label'] = testset['label'].astype(str)
print(train_df['label'].unique())
print(testset['label'].unique())
test_dataset = Dataset.from_pandas(testset[['text', 'label']])


def template(inp, out, rationale=''):
#     if rationale 
    conversation = [
        {"role": "user", "content": f"""sentiment analysis: '{inp.strip()}'"""},
    ]
#     print(out)
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    prompt = (prompt +str(out).strip()+'\n'+rationale.strip()).strip()
    print(prompt)
    return prompt
# , train_dataset[rationale_col]
# reformatted_output = [reformat(inp, out) for inp, out in zip(dataset['train']['words'], dataset['train']['tags'])]
if rationale_col == '':
    new_column_train = [template(inp, out) for inp, out in zip(train_dataset['text'], train_dataset['label'])]
else:
    new_column_train = [template(inp, out, rationale) for inp, out, rationale in zip(train_dataset['text'], train_dataset['label'], train_dataset[rationale_col])]
train_dataset= train_dataset.add_column("train_text", new_column_train)
# new_column_train = [template(inp, out) for inp, out in zip(test_dataset['text'], test_dataset['label'])]
# test_dataset= test_dataset.add_column("train_text", new_column_train)


# Load the individual metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.argmax(logits, axis=-1)
    print(logits, labels)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
    print('decoded_preds', decoded_preds)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred if pred.isdigit() else -1 for pred in decoded_preds]  # Replace non-digit predictions with '-1'
    decoded_labels = [label if label.isdigit() else -1 for label in decoded_labels]  # Replace non-digit labels with '-1'
    predictions = decoded_preds
    labels = decoded_labels
    print( f1.compute(predictions=predictions, references=labels, average=None)['f1'])
    print(set(decoded_preds))
    neg,neu,pos = f1.compute(predictions=predictions, references=labels, average=None)['f1']
    metrics_result = {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        "macro_f1": f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
#         "macro_precision": precision.compute(predictions=predictions, references=labels, average='macro')['precision'],
#         "macro_recall": recall.compute(predictions=predictions, references=labels, average='macro')['recall'],
        "f1_neg": neg,
        "f1_neu": neu,
        "f1_pos": pos

    }
    
    return metrics_result

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
#     gradient_checkpointing=True,
    warmup_steps=100,
    report_to=[],
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    optim="adamw_bnb_8bit",
    bf16=True,
    # gradient_accumulation_steps=2, # simulate larger batch sizes
    output_dir=f"results/{model_name_or_path.split('/')[-1]}_{rationale_col}v2",
    logging_strategy="epoch",
    dataloader_num_workers=4,
    save_total_limit=3,
    save_strategy='epoch',
#     eval_strategy='no',
)


trainer = SFTTrainer(
    model,
    packing=True, # pack samples together for efficient training
    max_seq_length=180, # maximum packed length
    args=training_args,
    train_dataset=train_dataset.shuffle(),
    compute_metrics=compute_metrics,
    peft_config=peft_config,
    dataset_text_field='train_text',
#     callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=3)]

)
trainer.train()
trainer.save_model()

# trainer.evaluate()


