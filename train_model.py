
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os

# Setup AMD environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

# Load model and tokenizer
model_name = "Qwen/Qwen3-VL-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)

# Load dataset
with open("./llama_factory_data/data/medalpaca.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare dataset
def format_example(example):
    text = f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
    return {"text": text}

formatted_data = [format_example(ex) for ex in data]
dataset = Dataset.from_list(formatted_data)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama_factory_data/medical_qwen3_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=3e-05,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=2,
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    max_grad_norm=1.0,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save model
trainer.save_model()
trainer.model.save_pretrained("./llama_factory_data/medical_qwen3_output")

print("Training completato!")
