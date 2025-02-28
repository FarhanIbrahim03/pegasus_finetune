import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load as load_metric  # Changed from datasets import load_metric
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load pre-trained model and tokenizer
model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Load dataset (example using CNN/DailyMail dataset)
# You can replace this with your own dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Define preprocessing function
def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlights"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=1024, 
        padding="max_length", 
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=128, 
            padding="max_length", 
            truncation=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Preprocess the dataset
processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Define evaluation metrics
rouge_metric = load_metric("rouge")  # Using evaluate.load instead of datasets.load_metric

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Replace -100 with the pad token id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 with the pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split()) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = rouge_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./pegasus-finetuned-summarizer",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Use mixed precision training if you have compatible GPU
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
)

# Initialize data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if training_args.fp16 else None
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the model
model_path = "./pegasus-finetuned-summarizer-final"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Optional: Evaluate on test set
test_results = trainer.evaluate(processed_datasets["test"])
print(f"Test results: {test_results}")

# Example of how to use the fine-tuned model for inference
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs.input_ids, 
        num_beams=4,
        min_length=30,
        max_length=128,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Test with a sample text
sample_text = """
Your long text to summarize goes here...
"""
print(generate_summary(sample_text))