from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import KFold
import jsonlines

import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

from trl import SFTTrainer


# Load dataset from JSONL file
data_list = []
with jsonlines.open('LegI-EX/data/ft_dataset.jsonl') as reader:
    for line in reader:
        data_list.append(line)

# Create DataFrame
df = pd.DataFrame(data_list)
data = Dataset.from_pandas(df)
data_dict = DatasetDict({"train": data})

# Define function to format data
def formatting_func(example):
    if example.get("context", "") != "":
        input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Input: \n"
        f"{example['context']}\n\n"
        f"### Response: \n"
        f"{example['response']}")

    else:
        input_prompt = (f"Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}")

    return {"text" : input_prompt}


model_id = "HuggingFaceH4/zephyr-7b-alpha"

qlora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda:0"

# Perform K-Fold Cross Validation
kfold = KFold(n_splits=3, shuffle=True, random_state=41)
for fold, (train_indices, test_indices) in enumerate(kfold.split(data_dict["train"])):
    train_data = data_dict["train"].select(train_indices)
    test_data = data_dict["train"].select(test_indices)
    train_data = train_data.map(formatting_func)
    test_data = test_data.map(formatting_func)

    # Fine-tune model
    supervised_finetuning_trainer = SFTTrainer(
        base_model,
        train_dataset=train_data,
        eval_dataset=test_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_steps=2500,
            output_dir=f"./checkpoints/prova66-{fold}",
            optim="paged_adamw_8bit",
            fp16=True,
        ),
        tokenizer=tokenizer,
        peft_config=qlora_config,
        dataset_text_field="text",
        max_seq_length=1024
    )

    supervised_finetuning_trainer.train()
    
    base_model.config.to_json_file(f"adapter_config-{fold}.json")
    base_model.push_to_hub(f"sberti/prova66-CV-{fold}", private=True)
    tokenizer.push_to_hub(f"sberti/prova66-CV-{fold}")
