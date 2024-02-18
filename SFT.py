#DATA PREPARATION
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd


import jsonlines

# Lista per memorizzare i dizionari da ciascuna riga del JSONL
data_list = []

with jsonlines.open('/home/mathias.berti/work/LLM/data/ft_dataset.jsonl') as reader:
    for line in reader:
        data_list.append(line)

# Creazione di un DataFrame utilizzando pandas
df = pd.DataFrame(data_list)
data = Dataset.from_pandas(df)
data_dict = DatasetDict({"train": data})
#print(data)


#data = load_dataset('json', data_files='/home/mathias.berti/work/LLM/ft_dataset.jsonl')
#data_reduced = data["train"]

data_reduced = data_dict["train"]
data_split = data_reduced.train_test_split(test_size=0.1)

print(data_split)

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

formatted_data = data_split.map(formatting_func)




import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

model_id = "HuggingFaceH4/zephyr-7b-alpha"

qlora_config = LoraConfig(
    r=8,
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
    quantization_config=bnb_config,
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'right'

print(base_model)
print(base_model.config.quantization_config)





from trl import SFTTrainer

supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=formatted_data['train'],
    eval_dataset=formatted_data['test'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=2000,
        output_dir="./SFTZephyr-EE",
        optim="paged_adamw_8bit",
        fp16=True,
    ),
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="text",
    max_seq_length=20000
)

supervised_finetuning_trainer.train()



base_model.config.to_json_file("adapter_config.json")
base_model.push_to_hub("sberti/prova", private=True)
tokenizer.push_to_hub("sberti/prova")