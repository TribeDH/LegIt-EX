import argparse
import time

from peft import get_peft_model
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import AutoTokenizer


def leggi_frasi_da_file(nome_file):
    with open(nome_file, 'r', encoding='utf-8') as file:
        frasi = [line.strip() for line in file.readlines() if line.strip()]
    return frasi


def main():
    
    def make_inference(instruction, context = None):
        if context:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: \n{instruction}\n\n### Response: \n"
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=1024)
        return (tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    
    models_list = ["provaCV-0"]

        
    for model_name in models_list:
        
        lora_config = LoraConfig.from_pretrained(f"sberti/{model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
    )
        
        tokenizer = AutoTokenizer.from_pretrained(f"sberti/{model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            lora_config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map={"":0}
        )

        model = get_peft_model(model, lora_config)
    
        start_time = time.time()  # Registra il tempo di inizio

        parser = argparse.ArgumentParser(description='Estrazione di entità legali da frasi.')
        parser.add_argument('file_txt', type=str, help='Il percorso del file txt contenente le frasi.')

        args = parser.parse_args()

        frasi_da_file = leggi_frasi_da_file(args.file_txt)
    
        start_time_2 = time.time()
    
        # Aggiungi un blocco per scrivere i risultati su un file
        with open(f"{model_name}_prova_results.txt", "w", encoding="utf-8") as output_file:
            for frase in frasi_da_file:    
    
                risultato = make_inference("Lets reason step by step. "
                     "You will be provided with a text. "
                     "Your task is to extract in the following italian legal text an event trigger, the entities involved, and the roles that entities have in the event or the relationships between them. "
                     "The entities I want to extract are:\n"
                     "legal concepts: all the legal concepts mentioned in the text;\n"
                     "references: every articles and laws mentioned in the text;\n"
                     "people, businesses and organizations;\n"
                     "common knowledge keywords relevant to the topic.\n"
                     "If any of the entities are not present in the text, the value will be the string none.\n"
                     "The roles and relationships I want to extract are:\n"
                     "agent: The entity or entities responsible for carrying out the legal action;\n"
                    "theme: The entity or entities which the legal action refers to;\n"
                     "regulates: articles or law and the entities which the law refers to;\n"
                     "If any of the roles are not present in the text, avoid writing it."
                     "Avoid repetitions of the same entities extracted.\n"
                     "Always produce the output without translating to english and in a list with this form:\n"
                     "event trigger: the verb or normalized verb that clearly expresses the occurrence of an event \n"
                     "legal concepts: list of the legal concepts extracted separated by a comma \n"
                     "references: list of the articles and laws extracted separated by a comma \n"
                     "actors: list of people, businesses and organizations extracted separated by a comma \n"
                     "keywords: list of the common knowledge keywords relevant to the topic extracted separated by a comma. \n"
                     "roles: list of the roles extracted produced always with the following form: (role name, argument 1, argument 2, ..., argument n). Here are some example cases:\n"
                     "(agent, the verb or normalized verb that clearly expresses the occurrence of an event, the entity or entities responsible for carrying out the legal action)\n"
                     "(theme, the verb or normalized verb that clearly expresses the occurrence of an event, the entity or entities which the legal action refers to)\n"
                     "(regulates, the legal article or law, the entities extracted in the text that are regulated by it\n"
                     "After the extraction, convert the output list to JSON format. "
                     "The text is:", frase)
        
                output_file.write(f"Risultato per la frase:\n{frase}\n\n{risultato}\n\n{'='*50}\n\n")
            
            
        end_time = time.time()  # Registra il tempo di fine
        elapsed_time = end_time - start_time  # Calcola il tempo trascorso con caricamento LLM
        elapsed_time_2 = end_time - start_time_2  # Calcola il tempo trascorso di sola elaborazione
    
        print(f"Tempo totale impiegato: {elapsed_time} secondi")
        print(f"Tempo sola estrazione impiegato: {elapsed_time_2} secondi")

    
    
    
if __name__ == "__main__":
    main()    