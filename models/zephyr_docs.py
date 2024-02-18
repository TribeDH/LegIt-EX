import argparse
import time

import torch
from transformers import pipeline


def leggi_frasi_da_file(nome_file):
    with open(nome_file, 'r', encoding='utf-8') as file:
        frasi = [line.strip() for line in file.readlines() if line.strip()]
    return frasi

def main():
    
    start_time = time.time()  # Registra il tempo di inizio

    parser = argparse.ArgumentParser(description='Estrazione di entità legali da frasi.')
    parser.add_argument('file_txt', type=str, help='Il percorso del file txt contenente le frasi.')

    args = parser.parse_args()

    frasi_da_file = leggi_frasi_da_file(args.file_txt)


    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16,                     device_map="auto")
    
    start_time_2 = time.time()

    # Aggiungi un blocco per scrivere i risultati su un file
    with open("eval_prova.txt", "w", encoding="utf-8") as output_file:
        for frase in frasi_da_file:
            messages = [
                {
                    "role": "system",
                    "content": "You are a friendly chatbot who always responds",
                },
                {"role": "user", "content": 
                 "Lets reason step by step. "
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
                 "The text is: " + frase},
            ]
            
            prompt_legal = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            outputs = pipe(prompt_legal, max_new_tokens=1024, do_sample=True, temperature=0.01, top_k=50, top_p=0.95)
            
            risultato = outputs[0]["generated_text"]
            output_file.write(f"Risultato per la frase:\n{frase}\n\n{risultato}\n\n{'='*50}\n\n")
            
            
    end_time = time.time()  # Registra il tempo di fine
    elapsed_time = end_time - start_time  # Calcola il tempo trascorso con caricamento LLM
    elapsed_time_2 = end_time - start_time_2  # Calcola il tempo trascorso di sola elaborazione
    
    print(f"Tempo totale impiegato: {elapsed_time} secondi")
    print(f"Tempo sola estrazione impiegato: {elapsed_time_2} secondi")
    

if __name__ == "__main__":
    main()
