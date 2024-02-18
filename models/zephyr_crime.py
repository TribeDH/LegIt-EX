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
    with open("crime-risultati.txt", "w", encoding="utf-8") as output_file:
        for frase in frasi_da_file:
    
                        
            messages = [
                {
                    "role": "system",
                    "content": "You are a friendly chatbot who always responds",
                },
                {"role": "user", "content": 
                 "Lets reason step by step. "
                 "You will be provided with a text. "
                 "Your task is to extract in the following italian crime news text an event trigger, the entities involved, and the roles that entities have in the event or the relationships between them. "
                 "The entities I want to extract are:\n"
                 "author of the theft; "
                 "victim of the theft; "
                 "place where the theft occurred; "
                 "the object stolen in the theft; "
                 "temporal expressions; "
                 "organizations. "
                 "If any of the entities are not present in the text, the value will be the string none.\n"
                 "Always produce the output without translating to english and in a list with this form:\n"
                 "event trigger: the verb or normalized verb that clearly expresses the occurrence of a theft event \n"
                 "author of the theft: the person or group of people who stole an object "
                 "victim of the theft: the person or business who got the object stolen "
                 "location: place where the theft occurred; "
                 "object: the object or objects stolen in the theft; "
                 "temporal expressions; "
                 "organizations. "
                 "After the extraction, convert the output list to JSON format. "
                 "The text is: " + frase}
            ]

            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            outputs = pipe(prompt, max_new_tokens=2048, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)
            
            risultato = outputs[0]["generated_text"]
            output_file.write(f"Risultato per la frase:\n{frase}\n\n{risultato}\n\n{'='*50}\n")
            
            
    end_time = time.time()  # Registra il tempo di fine
    elapsed_time = end_time - start_time  # Calcola il tempo trascorso con caricamento LLM
    elapsed_time_2 = end_time - start_time_2  # Calcola il tempo trascorso di sola elaborazione
    
    print(f"Tempo totale impiegato: {elapsed_time} secondi")
    print(f"Tempo sola estrazione impiegato: {elapsed_time_2} secondi")
    

if __name__ == "__main__":
    main()
