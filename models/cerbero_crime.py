import argparse
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


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

    model = AutoModelForCausalLM.from_pretrained("galatolo/cerbero-7b",                                                                                     bnb_4bit_compute_dtype=torch.float16, 
                                                  device_map="auto", 
                                                  load_in_4bit=True
)
    
    tokenizer = AutoTokenizer.from_pretrained("galatolo/cerbero-7b")

    start_time_2 = time.time()
    
    with open("cerbero-crime-risultati.txt", "w", encoding="utf-8") as output_file:
        for frase in frasi_da_file:
            
            prompt = """Questa è una conversazione tra un umano ed un assistente AI.
[|Umano|] Estrai tutte le entità coinvolte nella notizia di furto che segue. Le entità che voglio estrarre sono: autore del furto; vittima del furto; luogo in cui il furto è avvenuto; oggetto rubato nel furto; espressioni temporali; organizzazioni. Se una o più di queste entità non sono presenti nel testo, non estrarre nulla. Il testo è:
""" + frase + """
[|Assistente|]"""

            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
            with torch.no_grad():
                input_ids = input_ids.to('cuda')
               
            output_ids = model.generate(input_ids, max_new_tokens=256)

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            output_file.write(f"Risultato per la frase:\n{frase}\n\n{generated_text}\n\n{'='*50}\n")


    end_time = time.time()  # Registra il tempo di fine
    elapsed_time = end_time - start_time  # Calcola il tempo trascorso con caricamento LLM
    elapsed_time_2 = end_time - start_time_2  # Calcola il tempo trascorso di sola elaborazione
    
    print(f"Tempo totale impiegato: {elapsed_time} secondi")
    print(f"Tempo sola estrazione impiegato: {elapsed_time_2} secondi")
    

if __name__ == "__main__":
    main()
