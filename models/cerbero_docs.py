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
    
    with open("eval_prova.txt", "w", encoding="utf-8") as output_file:
        for frase in frasi_da_file:

            prompt = """Ragiona passo dopo passo. 
            Ti sarà fornito un testo. 
            Il tuo compito è estrarre nel seguente testo legale italiano un trigger di evento, le entità coinvolte e i ruoli che le entità hanno nell'evento o nelle relazioni tra di loro. 
            Le entità che desidero estrarre sono:
            concetti legali: tutti i concetti legali menzionati nel testo;
            riferimenti: tutti gli articoli e le leggi menzionati nel testo;
            persone, aziende e organizzazioni;
            parole chiave di conoscenza comune rilevanti per l'argomento.
            Se una qualsiasi delle entità non è presente nel testo, il valore sarà null.
            I ruoli e le relazioni che desidero estrarre sono:
            agente: l'entità o le entità responsabili dell'esecuzione dell'azione legale;
            tema: l'entità o le entità a cui si riferisce l'azione legale;
            regolamenta: articoli o leggi e le entità a cui la legge si riferisce;
            Se uno qualsiasi dei ruoli non è presente nel testo, evita di scriverlo.
            Evita ripetizioni delle stesse entità estratte.\n
            Produci sempre l'output senza tradurre in inglese e in una lista con questa forma:
            trigger dell'evento: il verbo o il verbo normalizzato che esprime chiaramente l'occorrenza di un evento
            concetti legali: elenco dei concetti legali estratti separati da virgola
            riferimenti: elenco degli articoli e delle leggi estratti separati da virgola
            attori: elenco di persone, aziende e organizzazioni estratti separati da virgola
            parole chiave: elenco delle parole chiave di conoscenza comune rilevanti per l'argomento estratte separate da virgola.
            ruoli: elenco dei ruoli estratti prodotto sempre con la seguente forma: (nome del ruolo, argomento 1, argomento 2, ..., argomento n). Ecco alcuni esempi:
            (agente, il verbo o il verbo normalizzato che esprime chiaramente l'occorrenza di un evento, l'entità o le entità responsabili dell'esecuzione dell'azione legale)
            (tema, il verbo o il verbo normalizzato che esprime chiaramente l'occorrenza di un evento, l'entità o le entità a cui si riferisce l'azione legale)
            (regolamenta, l'articolo o la legge, le entità estratte nel testo che sono regolamentate da esso
            Dopo l'estrazione, convertire l'elenco di output nel formato JSON.
            Il testo è: """ + frase + """
[|Assistente|]"""


            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
            with torch.no_grad():
                input_ids = input_ids.to('cuda')
               
            output_ids = model.generate(input_ids, max_new_tokens=1024)

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            output_file.write(f"Risultato per la frase:\n{frase}\n\n{generated_text}\n\n{'='*50}\n")


    end_time = time.time()  # Registra il tempo di fine
    elapsed_time = end_time - start_time  # Calcola il tempo trascorso con caricamento LLM
    elapsed_time_2 = end_time - start_time_2  # Calcola il tempo trascorso di sola elaborazione
    
    print(f"Tempo totale impiegato: {elapsed_time} secondi")
    print(f"Tempo sola estrazione impiegato: {elapsed_time_2} secondi")
    

if __name__ == "__main__":
    main()