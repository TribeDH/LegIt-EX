import os

import nltk

import json
from tqdm import tqdm



from sentence_splitter import SentenceSplitter, split_text_into_sentences
splitter = SentenceSplitter(language='it', non_breaking_prefix_file='non_break_prefix.txt')

import html2text
converter = html2text.HTML2Text()


tokenizer = nltk.tokenize.WhitespaceTokenizer()


docs_path = '/home/mathias.berti/shared/documenti_originali/documenti_totale.json'

docs = []
for i,doc in tqdm(enumerate((open(docs_path, 'r')))):
    if i <=550000:
        docs.append(json.loads(doc))
    else:
        break
        
        
def txt_preprocessing(testo):
    
    testo = testo.replace('&nbsp;', ' ')
    testo = testo.replace('---|--', ' ')    
    testo = testo.replace('---|---', ' ')
    testo = converter.handle(testo)    
    
    parole = testo.split()  # Dividi il testo in parole
    parole_modificate = []
        
    parola_corrente = ''
    
    for parola in parole:
        if parola.endswith('-'):
            parola_corrente += parola[:-1]  # Aggiungi la parte senza '-'
        elif parola_corrente:
            parola_corrente += parola  # Aggiungi la parte successiva della parola
            parole_modificate.append(parola_corrente)
            parola_corrente = ''
        else:
            parole_modificate.append(parola)
            
    for parola in parole:
        
        if parola.endswith('\n'):  # Verifica se la parola termina con '\n'
            parola_modificata = parola[:-1] + ' '  # Rimuovi '\n' e aggiungi uno spazio
        
        else:
            parola_modificata = parola  # Mantieni la parola invariata
        parole_modificate.append(parola_modificata)

    
    testo_modificato = ' '.join(parole_modificate)  # Unisci le parole con spazi
     
    return testo_modificato



import random

n = 1  # Puoi impostare questo valore a qualsiasi numero desiderato
random_docs = random.sample(docs, n)



for n, i in tqdm(enumerate(range(len(random_docs)))):
                 
    if 'testo' in random_docs[i]:
            
        testo_processato = txt_preprocessing(random_docs[i]['testo'])
        
        with open('docs_interi.txt', 'a', encoding='utf-8') as file:        
            file.write(testo_processato + '\n\n')