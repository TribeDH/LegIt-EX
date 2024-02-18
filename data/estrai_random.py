import pandas as pd
import random

df=pd.read_csv("/home/mathias.berti/work/Dati/df_def.csv", encoding='utf-8')

# Estrai casualmente dieci righe dalla colonna 'testo'
righe_casuali = random.sample(df['testo'].tolist(), 15)

with open('docs_sents.txt', 'a', encoding='utf-8') as file:
    for riga in righe_casuali:
        file.write(riga + '\n\n')
