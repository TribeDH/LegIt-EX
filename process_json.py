import json
import regex

# leggo il file con i risultati (testi e json mischiati)
with open("random_dataset.txt", "r") as f:
    test_str = f.read()

# con una regex individuo i json presenti nel testo
matches = regex.findall(r"\{(?:[^{}]|(?R))*\}", test_str)
print(type(matches))

# mi preparo a scrivere il file di output
with open("output.txt", "w") as f:

    # per ogni json individuato...
    for match in matches:
        print(match)
        # interpreto il json ottenendo un dizionario Python
        json_dict = json.loads(match)
        # codifico il json in stringa (unica riga)
        json_str = json.dumps(json_dict, ensure_ascii=False)
        # devo sostituire i doppi apici (") con la versione escape (\")
        json_str = json_str.replace('"', r'\"')
        # racchiudo la stringa tra doppi apici (pronta per essere copia e incollata)
        # scrivo su file e vado a capo
        f.write(f'"{json_str}"\n')