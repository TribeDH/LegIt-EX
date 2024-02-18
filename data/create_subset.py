import jsonlines
import random

def create_subsets(input_file, output_file1, output_file2):
    # Carica il dataset dal file JSONL
    dataset = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            dataset.append(obj)

    # Calcola la lunghezza del dataset
    dataset_length = len(dataset)

    # Mischiare il dataset
    random.shuffle(dataset)

    # Calcola le dimensioni dei subset
    subset1_length = int(0.66 * dataset_length)
    subset2_length = int(0.5 * subset1_length)

    # Crea il primo subset del 66% della dimensione totale
    subset1 = dataset[:subset1_length]

    # Crea il secondo subset del 50% della dimensione del primo subset
    subset2 = subset1[:subset2_length]

    # Salva il primo subset come file JSONL
    with jsonlines.open(output_file1, 'w') as writer:
        writer.write_all(subset1)

    # Salva il secondo subset come file JSONL
    with jsonlines.open(output_file2, 'w') as writer:
        writer.write_all(subset2)

    return subset1, subset2

# Esempio di utilizzo
input_file = 'final_dataset.jsonl'
output_file1 = 'subset66.jsonl'
output_file2 = 'subset33.jsonl'
subset1, subset2 = create_subsets(input_file, output_file1, output_file2)

print("Dimensione del primo subset:", len(subset1))
print("Dimensione del secondo subset:", len(subset2))
