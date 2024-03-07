# LegIt-EX
Note that due to copyright limitations, the original Maggioli
dataset can’t be shared, but a small number of sentences and documents
extracted from it are made available for replication of the experiment. Anyway,
with the appropriate modifications to the dataset path and data frame column
names, theoretically any dataset could be used in this process.

## Setup

Clone the repository from GitHub and install:
```
git clone https://github.com/TribeDH/LegIt-EX 
pip install -r requirements.txt
pip install -e ./
```

## Extraction

To run event extraction, different scripts for every model and document type
are provided due to schema variation in the models folder. The scripts take
the input sentences file as an external argument. Sample data from Maggioli
dataset is provided in data folder. For example, to run legal event extraction
with Zephyr-7b model:
```
cd models
python zephyr_docs.py LegItEX/data/sents_docs.txt
```
When using LegIt-EX model, remember to choose the name of the model
from https://huggingface.co/sberti.

## Fine-tuning

To fine-tune the model, two different scripts are provided: SFT.py for base
fine-tuning and CV-SFT.py for Cross Validation fine-tuning. In data folder is
possible to find the complete fine-tuning dataset and its subsets. Be careful
about updating the path to the chosen fine-tuning dataset.

#Evaluation

To perform evaluation of the models, just run event extraction for every
model in models folder specifying data/sents_eval.txt as input file and confront
the results with data/gold_pred.txt
