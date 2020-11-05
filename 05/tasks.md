# ECE324 Assignment 5: Tasks

'Home directory' of sorts for this assignment.

## Logistics
* DUE: Thursday 9PM
* POINTS: 100 Total
* SUBMISSION:
    - `assignment5.pdf`: Answers to questions.
    - `assign5.ipynb`: Code for project
        - I will be splitting this up into sections for ease of use.
    - [ ] `model_baseline.pt`, `model_rnn.pt`, and `modelrnn.pt` (assumed: this is a typo, one is supposed to be `model_cnn.pt`)
* MODELS:
    - [ ] MLP.
    - [ ] CNN.
    - [ ] RNN.

# 1: Problem Definition

* Subjective vs. Objective
* Tokenization, embedding, etc.

# 2: Environment

- [x] Install `torchtext` (general NLP processing) from [here](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- [x] Install `SpaCy` (tokenizing) from [here](https://spacy.io/)

# 3: Preparing the Data

## 3.1: Create Subsets

* Data provided on Quercus.
* `data.tsv`: Tab-separated-value file with two columns: `text`, `label`.
    - `text` = string (including punctuation).
    - `label` = binary value {0,1} -- 0 = objective, 1 = subjective.

- [x] Write a jupyter notebook `split_data.ipynb` to split data into 3 files:
    - [x] `train.tsv` -- 64% of the total data.
    - [x] `validation.tsv` -- 16% of the total data.
    - [x] `test.tsv` -- 20% of the data.
- [x] Ensuring that each subset is balanced.
    - [x] Print out the number of each class in each file. 
    - [x] Provide numbers in report.
- [ ] Create `overfit.tsv` -- 50 samples, equal class representation.

## 3.2: Process Input Data

*Code described here is already in `main.py`*.

Copy/paste into header of notebook. Goal is to create `train_iter`, `val_iter`, and `test_iter` objects using the TSV's from the last step.

# 4: Baseline Model and Training

*Code from `models.py` for creating the baseline model*

## 4.1: Loading GloVe Vector using Embedding Layer

- [ ] Follow pre-baked instructions from PDF.

## 4.2: Baseline Model

- [ ] Follow pre-baked instructions from PDF, copy/paste.

## 4.3: Training the Baseline


