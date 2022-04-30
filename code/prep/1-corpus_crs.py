"""
Construct training- and test-set corpuses
"""


import json
import copy
import random
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import pandas as pd
import tomotopy as tp

# Used for lemmatization
import spacy
## Use this one
### DOWNLOAD
# !python -m spacy download 'en_core_web_sm'
## Not this one
# !python -m spacy download 'en_core_web_trf'

# Used for stopwords
import nltk
### DOWNLOAD
# nltk.download("stopwords")

# Load English model, keeping only tagger component needed for lemmatizing
spacy_en = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])

def spacy_lemmatize_word(raw, user_data = None, nlp = spacy_en):
    """
    Run spacy's lemmatizer and return the first token's lemma.
    Intended only as the stemmer for tomotopy.utils.SimpleTokenizer().
    """
    import regex as re
    assert re.search(" ", raw) == None
    word = nlp(raw)
    out = word[0].lemma_
    return(out)


random.seed(575)

"""
Split data into training, test sets
"""

crs_labels = json.load(open('data/topic_labels/crs_labels.json'))

bill_fn = 'data/legislation/govinfo/bill_text.parquet'
bills   = pyarrow.Table.to_pandas(pq.read_pandas(bill_fn))

n_train   = round(len(bills) / 2)
# index of 50/50 split
split_index = set(random.sample(range(1, len(bills)), n_train))
# index of unlabeled bills
unlabel_index = np.where([billid not in crs_labels.keys()
                          for billid in bills['id']])[0].tolist()

# remove unlabeled bills from training set
train_index = list(set(split_index) - set(unlabel_index))
bills_train = bills.iloc[train_index]
bills_test = bills.drop(train_index)

len(bills_train); len(bills_test)

assert(len(bills_train) + len(bills_test) == len(bills))

len(split_index) - len(train_index)


"""
Build corpuses
"""

stopwords = set(nltk.corpus.stopwords.words('english'))

# Training corpus
bill_corpus = tp.utils.Corpus(tokenizer = tp.utils.SimpleTokenizer(stemmer = spacy_lemmatize_word),
                              stopwords = lambda x: len(x) <= 2 or x in stopwords)

test_corpus = copy.copy(bill_corpus)

bill_train_list = [(text, crs_labels[billid], {'labels':  crs_labels[billid]})
                   for text, billid in zip(bills_train['text'], bills_train['id'])
                   ]

bill_corpus.process(bill_train_list)
bill_corpus.save("data/legislation/corpus_train.pickle")

# Test corpus
bill_test_list = [(text, crs_labels[billid], {'labels':  crs_labels[billid]})
                  for text, billid in zip(bills_test['text'], bills_test['id'])
                  ]

test_corpus.process(bill_test_list)
test_corpus.save("data/legislation/corpus_test.pickle")
