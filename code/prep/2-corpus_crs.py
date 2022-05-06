"""
Prepare corpus of legislation with CRS labels
"""

import faulthandler
faulthandler.enable()

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

from code.pyfuncs import spacy_lemmatize_doc

# Load English model, keeping only tagger component needed for lemmatizing
spacy_en_sm = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])

random.seed(575)


crs_labels = json.load(open('data/topic_labels/crs_labels.json'))

bills_train = pyarrow.Table.to_pandas(
    pq.read_pandas('data/legislation/splits/bills_train_crs.parquet')
    )

bills_test = pyarrow.Table.to_pandas(
    pq.read_pandas('data/legislation/splits/bills_test_crs.parquet')
    )

"""
Build corpuses
"""

assert len(set(bills_train['id']) - set(crs_labels.keys())) == 0
assert len(set(bills_train['id']).intersection(bills_test['id'])) == 0

stopwords = set(nltk.corpus.stopwords.words('english')).union(
            set(["the", "section", "page"])
            )

# Spot checks...
spacy_lemmatize_doc(bills_train['text'].to_list()[12],
                    nlp = spacy_en_sm, stopwords = stopwords)

spacy_lemmatize_doc(bills_train['text'].to_list()[682],
                    nlp = spacy_en_sm, stopwords = stopwords)
                    

# Training corpus
corpus_train = tp.utils.Corpus(tokenizer = None,
                               stopwords = lambda x: len(x) <= 2 or x in stopwords)

[corpus_train.add_doc(
    words = spacy_lemmatize_doc(text, nlp = spacy_en_sm, stopwords = stopwords),
    labels = crs_labels[billid],
    billid = billid,
    ) for text, billid in zip(bills_train['text'], bills_train['id'])
    ]
    
# Find and concatenate ngrams
ngram_train = corpus_train.extract_ngrams(min_cf = 20, min_df = 20, max_cand = 1000000)
corpus_train.concat_ngrams(ngram_train)

corpus_train.save("data/legislation/corpus_train_crs.pickle")

# Repeat for test corpus, but don't pass labels to keep things clean
corpus_test = tp.utils.Corpus(tokenizer = None,
                              stopwords = lambda x: len(x) <= 2 or x in stopwords)

[corpus_test.add_doc(
    words = spacy_lemmatize_doc(text, nlp = spacy_en_sm, stopwords = stopwords),
    # Hide label in some keyword argument other than `labels`, so
    # it'll be accessible but impossible for tp models to find
    hidden_labels = cbp_labels[billid],
    billid = billid,
    ) for text, billid in zip(bills_test['text'], bills_test['id'])
    ]
    

corpus_test.save("data/legislation/corpus_test_crs.pickle")
