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
spacy_en_sm = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])

def spacy_lemmatize_doc(doc, nlp, stopwords):
    """
    Run spacy's lemmatizer on a document and keep only alpha-numeric terms of
    length 2 or greater AND not in stopwords.
    """
    import numpy as np
    
    # not using parser or NER, so should be safe to increase character limit
    nlp.max_length = 1000000000
    
    tokens = nlp(doc)
    lemmas = [token.lemma_ for token in tokens]
    
    alnum_ind = np.where([lemma.isalnum() for lemma in lemmas])[0].tolist()
    alnum_lemmas = [lemmas[i] for i in alnum_ind]
    
    not_stop_ind = np.where([len(lemma) > 2 and lemma not in stopwords for lemma in alnum_lemmas])[0].tolist()
    final_tokens = [alnum_lemmas[i] for i in not_stop_ind]
    
    out = " ".join(final_tokens)
    return out


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

stopwords = set(nltk.corpus.stopwords.words('english')).union(
            set(["the", "section", "page"])
            )

# Spot checks...
spacy_lemmatize_doc(bills_train['text'].to_list()[12],
                    nlp = spacy_en_sm, stopwords = stopwords)

spacy_lemmatize_doc(bills_train['text'].to_list()[682],
                    nlp = spacy_en_sm, stopwords = stopwords)

# Training corpus
bill_corpus = tp.utils.Corpus(tokenizer = None,
                              stopwords = lambda x: len(x) <= 2 or x in stopwords)

test_corpus = copy.copy(bill_corpus)


bill_train_list = [(spacy_lemmatize_doc(text, nlp = spacy_en_sm, stopwords = stopwords), 
                    crs_labels[billid], 
                    {'labels':  crs_labels[billid]})
                   for text, billid in zip(bills_train['text'], bills_train['id'])
                   ]

bill_corpus.process(bill_train_list)
bill_corpus.save("data/legislation/corpus_train.pickle")

# Test corpus
bill_test_list = [(spacy_lemmatize_doc(text, nlp = spacy_en_sm, stopwords = stopwords),
                    crs_labels[billid], 
                    {'labels':  crs_labels[billid]})
                  for text, billid in zip(bills_test['text'], bills_test['id'])
                  ]

test_corpus.process(bill_test_list)
test_corpus.save("data/legislation/corpus_test.pickle")
