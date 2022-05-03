"""
Construct training- and test-set corpuses with CRS labels
"""
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
    out = [alnum_lemmas[i] for i in not_stop_ind]
    
    return out


random.seed(575)


crs_labels = json.load(open('data/topic_labels/crs_labels.json'))

bill_fn = 'data/legislation/govinfo/bills_train.parquet' 

bills_train = pyarrow.Table.to_pandas(pq.read_pandas(bill_fn))

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


corpus_train.save("data/legislation/corpus_train.pickle")

# Test corpus
corpus_test_list = [spacy_lemmatize_doc(text, nlp = spacy_en_sm, stopwords = stopwords)
                    for text in bills_test['text']]
with open("data/legislation/corpus_test_list.json", "w") as outfile:
    json.dump(corpus_test_list, f, ensure_ascii = False, indent = 4)