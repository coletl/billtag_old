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
!python -m spacy download 'en_core_web_sm'
## Not this one
# !python -m spacy download 'en_core_web_trf'

# Used for stopwords
import nltk
### DOWNLOAD
# nltk.download("stopwords")

random.seed(575)

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

def spacy_lemma_tokens(raw, user_data, en = "en_core_web_sm"):
    # Load English model, keeping only tagger component needed for lemmatizing
    spacy_en = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
    doc = spacy_en(raw)
    # Lemmatize and paste doc back together
    lemmas = [token.lemma_ for token in doc]
    # out = " ".join(lemmas)
    return(lemmas)


stopwords = set(nltk.corpus.stopwords.words('english'))

bill_corpus = tp.utils.Corpus(tokenizer = tp.utils.SimpleTokenizer(stemmer = spacy_lemma_tokens),
                              stopwords = lambda x: len(x) <= 2 or x in stopwords)

test_corpus = copy.copy(bill_corpus)

bill_train_list = [(text, crs_labels[billid], {'labels':  crs_labels[billid]})
                   for text, billid in zip(bills_train['text'], bills_train['id'])
                   ]

bill_corpus.process(bill_train_list)
bill_corpus.save("data/legislation/corpus_train.pickle")

bill_test_list = [(text, crs_labels[billid], {'labels':  crs_labels[billid]})
                   for text, billid in zip(bills_test['text'], bills_test['id'])
                   ]

test_corpus.process(bill_test_list)
test_corpus.save("data/legislation/corpus_test.pickle")

mod_plda = tp.PLDAModel(tw = tp.TermWeight.IDF, corpus = bill_corpus,
                        min_cf = 0, min_df = 100,
                        latent_topics = 1, topics_per_label = 1,
                        # hyperparameters for dirichlet
                        alpha = 0.1, eta = 0.01,
                        seed = 575)

mod_plda.train(iter = 100)
# mod_plda.train(iter = 5000)

mod_plda.summary()
mod_plda.topic_label_dict
mod_plda.get_topic_words(topic_id=0)

