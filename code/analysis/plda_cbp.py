"""
Fit PLDA and SLDA models with topic labels from the Congressional Bills Project
"""

import json
import tomotopy as tp
import itertools as it
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import wandb

from code.pyfuncs import *

run = wandb.init(project = "billtag-cbp")


"""
Log hyperparameters
"""
wandb.config.iters = int(1e3)

wandb.config.tw = "IDF"

wandb.config.min_cf = 0
wandb.config.min_df = 100

wandb.config.latent_topics = 1
wandb.config.topics_per_label = 1

wandb.config.alpha = 0.1
wandb.config.eta = 0.01


"""
Corpora and CBP labels
"""

corpus_train = tp.utils.Corpus().load("data/legislation/corpus_train_cbp.pickle")

# Read in test corpus and DF of topic labels
corpus_test_cbp = tp.utils.Corpus().load("data/legislation/corpus_test_cbp.pickle")
bills_test_cbp = pq.read_pandas("data/legislation/splits/bills_test_cbp.parquet").to_pandas()


"""
Fit PLDA model
"""

plda_cbp = tp.PLDAModel(corpus = corpus_train,
                        tw = tp.TermWeight.IDF, 
                        min_cf = wandb.config.min_cf, 
                        min_df = wandb.config.min_df,
                        latent_topics = wandb.config.latent_topics, 
                        topics_per_label = wandb.config.topics_per_label,
                        alpha = wandb.config.alpha,
                        eta = wandb.config.eta,
                        seed = 575)

plda_cbp.train(iter = wandb.config.iters)

plda_cbp.save("models/plda_cbp.pickle")


"""
Construct test corpus
"""

# Repeat for test corpus, but don't pass labels to keep things clean
corpus_test = tp.utils.Corpus(tokenizer = None,
                              stopwords = lambda x: len(x) <= 2 or x in stopwords)

test_eta = plda_cbp.get_count_by_topics()/len(corpus_train)

[corpus_test.add_doc(
    words = spacy_lemmatize_doc(text, nlp = spacy_en_sm, stopwords = stopwords),
    billid = billid,
    # Hide label in some keyword argument other than `labels`, so
    # it'll be accessible but impossible for tp models to find
    hidden_labels = cbp_labels[billid],
    eta = test_eta,
    ) for text, billid in zip(bills_test_cbp['text'], bills_test_cbp['id'])
    ]


"""
Infer test-set topics from PLDA
"""

# plda_cbp = tp.PLDAModel.load("models/plda_cbp.pickle")
# plda_cbp.summary()    

# Inspect culture, which ends up as a catch-all for some reason
wandb.config.eta = test_eta

# Infer topics, store likelihoods separately
topic_df_plda, ll_plda = infer_labels(plda_cbp, corpus_test_cbp, idnm = "billid")

# Join with real labels
topic_df_plda = pd.merge(bills_test_cbp, topic_df_plda, how = "left", 
                         left_on = "id", right_on = "billid")


cross_tab = pd.crosstab(topic_df_plda['label'], topic_df_plda['topic_inferred'])

prop_correct = np.mean(topic_df_plda['label'] == topic_df_plda['topic_inferred'])

culture_inferred = np.mean(topic_df_plda['topic_inferred'] == "Culture")
latent_inferred = np.mean(topic_df_plda['topic_inferred'] == "Latent")

culture_actual = np.mean(topic_df_plda['label'] == "Culture")
latent_actual = np.mean(topic_df_plda['label'] == "Latent")

wandb.log(
    {
    "topic_df": topic_df_plda,
    "cross_tab": cross_tab,
    "prop_correct": prop_correct,
    "culture_inferred": culture_inferred,
    "latent_inferred": latent_inferred,
    "culture_actual": culture_actual,
    "latent_actual": latent_actual
    }
    )

wandb.log({"cbp.summary": plda_cbp.summary()})


# Top 10 words of each topic
labels = list(plda_cbp.topic_label_dict)
labels.append("Latent")

top_words_dict = dict()
for i in range(len(labels)): 
    top_words_dict[labels[i]] = pd.DataFrame(plda_cbp.get_topic_words(i),
                                             columns = ["word", "prob"]) 

# top_words = pd.concat(top_words_dict)

wandb.log(top_words_dict)


wandb.finish()
