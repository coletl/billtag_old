"""
Fit PLDA model with topic labels from the Congressional Research Service
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

run = wandb.init(project = "billtag-crs")


"""
Log hyperparameters
"""
wandb.config.iters = int(1e3)

wandb.config.tw = tp.TermWeight.IDF

wandb.config.min_cf = 0
wandb.config.min_df = 100

wandb.config.latent_topics = 1
wandb.config.topics_per_label = 1

wandb.config.alpha = 0.1
wandb.config.eta = 0.01

"""
Training corpus with CRS labels
"""

corpus_train = tp.utils.Corpus().load("data/legislation/corpus_train_crs.pickle")

"""
Fit PLDA model
"""
plda_crs = tp.PLDAModel(corpus = corpus_train,
                        tw = wandb.config.tw, 
                        min_cf = wandb.config.min_cf, 
                        min_df = wandb.config.min_df,
                        latent_topics = wandb.config.latent_topics, 
                        topics_per_label = wandb.config.topics_per_label,
                        alpha = wandb.config.alpha,
                        eta = wandb.config.eta,
                        seed = 575)

plda_crs.train(iter = wandb.config.iters)

plda_crs.save("models/plda_crs.pickle")
# plda_crs = tp.PLDAModel.load("models/plda_crs.pickle")


"""
Infer test-set topics from PLDA
"""

# plda_crs = tp.PLDAModel.load("models/plda_crs.pickle")
# plda_crs.summary()    

# Inspect culture, which ends up as a catch-all for some reason
plda_crs.get_topic_word_dist(20)


# Infer topics, store likelihoods separately
topic_df_plda, ll_plda = infer_labels(plda_crs, corpus_test_crs, idnm = "billid")


# Join with real labels
topic_df_plda = pd.merge(bills_test_crs, topic_df_plda, how = "left", 
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

plda_cbp.summary()

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
