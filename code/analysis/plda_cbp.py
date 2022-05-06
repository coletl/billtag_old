"""
Fit PLDA and SLDA models with topic labels from the Congressional Bills Project
"""

import json
import tomotopy as tp
import itertools as it

from code.pyfuncs import *

iters = int(1e4)

corpus_train = tp.utils.Corpus().load("data/legislation/corpus_train_cbp.pickle")

"""
Fit PLDA model
"""
plda_cbp = tp.PLDAModel(tw = tp.TermWeight.IDF, corpus = corpus_train,
                        min_cf = 0, min_df = 100,
                        latent_topics = 1, topics_per_label = 1,
                        # hyperparameters for dirichlet
                        alpha = 0.1, eta = 0.01,
                        seed = 575)

plda_cbp.train(iter = iters)

plda_cbp.save("models/plda_cbp.pickle")
# plda_cbp = tp.PLDAModel.load("models/plda_cbp.pickle")

"""
Fit SLDA model
"""

train_labels = set(it.chain(*[bill.labels for bill in corpus_train]))

slda_cbp = tp.SLDAModel(tw = tp.TermWeight.IDF, corpus = corpus_train,
                        min_cf = 0, min_df = 100,
                        k = len(train_labels),
                        # hyperparameters for dirichlet
                        alpha = 0.1, eta = 0.01,
                        seed = 575)


slda_cbp.train(iter = iters)

slda_cbp.save("models/slda_cbp.pickle")