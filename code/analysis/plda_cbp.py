"""
Fit PLDA and SLDA models with topic labels from the Congressional Bills Project
"""

import json
import tomotopy as tp

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
Apply model to test corpus
"""
plda_cbp.summary()
plda_cbp.topic_label_dict
plda_cbp.get_topic_words(topic_id=0)


corpus_test = tp.utils.Corpus().load("data/legislation/corpus_test_cbp.pickle")

corpus_test_plda_topics = plda_cbp.infer(corpus_test)

"""
Fit SLDA model
"""
slda_cbp = tp.SLDAModel(tw = tp.TermWeight.IDF, corpus = corpus_train,
                        min_cf = 0, min_df = 100,
                        # hyperparameters for dirichlet
                        alpha = 0.1, eta = 0.01,
                        seed = 575)

slda_cbp.train(iter = iters)

slda_cbp.save("models/slda_cbp.pickle")
# slda_cbp = tp.SLDAModel.load("models/slda_cbp.pickle")

"""
Apply model to test corpus
"""
slda_cbp.summary()
slda_cbp.topic_label_dict
slda_cbp.get_topic_words(topic_id=0)


corpus_test_slda_topics = slda_cbp.infer(corpus_test)
