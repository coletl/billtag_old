"""
Fit PLDA model
"""

import tomotopy as tp

iters = int(1e4)

corpus_train = load("data/legislation/corpus_train.pickle")
corpus_test = load("data/legislation/corpus_test.pickle")

plda_crs = tp.PLDAModel(tw = tp.TermWeight.IDF, corpus = corpus_train,
                        min_cf = 0, min_df = 100,
                        latent_topics = 1, topics_per_label = 1,
                        # hyperparameters for dirichlet
                        alpha = 0.1, eta = 0.01,
                        seed = 575)

plda_crs.train(iter = iters)

plda_crs.save("models/plda_crs.pickle")

plda_crs.summary()
plda_crs.topic_label_dict
plda_crs.get_topic_words(topic_id=0)

