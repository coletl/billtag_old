import tomotopy as tp

"""
Fit PLDA model
"""

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

