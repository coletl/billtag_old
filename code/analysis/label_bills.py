"""
Apply topic models to test documents
"""

import tomotopy as tp
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from code.pyfuncs import *

# Read in test corpus and DF of topic labels
corpus_test_cbp = tp.utils.Corpus().load("data/legislation/corpus_test_cbp.pickle")
bills_test_cbp = pq.read_pandas("data/legislation/splits/bills_test_cbp.parquet").to_pandas()


"""
Infer test-set topics from PLDA
"""

plda_cbp = tp.PLDAModel.load("models/plda_cbp.pickle")

    
plda_cbp.get_topic_word_dist(20)


# Infer topics, store likelihoods separately
topic_df_plda, ll_plda = infer_labels(plda_cbp, corpus_test_cbp, idnm = "billid")

# Join with real labels
topic_df_plda = pd.merge(bills_test_cbp, topic_df_plda, how = "left", 
                         left_on = "id", right_on = "billid")


pd.crosstab(topic_df_plda['label'], topic_df_plda['topic_inferred'])

np.mean(topic_df_plda['label'] == topic_df_plda['topic_inferred'])

"""
SLDA
"""

slda_cbp = tp.SLDAModel.load("models/slda_cbp.pickle")

slda_cbp.summary()
set(bills_test_cbp['label'])


# Infer topics, store likelihoods separately
topic_df_slda, ll_slda = infer_labels(slda_cbp, corpus_test_cbp, idnm = "billid")

# Join with real labels
topic_df_slda = pd.merge(bills_test_cbp, topic_df_slda, how = "left", 
                         left_on = "id", right_on = "billid")

np.mean(topic_df_slda['label'] == topic_df_slda['topic_inferred'])

pd.crosstab(topic_df_slda['label'], topic_df_slda['topic_inferred'])