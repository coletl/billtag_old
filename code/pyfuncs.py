"""
Python functions for fitting topic models to legislation
"""

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


    

def get_inferred_labels(corpus):
    import pandas
    import numpy
    from _tomotopy import _UtilsCorpus
    
    """
    Get inferred topic distributions in a DataFrame
    """
    
    assert isinstance(corpus, _UtilsCorpus)
    # assert isinstance(df_labels, pandas.DataFrame)
    # doc.labels is a list of tuples of (label, ll)
    
    # Get topic distribution for each document
    topic_dist_list = [doc.get_topic_dist() for doc in corpus]
    
    # Each document.labels is a list of tuples, each tuple is a (topic, ll)
    if hasattr(corpus[0], "labels"):
        label_list = corpus[0].labels
        labels = [tpl[0] for tpl in label_list]
        # add latent topic, if estimated
        if len(topic_dist_list[0]) == len(labels) + 1: labels.append("Latent")
    elif hasattr(corpus[0], "y"):
        label_list = corpus[0].y
        labels = [tpl[0] for tpl in label_list]
    else: labels = [i for i in range(len(corpus[0].get_topic_dist()))]
    
    topic_df = pandas.DataFrame(topic_dist_list, columns = labels)
     
    return topic_df




def infer_labels(tp_lda_model, corpus_test, idnm: str):
    import tomotopy
    import pandas
    
    assert isinstance(corpus_test, tomotopy.utils.Corpus)
    assert isinstance(tp_lda_model, tomotopy.PLDAModel) | isinstance(tp_lda_model, tomotopy.SLDAModel)
    
    corpus_test_infer, ll = tp_lda_model.infer(corpus_test)
    labels_infer_df = get_inferred_labels(corpus_test_infer)
    
    # Store column names for reordering
    cols = labels_infer_df.columns.values.tolist()
    
    # Best guess, assuming no ties
    label_max = labels_infer_df.idxmax(1)
    labels_infer_df['topic_inferred'] = label_max
    
    # ID column
    labels_infer_df[idnm] = [getattr(doc, idnm) for doc in corpus_test]
    
    # Set column order
    cols.insert(0, 'topic_inferred')
    cols.insert(0, idnm)
    labels_infer_df = labels_infer_df[cols]
    
    return labels_infer_df, ll
    
