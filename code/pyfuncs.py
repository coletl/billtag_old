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