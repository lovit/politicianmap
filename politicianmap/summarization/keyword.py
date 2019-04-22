import numpy as np


def to_proportion(bow, doc_idx):
    prop = np.asarray(bow[doc_idx,:].sum(axis=0))[0]
    prop = prop / prop.sum()
    return prop

def proportion_ratio(pos, ref):
    assert len(pos) == len(ref)
    ratio = pos / (pos + ref)
    ratio = np.nan_to_num(ratio)
    return ratio

def extract_keywords(bow, doc_idx, idx_to_vocab, margin=1, ref_size=2, topk1=100, topk2=30):
    """
    Arguments
    ---------
    bow : scipy.sparse.csr_matrix
        Bag-of-words model
    doc_idx : numpy.ndarray
        Shape = (n_dates,)
        dtype is numpy.int
    idx_to_vocab : list of str
        Index to vocab
    margin : int
        Date difference
    ref_size : float
        Scale factor between length of target period and length of reference period
    topk1 : int
        Number of keyword candidates by word occurrence
    topk2 : int
        Number of keywords selected from topk1 candidates by proportion ratio keyword score

    Returns
    -------
    list of tuple
        Each tuple is formed (term, keywords score, term proportion).

    Usage
    -----
        >>> doc_idx = [i for i in range(1511, 1519+1)]
        >>> keywords = extract_keywords(docvec, doc_idx, idx_to_vocab, margin=5, ref_size=5)
    """
    n_docs = bow.shape[0]
    # positive proportion
    pos = to_proportion(bow, doc_idx)

    # reference proportion
    period_begin = min(doc_idx)
    period_end = max(doc_idx)
    period_len = len(doc_idx)
    ref_idx = [i for i in range(max(0, period_begin - margin - int(period_len * ref_size)), period_begin)]
    ref_idx += [i for i in range(min(period_end + margin + int(period_len * ref_size), n_docs), n_docs)]
    ref = to_proportion(bow, ref_idx)
    ratio = proportion_ratio(pos, ref)

    # select candidates (frequent terms)
    candidates_idx = pos.argsort()[-topk1:]
    candidates_score = ratio[candidates_idx]

    # sort by distinctness
    keyword_idx = candidates_idx[candidates_score.argsort()[-topk2:]]
    keywords = [(idx, ratio[idx]) for idx in keyword_idx]
    keywords = [(idx_to_vocab[idx], score, pos[idx]) for idx, score in reversed(keywords)]

    return keywords