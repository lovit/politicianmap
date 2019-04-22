import numpy as np


def point_shift_distance(pdist, b, e, window=1):
    return pdist[b:e, e:e+window].mean() -  pdist[b:e,b:e].mean()

def shift_distance(pdist, window=1):
    w = window
    dist = np.zeros(pdist.shape[0])
    for e in range(w, dist.shape[0]-w):
        b = max(0, e - 3)
        dist[b] = point_shift_distance(pdist, b, e, window)
    return dist

def find_rectangular(pdist, threshold=0.4, min_length=2, max_length=20):
    """
    Arguments
    ---------
    pdist : numpy.ndarray
        Pairwise distance matrix
    threshold : float
        Minimum inner distance of segments
    min_length : int
        Minimum dates of a segment
    max_length : int
        Maximum dates of a segment

    Returns
    -------
    list of tuple
        Each tuple is (begin index, end index, length)
    """
    n = pdist.shape[0]
    segments = []
    b = 1
    n_iter = 0
    exception_iter = n * max_length
    while b < n:
        append = False
        n_iter += 1
        if n_iter > exception_iter:
            raise RuntimeError('Too much iteration. Fix bug.')
        if pdist[b,b-1] > threshold:
            b = b + 1
            continue
        for e in range(b+1, min(n, b+max_length)):
            if point_shift_distance(pdist, b, e) < threshold:
                continue
            if (e - b < min_length) or (pdist[b:e,b:e].mean() > threshold):
                b = b + 1
                break
            append = True
            segments.append((b, e, e - b))
            b = e
            break
        if not append:
            b = b + 1
    return segments
