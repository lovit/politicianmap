import pickle

class RepresentationGenerator:
    def __init__(self, doc2vec_path, tokenizer=None,
        alpha=0.025, min_alpha=0.01, steps=10):

        if tokenizer is None:
            tokenizer = lambda x:x.split()
        with open(doc2vec_path, 'rb') as f:
            self.doc2vec = pickle.load(f)
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.steps = steps

    def infer_docvec(self, docs, alpha=None, min_alpha=None, steps=None):
        if isinstance(docs, str):
            docs = [docs]
        alpha = self.alpha if alpha is None else alpha
        min_alpha = self.min_alpha if min_alpha is None else min_alpha
        steps = self.steps if steps is None else steps

        words = [w for doc in docs for w in self.tokenizer(doc)]
        vec = self.doc2vec.infer_vector(words, alpha=alpha,
            min_alpha=min_alpha, steps=steps)
        return vec
