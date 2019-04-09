class Tokenizer:
    """
        >>> docs = '이번 문장 은/R 이런 예시 이다/R'
        >>> tokenizer = Tokenizer()
        >>> tokenizer.tokenize(docs)

        $ ['이번', '문장', '은/R', '이런', '예시', '이다/R']
    """
    def __init__(self, *filters):
        if not filters:
            filters = [lambda x:x]
        self.filters = [f for f in filters]

    def __call__(self, doc):
        return self.tokenize(doc)

    def tokenize(self, doc):
        """
        Argument
        --------
        doc : str
            A document

        Returns
        -------
        list of str
            White space separated word sequence
        """
        words = doc.split()
        for f in self.filters:
            words = f(words)
        return words

class Stopwords:
    """
    >>> docs = '이번 문장 은/R 이런 예시 이다/R'
    >>> tokenizer = Tokenizer(
    >>>     Stopwords({'이런'})
    >>> )
    >>> tokenizer.tokenize(docs)

    $ ['이번', '문장', '은/R', '예시', '이다/R']
    """
    def __init__(self, stopwords):
        self.stopwords = stopwords
    def __call__(self, words):
        return [w for w in words if not (w in self.stopwords)]

class Tagfilter:
    """
        >>> docs = '이번 문장 은/R 이런 예시 이다/R'
        >>> tokenizer = Tokenizer(
        >>>     Stopwords({'이런'}),
        >>>     Tagfilter('/R')
        >>> )
        >>> tokenizer.tokenize(docs)

        $ ['이번', '문장', '예시']
    """
    def __init__(self, tags):
        self.tags = {t if t[0] == '/' else '/'+t for t in tags}
    def __call__(self, words):
        def has_tag(w):
            for t in self.tags:
                if t in w:
                    return True
            return False
        return [w for w in words if not has_tag(w)]
