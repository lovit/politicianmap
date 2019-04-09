from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix


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


class DateDocsDecorator:
    """
        >>> tokenizer = Tokenizer(Tagfilter({'/R'}))
        >>> date_news = DateDocsDecorator(news, min_doc=10)
    """
    def __init__(self, news, min_doc=10):
        self.news = news
        self.min_doc = min_doc
    def __iter__(self):
        for date, n_docs in self.news.date_to_ndocs:
            if n_docs < self.min_doc:
                continue
            docs = self.news.get_news(date, date)
            yield (date, docs)


def scan_vocabulary(date_docs, tokenizer, min_count=20):
    """
    Arguments
    ---------
    date_docs : Any type iterable data that yield (date, [doc])
    tokenizer : callable
        tokenizer(doc) : list of str
    min_count : int
        Minimum frequency of vocabulary

    Returns
    -------
    idx_to_vocab : list of str
        Each str stands for word
    vocab_to_idx : {str:int}
        Vocabulary index map

    Usage
    -----
        >>> tokenizer = Tokenizer(Tagfilter({'/R'}))
        >>> date_news = DateDocsDecorator(news, min_doc=10)
        >>> idx_to_vocab, vocab_to_idx = scan_vocabulary(date_news, tokenizer, min_count=5)
    """
    counter = defaultdict(int)
    for date, docs in date_docs:
        for doc in docs:
            for word in tokenizer(doc):
                counter[word] += 1
    counter = {term:count for term, count in counter.items() if count >= min_count}
    idx_to_vocab = [vocab for vocab in sorted(counter, key=lambda x:(-counter[x], x))]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def create_bow(date_docs, tokenizer, vocab_to_idx):
    """
    Arguments
    ---------
    date_docs : Any type iterable data that yield (date, [doc])
    tokenizer : callable
        tokenizer(doc) : list of str
    vocab_to_idx : {str:int}
        Vocabulary index

    Returns
    -------
    bow : scipy.sparse.csr_matrix
        (date, term) frequency matrix
    idx_to_date : list of str
        Each str stands for date

    Usage
    -----
        >>> tokenizer = Tokenizer(Tagfilter({'/R'}))
        >>> date_news = DateDocsDecorator(news, min_doc=10)
        >>> idx_to_vocab, vocab_to_idx = scan_vocabulary(date_news, tokenizer, min_count=5)
        >>> bow, idx_to_date = create_bow(date_news, tokenizer, vocab_to_idx)
    """
    idx_to_date = []
    rows, cols, data = [], [], []
    for date, docs in date_docs:
        # indexing row
        i = len(idx_to_date)
        idx_to_date.append(date)

        # count term frequency
        docs = news.get_news(date, date)
        tf = Counter(word for doc in docs for word in tokenizer(doc))
        for term, count in tf.items():
            j = vocab_to_idx.get(term, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j),
            data.append(count)
    bow = csr_matrix((data, (rows, cols)))
    return bow, idx_to_date
