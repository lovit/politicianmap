from glob import glob

class News:
    """
        >>> news = News('../data/politician/0/')
        >>> for i, doc in enumerate(news.get_news(begin_date = '2018-01-01', end_date='2018-01-03')):
        >>>     print(doc[:200], end='\n\n')
    """

    def __init__(self, dirname):
        self.dirname = dirname
        self.newspath = sorted(glob('{}/news/*.txt'.format(dirname)))
        self.indexpath = sorted(glob('{}/news/*.index'.format(dirname)))
        self.dates = [p.split('/')[-1][:10] for p in self.newspath]
        print('{} has news of {} dates, index of {} dates'.format(
            dirname, len(self.newspath), len(self.indexpath)))

    def get_news(self, begin_date=None, end_date=None):
        """
        Arguments
        ---------
        begin_date : str
            yyyy-mm-dd format
            If it is None, use first date of news in dirname
            Default is None
        end_date : str
            yyyy-mm-dd format
            If it is None, use first date of news in dirname
            Default is None

        Yields
        ------
        doc : str
            Doublespace line format document of a news
        """

        begin_date, end_date = self._set_date(begin_date, end_date)
        for doc in self._iter(begin_date, end_date, self.newspath):
            yield doc

    def get_index(self, begin_date=None, end_date=None):
        """
        Arguments
        ---------
        begin_date : str
            yyyy-mm-dd format
            If it is None, use first date of news in dirname
            Default is None
        end_date : str
            yyyy-mm-dd format
            If it is None, use first date of news in dirname
            Default is None

        Yields
        ------
        index : str
            Tap separated index (press/yy/mm/dd/article, category, date, title)
            For example,

                $ 421/2018/01/02/0003129236	100	2018-01-02 15:53	'NLL 파문' 대화록 유출 수사 빈손…검찰 "기소 없을듯"
        """

        begin_date, end_date = self._set_date(begin_date, end_date)
        for doc in self._iter(begin_date, end_date, self.indexpath):
            yield doc

    def _set_date(self, begin_date, end_date):
        if begin_date is None:
            begin_date = self.dates[0]
        if end_date is None:
            end_date = self.dates[-1]
        return begin_date, end_date

    def _iter(self, begin_date, end_date, paths):
        for path in paths:
            file_date = path.split('/')[-1][:10]
            if not (begin_date <= file_date <= end_date):
                continue
            with open(path, encoding='utf-8') as f:
                for doc in f:
                    yield doc

from glob import glob

dirname = '/workspace/data/politician/0/'
begin_date = '2018-01-01'
end_date='2018-01-03'

def parse_date(path):
    """
    Arguments
    ---------
    path : str
        File path

    Usage
    -----
        >>> path = '/workspace/data/politician/0/news/2013-01-02_politician.index'
        >>> parse_date(path)
        $ '2013-01-02'
    """
    return path.split('/')[-1][:10]

def line_counts(dirname, begin_date, end_date):
    """
    Arguments
    ---------
    dirname : str
        Directory path
    begin_date : str
        yyyy-mm-dd format
    end_date : str
        yyyy-mm-dd format

    Returns
    -------
    List of tuple
        It consists with (date, num lines)

    Usage
    -----
        >>> dirname = '/workspace/data/politician/0/'
        >>> begin_date = '2018-01-01'
        >>> end_date='2018-01-03'

        $ [('2018-01-01', 2),
           ('2018-01-02', 7),
            ('2018-01-03', 3)]
    """

    indexpath = sorted(glob('{}/news/*.index'.format(dirname)))
    indexpath = [p for p in indexpath if begin_date <= parse_date(p) <= end_date]
    counts = [(parse_date(p), line_count(p)) for p in indexpath]
    return counts

def line_count(path):
    n_line = 0
    with open(path, encoding='utf-8') as f:
        for _ in f:
            n_line += 1
    return n_line