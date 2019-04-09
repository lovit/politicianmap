import argparse
import os
import pickle
from glob import glob
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from politicianmap.utils import check_dir
from politicianmap.utils import News


class Input:
    def __init__(self, news, header='', min_doc=15):
        self.news = news
        self.n_iter = 0
        self.header = header
        self.min_doc = min_doc
        self.n_dates = len(news.dates)

    def __iter__(self):
        for i, (date, n_docs) in enumerate(self.news.date_to_ndocs):
            if i % 10000 == 0:
                print('\r{} iter = {}, date = {} / {} ...'.format(self.header, self.n_iter, i, self.n_dates), end='')
            if n_docs < self.min_doc:
                continue
            for doc in self.news.iter_news(date, date):
                for sent in doc.split('  '):
                    words = sent.split()
                    if len(words) <= 1:
                        continue
                yield TaggedDocument(words=words, tags=['{}_{}'.format(self.header, date)])
        self.n_iter += 1
        print('\r{0} iter = {1}, date = {2} / {2} was done'.format(self.header, self.n_iter, self.n_dates))


class AllInput:
    def __init__(self, tokenized_dir, debug=False, min_doc=15):
        self.tokenized_dir = tokenized_dir
        self.paths = glob(tokenized_dir+'/*/news/*.txt')
        self.n_files = len(self.paths)
        self.min_doc = min_doc
        print('num files = {}'.format(len(self.paths)))
        if debug:
            self.paths = self.paths[:100]
            print('debug mode. use only docs from 100 days')
        self.n_iter = 0

    def __iter__(self):
        for i, path in enumerate(self.paths):
            if i % 500 == 0:
                print('\r[Politician all] iter = {}, files = {} / {} ... '.format(self.n_iter, i+1, self.n_files), end='')
            with open(path, encoding='utf-8') as f:
                docs = [doc.strip() for doc in f]
            if len(docs) <= self.min_doc:
                continue
            idx, _, date = path.split('/')[-3:]
            date = date[:10]
            for doc in docs:
                for sent in doc.split('  '):                    
                    words = sent.split()
                    if len(words) <= 1:
                        continue
                    yield TaggedDocument(words=sent.split(), tags=['#{}_{}'.format(idx, date)])
        self.n_iter += 1
        print('\r[Politician all] iter = {0}, files = {1} / {1} was done '.format(self.n_iter, self.n_files))


def train_doc2vec_a_politician(data_dirname, index_dirnae, idx, debug):
    # variables for debug
    begin_date, end_date = '2018-01-01', '2018-01-10'

    # create News instance
    if debug:
        news = News('{}/{}/'.format(data_dirname, idx), '{}/{}/'.format(index_dirnae, idx), begin_date, end_date)
    else:
        news = News('{}/{}/'.format(data_dirname, idx), '{}/{}/'.format(index_dirnae, idx))

    doc2vec = Doc2Vec( Input(news, header='#%d'%idx), min_count=15 )
    doc2vec_path = '/workspace/lovit/politicianmap/doc2vec_models/doc2vec_docvec_politician_{}.pkl'.format(idx)
    check_dir(doc2vec_path)
    with open(doc2vec_path, 'wb') as f:
        pickle.dump(doc2vec, f)

def train_doc2vec_for_all_politician(data_dirname, debug):
    inputs = AllInput(data_dirname, debug=debug)
    doc2vec = Doc2Vec(inputs, min_count=(10 if debug else 20))
    doc2vec_path = '/workspace/lovit/politicianmap/doc2vec_models/doc2vec_docvec_politician_all.pkl'
    check_dir(doc2vec_path)
    with open(doc2vec_path, 'wb') as f:
        pickle.dump(doc2vec, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirname', type=str, default='/workspace/lovit/politicianmap/tokenized/')
    parser.add_argument('--index_dirnae', type=str, default='/workspace/data/politician_norm/')
    parser.add_argument('--politician', type=int, nargs='*', default=None)
    parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()
    data_dirname = os.path.abspath(args.data_dirname)
    index_dirnae = os.path.abspath(args.index_dirnae)
    debug = args.debug
    politician = args.politician
    if politician is None:
        politician = [i for i in range(20)]

    for idx in politician:
        train_doc2vec_a_politician(data_dirname, index_dirnae, idx, debug)
        print('trained politician {} doc2vec model'.format(idx), end='\n\n')
        if debug and idx >= 3:
            break

    train_doc2vec_for_all_politician(data_dirname, debug)
    print('trained all politician doc2vec model\nTerminated')

if __name__ == '__main__':
    main()
