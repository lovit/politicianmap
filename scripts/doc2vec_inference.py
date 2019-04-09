import argparse
import numpy as np
import os
import pickle
from politicianmap.utils import News, Tokenizer, Stopwords, Tagfilter, check_dir
from politicianmap.representation import RepresentationGenerator


def write_list(path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write('{}\n'.format(item))

def train_docvec_for_a_politician(data_dirname, index_dirname, idx, doc2vec_path, out_dirname, head, debug):
    # load trained doc2vec
    encoder = RepresentationGenerator(
        doc2vec_path,
        Tokenizer(Tagfilter({'R'}))
    )

    # variables for debug
    begin_date, end_date = '2018-01-01', '2018-01-10'

    # create News instance
    if debug:
        news = News('{}/{}/'.format(data_dirname, idx), '{}/{}/'.format(index_dirname, idx), begin_date, end_date)
    else:
        news = News('{}/{}/'.format(data_dirname, idx), '{}/{}/'.format(index_dirname, idx))

    date_docvecs = []
    n_dates = len(news.date_to_ndocs)
    for i, (date, n_docs) in enumerate(news.date_to_ndocs):
        if i % 100 == 99:
            print('\r[Politician {}, {}] Infering ... {} / {}'.format(idx, head, i+1, n_dates))
        if n_docs < 15:
            continue
        vec = encoder.infer_docvec(news.get_news(date, date), steps=100)
        date_docvecs.append((date, vec))

    dates, docvecs = zip(*date_docvecs)
    docvecs = np.vstack([vec.reshape(1,-1) for vec in docvecs])
    print('\r[Politician {}, {}] Infering was done. docvecs shape ={}'.format(idx, head, docvecs.shape))

    # write date
    path = '{}/date_{}.txt'.format(out_dirname, idx)
    check_dir(path)
    write_list(path, dates)
    # write doc vectors
    with open('{}/{}_{}_docvec.pkl'.format(out_dirname, head, idx), 'wb') as f:
        pickle.dump(docvecs, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirname', type=str, default='/workspace/lovit/politicianmap/tokenized/')
    parser.add_argument('--index_dirname', type=str, default='/workspace/data/politician_norm/')
    parser.add_argument('--doc2vec_model_dirname', type=str, default='/workspace/lovit/politicianmap/doc2vec_models/')
    parser.add_argument('--out_dirname', type=str, default='/workspace/lovit/politicianmap/docvec_infered/')
    parser.add_argument('--politician', type=int, nargs='*', default=None)
    parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()
    data_dirname = os.path.abspath(args.data_dirname)
    index_dirname = os.path.abspath(args.index_dirname)
    doc2vec_model_dirname = os.path.abspath(args.doc2vec_model_dirname)
    out_dirname = os.path.abspath(args.out_dirname)
    check_dir(out_dirname)

    debug = args.debug
    politician = args.politician
    if politician is None:
        politician = [i for i in range(20)]

    for idx in politician:
        head = 'each'
        doc2vec_path = '{}/doc2vec_docvec_politician_{}.pkl'.format(doc2vec_model_dirname, idx)
        train_docvec_for_a_politician(data_dirname, index_dirname, idx, doc2vec_path, out_dirname, head, debug)
        print('infered politician {} with each doc2vec model'.format(idx), end='\n\n')

        head = 'universial'
        doc2vec_path = '{}/doc2vec_docvec_politician_all.pkl'.format(doc2vec_model_dirname)
        train_docvec_for_a_politician(data_dirname, index_dirname, idx, doc2vec_path, out_dirname, head, debug)
        print('infered politician {} with universial doc2vec model'.format(idx), end='\n\n')
        if debug and idx >= 3:
            break

if __name__ == '__main__':
    main()
