import argparse
import os
from glob import glob
from politicianmap.utils import check_dir
from politicianmap.utils import News, DateDocsDecorator
from politicianmap.utils import Tokenizer, Tagfilter, scan_vocabulary, create_bow
from scipy.io import mmwrite


class MergedNews:
    def __init__(self, date_news_sequence):
        self.date_news_sequence = date_news_sequence
    def __iter__(self):
        for date_news in self.date_news_sequence:
            for date, docs in date_news:
                yield date, docs


def write_list(path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write('{}\n'.format(item))

def scan_universial_vocabulary(data_dirname, index_dirname, output_dirname, debug):
    # variables for debug
    begin_date, end_date = '2018-01-01', '2018-01-10'

    news_sequence = []
    for idx in range(20):
        if debug:
            news = News('{}/{}/'.format(data_dirname, idx), '{}/{}/'.format(index_dirname, idx), begin_date, end_date)
        else:
            news = News('{}/{}/'.format(data_dirname, idx), '{}/{}/'.format(index_dirname, idx))
        news_sequence.append(news)
    merged_data_news = MergedNews([DateDocsDecorator(news, min_doc=15) for news in news_sequence])
    tokenizer = Tokenizer(Tagfilter({'/R'}))
    idx_to_vocab, vocab_to_idx = scan_vocabulary(merged_data_news, tokenizer, min_count=5 if debug else 20)
    # vocab write
    write_list('{}/universial_vocab.txt'.format(output_dirname), idx_to_vocab)
    return vocab_to_idx

def train_bow_a_politician(data_dirname, index_dirname, output_dirname, idx, debug, vocab_to_idx=None):
    head = 'each' if vocab_to_idx is None else 'universial'

    # variables for debug
    begin_date, end_date = '2018-01-01', '2018-01-10'

    # create News instance
    if debug:
        news = News('{}/{}/'.format(data_dirname, idx), '{}/{}/'.format(index_dirname, idx), begin_date, end_date)
    else:
        news = News('{}/{}/'.format(data_dirname, idx), '{}/{}/'.format(index_dirname, idx))

    tokenizer = Tokenizer(Tagfilter({'/R'}))
    date_news = DateDocsDecorator(news, min_doc=10)
    if vocab_to_idx is None:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(date_news, tokenizer, min_count=5 if debug else 20)
    bow, idx_to_date = create_bow(date_news, tokenizer, vocab_to_idx)
    print('[Politician {}, {}]: bow shape = {}'.format(idx, head, bow.shape))

    # matrix write
    mmwrite('{}/{}_bow_{}.mtx'.format(output_dirname, head, idx), bow)
    # vocab write
    if head == 'each':
        write_list('{}/{}_vocab_{}.txt'.format(output_dirname, head, idx), idx_to_vocab)
    # date write
    write_list('{}/{}_date_{}.txt'.format(output_dirname, head, idx), idx_to_date)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirname', type=str, default='/workspace/lovit/politicianmap/tokenized/')
    parser.add_argument('--index_dirname', type=str, default='/workspace/data/politician_norm/')
    parser.add_argument('--output_dirname', type=str, default='/workspace/lovit/politicianmap/bow/')
    parser.add_argument('--politician', type=int, nargs='*', default=None)
    parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()
    data_dirname = os.path.abspath(args.data_dirname)
    index_dirname = os.path.abspath(args.index_dirname)
    output_dirname = os.path.abspath(args.output_dirname)
    debug = args.debug
    politician = args.politician
    if politician is None:
        politician = [i for i in range(20)]

    check_dir(output_dirname)
    univ_vocab_to_idx = scan_universial_vocabulary(data_dirname, index_dirname, output_dirname, debug)

    for idx in politician:
        train_bow_a_politician(data_dirname, index_dirname, output_dirname, idx, debug)
        print('trained politician {} bow model for each'.format(idx), end='\n\n')

        train_bow_a_politician(data_dirname, index_dirname, output_dirname, idx, debug, univ_vocab_to_idx)
        print('trained politician {} bow model with universial vocab'.format(idx), end='\n\n')
        if debug and idx >= 3:
            break

if __name__ == '__main__':
    main()
