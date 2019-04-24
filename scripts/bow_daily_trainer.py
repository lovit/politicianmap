import argparse
import os
from politicianmap.utils import check_dir
from politicianmap.utils import News, DateDocsDecorator
from politicianmap.utils import Tokenizer, Tagfilter, scan_vocabulary


def write_list(path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write('{}\n'.format(item))

def make_merged_date_news(data_dir, index_dir, index, min_doc=15, debug=True):
    # variables for debug
    begin_date, end_date = '2018-01-01', '2018-01-10'

    # instance class
    class MergedNews:
        def __init__(self, date_news_sequence):
            self.date_news_sequence = date_news_sequence
        def __iter__(self):
            for date_news in self.date_news_sequence:
                for date, docs in date_news:
                    yield date, docs

    # merging news for all idxs
    news_sequence = []
    for idx in index:
        if debug:
            news = News('{}/{}/'.format(data_dir, idx), '{}/{}/'.format(index_dir, idx), begin_date, end_date)
        else:
            news = News('{}/{}/'.format(data_dir, idx), '{}/{}/'.format(index_dir, idx))
        news_sequence.append(news)
    merged_data_news = MergedNews([DateDocsDecorator(news, min_doc=min_doc) for news in news_sequence])
    return merged_data_news

def scaning_vocabulary(date_news, debug=True):
    tokenizer = Tokenizer(Tagfilter({'/R'}))
    idx_to_vocab, vocab_to_idx = scan_vocabulary(date_news, tokenizer, min_count=5 if debug else 20)
    return idx_to_vocab, vocab_to_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/lovit/politicianmap/tokenized/')
    parser.add_argument('--index_dir', type=str, default='/workspace/data/politician_norm/')
    parser.add_argument('--output_dir', type=str, default='/workspace/lovit/politicianmap/bow_daily/')
    parser.add_argument('--min_doc_of_a_day', type=int, default=15)
    parser.add_argument('--index', type=int, nargs='*', default=None)
    parser.add_argument('--max_index', type=int, default=20)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--use_universal_vocab', dest='use_universal_vocab', action='store_true')

    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    index_dir = os.path.abspath(args.index_dir)
    output_dir = os.path.abspath(args.output_dir)

    min_doc = args.min_doc_of_a_day
    debug = args.debug
    index = args.index
    if index is None:
        index = [i for i in range(args.max_index)]

    # scan universal vocabulary
    merged_data_news = make_merged_date_news(data_dir, index_dir, index, min_doc, debug)
    idx_to_vocab_univ, vocab_to_idx_univ = scaning_vocabulary(merged_data_news, debug)
    path = '{}/universal_vocab.txt'.format(output_dir)
    check_dir(path)
    write_list(path, idx_to_vocab_univ)

    # scan their own vocabulary and vectorize

if __name__ == '__main__':
    main()
