import argparse
import os
from glob import glob
from politicianmap.utils import News
from politicianmap.utils import parse_date, line_counts, check_dir
from soynlp.tokenizer import LTokenizer


def iter_dates():
    months = [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11], [11, 12]]
    years = [y for y in range(2013, 2019)]
    dates = [('%d-%02d-01' % (y, m[0]), '%d-%02d-31' % (y, m[-1]))
             for y in years for m in months]
    dates.append(('2019-01-01', '2019-03-10'))
    return dates

def load_dictionary(noun_dirname, idx):
    dictionary_paths = sorted(glob('{}/{}/*'.format(noun_dirname, idx)))

    def date(path):
        bd, ed = path.split('/')[-1].split('_')
        return bd, ed

    def load(path):
        with open(path, encoding='utf-8') as f:
            nouns = [line.strip().split() for line in f]
            # (noun, frequency, score)
            nouns = {w:float(s) for w, _, s in nouns}
        return nouns

    dictionaries = [(*date(p), load(p)) for p in dictionary_paths]
    return dictionaries

def select_dictionary(path, dictionaries):
    date = parse_date(path)
    for bd, ed, dictionary in dictionaries:
        if bd <= date <= ed:
            return dictionary
    raise ValueError('Not found available dictionary')

def tokenize(docs, dictionary):
    tokenizer = LTokenizer(scores = dictionary)

    def tokenize_sent(sent):
        sent_ = []
        for l, r in tokenizer.tokenize(sent, flatten=False):
            sent_.append(l)
            if r:
                sent_.append(r+'/R')
        return ' '.join(sent_)

    docs = ['  '.join([tokenize_sent(sent) for sent in doc.split('  ')]) for doc in docs]
    return docs

def load_docs(path):
    with open(path, encoding='utf-8') as f:
        docs = [line.strip() for line in f]
    return docs

def write_docs(docs, path):
    check_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write('{}\n'.format(doc))

def tokenize_a_politician(data_dirname, noun_dirname, idx, dest_dirname, debug):
    paths = sorted(glob('{}/{}/news/*.txt'.format(data_dirname, idx)))
    if debug:
        paths = paths[:3]
    dictionaries = load_dictionary(noun_dirname, idx)

    for inpath in paths:
        outpath = dest_dirname + inpath[len(data_dirname):]
        dictionary = select_dictionary(inpath, dictionaries)
        docs = load_docs(inpath)
        docs = tokenize(docs, dictionary)
        write_docs(docs, outpath)
        print('tokenized {}'.format(inpath))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirname', type=str, default='/workspace/data/politician_norm/')
    parser.add_argument('--noun_dirname', type=str, default='/workspace/lovit/politicianmap/noun_extraction/')
    parser.add_argument('--dest_dirname', type=str, default='/workspace/lovit/politicianmap/tokenized/')
    parser.add_argument('--politician', type=int, nargs='*', default=None)
    parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()
    data_dirname = os.path.abspath(args.data_dirname)
    noun_dirname = os.path.abspath(args.noun_dirname)
    dest_dirname = os.path.abspath(args.dest_dirname)
    debug = args.debug
    politician = args.politician
    if politician is None:
        politician = [i for i in range(20)]

    for idx in politician:
        tokenize_a_politician(data_dirname, noun_dirname, idx, dest_dirname, debug)
        print()
        if debug and idx >= 3:
            break

if __name__ == '__main__':
    main()
