import argparse
import re
import os
from glob import glob
from shutil import copyfile
from politicianmap.utils import check_dir


doublespace_pattern = re.compile('\s+')
text_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9]')

def only_text(sent):
    return doublespace_pattern.sub(' ',text_filter.sub(' ', sent)).strip()

def normalize(doc):
    doc = [only_text(sent) for sent in doc.split('  ')]
    doc = [sent for sent in doc if sent]
    if not doc:
        return ''
    return '  '.join(doc)

def normalize_a_politician(source_dir, idx, dest_dir, debug):
    paths = sorted(glob('{}/{}/news/*.txt'.format(source_dir, idx)))
    if debug:
        paths = paths[:10]

    for inpath in paths:
        outpath = dest_dir + inpath[len(source_dir):]
        check_dir(outpath)
        with open(inpath, encoding='utf-8') as f:
            docs = [doc.strip() for doc in f]

        docs = [normalize(doc) for doc in docs]
        with open(outpath, 'w', encoding='utf-8') as f:
            for doc in docs:
                f.write('{}\n'.format(doc))

        index_source = inpath[:-3] + 'index'
        index_dest = dest_dir + index_source[len(source_dir):]
        copyfile(index_source, index_dest)
        print('normalized {}'.format(inpath))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='/workspace/data/politician/')
    parser.add_argument('--dest_dir', type=str, default='/workspace/data/politician_norm/')
    parser.add_argument('--politician', type=int, nargs='*', default=None)
    parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()
    source_dir = os.path.abspath(args.source_dir)
    dest_dir = os.path.abspath(args.dest_dir)
    debug = args.debug
    politician = args.politician
    if politician is None:
        politician = [i for i in range(20)]

    for idx in politician:
        normalize_a_politician(source_dir, idx, dest_dir, debug)

if __name__ == '__main__':
    main()
