import argparse
import os
from politicianmap.utils import check_dir
from politicianmap.utils import News
from soynlp.noun import LRNounExtractor_v2


def iter_dates():
    months = [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11], [11, 12]]
    years = [y for y in range(2013, 2019)]
    dates = [('%d-%02d-01' % (y, m[0]), '%d-%02d-31' % (y, m[-1]))
             for y in years for m in months]
    dates.append(('2019-01-01', '2019-03-10'))
    return dates

def noun_extraction(data_dirname, index_dirname, noun_dirname, idx, debug=False):
    for begin_date, end_date in iter_dates():
        news = News(
            '{}/{}/'.format(data_dirname, idx),
            '{}/{}/'.format(index_dirname, idx),
            begin_date, end_date
        )
        noun_extractor = LRNounExtractor_v2(extract_compound=True, verbose=False)
        noun_score = noun_extractor.train_extract(news)
        path = '{}/{}/{}_{}'.format(noun_dirname, idx, begin_date, end_date)
        check_dir(path)
        with open(path, 'w', encoding='utf-8') as f:
            for noun, score in sorted(noun_score.items(), key=lambda x:-x[1].frequency):
                f.write('%s %d %.4f\n' % (noun, score.frequency, score.score))
        print('noun extraction idx = {}, {} - {} done\n'.format(idx, begin_date, end_date))
        if debug:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirname', type=str, default='/workspace/data/politician_norm/')
    parser.add_argument('--index_dirname', type=str, default='/workspace/data/politician_norm/')
    parser.add_argument('--noun_dirname', type=str, default='/workspace/lovit/politicianmap/noun_extraction/')
    parser.add_argument('--politician', type=int, nargs='*', default=None)
    parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()
    data_dirname = os.path.abspath(args.data_dirname)
    index_dirname = os.path.abspath(args.index_dirname)
    noun_dirname = os.path.abspath(args.noun_dirname)
    debug = args.debug
    politician = args.politician
    if politician is None:
        politician = [i for i in range(20)]

    for idx in politician:
        noun_extraction(data_dirname, index_dirname, noun_dirname, idx, debug)
        print('-' *40, end='\n\n')

if __name__ == '__main__':
    main()
