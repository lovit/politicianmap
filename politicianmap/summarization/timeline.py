import math
from .keyword import extract_keywords
from .keysentence import summarize


def summarize_timeline(news, idx_to_date, segments, docvec, idx_to_vocab, penalty, margin=10,
    ref_size=5, use_bothside=False, diversity=0.6, num_candidates=300, num_keywords=60, num_keysents=5):

    timeline = []
    n_segments = len(segments)
    for i, (b, e, _) in enumerate(segments):
        doc_idx = [i for i in range(b, e)]
        keywords = extract_keywords(docvec, doc_idx, idx_to_vocab,
            margin=margin, ref_size=ref_size, use_bothside=use_bothside,
            topk1=num_candidates, topk2=num_keywords)
        if not keywords:
            continue
        b_date, e_date = idx_to_date[b], idx_to_date[e-1]
        docs = news.get_news(b_date, e_date)
        sents = [sent for doc in docs for sent in doc.split('  ')]
        vocab_score = {word:score for word, score, _ in keywords}
        keysentences = summarize(vocab_score, sents, penalty=penalty, topk=num_keysents, diversity=diversity)
        timeline.append((b_date, e_date, keywords, keysentences))
        print('\rsummarizing {} / {} segments ...'.format(i, n_segments), end='')
    print('\rsummarizing {0} segments was done   '.format(n_segments))
    return timeline


class PenaltyFunction:
    def __init__(self, min_len=15, max_len=25, including_terms=None, including_term_penalty=0.4):
        if including_terms is None:
            including_terms = {}
        elif isinstance(including_terms, str):
            including_terms = {including_terms}

        self.min_len = min_len
        self.max_len = max_len
        self.including_terms = including_terms
        self.including_term_penalty = including_term_penalty

    def __call__(self, sent):
        return self.penalty(sent)

    def penalty(self, sent):
        def has_pos_term(sent):
            for term in self.including_terms:
                if term in sent:
                    return True
            return False

        if not sent.strip():
            return 2

        n_words = len(sent.split())
        if (self.min_len <= n_words <= self.max_len) and sent[-1] == '다':
            penalty = 0
        else:
            penalty = 2 + math.log(abs(n_words - self.max_len) + 1)

        if not has_pos_term(sent):
            penalty += self.including_term_penalty
        return penalty

def highlight_keyword(sent, keywords):
    keywords = {w if w[0] == ' ' else (' '+w) for w in keywords}
    for keyword in keywords:
        sent = sent.replace(keyword, ' [%s]' % keyword.strip())
    return sent

def as_html(timeline, only_keysent_term=False):
    tr = '<tr><th class="tg-j4kc">Period</th><th class="tg-uqo3">Keywords</th><th class="tg-uqo3">Key sentences</th></tr>'
    even_tr = '\n<tr><td class="tg-baqh">{}<br>~<br>{}</td><td class="tg-0lax">{}</td><td class="tg-0lax">{}</td></tr>'
    odd_tr = '\n<tr><td class="tg-uqo3">{}<br>~<br>{}</td><td class="tg-kftd">{}</td><td class="tg-kftd">{}</td></tr>'
    for i, (b_date, e_date, keywords, keysentences) in enumerate(timeline):
        keysents = ' '.join(keysentences)
        keywords = [w for w, _, _ in keywords]
        if only_keysent_term:
            keywords = [w for w in keywords if w in keysents]
        keywords_str = keyword_to_str(keywords)
        keysents_str = keysent_to_str(keysentences)
        if i % 2 == 0:
            tr += even_tr.format(b_date, e_date, keywords_str, keysents_str)
        else:
            tr += odd_tr.format(b_date, e_date, keywords_str, keysents_str)
    return HTML_TEMPLATE % (tr)

def keyword_to_str(keywords, n_words_in_line=5):
    n = n_words_in_line
    strs = ''.join(w+', ' if i % n != (n-1) else w+'<br>' for i, w in enumerate(keywords)).strip()
    if strs and strs[-1] == ',':
        strs = strs[:-1]
    return strs

def keysent_to_str(keysentences):
    return '<br>'.join('- %s' % sent for sent in keysentences)

HTML_TEMPLATE = """<html><body>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-uqo3{background-color:#efefef;text-align:center;vertical-align:top}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-kftd{background-color:#efefef;text-align:left;vertical-align:top}
.tg .tg-j4kc{background-color:#efefef;text-align:center}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
%s
</table>
</body></html>"""
