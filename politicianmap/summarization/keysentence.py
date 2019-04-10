from politicianmap.tokenizer import recover_rtokenized_sent


def select_keysentence(keywords, docs, topk=5):
    def scoring(sent):
        words = sent.split()
        if not (5 <= len(words) <= 15):
            return 0
        score = 0
        for word in words:
            score += word_to_score.get(word, 0)
        return score

    word_to_score = {word:score for word, score, _ in keywords}
    sent_score = [(sent, scoring(sent)) for doc in docs for sent in doc.split('  ')]
    best_sents = [sent for sent, _ in sorted(sent_score, key=lambda x:-x[1])][:topk]
    best_sents = [recover_rtokenized_sent(sent) for sent in best_sents]
    return best_sents