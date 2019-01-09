import re
import numpy as np


limit_count = 10


def split_comment(data):
    res = []
    for i, l in enumerate(data):
        ident = l[0]
        phrase = l[1]
        res.append((ident, np.asarray(re.findall(r"[\w]+|[.,!?;']", phrase)))) # Sépare ponctution et mots de la séquence
    return res


def mk_vocab(data):
    vocab = {}
    count = {}
    for (ident, p) in data:
        for w in p:
            if w not in count:
                count[w] = 1
            else:
                count[w] += 1
    for (ident, p) in data:
        for w in p:
            if w not in vocab and count[w] >= limit_count:
                vocab[w] = len(vocab)
    return count, vocab


def words_to_index(word_array, vocab, count):
    return np.array([vocab[w] for w in word_array if count[w] >= limit_count])


def pass_data_to_word_idx(data, vocab, count):
    ident_sent = []
    all_sent = []
    for ident, s in data:
        all_sent.append(words_to_index(s, vocab, count))
        ident_sent.append(ident)
    return np.asarray(ident_sent), np.asarray(all_sent)

