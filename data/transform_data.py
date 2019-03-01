import re
import numpy as np


limit_count = 10


padding = "<pad>"


def split_comment(data):
    res = []
    for i, l in enumerate(data):
        ident = l[0]
        phrase = l[1].lower()
        res.append((ident, np.asarray(re.findall(r"[\w]+", phrase)))) # Sépare ponctution et mots de la séquence |[.,!?;']
    return res


def mk_vocab(data):
    vocab = {padding: 0}
    count = {}
    for (ident, p) in data:
        for w in p:
            if w not in count:
                count[w] = 1
            else:
                count[w] += 1
    for (ident, p) in data:
        for w in p:
            if w not in vocab and count[w] > limit_count:
                vocab[w] = len(vocab)
    return count, vocab


def words_to_index(word_array, vocab, count):
    return np.array([vocab[w] for w in word_array if w in vocab and w in count and count[w] > limit_count])


def pass_data_to_word_idx(data, vocab, count):
    ident_sent = []
    all_sent = []
    for ident, s in data:
        all_sent.append(words_to_index(s, vocab, count))
        ident_sent.append(ident)
    return np.asarray(ident_sent), np.asarray(all_sent)


def get_max_len_sent(sentences):
    return max(map(len, sentences))


def pad_sentences(sentences, max_len, pad_idx):
    res = []
    for s in sentences:
        if s.shape[0] < max_len:
            new_s = np.lib.pad(s, (0, max_len - s.shape[0]), 'constant', constant_values=(pad_idx,))
        else:
            # la longueur max sur le test est de plus de 1400 mots...
            new_s = s[:max_len]
        res.append(new_s)
    return np.asarray(res)


def filter_test_data(data, labels):
    tmp = [(d, l) for d, l in zip(data, labels) if min(l) != -1]
    res_data = []
    res_labels = []
    for d, l in tmp:
        res_data.append(d)
        res_labels.append(l)
    return res_data, np.asarray(res_labels)
