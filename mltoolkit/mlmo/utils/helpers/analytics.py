from nltk.util import ngrams as compute_ngrams
import numpy as np
from collections import OrderedDict

X_AND_X_PROP = 'x_and_x_prop'
X_AND_X_COUNT = 'x_and_x_count'
UN_SENT_PROP_PROP = 'un_sent_prop'
AVG_SEQ_LEN = 'avg_seq_len'
UN_SENTS = 'un_sents'
TOTAL_SENTS = 'total_sents'


def ngram_seq_analysis(seqs, tokenizer, sent_splitter,
                       n_grams_to_comp=(2, 3, 4, 5)):
    """
    Performs sequence repetition analytics based on:
        1. Unique N-grams proportion
        2. Unique sentences proportion
        3. X and X pattern (e.g. good and good) - the count of detected patterns

    At the moment the analytics are mainly on the level of individual sequences.
    N-grams are computed considering sentences.

    :param seqs: list/array of sequence strings.
    :param tokenizer: function for splitting strings to list of tokens.
    :param sent_splitter: function for splitting strings to list of sentence
                          strings.
    :param n_grams_to_comp: what n-grams to consider for analysis.
    :return: list with tuples containing aggregated over the number of sequences
             stats.
    """
    n_gram_str_fn = lambda x: "un_%dgr_prop" % x
    seqs_sents = [sent_splitter(seq_sents_tokens) for seq_sents_tokens in seqs]
    # seqs_sents_tokens is a triple nested list
    seqs_sents_tokens = [[tokenizer(sent) for sent in sents] for sents
                         in seqs_sents]
    # for each sequence it's the number of unique n-grams / total n-grams
    stats = OrderedDict()
    for ngr in n_grams_to_comp:
        stats[n_gram_str_fn(ngr)] = []

    # special repetition pattern observed in the generated sequences
    stats[X_AND_X_PROP] = []

    total_seq_len = 0.
    for seq_sents_tokens in seqs_sents_tokens:

        # n-gram related statistics
        for ngr in n_grams_to_comp:
            n_grams = []
            for sent_toks in seq_sents_tokens:
                n_grams += list(compute_ngrams(sent_toks, ngr))
            avg_un_ngrams = float(len(set(n_grams))) / len(n_grams) if len(
                n_grams) > 0 else 0.
            stats[n_gram_str_fn(ngr)].append(avg_un_ngrams)

        # x and x patterns and seq lens
        x_and_x_count = 0
        for sent_toks in seq_sents_tokens:
            x_and_x_count += count_x_and_x_patterns(sent_toks)
            total_seq_len += len(sent_toks)
        stats[X_AND_X_PROP].append(x_and_x_count)

    # computing sentence related analytics
    stats[UN_SENT_PROP_PROP] = []
    total_un_sents = 0
    total_sents = 0
    for seq_sents in seqs_sents:
        # remove the last ./!/? if it's present
        un_sents = set()
        for sent in seq_sents:
            if sent[-1] in [".", "!", "?"]:
                sent = sent[:-1]
            un_sents.add(sent)
        total_un_sents += len(un_sents)
        total_sents += len(seq_sents)
        avg_un_sents_prop = float(len(un_sents)) / len(seq_sents) if len(
            seq_sents) > 0 else 0.
        stats[UN_SENT_PROP_PROP].append(avg_un_sents_prop)

    # averaging over the number of seqs
    res = [(k, np.mean(v)) for k, v in stats.items()]

    # extra stats
    res.append((UN_SENTS, total_un_sents))
    res.append((TOTAL_SENTS, total_sents))
    res.append((AVG_SEQ_LEN, total_seq_len / len(seqs)))
    res.append((X_AND_X_COUNT, np.sum(stats[X_AND_X_PROP])))

    return res


def count_x_and_x_patterns(tokens):
    x_and_x_count = 0
    for i in range(len(tokens)):
        if i != 0 and i + 1 < len(tokens) and tokens[i] == "and" and \
                tokens[i - 1] == tokens[i + 1]:
            x_and_x_count += 1
    return x_and_x_count
