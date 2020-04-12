import numpy as np


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def score_candidates(target_reprs, cand_reprs, score_func):
    """
    Scores every target repr against every candidate repr using a provided
    score function.
    :return: scores: [target_count, candidate_count]
    """
    scores = np.empty((len(target_reprs), len(cand_reprs)), dtype='float32')
    for tar_indx, tar_repr in enumerate(target_reprs):
        for sour_indx, cand_repr in enumerate(cand_reprs):
            scores[tar_indx, sour_indx] = score_func(tar_repr, cand_repr)
    return scores


def select_cands_by_th(slot_scores, th=0.4):
    """
    Selects candidates(indxs) based on their avg. scores exceeding th.

    :param slot_scores: [source_slot_segms_count, cand_segms_count].
    :param th: threshold value.
    :return indices.
    """
    avg_sim = slot_scores.mean(axis=0)
    return np.where(avg_sim > th)


def select_cands_by_rank(slot_scores, top_n=1, order='descending'):
    """
    Selects a fixed number of unique candidates per source seq. by sorting their
    scores in a set order.

    :param slot_scores: [source_slot_segs_count, cand_seqs_count].
    :param top_n: candidates per source segments.
    :param order: descending or ascending.
    :return indices of selected candidates.
    """
    assert order in ['descending', 'ascending']
    if order == 'descending':
        s = - slot_scores
    else:
        s = slot_scores
    indxs = np.argsort(s, axis=1)[:, :top_n]
    indxs = np.unique(indxs.reshape((-1,)))
    return indxs
