from collections import OrderedDict


class DecState(object):
    """
    Standardized decoder state used by higher level decoders, such as the
    beam and gumbel softmax decoders.
    """

    def __init__(self, word_scores=None, rec_vals=None, coll_vals=None):
        """
        :param word_scores: PyTorch tensor with scores, probs, or log probs over
                            next words depending on the decoder being used.
                            For the initial state should not be provided.
        :param rec_vals: a dict of values that should be fed back to the
                         decoder. Dict keys should match the decoder's signature.
        :param coll_vals: a dict of values that should be collected by an external
                          module. Those values can later be used for logging or
                          another analysis.
        """
        if rec_vals is not None:
            assert isinstance(rec_vals, (dict, OrderedDict))
        if coll_vals is not None:
            assert isinstance(coll_vals, (dict, OrderedDict))
        self.word_scores = word_scores
        self.rec_vals = rec_vals
        self.coll_vals = coll_vals
