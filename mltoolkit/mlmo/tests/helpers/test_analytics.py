import unittest
from mltoolkit.mlmo.utils.helpers.analytics import ngram_seq_analysis
from nltk import sent_tokenize, word_tokenize
from numpy import isclose


class TestAnalytics(unittest.TestCase):

    def test_proportions_ngram_seq_analysis(self):
        """Tests ngram computation of n-gram and sequence proportions."""

        seqs = [
            "one two three. four five fix!",
            "one two. one two. three four",

        ]
        exp_metr = {
            "un_1gr_prop": (1. + 5. / 8.) / 2.,
            "un_2gr_prop": (1. + 3 / 5.) / 2.,
            "un_3gr_prop": (1. + 1. / 2) / 2.,
            "un_sent_prop": (1. + 2. / 3) / 2.,
            "x_and_x_count": 0
        }

        act_metr = ngram_seq_analysis(seqs, sent_splitter=sent_tokenize,
                                      tokenizer=word_tokenize,
                                      n_grams_to_comp=(1, 2, 3))
        act_metr = dict(act_metr)

        for k, v in exp_metr.items():
            self.assertTrue(k in act_metr)
            self.assertTrue(isclose(v, act_metr[k]))

    def test_x_and_x_ngram_seq_analysis(self):
        """Tests ngram x_and_x patterns"""

        seqs = [
            "good and good.",
            "it was good",
            "the weather was wonderful and wonderful."

        ]
        exp_metr = {
            "x_and_x_count": 2
        }

        act_metr = ngram_seq_analysis(seqs, sent_splitter=sent_tokenize,
                                      tokenizer=word_tokenize)
        act_metr = dict(act_metr)

        for k, v in exp_metr.items():
            self.assertTrue(k in act_metr)
            self.assertTrue(isclose(v, act_metr[k]))


if __name__ == '__main__':
    unittest.main()
