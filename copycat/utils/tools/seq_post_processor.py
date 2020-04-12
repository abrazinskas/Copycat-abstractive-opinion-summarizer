from logging import getLogger
import os

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class SeqPostProcessor(object):
    """
    Post-processes sequences by performing true-casing and de-tokenization
    to make them look more like human written.
    """

    def __init__(self, tokenizer=None, detokenizer=None, sent_splitter=None,
                 tcaser=None):
        """
        :param tokenizer: if provided will tokenize and concatenate input
                          tokens.
        :param detokenizer: used to make the sequence look more like human
                            written.
        :param sent_splitter: a function that splits a string to sentences.
                              Used for capitalisation of first words if provided.
        :param tcaser: a true casing function.
        """
        super(SeqPostProcessor, self).__init__()
        self.tk = tokenizer
        self.dtk = detokenizer
        self.ss = sent_splitter
        self.tc = tcaser

    def __call__(self, seq):
        """
        :param seq: list of tokens or a string
        """
        if self.tk:
            text = " ".join(self.tk(seq))
        else:
            text = seq

        #   1. true casing
        if self.tc:
            text = self.tc(text)

        #   2. de-tokenizing in order to make it look like a human created text
        if self.dtk:
            text = self.dtk(text.split())

        #   3. capitalizing the first word in each sentence
        if self.ss:
            text = self._capitalize_first_word(text)

        return text

    def _capitalize_first_word(self, my_str):
        """Capitalizes the first word in each sentence."""
        out = " ".join([sent[0].capitalize() + sent[1:] for sent in
                        self.ss(my_str)])
        return out
