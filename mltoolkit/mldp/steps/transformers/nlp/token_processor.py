from mltoolkit.mldp.utils.helpers.validation import validate_field_names
from mltoolkit.mlutils.helpers.general import listify
from mltoolkit.mldp.steps.transformers.base_transformer import BaseTransformer
import numpy as np


class TokenProcessor(BaseTransformer):
    """
    Performs an in-place tokenization of field values (string sequences) without
    creation of new fields. Also, allows to clean tokens, and match them to
    know symbols.

    Works only for 2D data (batch_size x 1D_string_sequences).
    """

    def __init__(self, fnames, tokenization_func=None,
                 token_cleaning_func=None, token_matching_func=None,
                 lower_case=True, **kwargs):
        """
        :param fnames: str or list of string corresponding to fields that
                            should be tokenized.
        :param tokenization_func: a function that splits string sequences into
                                  sequences of tokens. The form should be:
                                  x -> y where x is a str and y is a list/array
                                  of tokens.
        :param token_cleaning_func: the function responsible for normalization
                                    of tokens, elimination of unwanted
                                    characters, etc. format: x -> y, where x is
                                    a str token, and y is a clean str token.
        :param token_matching_func: a function that matches raw text tokens to
                                    to a special set of tokens. E.g. to twitter
                                    emoticons ':)' -> '<POSIT_EMOT>'.
                                    The format: x -> y, where x is a str token,
                                    and y is either False, if it does not match
                                    or a string token otherwise.
        :param lower_case: whether to lower-case strings before tokenization.
        """
        try:
            validate_field_names(fnames)
        except Exception as e:
            raise e
        msg = "Please provide a valid callable %s function."
        if tokenization_func is None:
            tokenization_func = lambda x: x.split()
        if not callable(tokenization_func):
            raise ValueError(msg % "tokenization")
        if token_cleaning_func is not None and not callable(
                token_cleaning_func):
            raise ValueError(msg % "token cleaning")
        if token_matching_func is not None and not callable(
                token_matching_func):
            raise ValueError(msg % "token matching")

        super(TokenProcessor, self).__init__(**kwargs)
        self.field_names = listify(fnames)
        self.tokenization_func = tokenization_func
        self.token_cleaning_func = token_cleaning_func
        self.token_matching_func = token_matching_func
        self.lower_case = lower_case

    def _transform(self, data_chunk):
        """Tokenizes string sequences by converting them to lists of tokens."""
        for fn in self.field_names:
            fv = data_chunk[fn]
            new_fv = []
            for i, field_unit in enumerate(fv):
                tokens = self.tokenization_func(field_unit)
                clean_tokens = []
                for token in tokens:
                    # trying to match raw tokens to special tokens
                    if self.token_matching_func:
                        match_res = self.token_matching_func(token)
                        try:
                            self._validate_token_matching_func_output(match_res)
                        except Exception as e:
                            raise e

                        if match_res:
                            clean_tokens.append(match_res)
                            continue

                    if self.token_cleaning_func:
                        token = self.token_cleaning_func(token)
                        if not token:
                            continue

                        # trying to match raw tokens to special tokens again
                        # because now tokens are clean
                        if self.token_matching_func:
                            match_res = self.token_matching_func(token)
                            try:
                                self._validate_token_matching_func_output(
                                    match_res)
                            except Exception as e:
                                raise e

                            if match_res:
                                clean_tokens.append(match_res)
                                continue
                    if self.lower_case:
                        token = token.lower()
                    if token:
                        clean_tokens.append(token)
                new_fv.append(clean_tokens)
            data_chunk[fn] = new_fv
        return data_chunk

    @staticmethod
    def _validate_token_matching_func_output(output):
        if not (output is False or isinstance(output, str)):
            raise ValueError("The provided token_matching_func is invalid."
                             " It should return either False if a token is not"
                             " matched, a string otherwise.")
