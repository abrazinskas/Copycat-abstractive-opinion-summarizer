import unittest
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor
from mltoolkit.mldp.utils.tools import DataChunk
import numpy as np
import re


class TestTokenProcessor(unittest.TestCase):

    def test_output(self):
        """
        Testing a simple scenario when a token matching function, cleaner, and a
        simple token splitter are used.
        """
        field_name = "dummy"
        special_token = "<ANIMAL>"
        lower_case = True
        tok_mat_func = lambda x: token_matching_func(x, special_token)
        token_cleaning_func = lambda x: re.sub(r'[?!,.]', '', x)
        tokenization_func = lambda x: x.split()

        input_seqs = ["Hello, this is my dog!",
                      "A dummy sentence for tokenization.",
                      "What a lovely puppy!"]
        input_data_chunk = DataChunk(**{field_name: input_seqs})
        expect_seqs = [["hello", "this", "is", "my", special_token],
                       ["a", "dummy", "sentence", "for",
                        "tokenization"],
                       ["what", "a", "lovely", special_token]]
        expected_data_chunk = DataChunk(**{field_name: expect_seqs})

        tokenizer = TokenProcessor(field_name,
                                   tokenization_func=tokenization_func,
                                   token_cleaning_func=token_cleaning_func,
                                   token_matching_func=tok_mat_func,
                                   lower_case=lower_case)
        actual_data_chunk = tokenizer(input_data_chunk)
        self.assertTrue(expected_data_chunk == actual_data_chunk)


def token_matching_func(x, special_token):
    return special_token if x in ["dog", "puppy"] else False


if __name__ == '__main__':
    unittest.main()
