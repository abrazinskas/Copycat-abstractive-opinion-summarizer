import numpy as np
from mltoolkit.mldp.utils.tools import DataChunk
import lorem
from nltk import word_tokenize


def create_chunk(size, fname):
    """Creates a data-chunk with one field populated by lorem ipsum text."""
    dc = DataChunk(**{fname: np.zeros(size, dtype='object')})
    for indx in range(size):
        dc[indx, fname] = word_tokenize(lorem.text())
    return dc
