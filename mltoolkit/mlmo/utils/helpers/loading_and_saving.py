import pickle
from pickle import UnpicklingError
from logging import getLogger
import numpy as np
from numpy import random
import codecs

random.seed(42)
logger = getLogger(__name__)


def load_params_from_pkl(params_dump_file_path):
    """
    Loads parameters from a pickle _dump file.

    :param params_dump_file_path: self-explanatory
    :return dict of param_name => param
    """
    coll = {}
    f = open(params_dump_file_path, 'rb')
    while True:
        try:
            param_name, param_val = pickle.load(f)
            coll[param_name] = param_val
        except (EOFError, UnpicklingError):
            break
    f.close()
    return coll


def load_embeddings(embeddings_file_path, vocab, init_type='uniform',
                    dtype='float32', encoding='utf-8'):
    """
    Initializes and input matrix with embeddings e.g. from word2vec.
    Assuming that not all words will be present in the embeddings file, so
    the matrix with be initialized with noise.

    :param embeddings_file_path: file_path of an embeddings file in the format:
                            (name \space embedding).
    :param vocab: vocabulary that allows to access assigned token ids.
    :param init_type: normal or uniform (str).
    :param dtype: desired type of return numpy matrix.
    :param encoding: encoding of the input file.
    :return: numpy 2D matrix.

    """
    # will init emb. matr. after reading the first word embedding (need dims)
    if init_type not in ['uniform', 'normal']:
        raise ValueError("Please provide a valid 'init_type' ('uniform' or"
                         " 'normal').")

    init_func = random.normal if init_type == 'normal' else random.uniform
    embeddings = None
    matches = 0

    with codecs.open(embeddings_file_path, encoding=encoding) as f:
        for line in f:
            parts = line.strip().split(" ")
            token, embedding = parts[0], parts[1:]

            # init the emb. container
            if embeddings is None:
                embeddings = init_func(size=(len(vocab), len(embedding)))
                embeddings = embeddings.astype(dtype=dtype)

            if token in vocab:
                matches += 1
                word_id = vocab[token].id
                embeddings[word_id] = np.array(embedding, dtype=dtype)

    match_prop = 100. * matches / float(len(vocab))

    logger.info("Loaded embeddings from '%s'" % embeddings_file_path)
    logger.info('Filled %.3f %% of the current embeddings' % match_prop)

    return embeddings


def save(obj, file_path):
    file = open(file_path, 'wb+')
    pickle.dump(obj=obj, file=file, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_path):
    file = open(file_path, 'rb+')
    return pickle.load(file)
