from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mlutils.helpers.general import sort_hash, flatten, listify
from mltoolkit.mldp.utils.helpers.validation import validate_field_names
from mltoolkit.mlutils.tools import SignedObject
from mltoolkit.mldp.utils.constants.vocabulary import PAD, UNK, START, END
import itertools
import logging
import codecs
import os
import numpy as np

try:
    import re2 as re
except ImportError:
    import re
logger_name = 'vocabulary'
logger = logging.getLogger(logger_name)

DEFAULT_SPECIAL_TOKENS = {PAD, UNK, START, END}


class _Symbol:
    def __init__(self, token, id, count):
        self.token = token
        self.id = id
        self.count = count


class Vocabulary(SignedObject):
    """
    A general purpose vocabulary class that allows to efficiently create and
    access symbol objects.
    Each symbol object has:
                          1. id - unique integer identifier
                          2. token - string representation of the token. E.g. it 
                                     could be the string name of a label or
                                     letters of a word.
                          3. count - the number of times a token was observed.
    Symbols can be accessed by querying based on 'id' or 'token'. Creation of
    vocabularies is based on data-chunk iterables, such as a reader or a full
    data pipeline.
    """

    def __init__(self, data_chunk_iterable=None, name_prefix="",
                 special_tokens=None):
        """
        :param data_chunk_iterable: self-explanatory, do not provide if not
                                    planning to create vocabs but rather load.
        :param name_prefix: will be added to msg title in  __str__ out if
                            provided.
        :param special_tokens: list of special tokes/symbols to add to top of
            vocabulary.
        """
        if data_chunk_iterable is not None and \
                not hasattr(data_chunk_iterable, "iter"):
            raise ValueError("Please pass a valid data chunk iterable. It has"
                             " to have the iter() method.")
        super(Vocabulary, self).__init__(name_prefix=name_prefix)
        self._data_chunk_iterable = data_chunk_iterable
        # below are internal vars
        self._total_count = 0
        self._id_to_symbol = []
        self._token_to_symbol = {}
        self.special_symbols = {}
        # adjusting scraper
        self.scraper.scrape_obj_vals = False

        if special_tokens is not None:
            for ss in special_tokens:
                self.add_special_symbol(ss)

    def load(self, vocab_file_path, min_count=1, max_size=None,
             add_default_special_symbols=True, sep=' ', encoding='utf-8'):
        """
        Loads vocabulary from a saved file, where the format is assumed to be
        {token}{sep}{count}.

        :param vocab_file_path: self-explanatory.
        :param min_count: the minimum frequency of a token to be loaded.
        :param max_size: maximum number of symbols to load from a file.
        :param add_default_special_symbols: whether after loading add default
                                            symbols, such as <PAD> and <UNK>.
        :param sep: separator in the input file between tokens and counts.
        :param encoding: self-explanatory.
        """
        logger.info("Loading a vocabulary from '%s'. min_count: %d,"
                    " max_vocab_size: %s." % (vocab_file_path, min_count,
                                              str(max_size)))
        with codecs.open(vocab_file_path, encoding=encoding) as f:
            for i, entry_line in enumerate(f):
                if max_size and i >= max_size:
                    break

                # NOTE: I use rstrip below because it causes issues when vocab
                # entry has an empty space as character.
                fields = entry_line.rstrip().split(sep)
                if len(fields) > 2:
                    raise ValueError("The file: '%s' has an incorrect"
                                     " structure, as one or more entries have "
                                     "more than two attributes. It must be in"
                                     " the format {token}{sep}{count}. "
                                     "Alternatively the passed separator '%s'"
                                     " is wrong." % (vocab_file_path, sep))
                token, count = fields[0], int(fields[1])

                if count >= min_count:
                    symbol = self.add_symbol(token, count=count)
                    if match_special_symbol(token):
                        self.special_symbols[token] = symbol

        if add_default_special_symbols:
            self.add_special_symbols(DEFAULT_SPECIAL_TOKENS)
        logger.info("Loaded the vocabulary.")

    def create(self, data_source, data_fnames, min_count=1, max_size=None,
               add_default_special_symbols=True):
        """
        Create vocabulary by passing data_source to the corresponding data-chunk
        iterable and fetching chunks out of it.

        Assumes that tokens are strings, if they are not, it will try to convert
        them to strings.

        :param data_source: dictionary of attributes that should be passed to
                            the data_chunk_iterable.
        :param data_fnames: String or List of (string) attributes that map
                            to the symbols which should be used to create
                            the vocabulary.
        :param min_count: minimum frequency of a token to be added to the
                          vocabulary.
        :param max_size: maximum number of symbols to store to the vocabulary.
        :param add_default_special_symbols: whether default symbols,
                                    such as <PAD> and <UNK> should be added.
                                    In some cases, e.g. labels vocab
                                    those symbols are not necessary.
        """
        try:
            validate_field_names(data_fnames)
        except Exception as e:
            raise e

        data_fnames = listify(data_fnames)
        dfn_formatted_str = ', '.join(["'%s'" % dfn for dfn in data_fnames])
        logger.info("Creating a vocabulary from %s data_source, and %s"
                    " chunk field(s). min_count: %d, max_vocab_size: %s." %
                    (data_source,
                     dfn_formatted_str,
                     min_count,
                     str(max_size)))
        temp_token_to_count = {}
        for data_chunk in self._data_chunk_iterable.iter(**data_source):
            for data_attr in data_fnames:
                for tokens in data_chunk[data_attr]:

                    if not isinstance(tokens, (list, np.ndarray)):
                        tokens = [tokens]

                    for token in flatten(tokens):
                        if token == '':
                            continue

                        if not isinstance(token, (int, float, str)):
                            raise TypeError("Token is not of a correct type"
                                            " (should be int, float, str,"
                                            " unicode).")

                        if isinstance(token, (int, float)):
                            token = str(token)

                        if token not in temp_token_to_count:
                            temp_token_to_count[token] = 0
                        temp_token_to_count[token] += 1

        # populate the collectors
        for token, count in sort_hash(temp_token_to_count, by_key=False):
            if max_size and len(self) >= max_size:
                break
            if count >= min_count:
                symbol = self.add_symbol(token, count)
                self._total_count += count
                if match_special_symbol(token):
                    self.special_symbols[token] = symbol
        if add_default_special_symbols:
            self.add_special_symbols(DEFAULT_SPECIAL_TOKENS)

        logger.info("Created the vocabulary.")
        logging.info("Total word count: %d." % self._total_count)
        logging.info("Vocab size: %d." % len(self))

    def write(self, file_path, sep=' ', encoding='utf-8'):
        """
        Writes the vocabulary to a plain text file where each line is of the
        form: {token}{sep}{count}. Default special symbols are not written.

        :param file_path: self-explanatory.
        :param sep: self-explanatory.
        :param encoding: self-explanatory.
        """
        safe_mkfdir(file_path)
        with codecs.open(file_path, 'w', encoding=encoding) as f:
            for symbol in self:
                token = symbol.token
                count = str(symbol.count)
                try:
                    str_entry = sep.join([token, count])
                    f.write(str_entry)
                    f.write("\n")
                except Exception:
                    logger.fatal(
                        "Below entry produced a fatal error in write().")
                    logger.fatal(symbol.token)
                    raise ValueError("Could not process a token.")
        logger.info("Vocabulary is written to: '%s'." % file_path)

    def load_or_create(self, vocab_file_path, data_source, data_fnames,
                       min_count=1, max_size=None,
                       add_default_special_symbols=True, sep=" ",
                       encoding='utf-8'):
        """
        A convenience function that either creates a vocabulary or loads it if
        it already exists.
        """
        if os.path.isfile(vocab_file_path):
            self.load(vocab_file_path, min_count=min_count, max_size=max_size,
                      add_default_special_symbols=add_default_special_symbols,
                      sep=sep, encoding=encoding)
        else:
            self.create(data_source=data_source, max_size=max_size,
                        add_default_special_symbols=add_default_special_symbols,
                        data_fnames=data_fnames, min_count=min_count)
            self.write(vocab_file_path, sep=sep, encoding=encoding)

    def get_sign_attrs(self):
        attrs = self.scraper.scrape(self)
        attrs['vocab_size'] = len(self)
        return attrs

    @property
    def size(self):
        return len(self)

    def __len__(self):
        return len(self._id_to_symbol)

    def __contains__(self, entry):
        """
        A generic containment function that assumes item to be either int, str,
        or a symbol object.
        """
        if isinstance(entry, _Symbol):
            return entry.token in self._token_to_symbol
        if isinstance(entry, str):
            return entry in self._token_to_symbol
        if isinstance(entry, int):
            return len(self._id_to_symbol) > entry
        raise ValueError('Input argument is not of a correct type.')

    def __iter__(self):
        for symbol in self._id_to_symbol:
            yield symbol

    def __getitem__(self, entry):
        """
        A generic get method that will return <UNK> special symbol object if
        word(s) are not in the vocabulary (unless the special symbol does not
        exist).

        :param entry: either an int (id) or string (token) or a list of
                      str (tokens).
        :return: symbol object.
        """
        if isinstance(entry, str):
            if entry in self:
                return self._token_to_symbol[entry]
            else:
                if UNK in self:
                    return self[UNK]
                else:
                    raise ValueError(
                        "Item '%s' is not present in the vocabulary." % (
                            str(entry)))
        if isinstance(entry, (int, np.integer)):
            return self._id_to_symbol[entry]
        if isinstance(entry, (list, np.ndarray)):
            return [self[w] for w in entry]
        logger.fatal("Below entry produced a fatal error in __getitem__().")
        logger.fatal(type(entry))
        logger.fatal(entry)
        raise ValueError('Input argument is not of a correct type.')

    def add_symbol(self, token, count=1):
        """
        Adds an entry to the collection or updates its count if it already
        present. TODO: think if updating makes sense.

        :return: created or updated symbol object.
        """
        if not isinstance(token, str):
            raise TypeError("Token must be a string!")
        if token in self:
            symbol = self[token]
            self._total_count += count
            self._total_count -= symbol.count
            symbol.count = count
        else:
            n = len(self._token_to_symbol)
            symbol = _Symbol(token, id=n, count=count)
            self._token_to_symbol[token] = symbol
            self._id_to_symbol.append(symbol)
            self._total_count += count
        return symbol

    def add_special_symbol(self, token, count=0):
        """Appends or updates a special symbol."""
        count = self[token].count if token in self else count
        self.special_symbols[token] = self.add_symbol(token, count)

    def add_special_symbols(self, special_symbols):
        """Appends or updates special symbols."""
        for token in special_symbols:
            self.add_special_symbol(token)


def match_special_symbol(token):
    """Checks whether the passed token matches the special symbols format."""
    return re.match(r'<[A-Z]+>', token)
