from mltoolkit.mldp.steps.transformers.base_transformer import BaseTransformer
from mltoolkit.mldp.utils.tools import Vocabulary
from collections import OrderedDict
from mltoolkit.mldp.utils.helpers.validation import validate_field_names_mapping
import numpy as np


class VocabMapper(BaseTransformer):
    """
    Maps in-place field values to vocabulary symbol attributes without creation
    of new fields.

    For example, it can be used to convert str tokens to vocabulary ids and
    vice-versa.

    Assumes that field values are np.arrays with multi-dimensional data
    Namely, lists, tuples, or np.arrays or plain ints, strings.

    For multi-dimensional data, it will try to merge it into numpy arrays
    so the output shape might be different from the input shape.
    """

    def __init__(self, field_names_to_vocabs, symbols_attr='id', **kwargs):
        """
        :param field_names_to_vocabs: dict of mappings to vocab objects.
        :param symbols_attr: what attribute to select from vocabulary symbol
                             objects. E.g. it can be 'token' or 'id'.
        """
        try:
            validate_field_names_mapping(field_names_to_vocabs, Vocabulary)
        except Exception as e:
            raise e

        super(VocabMapper, self).__init__(**kwargs)
        self.symbols_attr = symbols_attr
        self.field_names_to_vocabs = field_names_to_vocabs

    def _transform(self, data_chunk):
        for fn, vocab in self.field_names_to_vocabs.items():
            data_chunk[fn] = self._map_rec(data_chunk[fn], vocab)
        return data_chunk

    def _map_rec(self, fv, vocab):
        """Recursively Maps field values to vocabulary symbol attributes."""
        if isinstance(fv, (list, np.ndarray)):
            tmp = [None] * len(fv)
        elif isinstance(fv, tuple):
            tmp = tuple([None] * len(fv))
        else:
            return getattr(vocab[fv], self.symbols_attr)
        for indx in range(len(fv)):
            tmp[indx] = self._map_rec(fv[indx], vocab)
        if isinstance(fv, np.ndarray):
            try:
                tmp = np.array(tmp)
            except ValueError:
                # sometimes there are issues with arrays merging
                new_tmp = np.empty(len(tmp), dtype="object")
                for i in range(len(tmp)):
                    new_tmp[i] = tmp[i]
                return new_tmp
        return tmp

    def get_sign_attrs(self):
        attrs = self.scraper.scrape(self)
        field_names_to_vocabs = {k: "".join(v.get_title().split(" "))
                                 for k, v in self.field_names_to_vocabs.items()}
        attrs['field_names_to_vocabs'] = field_names_to_vocabs
        return attrs
