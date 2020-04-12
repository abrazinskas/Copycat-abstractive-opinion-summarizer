from mltoolkit.mldp.steps.transformers.base_transformer import BaseTransformer
from mltoolkit.mldp.utils.helpers.validation import validate_field_names
from mltoolkit.mlutils.helpers.general import listify


class FieldSelector(BaseTransformer):
    """
    Preserves a subset of fields from data-chunk, drops the rest in-place.
    """

    def __init__(self, fnames, **kwargs):
        """
        :param fnames: str or list of str names that should represent
                            fields that should be selected from data-chunks.
                            Other fields are discarded.
        """
        try:
            validate_field_names(fnames)
        except Exception as e:
            raise e

        super(FieldSelector, self).__init__(**kwargs)
        self.fnames = listify(fnames)

    def _transform(self, data_chunk):
        fn_set = set(self.fnames)
        for fn in fn_set:
            if fn not in data_chunk:
                raise ValueError(
                    "The data-chunk does not have the `%s` field." % fn)
        fnames_to_del = [fn for fn in data_chunk if fn not in fn_set]
        for fn in fnames_to_del:
            del data_chunk[fn]
        return data_chunk
