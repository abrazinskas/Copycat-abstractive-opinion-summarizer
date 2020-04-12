from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mldp.utils.tools import DataChunk


class FieldRenamer(BaseTransformer):

    def __init__(self, old_to_new_fnames_map, **kwargs):
        super(FieldRenamer, self).__init__(**kwargs)
        self.old_to_new_fnames = old_to_new_fnames_map

    def _transform(self, data_chunk):
        new_dc = DataChunk()
        for k, v in data_chunk.items():
            if k in self.old_to_new_fnames:
                k = self.old_to_new_fnames[k]
            new_dc[k] = v
        return new_dc
