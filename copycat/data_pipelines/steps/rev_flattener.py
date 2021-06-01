from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mldp.utils.tools import DataChunk
from copycat.utils.fields.model import ModelF
from copy import copy


class ReviewFlattener(BaseTransformer):
    """Flattens data-units which have multiple reviews per column.
    Retains the original `group_id`. This step is used in inference.
    """

    def __init__(self, group_id_fname, rev_fnames):
        super(ReviewFlattener, self).__init__()
        self.group_id_fname = group_id_fname
        self.rev_fnames = rev_fnames

    def _transform(self, data_chunk):
        new_dc = DataChunk()

        new_dc[ModelF.SUMM_GROUP_ID] = copy(data_chunk[self.group_id_fname])
        new_dc[ModelF.GROUP_ID] = []
        new_dc[ModelF.REV] = []
        for du in data_chunk.iter():
            for rev_fn in self.rev_fnames:
                new_dc[ModelF.GROUP_ID].append(du[self.group_id_fname])
                new_dc[ModelF.REV].append(du[rev_fn])
        return new_dc
