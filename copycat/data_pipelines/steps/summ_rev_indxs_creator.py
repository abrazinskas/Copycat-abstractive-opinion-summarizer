from mltoolkit.mldp.steps.transformers import BaseTransformer
from copycat.utils.helpers.data import group_vals_by_keys
from mltoolkit.mldp.utils.helpers.nlp.sequences import pad_sequences as pad_seqs
from copycat.utils.fields import ModelF
import numpy as np


class SummRevIndxsCreator(BaseTransformer):
    """Creates the necessary index fields to perform summarization of reviews.

    Creates special fields:
        1. GROUP_REV_INDXS: padded indices of data-units that belong
                           to the same group, and thus should be summarizer
                           all together.
        2. GROUP_REV_INDXS_MASK: contains binary value for masking dummy reviews
                                of groups that don't have the maximum number
                                of reviews.
        3. SUMM_PRODUCT_ID: self-explanatory.
        4. SUMM_CATEGORY: categories of groups that are summarized.

        Preserves the sequential order present in data-chunks.

    Makes data-chunks 'invalid' as summaries number will differ from the reviews
    number.
    """

    def __init__(self, group_id_fname, category_fname, **kwargs):
        super(SummRevIndxsCreator, self).__init__(**kwargs)
        self.group_id_fname = group_id_fname
        self.category_fname = category_fname

    def _transform(self, data_chunk):
        group_ids = data_chunk[self.group_id_fname]
        categories = data_chunk[self.category_fname]

        group_map = group_vals_by_keys(range(len(group_ids)), group_ids)
        gr_ids = list(group_map.keys())
        gr_indxs = list(group_map.values())

        # the gr_indxs are indxs reviews that belong to the same group
        padded_indxs, mask = pad_seqs(gr_indxs, pad_symbol=0,
                                      padding_mode='right')

        prod_to_cat = group_vals_by_keys(categories, group_ids)
        summ_cats = [prod_to_cat[gr_id][0] for gr_id in gr_ids]

        data_chunk[ModelF.SUMM_CAT] = np.array(summ_cats)
        data_chunk[ModelF.SUMM_GROUP_ID] = np.array(gr_ids)
        data_chunk[ModelF.GROUP_REV_INDXS] = padded_indxs
        data_chunk[ModelF.GROUP_REV_INDXS_MASK] = mask

        return data_chunk
