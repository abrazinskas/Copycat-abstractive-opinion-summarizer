from mltoolkit.mldp.steps.transformers import BaseTransformer
from copycat.utils.helpers.data import group_vals_by_keys
from mltoolkit.mldp.utils.helpers.nlp.sequences import pad_sequences as pad_seqs
from copycat.utils.fields import ModelF


class GoldSummRevIndxsCreator(BaseTransformer):
    """
    The step is specific to data that has golden summaries, which are passed
    along the pipeline. E.g., Yelp or Amazon gold datasets.

    It will align golden summaries to reviews (by creating indxs), which come
    sorted, while the golden summaries retain their original order.

    I can't reuse a similar step (prod_rev_indxs_creator) because it produces
    summaries by scanning prod_ids of reviews, which come sorted. This step
    iterates over SUMM_GROUP_ID field and grabs reviews that belong to that group.

    If I do the same operation for indxs creation as I do when summaries are not
    available, then I might produce alignment mismatches. In order words,
    I would not be able to tell which summaries are first, which are second
    based on sorted prod_ids. Because golden summaries are not sorted!

    Creates special fields:
        1. GROUP_REV_INDXS: that contain indices (padded) of data units that
                            belong to the same group
        2. GROUP_REV_INDXS_MASK: padding mask for those data-units.
    """

    def __init__(self, group_id_fname, **kwargs):
        super(GoldSummRevIndxsCreator, self).__init__(**kwargs)
        self.group_id_fname = group_id_fname

    def _transform(self, data_chunk):
        prod_ids = data_chunk[self.group_id_fname]
        summ_prod_ids = data_chunk[ModelF.SUMM_GROUP_ID]

        groups = group_vals_by_keys(range(len(prod_ids)), prod_ids)

        aligned_rev_indxs = []
        for summ_group_id in summ_prod_ids:
            aligned_rev_indxs.append(groups[summ_group_id])

        # the gr_indxs are indxs reviews that belong to the same group
        padded_rev_indxs, mask = pad_seqs(aligned_rev_indxs, pad_symbol=0)

        data_chunk[ModelF.GROUP_REV_INDXS] = padded_rev_indxs
        data_chunk[ModelF.GROUP_REV_INDXS_MASK] = mask

        return data_chunk
