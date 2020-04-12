from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mldp.utils.tools import DataChunk
from copycat.utils.fields import AmazonEvalF, ModelF


class AmazonTransformer(BaseTransformer):
    """
    Almost exactly the same as YelpTransformer but here I take into account that
    each product has three summaries, and that categories are given.
    """

    def __init__(self):
        super(AmazonTransformer, self).__init__()

    def _transform(self, data_chunk):
        fields_to_copy = [AmazonEvalF.PROD_ID, AmazonEvalF.CAT]
        new_dc = DataChunk(**{fn: [] for fn in fields_to_copy})

        new_dc[ModelF.SUMMS] = [[summ1, summ2, summ3]
                                for summ1, summ2, summ3 in
                                zip(data_chunk[AmazonEvalF.SUMM1],
                                    data_chunk[AmazonEvalF.SUMM2],
                                    data_chunk[AmazonEvalF.SUMM3])]

        new_dc[ModelF.SUMM_CAT] = data_chunk[AmazonEvalF.CAT]
        new_dc[ModelF.SUMM_GROUP_ID] = data_chunk[AmazonEvalF.PROD_ID]

        # splitting data-units by the reviews field. I.e. each unit will have
        # one review associated with it

        new_dc[ModelF.REV] = []
        for du in data_chunk.iter():
            for rev_fn in AmazonEvalF.REVS:
                new_dc[ModelF.REV].append(du[rev_fn])
                # copying the rest
                for c_fn in fields_to_copy:
                    new_dc[c_fn].append(du[c_fn])

        return new_dc
