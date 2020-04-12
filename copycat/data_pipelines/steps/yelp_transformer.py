from mltoolkit.mldp.steps.transformers import BaseTransformer
from copycat.utils.fields import ModelF, YelpEvalF
from mltoolkit.mldp.utils.tools import DataChunk


class YelpTransformer(BaseTransformer):
    """Specific transformer for the YELP evaluation data.

    It performs the following operations:
        1. splits data-units by reviews. I.e. each data-unit has one review.
        2. creates dummy SUMMARY_CATEGORY, and CATEGORY not to break the later
           logic in the dev. interface.
        3. creates SUMMARY_GROUP_ID that stores the id of the business that
           summary belongs to. (Later used in the dev. interface)
        4. wraps each summary by a list as there is one summary per business

    Produces 'invalid' data-chunks as the number of summaries will be different
    from the number of input reviews.
    """

    def _transform(self, data_chunk):
        fields_to_copy = [YelpEvalF.BUSINESS_ID]
        new_dc = DataChunk(**{fn: [] for fn in fields_to_copy})

        summ_cats = ["no_cat" for _ in range(len(data_chunk))]

        # wrapping each summary to a list as there is only one summary per
        # business
        new_dc[ModelF.SUMMS] = [[summ] for summ in data_chunk[YelpEvalF.SUMM]]

        new_dc[ModelF.SUMM_CAT] = summ_cats
        new_dc[ModelF.SUMM_GROUP_ID] = data_chunk[YelpEvalF.BUSINESS_ID]

        # splitting data-units by the reviews field. I.e. each unit will have
        # one review associated with it

        new_dc[ModelF.REV] = []
        for du in data_chunk.iter():
            for rev_fn in YelpEvalF.REVS:
                new_dc[ModelF.REV].append(du[rev_fn])
                # copying the rest
                for c_fn in fields_to_copy:
                    new_dc[c_fn].append(du[c_fn])

        # adding dummy category field
        cat_fvals = ["no_cat" for _ in range(len(new_dc))]
        new_dc[ModelF.CAT] = cat_fvals

        return new_dc
