from mltoolkit.mldp.steps.transformers import BaseTransformer
from copycat.utils.fields import ModelF
from copycat.utils.helpers.data import reverse_mapping, \
    get_all_but_one_rev_indxs, \
    create_optimized_selector
import numpy as np


class RevMapper(BaseTransformer):
    """Creates index mappings for each review to other reviews.

        1. REV_TO_GROUP: mapping from reviews to their original groups.
        2. OTHER_REVS: mapping from reviews to other reviews(indxs) that belong
                        to the same group.
        3. OTHER_REVS_MASK: masking of review indxs if some groups have
                             unequal number of units in them.
        4. OTHER_REV_STATES: indxs of non-masked states of concatenated review
                             states. This one is used to select more compact
                             tensor for attention.
        5. OTHER_REV_STATES_MASK: mask for the above that cancels indxs of dummy
                                  states that are used as padding.
    """

    def __init__(self, rev_mask_fname, group_rev_indxs_fname,
                 group_rev_mask_fname, **kwargs):
        super(RevMapper, self).__init__(**kwargs)
        self.rev_mask_fname = rev_mask_fname
        self.group_rev_indxs_fname = group_rev_indxs_fname
        self.group_rev_mask_fname = group_rev_mask_fname

    def _transform(self, data_chunk):
        revs_mask = data_chunk[self.rev_mask_fname]
        group_rev_indxs = data_chunk[self.group_rev_indxs_fname]
        group_revs_mask = data_chunk[self.group_rev_mask_fname]

        # reversing mapping
        rev_to_group = reverse_mapping(group_rev_indxs, group_revs_mask)
        data_chunk[ModelF.REV_TO_GROUP_INDX] = rev_to_group

        # creating an optimized selector of states and word ids for attention
        rev_to_others, \
        rev_to_others_mask = get_all_but_one_rev_indxs(group_rev_indxs,
                                                       group_revs_mask)

        data_chunk[ModelF.OTHER_REV_INDXS] = rev_to_others
        data_chunk[ModelF.OTHER_REV_INDXS_MASK] = rev_to_others_mask

        other_revs_mask = revs_mask[rev_to_others] * rev_to_others_mask[:, :,
                                                     np.newaxis]
        other_revs_mask = other_revs_mask.reshape(
            (other_revs_mask.shape[0], -1))

        other_states, other_states_mask = create_optimized_selector(
            other_revs_mask)
        data_chunk[ModelF.OTHER_REV_COMP_STATES] = other_states
        data_chunk[ModelF.OTHER_REV_COMP_STATES_MASK] = other_states_mask

        return data_chunk
