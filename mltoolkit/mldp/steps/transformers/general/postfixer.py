from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np


class Postfixer(BaseTransformer):
    """Adds a special postfix to each entry of a field.

    The step records the unique identifiers in chunks, and adds a count postfix
    to each data unit's `id_fname`. Assumes that data-units contain the same
    `id_fname` values.
    """

    def __init__(self, id_fname, **kwargs):
        """
        :param id_fname: unique identifier for an entity, e.g., `group_id` or
            `product_id`.
        """
        super(Postfixer, self).__init__(**kwargs)
        self.id_fname = id_fname
        self._id_to_count = {}

    def _transform(self, data_chunk):
        id_fvals = data_chunk[self.id_fname]
        assert len(np.unique(id_fvals)) == 1

        id = data_chunk[0, self.id_fname]
        if id not in self._id_to_count:
            self._id_to_count[id] = 0
        self._id_to_count[id] += 1

        count = self._id_to_count[id]
        new_id = "%s_%d" % (id, count)

        for indx in range(len(id_fvals)):
            data_chunk[indx, self.id_fname] = new_id

        return data_chunk
