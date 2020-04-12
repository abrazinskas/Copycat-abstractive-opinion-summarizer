from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mlutils.helpers.general import argsort
import numpy as np


class ChunkSorter(BaseTransformer):
    """Sorts data-chunks by one field."""

    def __init__(self, field_name, order='descending', fields_to_sort=None,
                 **kwargs):
        """
        :param field_name: name of the target/key field, based on which other
                           fields will be sorted.
        :param order: the order of sorting.
        :param fields_to_sort: if a list of field names is provided, will only
                               sort those specific fields. All otherwise.
        """
        if order not in ['ascending', 'descending']:
            raise ValueError("Please provide a valid 'order' param"
                             " ('ascending' or 'descending').")
        if fields_to_sort is not None and not isinstance(fields_to_sort, list):
            raise TypeError("Please provide a 'list' of field names to be "
                            "sorted based on the target field. Otherwise set "
                            "'fields_to_sort' to 'None'.")
        super(ChunkSorter, self).__init__(**kwargs)
        self.field_name = field_name
        self.order = order
        self.fields_to_sort = fields_to_sort
        if isinstance(self.fields_to_sort, list):
            self.fields_to_sort.append(self.field_name)

    def _transform(self, data_chunk):
        indxs = argsort(data_chunk[self.field_name], order=self.order)
        field_names = self.fields_to_sort if self.fields_to_sort else data_chunk.keys()
        for field_name in field_names:
            fv = data_chunk[field_name]
            if isinstance(fv, np.ndarray):
                fv = fv[indxs]
            elif isinstance(fv, list):
                fv = [fv[indx] for indx in indxs]
            else:
                raise TypeError("All field values must be of 'list' or 'array'"
                                " type.")
            data_chunk[field_name] = fv
        return data_chunk
