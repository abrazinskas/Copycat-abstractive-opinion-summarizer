from mltoolkit.mldp.steps.transformers.base_transformer import BaseTransformer
from mltoolkit.mldp.utils.helpers.validation import validate_field_names


class FunctionApplier(BaseTransformer):
    """
    Applies a function to elements of specific fields.
    E.g., it can a logarithmic transformation that is applied to the 'sales'
    field.
    """

    def __init__(self, field_name_to_func, **kwargs):
        """
        :param field_name_to_func: a dict of mappings, where values are
                                  functions of the form: x -> y.
        """
        try:
            validate_field_names(list(field_name_to_func.keys()))
        except Exception as e:
            raise e
        for f in field_name_to_func.values():
            if not callable(f):
                raise ValueError("Please provide all valid callable functions.")

        super(FunctionApplier, self).__init__(**kwargs)
        self.field_name_to_func = field_name_to_func

    def _transform(self, data_chunk):
        """
        :param data_chunk: self-explanatory.
        :return data_chunk: same as input but with modified values.
        """
        for field_name, func in self.field_name_to_func.items():
            data_chunk[field_name] = func(data_chunk[field_name])
        return data_chunk
