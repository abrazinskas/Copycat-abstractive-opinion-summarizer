from mltoolkit.mldp.utils.tools.dc_writers import CsvWriter, JsonWriter


def get_writer(f, repr_funcs, format='json', **kwargs):
    """
    :param f: opened file where data-chunks should to be written.
    :param repr_funcs: dict of field names mapping to functions that
                      should be used to obtain str. reprs of field values.
    :param format: format in which data-chunks should be written.
    :param kwargs: writer specific additional parameters.
    """
    if format == "json":
        return JsonWriter(f=f, repr_funcs=repr_funcs, **kwargs)
    if format == 'csv':
        return CsvWriter(f=f, repr_funcs=repr_funcs, **kwargs)
    raise ValueError("Please provide a valid format.")
