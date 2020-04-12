# TERMINATION_TOKEN: indicates that data processing has finished and no more
#                    chunks should be expected by a producer (e.g. reader).
TERMINATION_TOKEN = "<DONE>"
# EMPTY_CHUNK: indicates that a data-chunk contains no information and can be
#                        safely ignored by processing steps.
EMPTY_CHUNK = "<EMPTY>"
# GROUPING_FNAMES: is used in data-chunks json dumping to indicate what where
#                  the fields used to aggregate/group data-units. The order of
#                  field names is preserved.
GROUPING_FNAMES = '<GROUPING_FNAMES>'
