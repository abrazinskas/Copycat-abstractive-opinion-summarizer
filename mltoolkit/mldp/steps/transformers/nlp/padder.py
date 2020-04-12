from mltoolkit.mldp.steps.transformers.base_transformer import BaseTransformer
from mltoolkit.mldp.utils.helpers.validation import validate_field_names
from mltoolkit.mldp.utils.helpers.nlp.sequences import pad_sequences as ps
from mltoolkit.mlutils.helpers.general import listify
import numpy as np


class Padder(BaseTransformer):
    """
    This transformer pads sequences with a special pad_symbol to assure that
    all sequences are of the same length.

    Creates a separate field for each field that is padded.

    This transformer is useful if a model expects tensors as input.

    Works only for 2D data (i.e. batch_size x 1D_sequences)
    and 3D data (i.e. batch_size x num_seqs x 1D_sequences).
    """

    def __init__(self, fname, new_mask_fname, pad_symbol, symbol_to_mask=None,
                 padding_mode='both', axis=1, **kwargs):
        """
        :param fname: str or list of str names that should represent
                            fields that should be padded.
        :param new_mask_fname: str or list of str names that will contain
                                masks.
        :param pad_symbol: a symbol that should be used for padding.
        :param symbol_to_mask: a symbol(token) that should be masked in
                               sequences. E.g. Can be used to mask <UNK> tokens.
        :param padding_mode: left, right, or both. Defines the side to which
                             padding symbols should be appended.
        :param axis: defines an axis of data to which padding should be applied.
                     Currently only axes 1 or 2 are supported.
        """
        try:
            validate_field_names(fname)
        except Exception as e:
            raise e

        super(Padder, self).__init__(**kwargs)
        self.fnames = listify(fname)
        self.mask_fnames = listify(new_mask_fname)
        assert (len(self.mask_fnames) == len(self.fnames))
        self.pad_symbol = pad_symbol
        self.symbol_to_mask = symbol_to_mask
        self.padding_mode = padding_mode
        self.axis = axis

    def _transform(self, data_chunk):
        for fn, m_fn in zip(self.fnames, self.mask_fnames):
            fv = data_chunk[fn]
            if self.axis == 1:
                padded_seqs, mask = ps(fv,
                                       pad_symbol=self.pad_symbol,
                                       mask_present_symbol=self.symbol_to_mask,
                                       padding_mode=self.padding_mode)
            else:
                # TODO: consider if it's truly useful and desirable logic
                padded_seqs = np.empty(len(fv), dtype="object")
                mask = np.empty(len(fv), dtype="object")
                for i, el in enumerate(fv):
                    c_pad_seqs, c_mask = ps(el,
                                            pad_symbol=self.pad_symbol,
                                            mask_present_symbol=self.symbol_to_mask,
                                            padding_mode=self.padding_mode
                                            )
                    padded_seqs[i] = c_pad_seqs
                    mask[i] = c_mask
            data_chunk[fn] = padded_seqs
            data_chunk[m_fn] = mask
        return data_chunk
