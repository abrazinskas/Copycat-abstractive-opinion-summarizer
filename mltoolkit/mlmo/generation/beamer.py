import torch as T
from torch import Tensor
from mltoolkit.mlmo.utils.helpers.pytorch.data import adjust_tensor_to_beam_size
from mltoolkit.mlutils.helpers.general import merge_dicts
from mltoolkit.mlmo.utils.tools import DecState, BeamSearch
from mltoolkit.mlmo.utils.helpers.general import collect_arts


class Beamer(object):
    """
    Wrapper class for PyTorch beam search that works with recurring values that
    need to be passed back to the generator/decoder.
    """

    def __init__(self, decoding_func, start_id, end_id, device='cpu',
                 validate_dec_out=True, len_norm=False, excl_ids=None,
                 n_best=1, beam_size=1, **kwargs):
        """
        :param decoding_func: a function that inputs:
                                  1. prev token ids [batch_size * beam_size, 1]
                                  2. recurring values (e.g. hidden states)
                              and outputs DecState object with at least:
                                  1. word log probs [batch_size * beam_size, 1,
                                                    vocab_size]
                                  2. hidden state [batch_size * beam_size,
                                                  hidden_dim]
        :param start_id: id of the symbol that starts sequences.
        :param end_id: id of the symbol that ends sequences.
        :param device: self-explanatory.
        :param validate_dec_out: whether to throw an error if the decoding function
                                outputs values above [-inf, 0].
        :param len_norm: whether sequence scores should be normalized
                         by length. Such that the search would not favor shorter
                         sequences.
        :param excl_ids: list of ids that should be excluded from scores.
        :param n_best: determines how many candidates in the search are necessary
                       to find in order for search to be considered as finished.
        :param beam_size: the number of candidates that are considered at every
                          step.
        """
        super(Beamer, self).__init__()
        self.decoding_func = decoding_func
        self.device = device
        self.beam_size = beam_size
        self._beam_search_constr = lambda bs, min_lens: BeamSearch(
            beam_size=beam_size,
            batch_size=bs,
            start_id=start_id,
            end_id=end_id,
            device=device, len_norm=len_norm,
            min_lens=min_lens,
            excl_ids=excl_ids, n_best=n_best,
            **kwargs)
        self.val_dec_out = validate_dec_out

    def __call__(self, init_dec_state, max_steps, minimum=None, min_lens=None,
                 **dec_kwargs):
        """
        Decodes sequence token ids based on the initial state until the maximum
        number of steps is reached.

        :param init_dec_state: DecState object containing the initial state of
                               the decoder.
        :param max_steps: the number of times the decoder is executed before
                          collection of completed hypotheses starts.
        :param minimum: the minimum numbers of outputs (even if incomplete).
        :param min_lens: minimum lengths of decoded sequences.
        :param dec_kwargs: additional parameters that are passed on the decoding
                          function directly. E.g. values to attend. Will try
                          to adjust them to `beam_size`.
        :return: best_seqs: list of lists, where each contains word ids of the
                            best (completed) hypotheses.
                 best_coll_vals(opt): list of lists, elements are PyTorch
                                      elements.
        """
        assert max_steps > 0
        if not isinstance(init_dec_state, DecState) or \
                init_dec_state.rec_vals is None or \
                not len(init_dec_state.rec_vals):
            raise TypeError("Please provide a valid decoder's initial state"
                            " object.")
        first_key = next(iter(init_dec_state.rec_vals.keys()))
        bs = len(init_dec_state.rec_vals[first_key])
        beam_search = self._beam_search_constr(bs, min_lens)

        # replicating recurring values and kwargs to adjust to the beam size
        init_dec_state = _adjust_init_state_to_beam(init_dec_state,
                                                    self.beam_size)
        dec_kwargs = _adjust_kwargs_to_beam(dec_kwargs,
                                            beam_size=self.beam_size)
        rec_vals = init_dec_state.rec_vals
        coll_vals = {}  # additional artifacts that are produced by the decoder

        for t in range(max_steps):
            if beam_search.done():
                break

            if t > 0:
                assert rec_vals is not None
                # shuffling recurring values based on back-pointers
                bp = beam_search.get_current_origin()
                for k, v in rec_vals.items():
                    rec_vals[k] = _bp_shuffle(v, bp, beam_size=self.beam_size)

            # merging static params and the recurring ones
            new_kwargs = merge_dicts(rec_vals, dec_kwargs)

            prev_word_ids = beam_search.get_current_state()
            prev_word_ids = prev_word_ids.view(bs * self.beam_size, 1)
            out = self.decoding_func(prev_word_ids, **new_kwargs)

            assert isinstance(out, DecState)
            word_log_probs = out.word_scores
            rec_vals = out.rec_vals

            # collecting artifacts if produced by the decoder
            if out.coll_vals:
                collect_arts(coll_vals, out.coll_vals)

            if self.val_dec_out and not (word_log_probs <= 0. + 1e-4).all():
                raise ValueError("Please adjust the decoding function as it "
                                 "should provide valid log-probabilities.")

            word_log_probs = word_log_probs.view(bs, self.beam_size, -1)
            beam_search.advance(word_log_probs)

        if coll_vals:
            # concatenating artifacts produced by the decoder over time-steps
            coll_vals = {k: T.stack(v, dim=0) for k, v in coll_vals.items()}

        res = beam_search.get_finished_best(minimum=minimum, **coll_vals)

        if coll_vals:
            sel_seq = res[0]
            sel_coll_vals = res[1]
        else:
            sel_seq = res
            sel_coll_vals = None

        sel_seq = _dec_out_formatter(sel_seq)

        return sel_seq, sel_coll_vals


def _bp_shuffle(tensor, bp, beam_size):
    """Shuffles 2D and 3D tensor according to back-pointers."""
    # TODO: make generalized to any number of dimensions
    bs = tensor.size(0) // beam_size
    if len(tensor.shape) == 2:
        tensor = tensor.view(bs, beam_size, -1)
        tensor = tensor[T.arange(bs).reshape((-1, 1)), bp].view(bs * beam_size,
                                                                -1)
    elif len(tensor.shape) == 3:
        dim1 = tensor.size(1)
        tensor = tensor.view(bs, beam_size, dim1, -1)
        tensor = tensor[T.arange(bs).reshape((-1, 1)), bp].view(bs * beam_size,
                                                                dim1, -1)
    elif len(tensor.shape) == 4:
        dim1 = tensor.size(1)
        dim2 = tensor.size(2)
        tensor = tensor.view(bs, beam_size, dim1, dim2, -1)
        tensor = tensor[T.arange(bs).unsqueeze(1), bp].view(bs * beam_size,
                                                            dim1,
                                                            dim2, -1)
    else:
        raise ValueError("At the moment the decoder does not support 4D+"
                         " tensors.")
    return tensor


def _adjust_kwargs_to_beam(kwargs, beam_size):
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            kwargs[k] = adjust_tensor_to_beam_size(v, beam_size=beam_size)
    return kwargs


def _adjust_init_state_to_beam(init_dec_state, beam_size):
    """
    Replicates recurring values in the initial decoder's state based on the
    beam size.
    """
    for k, v in init_dec_state.rec_vals.items():
        init_dec_state.rec_vals[k] = adjust_tensor_to_beam_size(v, beam_size)
    return init_dec_state


def _dec_out_formatter(out):
    """Converts PyTorch int sequences (word ids) to Python integers."""
    res = []
    for seq in out:
        tmp = []
        for elem in seq:
            elem = elem.to('cpu').numpy()
            if len(elem.shape) == 0:
                elem = elem.item()
            tmp.append(elem)
        res.append(tmp)
    return res
