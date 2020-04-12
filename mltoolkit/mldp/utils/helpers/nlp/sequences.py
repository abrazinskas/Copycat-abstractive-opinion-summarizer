import numpy as np


def pad_sequences(sequences, pad_symbol, max_length=None,
                  mask_present_symbol=None, padding_mode='both'):
    """
    Pads a collection of sequences. Will work only for two dimensional data.

    :param sequences:  list or np array, which has sequences (lists or np arrays)
                       that should be padded with respect to the max_length.
    :param pad_symbol: a symbol that should be used for padding.
    :param max_length: length of the desired padded sequences. If it's not
                       provided the length of the longest sequence will be used.
    :param mask_present_symbol: a symbol(token) that should be masked in
                                sequences. E.g. Can be used to mask <UNK> tokens.
    :param padding_mode: left, right, or both. Defines the side to which
                         padding symbols should be appended.
    :return: 2D numpy array with padded sequences, and 2D binary mask
            (numpy float).
    """
    if not isinstance(sequences, (list, np.ndarray)):
        raise TypeError("Please provide a valid collection of sequences."
                        " It must be np array or list.")
    padded_sentences, masks = [], []

    if max_length is None:
        max_length = 0
        for seq in sequences:
            max_length = max(max_length, len(seq))
    if max_length == 0:
        raise ValueError("Can't pad empty sequences.")

    for sequence in sequences:
        if not isinstance(sequence, (list, np.ndarray)):
            raise ValueError("All sequences must be lists or 1D numpy arrays.")

        x, m = pad_sequence(sequence, pad_symbol, max_length=max_length,
                            mask_present_symbol=mask_present_symbol,
                            padding_mode=padding_mode)
        padded_sentences.append(x)
        masks.append(m)
    return np.array(padded_sentences), np.array(masks, dtype="float32")


def pad_sequence(sequence, pad_symbol, max_length,
                 mask_present_symbol=None, padding_mode='both'):
    """
    :param sequence: self-explanatory.
    :param pad_symbol: a symbol that should be used for padding.
    :param max_length: length of the desired padded sequence.
    :param mask_present_symbol: a symbol(token) that should be masked in the
                                sequence. E.g. Can be used to mask <UNK> tokens.
    :param padding_mode: left, right, or both. Defines the side to which
                         padding symbols should be appended.
    :return: 1D array with padded sequence, 1D float binary numpy array mask.
    """
    assert padding_mode in ['left', 'both', 'right']
    pad_number = max_length - len(sequence)

    # creating an initial mask where all elements are ones
    # (opt: except the currently present symbols)
    mask = [s != mask_present_symbol for s in sequence] if \
        mask_present_symbol is not None else [1] * len(sequence)

    # perform truncation if the sentence is too long
    if pad_number < 0:
        res = sequence[:max_length]
        mask = mask[:max_length]
        return res, mask

    nr_left_pads = 0
    nr_right_pads = 0
    # compute the number of pads necessary for each side
    if padding_mode == 'left':
        nr_left_pads = pad_number
    if padding_mode == 'right':
        nr_right_pads = pad_number
    if padding_mode == 'both':
        nr_right_pads = pad_number // 2
        nr_left_pads = pad_number // 2
        if pad_number % 2 == 1:
            nr_right_pads += 1

    sequence = pad(sequence, pad_symbol, nr_left_pads=nr_left_pads,
                   nr_right_pads=nr_right_pads)
    mask = pad(mask, 0.0, nr_left_pads=nr_left_pads,
               nr_right_pads=nr_right_pads)

    return sequence, mask


def pad(x, pad_symbol, nr_left_pads=0, nr_right_pads=0):
    # TODO: this function seems all wrong as it convert everything to numpy
    # TODO: need to rewrite it
    if nr_left_pads > 0:
        left_padding = [pad_symbol] * nr_left_pads
        if len(x):
            x = np.concatenate([left_padding, x])
        else:
            x = np.array(left_padding)
    if nr_right_pads > 0:
        right_padding = [pad_symbol] * nr_right_pads
        if len(x):
            x = np.concatenate([x, right_padding])
        else:
            x = np.array(right_padding)
    return x


def compute_windows(elems, window_size=2, step_size=1, only_full_windows=False):
    """
    Computes rolling windows over the elements.

    :param elems: list of numpy array with elements.
    :param window_size: self-explanatory.
    :param step_size: self-explanatory.
    :param only_full_windows: if set to True guarantees that all windows will
                              be of the same size.
    :return: list of window elements.
    """
    if not isinstance(elems, (list, np.ndarray)):
        raise TypeError("The provided elems must be a list or numpy array.")
    if window_size < 0:
        raise ValueError("Please provide a positive window_size.")
    if step_size < 0:
        raise ValueError("Please provide a positive step_size.")

    if (len(elems) < window_size and not only_full_windows) or \
            len(elems) == window_size:
        return [elems[0: window_size]]

    window_elems = []
    for i in range(0, len(elems), step_size):
        if i + window_size > len(elems) and only_full_windows:
            break
        window_elems.append(elems[i: i + window_size])

    return window_elems
