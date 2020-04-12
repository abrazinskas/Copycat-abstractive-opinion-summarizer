# search/decoding related helper functions


def traverse_table(time_step, beam_indx, back_pointers, elems_table,
                   format_func=None):
    """
    Walks back to construct the full hypothesis by traversing the passed table.

    :param time_step: the last time-step of the best candidate
    :param beam_indx: the beam index of the best candidate in the last time-step
    :param back_pointers: array [steps]
    :param elems_table: array of elements to traverse [steps, elements_count],
                        e.g. vocabulary word ids
    :param format_func: a function to format elements
    :return: hypothesis list of the size 'time_step'
    """
    hyp = []
    for j in range(len(back_pointers[:time_step]) - 1, -1, -1):
        elem = elems_table[j + 1][beam_indx]
        elem = elem if format_func is None else format_func(elem)
        hyp.append(elem)
        beam_indx = back_pointers[j][beam_indx]
    elem = elems_table[0][beam_indx]
    elem = elem if format_func is None else format_func(elem)
    hyp.append(elem)
    return hyp[::-1]


def adjust_tensor_to_beam_size(tens, beam_size):
    """Replicates tensor values for each beam over the first dim."""
    bs = tens.size(0)
    if len(tens.shape) == 3:
        s = tens.size(1)
        tens = tens.unsqueeze(1).repeat((1, beam_size, 1, 1))
        tens = tens.view(bs * beam_size, s, -1)
    elif len(tens.shape) == 2:
        s = tens.size(1)
        tens = tens.unsqueeze(1).repeat((1, beam_size, 1))
        tens = tens.view(bs * beam_size, s)
    elif len(tens.shape) == 1:
        tens = tens.unsqueeze(1).repeat((1, beam_size))
        tens = tens.view(bs * beam_size)
    else:
        raise NotImplementedError
    return tens


def find_mirror_next(seq, max_window_size, mirror_centre):
    """Find the next token that will lead to a mirror pattern.

    Searches in the range of `window_size` tokens to the left from the
    `mirror_centre` if it's found.

    E.g., in [a, b, AND, a] the next token should be 'b' to create a mirror
    pattern.

    Args:
        seq (list): list of tokens or ids
        max_window_size (int): maximum span of search from the found
            `mirror_centre`.
        mirror_centre (list): list of tokens/ids that should be searched for as
            centres of mirror patterns.

    Returns:
        next unit that that will lead to a mirror pattern.
        if no units in the `max_window_size` are in `mirror_centre`, then it
        will return None.
    """
    assert max_window_size > 0

    for i in range(1, max_window_size + 1):
        if len(seq) < 2 * i:
            continue
        centre_indx = len(seq) - i
        if seq[centre_indx] in mirror_centre:
            left = seq[centre_indx - i:centre_indx - 1]
            right = seq[centre_indx + 1:]
            next_unit = seq[centre_indx - 1]
            if left == right:
                return next_unit
