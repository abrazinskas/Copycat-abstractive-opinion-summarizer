import re
from html.entities import name2codepoint


def format_big_box(message, ws_offset=20):
    """
    Formats a message by wrapping it into a big box of # and white space
    symbols. Works for multi-line message (splat by \n).

    :param message: self-explanatory.
    :param ws_offset: two sided white space offset wrapping the message.
    """
    msg_lines = message.split("\n")
    max_msg_line_len = max([len(line) for line in msg_lines])

    total_len = max_msg_line_len + ws_offset

    # 2 is added because of # on both sides
    top_or_bottom = '#' * (total_len + 2)
    left_side_offset = ws_offset // 2
    right_side_offset = ws_offset // 2
    if ws_offset % 2 == 1:
        right_side_offset += 1

    left_padding = ' ' * left_side_offset
    center_lines = []
    for line in msg_lines:
        # adjusting the right padding to the maximum line len
        extra_padding = max_msg_line_len - len(line)
        right_padding = ' ' * (right_side_offset + extra_padding)
        center_lines.append("#%s%s%s#" % (left_padding, line, right_padding))
    center = "\n".join(center_lines)
    mess = "\n%s\n%s\n%s\n" % (top_or_bottom, center, top_or_bottom)

    return mess


def format_small_box(message, ws_offset=6, box_width=40):
    """
    Formats the message into a one line string offset by # and white space
    symbols.

    :param message: self-explanatory.
    :param ws_offset: two sided white space offset wrapping the message.
    :param box_width: the total width of the box.
    """
    msg = "\n"
    hashes_count = max(box_width - (len(message) + ws_offset), 0)
    left_side = hashes_count // 2
    right_side = hashes_count // 2
    if hashes_count % 2 == 1:
        right_side += 1
    left_ws_offset = ws_offset // 2
    right_ws_offset = ws_offset // 2
    if ws_offset % 2 == 1:
        right_ws_offset += 1

    msg += ''.join(['#' * left_side, ' ' * left_ws_offset,
                    message,
                    ' ' * right_ws_offset,
                    '#' * right_side]) + "\n"
    return msg


def format_dict(dic, indent=0):
    """
    Formats dictionary as a string of key value pairs. Each pair is printed on
    a new line.
    """
    msg = ""
    for param_name, param_value in dic.items():
        msg += ' ' * indent + (param_name + ": " + str(param_value) + '\n')
    return msg


def format_to_standard_msg_str(parent_title, parent_dict, parent_ws_offset=40,
                               indent=2, children_titles=None,
                               children_dicts=None, children_ws_offset=6,
                               ):
    """Creates a standard print-out used for automatic objects documentation."""
    msg = ""
    msg += format_big_box(parent_title, ws_offset=parent_ws_offset)
    msg += "\n"
    msg += format_dict(parent_dict, indent=indent)

    parent_box_width = len(parent_title) + parent_ws_offset + 2

    if children_titles and children_dicts:
        assert len(children_titles) == len(children_dicts)
        for title, child_dict in zip(children_titles, children_dicts):
            msg += format_signature(title, attrs=child_dict, indent=indent,
                                    ws_offset=children_ws_offset,
                                    box_width=parent_box_width)
    msg += format_big_box("", ws_offset=len(parent_title) + parent_ws_offset)
    return msg


def format_signature(title, attrs, indent=0, **format_small_box_kwargs):
    """A common formatting for conversion of signature to str."""
    msg = format_small_box(title, **format_small_box_kwargs)
    msg += "\n"
    msg += format_dict(attrs, indent=indent)
    return msg


def format_title(title, name_prefix=None, capitalize_prefix=True):
    if name_prefix:
        if capitalize_prefix:
            name_prefix = name_prefix.capitalize()
        title = "%s %s" % (name_prefix, title)
    return title


def metrics_to_str(metrics, prefix=""):
    """
    Converts metrics dictionary into a human readable string.
    
    :param metrics: dict where keys are metric names and values are scores.
    :param prefix: self-explanatory.
    :return: str representation.
    """
    my_str = ", ".join(["%s: %.3f" % (k, v) for k, v in metrics.items()])
    if prefix:
        my_str = prefix + " " + my_str
    return my_str


def unescape(text):
    """
    Removes (replaces) HTML or XML character references and entities from a
    string.
    """

    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return chr(int(text[3:-1], 16))
                else:
                    return chr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = chr(name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text  # leave as is

    return re.sub("&#?\w+;", fixup, text)


def format_stats(stats):
    """Formats a multidimensional dictionary with stats to a string.

    Args:
        stats (dict): multidimensional stats dict. 

    Returns: string

    """
    formatted_parts = ["\n"]
    for k, v in stats.items():
        pad = "".join(['='] * 5)
        sub_title = "%s %s %s \n" % (pad, k, pad)
        formatted_parts.append(sub_title)
        formatted_parts.append(format_dict(v, indent=2))
        formatted_parts.append("\n")
    formatted_str = "".join(formatted_parts)
    return formatted_str
