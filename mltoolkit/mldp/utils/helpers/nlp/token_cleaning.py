# -*- coding: utf-8 -*-
import unicodedata

try:
    import re2 as re
except ImportError:
    import re


def twitter_text_cleaner(token):
    return re.sub(r'[^\w_?!$@#:/\-()><]|[?$@#]{2,}', "", token)


def deal_with_accents(my_str):
    """Removes/replaces strange symbols like é."""
    my_str = my_str.decode() if isinstance(my_str, str) else my_str
    return unicodedata.normalize('NFD', my_str)
