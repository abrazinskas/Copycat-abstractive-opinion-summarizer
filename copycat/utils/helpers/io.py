from copycat.utils.fields import InfDataF
import re


def get_rev_number(input_file_path, sep="\t", encoding='utf-8'):
    """Reads the number of reviews based on the CSV header."""
    patt = re.compile(f'^{InfDataF.REV_PREFIX}\d+$')
    with open(input_file_path, encoding=encoding) as f:
        for l in f:
            rev_cols = [c for c in l.split(sep) if patt.match(c)]
            return len(rev_cols)


def read_text_file(input_file_path, encoding='utf-8'):
    coll = []
    with open(input_file_path, encoding=encoding) as f:
        for l in f:
            l = l.strip()
            if l:
                coll.append(l)
    return coll
