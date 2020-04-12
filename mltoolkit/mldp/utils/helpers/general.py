import itertools


def gen_permuted_pairs(lst):
    """Generate all possible non-repeating permutations."""
    for pair in itertools.permutations(lst, 2):
        yield pair
