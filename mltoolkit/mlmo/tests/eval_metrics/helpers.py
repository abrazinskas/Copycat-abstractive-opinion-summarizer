import lorem


def get_fixture_refs_and_hyps():
    """Returns 3 ref and hyp summeries consisting of single sentences."""
    refs = ["one two three", "four five six", "seven eight ten"]
    hyps = ["two three 2 3", "four five six", "seven ten"]
    return hyps, refs


def generate_sents(n=1):
    return [lorem.sentence() for _ in range(n)]
