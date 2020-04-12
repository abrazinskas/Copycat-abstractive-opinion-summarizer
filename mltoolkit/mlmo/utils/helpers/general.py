def isnan(x):
    return x != x


def collect_arts(coll, arts):
    """Collects in-place recently produced artifacts to `coll`."""
    for k, v in arts.items():
        if k not in coll:
            coll[k] = []
        coll[k].append(v)
