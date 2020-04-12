class KlMonAnnealing(object):
    """Allows to compute KLD monotonically increasing annealing scalar."""

    def __init__(self, ann_batches, start=0., end=1.):
        self.ann_batches = ann_batches
        self.start = start
        self.end = end
        self._kl_ann_indx = 0

    def __call__(self, increment_indx=False):
        """Computes and returns the annealing scale for the KL term."""
        if not self.ann_batches or self.ann_batches == 0:
            return 1.
        else:
            curr_kl = self.start
            if increment_indx:
                self._kl_ann_indx += 1
                prop = float(self._kl_ann_indx) / self.ann_batches
                curr_kl += (self.end - self.start) * prop
            return min(self.end, curr_kl)
