class KlCycAnnealing(object):
    """
    Cycling Kullback-Leibler annealing scalar.
    Based on 'Cyclical Annealing Schedule: A Simple Approach to Mitigating KL
    Vanishing'
    '"""

    def __init__(self, t, m=4, r=0.5, max_val=1.):
        """
        :param t: total number of training iterations.
        :param m: total number of cycles.
        :param r: proportion used to increase beta within a cycle.
        :param max_val: maximum values that the annealing constant can attain.
        """
        self.t = t
        self.m = m
        self.r = r
        self.max_val = max_val
        self._kl_ann_indx = 0

    def __call__(self, increment_indx=False):
        """Computes and returns the annealing scale for the KL term."""
        if self._kl_ann_indx == 0 and not increment_indx:
            return 0.

        if increment_indx:
            self._kl_ann_indx += 1

        tau = ((self._kl_ann_indx - 1) % round(self.t / self.m)) / \
              (self.t / self.m)

        if tau <= self.r:
            val = self._ann_func(tau)
        else:
            val = 1.

        return self.max_val * val

    def _ann_func(self, tau):
        return tau / self.r
