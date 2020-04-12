from mltoolkit.mlutils.tools.signed_object import SignedObject


class BaseMetric(SignedObject):
    """
    Parent metrics class, where metrics are used to calculate different
    evaluation scores on data. For example, accuracy or exact matches.

    Each metric is calculated in the on-line fashion. Where predicted and true
    labels are fed iteratively to the 'accum' method that usually returns
    temporary results, and stores statistics for later aggregation.

    For example, for accuracy metric, we might aggregate the number of correct
    predictions, and the total number of labels. Later, we aggregate those
    statistics to compute the final accuracy score.
    """

    def aggr(self):
        """
        Calculates the final aggregated results based on collected information.

        :return: dict where keys are metric names, and values are scores.
        """

    def accum(self, **kwargs):
        """
        Collects local statistics based on provided data that are later used for
        aggregation.

        :return: dict where keys are metric names, and values are scores.
        """
        raise NotImplementedError
