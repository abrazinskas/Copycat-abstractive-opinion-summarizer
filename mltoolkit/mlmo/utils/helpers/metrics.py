def comp_f1(precision, recall):
    """Computes F1 score based on scalar precision and recall."""
    return 2. * recall * precision / (recall + precision) \
        if (recall + precision) > 0. else 0.


def comp_recall(nr_correct, nr_total):
    return nr_correct / nr_total if nr_total > 0. else 0.


def comp_precision(nr_correct, nr_predicted):
    return nr_correct / nr_predicted if nr_predicted > 0. else 0.


def comp_recall_precision_f1(nr_correct, nr_predicted, nr_total):
    """
    Computes binary recall, precision, and f1 based on input scalar statistics.

    :param nr_correct: the number of correct true class' predictions.
    :param nr_predicted: the number of predictions of true class.
    :param nr_total: the total number of units of the true class.
    """
    recall = comp_recall(nr_correct, nr_total)
    precision = comp_precision(nr_correct, nr_predicted)
    f1 = comp_f1(precision, recall)
    return recall, precision, f1
