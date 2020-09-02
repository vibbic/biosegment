import numpy as np
from scipy.spatial.distance import directed_hausdorff


def jaccard(y_true, y_pred, w=None):
    """
    Jaccard index between two segmentations

    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :return: the Jaccard index
    """

    # check mask
    if w is None:
        w = np.ones_like(y_true, dtype='bool')
    y_true = y_true[w]
    y_pred = y_pred[w]

    # binarize
    y_true = (y_true > 0.5).astype('int')
    y_pred = (y_pred > 0.5).astype('int')

    # compute jaccard score
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    return intersection / union


def dice(y_true, y_pred, w=None):
    """
    Dice coefficient between two segmentations

    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :return: the Jaccard index
    """

    j = jaccard(y_true, y_pred, w=w)

    return 2 * j / (1 + j)


def accuracy_metrics(y_true, y_pred, w=None):
    """
    Accuracy metrics between two segmentations (accuracy, balanced accuracy, precision, recall and f1-score)

    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :return: the Jaccard index
    """

    # check mask
    if w is None:
        w = np.ones_like(y_true, dtype='bool')
    y_true = y_true[w]
    y_pred = y_pred[w]

    # binarize
    y_true = (y_true > 0.5).astype('int')
    y_pred = (y_pred > 0.5).astype('int')

    # compute accuracy metrics
    tp = np.sum(y_true * y_pred)
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = (recall + specificity) / 2
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, balanced_accuracy, precision, recall, f1


def hausdorff_distance(x, y):
    """
    Hausdorff distance between two segmentations

    :param x: array
    :param y: array
    :return: the hausdorff distance
    """

    # binarize
    x = x > 0.5
    y = y > 0.5

    hd_0 = 0
    hd_1 = 0
    hd = 0
    for i in range(x.shape[0]):
        hd_0 += directed_hausdorff(x[i, ...], y[i, ...])[0]
        hd_1 += directed_hausdorff(y[i, ...], x[i, ...])[0]
        hd += max(hd_0, hd_1)
    hd_0 /= x.shape[0]
    hd_1 /= x.shape[0]
    hd /= x.shape[0]

    return hd, hd_0, hd_1
