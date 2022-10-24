"""Binary confusion matrix and utilities."""
import numpy as np

class BinaryCM:
    """Binary confusion matrix.

    Represents a confusion matrix in which the columns represent the true labels and the rows the predictions.

    [[True positives        False positives]
     [False negatives       True negatives ]]
    """
    def __init__(self, tp: int, fp: int, fn: int, tn: int):
        self._tp = tp
        self._fp = fp
        self._fn = fn
        self._tn = tn

    @property
    def tp(self):
        return self._tp

    @property
    def fp(self):
        return self._fp

    @property
    def fn(self):
        return self._fn

    @property
    def tn(self):
        return self._tn

    def numpy(self):
        return np.array([self.tp, self.fp, self.fn, self.tn], dtype=np.int32)

    def __str__(self):
        return str(self.numpy())

    def recall(self):
        return self.tp/(self.tp + self.fn)

    def precision(self):
        return self.tp/(self.tp + self.fp)

    def f1score(self):
        p = self.precision()
        r = self.recall()
        return 2*(p*r)/(p + r)


def convert_to_binary(confusion_matrix: np.ndarray, class_index: int):
    """Convert a numpy confusion matrix into a binary confusion matrix."""
    # Verify that the given confusion is of rank two.
    if len(confusion_matrix.shape) != 2:
        raise ValueError("Confusion matrix must be of rank 2, but a matrix of rank {len(confusion_matrix.shape)} was given.")
    tp = confusion_matrix[class_index, class_index]
    fp = np.sum(np.delete(confusion_matrix[class_index], class_index))
    fn = np.sum(np.delete(confusion_matrix, class_index, axis=0)[:, class_index])
    tn = np.sum(np.delete(np.delete(confusion_matrix, class_index, axis=0), class_index, axis=1))
    return BinaryCM(tp=tp, fp=fp, fn=fn, tn=tn)


if __name__ == "__main__":
    cm = np.array([[100, 10, 1], [2, 2, 2], [1, 10, 100]])
    bcm = convert_to_binary(cm, 0)
    print(bcm)
