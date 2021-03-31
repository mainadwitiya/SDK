import math

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

class MultiLabelClassificationMetrics(object):
    r"""
    Evaluation Metrics for multilabel classification
    Args:
        predictions: list,
            list of classes predicted for each record

        ground_truth: list,
            list of ground_truth classes for each record
    """

    def __init__(self, predictions, ground_truth):
        self.predictions = predictions
        self.ground_truth = ground_truth
        self._compute_metrics(predictions, ground_truth)

    def _compute_metrics(self, predictions, ground_truth):
        self.accuracy_result = accuracy_score(ground_truth, predictions)
        self.hamming_l= hamming_loss(ground_truth, predictions)

        multilabel_confusion_mat = multilabel_confusion_matrix(ground_truth, predictions)

        self.output = {}
        for k, row in enumerate(multilabel_confusion_mat):
            tn, fp, fn, tp = row.ravel()
            self.output[k] = {'TN': tn , 'FP': fp, 'FN': fn, 'TP': tp}

        TP, FP, TN, FN = 0, 0, 0, 0 # micro
        self.precision_macro, self.recall_macro = 0.0, 0.0 # macro
        for c, mat in self.output.items():
            TP += mat['TP']
            FP += mat['FP']
            TN += mat['TN']
            FN += mat['FN']

            p = mat['TP'] / (mat['TP'] + mat['FP'])
            r = mat['TP'] / (mat['TP'] + mat['FN'])

            self.precision_macro = self.precision_macro + p if not math.isnan(p) else self.precision_macro
            self.recall_macro = self.recall_macro + r if not math.isnan(r) else self.recall_macro
        
        self.precision_micro = TP / (TP + FP)
        self.recall_micro = TP / (TP + FN)
        self.f1_micro = 2 * (self.precision_micro * self.recall_micro) / (self.precision_micro + self.recall_micro)

        self.precision_micro = self.precision_micro if not math.isnan(self.precision_micro) else 0
        self.recall_micro = self.recall_micro if not math.isnan(self.recall_micro) else 0
        self.f1_micro = self.f1_micro if not math.isnan(self.f1_micro) else 0

        self.precision_macro = self.precision_macro / len(multilabel_confusion_mat)
        self.recall_macro = self.recall_macro / len(multilabel_confusion_mat)

        self.f1_macro = 2 * (self.precision_macro * self.recall_macro) / (self.precision_macro + self.recall_macro)
        self.f1_macro = self.f1_macro if not math.isnan(self.f1_macro) else 0

    def get_accuracy(self):
        r"""
        Returns:
            accuracy as a float
        """
        return self.accuracy_result

    def get_f1_score_micro(self):
        r"""
        Returns:
            F1 score micro
        """
        return self.f1_micro

    def get_f1_score_macro(self):
        r"""
        Returns:
            F1 score macro
        """
        return self.f1_macro

    def get_precision_micro(self):
        r"""
        Returns:
            precision micro
        """
        return self.precision_micro

    def get_precision_macro(self):
        r"""
        Returns:
            precision macro
        """
        return self.precision_macro

    def get_recall_micro(self):
        r"""
        Returns:
            recall micro
        """
        return self.recall_micro

    def get_recall_macro(self):
        r"""
        Returns:
            recall macro
        """
        return self.recall_macro

    def get_confusion_matrix(self):
        r"""
        Returns:
            confusion matrix as a list
        """
        return self.output

    def get_hamming_loss(self):
        r"""
        Returns:
            hamming loss
        """
        return self.hamming_l
