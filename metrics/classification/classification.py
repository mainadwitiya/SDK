import sklearn.metrics
import numpy as np

class ClassificationMetrics(object):
    r"""
    Evaluation Metrics for classification

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
        self.confusion_matrix = sklearn.metrics.confusion_matrix(ground_truth, predictions)
        self.FP = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        self.FN = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        self.TP = self.confusion_matrix.diagonal()
        self.TN = self.confusion_matrix.sum() - (self.FP + self.FN + self.TP)

        self.precision = self.TP / (self.TP + self.FP)
        self.precision[np.isnan(self.precision)] = 0
        self.recall = self.TP / (self.TP + self.FN)
        self.recall[np.isnan(self.recall)] = 0
        self.accuracy = sklearn.metrics.accuracy_score(ground_truth, predictions)
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        self.f1_score[np.isnan(self.f1_score)] = 0
        
        self.label_disagreement = {k: v.round(3) for k, v in enumerate(self.FP / (self.FP + self.TN))}
        self.acc_per_class = {k: v.round(3) for k, v in enumerate(self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=1))}
        self.f1_score_per_class = {k: v for (k, v) in enumerate(self.f1_score)}
        self.f1_score = self.f1_score.mean()

        self.precision_per_class = {k: v for (k, v) in enumerate(self.precision)}
        self.precision = self.precision.mean()

        self.recall_per_class = {k: v for (k, v) in enumerate(self.recall)}
        self.recall = self.recall.mean()

        self.confusion_matrix = self.confusion_matrix.tolist()

    def get_accuracy(self):
        r"""
        Returns:
            accuracy as a float
        """
        return self.accuracy

    def get_f1_score(self):
        r"""
        Returns:
            F1 score as a float
        """
        return self.f1_score

    def get_f1_score_per_class(self):
        r"""
        Returns:
            F1 score per class as a dict
        """
        return self.f1_score_per_class

    def get_precision_per_class(self):
        r"""
        Returns:
            precision per class as a dict
        """
        return self.precision_per_class

    def get_precision(self):
        r"""
        Returns:
            precision as a float
        """
        return self.precision

    def get_recall_per_class(self):
        r"""
        Returns:
            recall per class as a dict
        """
        return self.recall_per_class

    def get_recall(self):
        r"""
        Returns:
            recall as a float
        """
        return self.recall

    def get_confusion_matrix(self):
        r"""
        Returns:
            confusion matrix as a list
        """
        return self.confusion_matrix

    def get_acc_per_class(self):
        r"""
        Returns:
            acc_per_class as a dict
        """
        return self.acc_per_class

    def get_label_disagreement(self):
        r"""
        Returns:
            label disagreement as a dict
        """
        return self.label_disagreement
