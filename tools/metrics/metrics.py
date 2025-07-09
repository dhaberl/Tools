#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 15 21:52 2021

Performance metrics for tabular data classification

Content:
    - Performance estimation using Monte Carlo cross validation with multiple metrics
        - Positive predictive value
        - Negative predictive value
        - Balanced accuracy
        - Accuracy
        - Sensitivity
        - Specificity
        - ROC AUC

@author: cspielvogel
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm


def positive_predictive_value(y_true, y_pred):
    """
    Calculate positive predictive value for confusion matrix with any number of classes
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
    :return: Float between 0 and 1 indicating the positive predictive value
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Binary classification
    if cm.shape == (2, 2):
        # Handle division by zero, (tp + fp) = 0
        if (cm[1][1] + cm[0][1]) == 0:
            ppv = 0
        else:
            ppv = cm[1][1] / (cm[1][1] + cm[0][1])

    # Multiclass classification
    else:
        classwise_ppv = []
        for i in np.arange(len(cm)):
            tp = cm[i][i]
            fp_and_tp = cm[:, i]
            if np.sum(fp_and_tp) == 0:
                classwise_ppv.append(0)
            else:
                classwise_ppv.append(tp / np.sum(fp_and_tp))
            ppv = np.sum(classwise_ppv) / len(classwise_ppv)

    return ppv


def negative_predictive_value(y_true, y_pred):
    """
    Calculate negative predictive value for confusion matrix with any number of classes
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
    :return: Float between 0 and 1 indicating the negative predictive value
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Binary classification
    if cm.shape == (2, 2):
        # Handle division by zero, (tn + fn) = 0
        if (cm[0][0] + cm[1][0]) == 0:
            npv = 0
        else:
            npv = cm[0][0] / (cm[0][0] + cm[1][0])

    # Multiclass classification
    else:
        # For multiclass classification NPV = PPV
        npv = positive_predictive_value(y_true, y_pred)

    return npv


def balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy for confusion matrix with any number of classes
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
    :return: Float between 0 and 1 indicating the balanced accuracy
    """

    bacc = 0.5 * (specificity(y_true, y_pred) + sensitivity(y_true, y_pred))

    return bacc


def accuracy(y_true, y_pred):
    """
    Calculate accuracy for confusion matrix with any number of classes
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
    :return: Float between 0 and 1 indicating the accuracy
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Return accuracy
    return np.sum(np.diagonal(cm)) / np.sum(cm)


def sensitivity(y_true, y_pred):
    """
    Calculate sensitivity (=Recall/True positive rate) for confusion matrix with any number of classes
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :return: Float between 0 and 1 indicating the sensitivity
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Binary classification
    if cm.shape == (2, 2):
        sns = cm[1][1] / (cm[1][1] + cm[1][0])

    # Multiclass classification
    else:
        classwise_sns = []
        for i in np.arange(len(cm)):
            tp = cm[i][i]
            fn_and_tp = cm[i, :]
            if np.sum(fn_and_tp) == 0:
                classwise_sns.append(0)
            else:
                classwise_sns.append(tp / np.sum(fn_and_tp))
            sns = np.sum(classwise_sns) / len(classwise_sns)

    return sns


def specificity(y_true, y_pred):
    """
    Calculate specificity (=True negative rate) for confusion matrix with any number of classes
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :param y_pred: numpy.ndarray of 1 dimension or list indicating the predicted classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :return: Float between 0 and 1 indicating the specificity
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Binary classification
    if cm.shape == (2, 2):
        spc = cm[0][0] / (cm[0][0] + cm[0][1])

    # Multiclass classification
    else:
        classwise_spc = []
        for i in np.arange(len(cm)):
            t = [cm[j][j] for j in np.arange(len(cm))]
            del t[i]
            tn = np.sum(t)
            f = list(cm[i])
            del f[i]
            fp = np.sum(f)
            if (tn + fp) == 0:
                classwise_spc.append(0)
            else:
                classwise_spc.append(tn / (tn + fp))
        spc = np.sum(classwise_spc) / len(classwise_spc)

    return spc


def roc_auc(y_true, y_prob, average="macro"):
    """
    Calculation of area under the receiver operating characteristic curve (ROC AUC)
    :param y_true: numpy.ndarray of 1 dimension or list indicating the actual classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_pred
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :param y_prob: numpy.ndarray of 1 dimension or list indicating the probability of classes for a set of instances
                   CAVE: Instances must be in the same order as in parameter y_true
                   CAVE: For binary classifications, class encoded by the numerically smaller integer will be assumed
                         as negative class while the greater integer will be assumed as positive class
    :param average: String 'micro', 'macro', 'samples' or 'weighted'; default is 'macro'
                    If None, the scores for each class are returned. Otherwise, this determines the type of averaging
                    performed on the data: Note: multiclass ROC AUC currently only handles the 'macro' and 'weighted'
                    averages.
                    'micro':
                    Calculate metrics globally by considering each element of the label indicator matrix as a label.
                    'macro':
                    Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance
                     into account.
                    'weighted':
                    Calculate metrics for each label, and find their average, weighted by support (the number of true
                    instances for each label).
                    'samples':
                    Calculate metrics for each instance, and find their average.
                    See also https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    :return: Float between 0 and 1 indicating the ROC AUC
    """

    return roc_auc_score(y_true, y_prob, average=average, multi_class="ovr")


def get_classification_dict(y_true, y_pred, y_proba):
    acc = accuracy(y_true, y_pred)
    sns = sensitivity(y_true, y_pred)
    spc = specificity(y_true, y_pred)
    ppv = positive_predictive_value(y_true, y_pred)
    npv = negative_predictive_value(y_true, y_pred)
    bac = balanced_accuracy(y_true, y_pred)
    auc = roc_auc(y_true, y_proba)
    return {
        "acc": [acc],
        "sns": [sns],
        "spc": [spc],
        "bac": [bac],
        "ppv": [ppv],
        "npv": [npv],
        "auc": [auc],
    }


def bootstrap_confidence_interval_performances(
    y_true,
    y_pred,
    y_proba,
    percent=95,
    n_iterations=1000,
    percent_samples=1.0,
    decimals=3,
):
    n_samples = int(np.round(len(y_true) * percent_samples, 0))

    metrics_dict = {
        "acc": [],
        "sns": [],
        "spc": [],
        "bac": [],
        "ppv": [],
        "npv": [],
        "auc": [],
    }

    pbar = tqdm(range(n_iterations))
    for i in pbar:
        pbar.set_description(desc="Bootstrapping")

        sample_indices = np.arange(0, len(y_pred))
        selected_indices = resample(
            sample_indices, n_samples=n_samples, stratify=y_true
        )

        selected_y_true = y_true[selected_indices]
        selected_y_pred = y_pred[selected_indices]
        selected_y_proba = y_proba[selected_indices]

        perf_dict = get_classification_dict(
            selected_y_true, selected_y_pred, selected_y_proba
        )
        metrics_dict["acc"].append(perf_dict["acc"][0])
        metrics_dict["sns"].append(perf_dict["sns"][0])
        metrics_dict["spc"].append(perf_dict["spc"][0])
        metrics_dict["bac"].append(perf_dict["bac"][0])
        metrics_dict["ppv"].append(perf_dict["ppv"][0])
        metrics_dict["npv"].append(perf_dict["npv"][0])
        metrics_dict["auc"].append(perf_dict["auc"][0])

    # Compute mean and confidence intervals
    for metric in metrics_dict:
        stats = metrics_dict[metric]
        lower, upper = confidence_interval(stats, percent=percent)

        res = (
            f"{np.mean(stats):.{decimals}f} ({lower:.{decimals}f}-{upper:.{decimals}f})"
        )
        metrics_dict[metric] = [res]
    return metrics_dict


def confidence_interval(data, percent=95):
    alpha = percent / 100
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(data, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(data, p))

    return lower, upper
