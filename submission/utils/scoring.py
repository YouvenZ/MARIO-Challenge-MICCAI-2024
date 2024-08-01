# Function to calculate Specificity metric
import numpy as np
from sklearn.metrics import confusion_matrix


def specificity(class_ground_truth, class_prediction):
    """
    Calculates the Specificity metric for a binary classification problem.

    Args:
        class_ground_truth (array-like): Array of true class labels.
        class_prediction (array-like): Array of predicted class labels.

    Returns:
        float: The average Specificity score.
    """

    eps = 0.000001  # Add a small value to avoid division by zero
    cnf_matrix = confusion_matrix(class_ground_truth, class_prediction)

    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    # Cast all values to float to avoid type errors
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Calculate Specificity for each class and average them
    spe = TN / (TN + FP + eps)
    spe = spe.mean()

    return spe