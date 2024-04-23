from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, cohen_kappa_score
import numpy as np
import pandas as pd


# Function to calculate Specificity metric
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


# Function to calculate multiple evaluation metrics
 
def score_aggregates(df_solution, df_prediction):
    """
    Calculates F1 score, Matthews Correlation Coefficient, and Specificity for a classification task.

    Args:
        df_solution (pandas.DataFrame): DataFrame containing ground truth labels.
        df_prediction (pandas.DataFrame): DataFrame containing predicted labels.

    Returns:
        dict: Dictionary containing F1 score, Matthews Correlation Coefficient, Quadratic-weighted_Kappa, and Specificity metrics.
    """
    cols_prediction = df_prediction.columns

    assert ('case' in cols_prediction) , "Column case not present inside prediction dataframe"
    assert ('prediction' in cols_prediction) , "Column prediction not present inside prediction dataframe"

    
    assert len(df_solution) == len(df_prediction),f"Dataframes must have the same length: {len(df_solution)} (solution) ≠ {len(df_prediction)} prediction"
    assert set(df_solution["case"].values) ^ set(df_prediction["case"].values) == set(),"Dataframes must have the same unique id"

    # Merge solution and prediction DataFrames based on the "case" column
    df_merge = pd.merge(df_solution, df_prediction, on="case")

    solution_cases = df_merge["label"].values
    prediction_cases = df_merge["prediction"].values

    # Calculate evaluation metrics
    return {
        "F1_score": f1_score(solution_cases, prediction_cases, average="micro"),
        "Rk-correlation": matthews_corrcoef(solution_cases, prediction_cases),
        "Quadratic-weighted_Kappa": cohen_kappa_score(solution_cases,prediction_cases,weights="quadratic"),
        "Specificity": specificity(solution_cases, prediction_cases),
    }