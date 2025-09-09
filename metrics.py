import numpy as np
from scipy import spatial
from sklearn.metrics import roc_auc_score


def pairwise_l2(arr1, arr2):
    return np.array(spatial.distance.cdist(arr1, arr2))


def mean_auc(y_pred, y_truth):
    """
    Compute average AUC over columns with at least one positive
    and one negative element
    """
    selected_indexes = [
        index
        for index in range(y_truth.shape[1])
        if (0 in y_truth[:, index]) and (y_truth[:, index].sum())
    ]

    return roc_auc_score(y_truth[:, selected_indexes], y_pred[:, selected_indexes])

def top_k_metric(y_true, y_pred, k, is_argsort=False):
    """
    Calculate the Top-K metric for a recommendation system.

    Parameters:
    - y_true: True matching labels (ground truth)
    - y_pred: Distance matrix or predicted scores (lower values are better)
    - k: Accepted top-k matches
    - is_sorted: If True, y_pred is assumed to be the result of np.argsort(y_pred, axis=1)

    Returns:
    - top_k_score: top-k metric score
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    # Sort the predictions (y_pred) and get the indices that would sort it
    sorted_indices = (y_pred if is_argsort else np.argsort(y_pred, axis=1))

    top_k_matches = 0

    for i in range(len(y_true)):
        true_label = y_true[i]
        top_k_indices = sorted_indices[i, :k]

        if true_label in top_k_indices:
            top_k_matches += 1

    top_k_score = top_k_matches / len(y_true)

    return top_k_score


def mix_match(similarity):
    accuracies = []
    for row_index in range(len(similarity)):
        current_row_accumulator = 0
        for col_index in range(len(similarity[row_index])):
            if col_index == row_index:
                continue
            else:
                if similarity[row_index][row_index] > similarity[row_index][col_index]:
                    current_row_accumulator += 1

        accuracies.append(current_row_accumulator / (len(similarity[row_index])-1))

    return np.mean(accuracies)
