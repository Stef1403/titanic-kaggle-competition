import numpy as np
import pandas as pd

def euclidean_dist_pre(row_1, row_2, mask):
    """
    Compute Euclidean distance between two rows using a mask to skip NaNs.
    """
    squared_diffs = (row_1[mask] - row_2[mask]) ** 2
    return np.sqrt(np.sum(squared_diffs))

def manhattan_dist(row_1, row_2, mask):
    """
    Compute Manhattan distance between two rows using a mask to skip NaNs.
    """
    abs_diffs = np.abs(row_1[mask] - row_2[mask])
    return np.sum(abs_diffs)

def cosine_similarity(row_1, row_2, mask):
    """
    Compute cosine similarity between two rows using a mask to skip NaNs.
    """
    dot_product = np.dot(row_1[mask], row_2[mask])
    norm_row_1 = np.sqrt(np.sum(row_1[mask] ** 2))
    norm_row_2 = np.sqrt(np.sum(row_2[mask] ** 2))
    return dot_product / (norm_row_1 * norm_row_2)

def cosine_distance(row_1, row_2, mask):
    """
    Compute cosine distance between two rows using a mask to skip NaNs.
    """
    return 1 - cosine_similarity(row_1, row_2, mask)

def chebyshev_dist(row_1, row_2, mask):
    """
    Compute Chebyshev distance between two rows using a mask to skip NaNs.
    """
    abs_diffs = np.abs(row_1[mask] - row_2[mask])
    return np.max(abs_diffs)

def get_nearest_neighbors_pre(train_values, train_masks, test_values, test_mask, n_neighbors, dist_measure):
    """
    Find the nearest neighbors of a test row within the training data.
    """
    distances = []
    
    for i, (train_row, train_mask) in enumerate(zip(train_values, train_masks)):
        valid_mask = test_mask & train_mask
        if (dist_measure == "cos"):
            dist = cosine_distance(train_row, test_values, valid_mask)
        elif (dist_measure == "manh"):
            dist = manhattan_dist(train_row, test_values, valid_mask)
        elif (dist_measure == "cheb"):
            dist = chebyshev_dist(train_row, test_values, valid_mask)
        else:
            dist = euclidean_dist_pre(train_row, test_values, valid_mask)
        distances.append((i, dist))
    
    distances.sort(key=lambda tup: tup[1])
    return distances[:n_neighbors]


def make_prediction(train_values, train_masks, train_labels, test_row, n_neighbors, dist_measure):
    """
    Predict the label for a test row using k-nearest neighbors.
    """
    test_values = np.asarray(test_row[:-1])
    test_mask = ~pd.isna(test_values)
    
    neighbors = get_nearest_neighbors_pre(train_values, train_masks, test_values, test_mask, n_neighbors, dist_measure)
    neighbors_rows = [train_labels.iloc[i] for i, _ in neighbors]  # Use .iloc for integer indexing
    
    output_values = [row['Survived'] for row in neighbors_rows]
    return max(set(output_values), key=output_values.count)

def make_prediction_shep(train_values, train_masks, train_labels, test_row, n_neighbors, dist_measure):
    """
    Predict the label for a test row using k-nearest neighbors.
    """
    test_values = np.asarray(test_row[:-1])
    test_mask = ~pd.isna(test_values)
    
    neighbors = get_nearest_neighbors_pre(train_values, train_masks, test_values, test_mask, n_neighbors, dist_measure)
    neighbor_indices, distances = zip(*neighbors)
    
    # Extract the labels of the nearest neighbors
    neighbor_labels = [train_labels.iloc[i]['Survived'] for i in neighbor_indices]
    
    # Compute the weights for each neighbor
    # Adding a small constant to avoid division by zero
    weights = 1 / (np.array(distances) + 1e-5)
    
    # Aggregate weighted votes
    weighted_votes = {}
    for label, weight in zip(neighbor_labels, weights):
        label = int(label)
        if label in weighted_votes:
            weighted_votes[label] += weight
        else:
            weighted_votes[label] = weight
    
    # Predict the class with the highest aggregated weight
    predicted_class = max(weighted_votes, key=weighted_votes.get)
    predicted_class = int(predicted_class)
    return predicted_class


def process_test_row(train_values, train_masks, train_labels, row, n_neighbors, dist_measure):
    return make_prediction_shep(train_values, train_masks, train_labels, row, n_neighbors, dist_measure)

