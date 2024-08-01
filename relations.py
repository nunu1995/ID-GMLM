import torch
import numpy as np


def gaussian_kernel(vector_a, vector_b, sigma_a, sigma_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    euclidean_distance_squared = np.sum((vector_a - vector_b) ** 2)
    similarity = np.exp(-euclidean_distance_squared / (sigma_a * sigma_b))
    return similarity


def paire_compaire(labels):
    pairs = []
    target_scores = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] > labels[j]:
                pairs.append([i, j])
                target_scores.append(1.0)
            elif labels[i] < labels[j]:
                pairs.append([i, j])
                target_scores.append(0.0)

    pairs = torch.tensor(pairs, dtype=torch.long)
    target_scores = torch.tensor(target_scores, dtype=torch.float)
    return pairs, target_scores


def criteria_label(alternatives, labels):
    criteria_importance = []
    for i in range(alternatives.shape[1]):
        corr_coef = np.corrcoef(alternatives[:, i], labels)[0, 1]
        criteria_importance.append(corr_coef)

    num_criteria = len(criteria_importance)
    criteria_relations = torch.zeros((num_criteria, num_criteria))
    for i in range(num_criteria):
        for j in range(num_criteria):
            if i != j:
                criteria_relations[i, j] = criteria_importance[i] * criteria_importance[j]
    return criteria_relations