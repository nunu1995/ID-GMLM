import numpy as np
import torch
import math


def aCC(output, label):
    _, indices = torch.max(output, dim=1)
    corr = torch.sum(indices == label)
    return corr.item() * 1.0 / len(label)


def dCG(scores):
    v = 0
    for dcgi in range(len(scores)):
        v += (np.power(2, scores[dcgi]) - 1) / np.log2(dcgi+2)
    return v


def iDCG(scores):
    best_scores = sorted(scores)[::-1]
    return dCG(best_scores)


def nDCG(scores):
    return dCG(scores)/iDCG(scores)


def nDCG_k(scores, k):
    scores_k = scores[:k]
    fenzi = dCG(scores_k)
    fenmu = dCG(sorted(scores)[::-1][:k])
    return fenzi/fenmu


def cINDEX(predicted, true):
    correct_count = 0
    incorrect_count = 0

    for i in range(len(predicted)):
        for j in range(len(true)):
            if i != j:
                if predicted[i] >= predicted[j] and true[i] >= true[j] or \
                   predicted[i] <= predicted[j] and true[i] <= true[j] or \
                   predicted[i] == predicted[j] and true[i] == true[j]:
                    correct_count += 1
                else:
                    incorrect_count += 1
    return correct_count / (correct_count + incorrect_count)


PrecisionRel = {2: 1, 1: 0}


def aP(rates):
    numRelevant = 0
    avgPrecision = 0.0
    for iPos in range(len(rates)):
        if PrecisionRel[rates[iPos]] == 1:
            numRelevant += 1
            avgPrecision += (numRelevant / (iPos + 1))
    return 0.0 if numRelevant == 0 else avgPrecision / numRelevant


def aPK(rates, k):
    numRelevant = 0
    avgPrecision = 0.0
    for iPos in range(k):
        if PrecisionRel[rates[iPos]] == 1:
            numRelevant += 1
            avgPrecision += (numRelevant / (iPos + 1))
    return 0.0 if numRelevant == 0 else avgPrecision / numRelevant


def sPEAR(predicted, true):
    sparM = []
    spaD1 = []
    spaD2 = []
    for i in range(len(predicted)):
        sparM.append((predicted[i] - np.mean(predicted)) * (true[i] - np.mean(true)))
        spaD1.append((predicted[i] - np.mean(predicted)) ** 2)
        spaD2.append((true[i] - np.mean(true)) ** 2)
    return np.sum(sparM) / (math.sqrt(np.sum(spaD1)) * math.sqrt(np.sum(spaD2)))




