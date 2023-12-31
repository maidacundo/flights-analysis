import numpy as np

def percentile_graph(scores, passengers, num=10):
    limits = np.linspace(0, 100, num=num+1)
    percentile_scores = []
    for i in range(len(limits) - 1):
        start = limits[i]
        end = limits[i + 1]
        relevant_scores = [scores[k] for k in passengers if passengers[k] > start and passengers[k] <= end]
        percentile_scores.append(np.mean(relevant_scores))
    return percentile_scores

def weighted_average(scores, weights):
    relevant_weights = [weights[k] for k in scores.keys()]
    if sum(relevant_weights) == 0:
        return 0
    else:
        return sum([scores[k] * weights[k] for k in scores.keys()]) / sum(relevant_weights)