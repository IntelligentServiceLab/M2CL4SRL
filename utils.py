import torch
import random
import numpy as np
from torch_geometric.utils import structured_negative_sampling
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')









def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """

    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

# def MAP_ATk(groundTruth, r, k):
#     """Computers MAP @ k
#
#     Args:
#         groundTruth (list): list of lists containing highly rated items of each user
#         r (list): list of lists indicating whether each top k item recommended to each user
#             is a top k ground truth item or not
#         k (intg): determines the top k items to compute precision and recall on
#
#     Returns:
#         float: MAP @ k
#     """
#
#     c=1
#     AP_at_k = []  # list to store the average precision at k for each user
#     for i in range(len(groundTruth)):  # for each user
#         top_k = r[i]  # get the top k items recommended to the user
#         num_correct_pred = 0  # number of correctly predicted items so far
#         precisions = []  # list to store the precision at each relevant item
#         for j in range(len(top_k)):  # for each item in the top k
#             if top_k[j] > 0:  # if the item is a relevant item
#                 num_correct_pred += 1  # increment the number of correctly predicted items
#                 precision = num_correct_pred / (j + 1)  # compute the precision at this item
#                 precisions.append(precision)  # add the precision to the list
#         if precisions:  # if there is at least one relevant item in the top k
#             AP_at_k.append(sum(precisions) / len(precisions))  # compute the average precision at k for this user
#         else:  # if there are no relevant items in the top k
#             AP_at_k.append(0)  # the average precision at k for this user is 0
#     MAP_at_k = sum(AP_at_k) / len(AP_at_k)  # compute the mean average precision at k
#
#     a=1
#
#     return MAP_at_k

def compute_AP_at_k(r, groundTruth, k):
    if len(r)>k:
        r = r[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(r):
        if p and p not in r[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if num_hits == 0.0:
        return 0.0

    return score / min(len(groundTruth), k)

def compute_MAP_at_k(r_list, groundTruth, k):
    all_scores = []
    for r, gt in zip(r_list, groundTruth):
        score = compute_AP_at_k(r, gt, k)
        all_scores.append(score)

    return sum(all_scores) / len(all_scores)



def HitRate_ATk(groundTruth, r, k):
    """Computers hit rate @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        float: hit rate @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))])
    hit_rate = torch.mean(num_correct_pred / user_num_liked)
    return hit_rate.item()




def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

