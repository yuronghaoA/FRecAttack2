import torch
from parse import args
import numpy as np


def evaluate_recall(rating, ground_truth):
    recall = []
    for k in eval(args.top_k):
        _, rating_k = torch.topk(rating, k)
        rating_k = rating_k.tolist()

        hit = 0
        for i, v in enumerate(rating_k):
            if v in ground_truth:
                hit += 1

        rec_ = hit / len(ground_truth)
        recall.append(rec_)
    return torch.tensor(recall)


def evaluate_precision(rating, ground_truth):
    precision = []
    for k in eval(args.top_k):
        _, rating_k = torch.topk(rating, k)
        rating_k = rating_k.tolist()

        hit = 0
        for i, v in enumerate(rating_k):
            if v in ground_truth:
                hit += 1

        prec_ = hit / k
        precision.append(prec_)
    return torch.tensor(precision)


def evaluate_ndcg(rating, ground_truth):
    ndcg = []
    for k in eval(args.top_k):
        _, rating_k = torch.topk(rating, k)
        rating_k = rating_k.tolist()
        dcg, idcg = 0., 0.

        for i, v in enumerate(rating_k):
            if i < len(ground_truth):
                idcg += (1 / np.log2(2 + i))
            if v in ground_truth:
                dcg += (1 / np.log2(2 + i))

        ndcg_ = dcg / idcg
        ndcg.append(ndcg_)
    return torch.tensor(ndcg)
