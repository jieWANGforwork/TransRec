import numpy as np
import torch

def hr(label, pos, k = 10):
    pos = len(pos)-pos[label]+1
    if pos<=k:
        return 1
    else:
        return 0

def auc_score(label, pos):
    m = 1
    n = len(pos)-1
    s = m * n
    auc = (pos[label] - (m * (m + 1) / 2)) / s
    return auc.to(torch.device("cpu")).detach().numpy()

def recall_score(y_true, y_score, k = 10):
    m = np.sum(y_true == 1)
    order = np.argsort(y_score)[::-1]
    y_pre = np.take(y_true, order[:k])
    tp = np.sum(y_pre == 1)
    return tp / m


def dcg_score(label, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    t_pos = np.where(order==label)
    if t_pos[0]>=k:
        return 0
    else:
        return 1/np.log2(t_pos[0] + 2)
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(label, pos, k=10):
    pos = len(pos)-pos[label]+1
    if pos>k:
        return 0
    else:
        s = 1/torch.log2(pos + 1)
        return s.to(torch.device("cpu")).detach().numpy()

def mrr_score(label, pos):
    pos = len(pos)-pos[label]+1
    rr_score = 1 / pos
    return rr_score.to(torch.device("cpu")).detach().numpy()


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)
