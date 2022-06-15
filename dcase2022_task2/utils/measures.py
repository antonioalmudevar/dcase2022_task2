import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def cross_entropy(pred, labels):
    return -torch.sum(pred.log_softmax(dim=-1)*labels, axis=-1)

def cosine_distance(a, b):
    norm_a = torch.norm(a, dim=len(a.shape)-1)
    norm_b = torch.norm(b, dim=len(b.shape)-1)
    return 1-torch.matmul(b,a)/(norm_a*norm_b)

def auc(normal, anomaly, p=1):
    if len(normal)==0 or len(anomaly)==0:
        return 0.5
    else:
        labels = np.concatenate((np.zeros(len(normal)), np.ones(len(anomaly))))
        pred = np.concatenate((normal, anomaly))
        return roc_auc_score(labels, pred, max_fpr=p)

def mse(x, y):
    return ((x-y)**2).reshape(x.shape[0],-1).mean(axis=1)