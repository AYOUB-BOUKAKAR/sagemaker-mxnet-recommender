"""M3 - metriques : RMSE/MAE (notes) et Precision/Recall/NDCG/Hit @K (classement)."""
import numpy as np


def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def ranking_metrics(scores, train, target, k=10, thr=4):
    # scores : matrice (n_users, n_items), plus grand = mieux
    # on retire les films deja vus dans train, pertinent = note >= thr dans target
    seen = train.groupby("user")["item"].apply(set).to_dict()
    rel_all = target[target["rating"] >= thr].groupby("user")["item"].apply(set).to_dict()
    prec, rec, ndcg, hit = [], [], [], []
    for u, rel in rel_all.items():
        s = scores[u].astype(float).copy()
        if u in seen:
            s[list(seen[u])] = -np.inf
        top = np.argsort(-s, kind="stable")[:k]
        hits = np.array([1 if i in rel else 0 for i in top])
        dcg = (hits / np.log2(np.arange(len(top)) + 2)).sum()
        idcg = (1 / np.log2(np.arange(min(len(rel), k)) + 2)).sum()
        prec.append(hits.sum() / k)
        rec.append(hits.sum() / len(rel))
        ndcg.append(dcg / idcg)
        hit.append(1 if hits.sum() > 0 else 0)
    return {"P@%d" % k: np.mean(prec), "R@%d" % k: np.mean(rec),
            "NDCG@%d" % k: np.mean(ndcg), "Hit@%d" % k: np.mean(hit)}