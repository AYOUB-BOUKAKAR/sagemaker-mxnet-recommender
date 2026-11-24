"""M4b - meme modele MF mais perte BPR (classement), early stopping sur NDCG@10 val."""
import argparse, json, os, time
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd
from reco.model import MF, score_matrix
from reco.metrics import ranking_metrics

D = "data/processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--thr", type=int, default=4)       # note min pour compter comme positif
    ap.add_argument("--out", default="models/bpr")
    a = ap.parse_args()

    mx.random.seed(0)
    rng = np.random.RandomState(0)
    ctx = mx.cpu()
    train = pd.read_csv(f"{D}/train.csv")
    val = pd.read_csv(f"{D}/val.csv")
    items = pd.read_csv(f"{D}/items.csv")
    n_users, n_items = int(train["user"].max()) + 1, len(items)

    seen = np.zeros((n_users, n_items), dtype=bool)     # films deja vus (pour les negatifs)
    seen[train["user"].values, train["item"].values] = True
    pos = train[train["rating"] >= a.thr]
    pu = pos["user"].values
    pi = pos["item"].values

    net = MF(n_users, n_items, a.k, 0.0, 0.0)           # mean=0, ca ne change pas le classement
    net.initialize(mx.init.Normal(0.05), ctx=ctx)
    net.bu.weight.initialize(mx.init.Zero(), ctx=ctx, force_reinit=True)
    net.bi.weight.initialize(mx.init.Zero(), ctx=ctx, force_reinit=True)
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": a.lr, "wd": a.wd})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best, bad = -1, 0
    for ep in range(a.epochs):
        t0 = time.time()
        # negatifs tires au hasard, on retire ceux deja vus
        neg = rng.randint(0, n_items, len(pu))
        for _ in range(5):
            m = seen[pu, neg]
            if not m.any():
                break
            neg[m] = rng.randint(0, n_items, m.sum())
        order = rng.permutation(len(pu))
        tot = 0
        for s in range(0, len(order), a.bs):
            idx = order[s:s + a.bs]
            u = mx.nd.array(pu[idx], ctx=ctx)
            ip = mx.nd.array(pi[idx], ctx=ctx)
            ineg = mx.nd.array(neg[idx], ctx=ctx)
            with autograd.record():
                diff = net(u, ip) - net(u, ineg)
                loss = -mx.nd.log(mx.nd.sigmoid(diff) + 1e-10)
            loss.backward()
            trainer.step(len(idx))
            tot += loss.sum().asscalar()
        sc = score_matrix(net, n_users, n_items)
        r = ranking_metrics(sc, train, val, k=10)
        nd = r["NDCG@10"]
        print(f"ep {ep+1:3d} | loss {tot/len(pu):.4f} | val NDCG@10 {nd:.4f} Hit@10 {r['Hit@10']:.4f} | {time.time()-t0:.1f}s")
        if nd > best + 1e-4:
            best, bad = nd, 0
            net.save_parameters(a.out + ".params")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break

    net.load_parameters(a.out + ".params", ctx=ctx)
    json.dump({"n_users": n_users, "n_items": n_items, "k": a.k, "mean": 0.0, "val_ndcg": best},
              open(a.out + ".json", "w"))
    sc = score_matrix(net, n_users, n_items)
    r = ranking_metrics(sc, train, val, k=10)
    print("\nMEILLEUR (BPR) val :", "  ".join(f"{x} {y:.4f}" for x, y in r.items()))


if __name__ == "__main__":
    main()