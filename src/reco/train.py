"""M4 - entrainement du MF avec early stopping sur le RMSE de validation."""
import argparse, json, os, time
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd
from reco.model import MF, score_matrix
from reco.metrics import rmse, mae, ranking_metrics

D = "data/processed"


def predict(net, df, ctx):
    u = mx.nd.array(df["user"].values, ctx=ctx)
    i = mx.nd.array(df["item"].values, ctx=ctx)
    p = net(u, i).asnumpy()
    return np.clip(p, 1, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--out", default="models/mf")
    a = ap.parse_args()

    mx.random.seed(0)
    np.random.seed(0)
    ctx = mx.cpu()
    train = pd.read_csv(f"{D}/train.csv")
    val = pd.read_csv(f"{D}/val.csv")
    items = pd.read_csv(f"{D}/items.csv")
    n_users, n_items = int(train["user"].max()) + 1, len(items)
    mean = float(train["rating"].mean())

    net = MF(n_users, n_items, a.k, mean, a.drop)
    net.initialize(mx.init.Normal(0.05), ctx=ctx)
    net.bu.weight.initialize(mx.init.Zero(), ctx=ctx, force_reinit=True)
    net.bi.weight.initialize(mx.init.Zero(), ctx=ctx, force_reinit=True)
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": a.lr, "wd": a.wd})
    loss_fn = gluon.loss.L2Loss()

    ds = gluon.data.ArrayDataset(train["user"].values.astype("float32"),
                                 train["item"].values.astype("float32"),
                                 train["rating"].values.astype("float32"))
    loader = gluon.data.DataLoader(ds, batch_size=a.bs, shuffle=True)

    best, bad = 9.9, 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ep in range(a.epochs):
        t0 = time.time()
        tot = 0
        for u, i, r in loader:
            with autograd.record():
                loss = loss_fn(net(u, i), r)
            loss.backward()
            trainer.step(u.shape[0])
            tot += loss.sum().asscalar()
        v = rmse(val["rating"].values, predict(net, val, ctx))
        print(f"ep {ep+1:2d} | loss {tot/len(train):.4f} | val RMSE {v:.4f} | {time.time()-t0:.1f}s")
        if v < best - 1e-4:
            best, bad = v, 0
            net.save_parameters(a.out + ".params")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break

    # on recharge le meilleur modele
    net.load_parameters(a.out + ".params", ctx=ctx)
    meta = {"n_users": n_users, "n_items": n_items, "k": a.k, "mean": mean, "val_rmse": best}
    json.dump(meta, open(a.out + ".json", "w"))
    p = predict(net, val, ctx)
    print(f"\nMEILLEUR val RMSE {best:.4f} | MAE {mae(val['rating'].values, p):.4f}")
    sc = score_matrix(net, n_users, n_items)
    r = ranking_metrics(sc, train, val, k=10)
    print("  ".join(f"{x} {y:.4f}" for x, y in r.items()))


if __name__ == "__main__":
    main()