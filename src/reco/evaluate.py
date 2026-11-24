"""M4c - evaluation finale sur le TEST (a lancer une seule fois, config deja choisie sur val)."""
import json
import numpy as np
import pandas as pd
import mxnet as mx
from reco.model import MF, score_matrix
from reco.metrics import rmse, mae, ranking_metrics
from reco.baselines import item_mean_table

D = "data/processed"


def load_net(path):
    meta = json.load(open(path + ".json"))
    net = MF(meta["n_users"], meta["n_items"], meta["k"], meta["mean"], 0.0)
    net.load_parameters(path + ".params", ctx=mx.cpu())
    return net, meta


def main():
    train = pd.read_csv(f"{D}/train.csv")
    val = pd.read_csv(f"{D}/val.csv")
    test = pd.read_csv(f"{D}/test.csv")
    items = pd.read_csv(f"{D}/items.csv")
    n_items = len(items)
    n_users = int(train["user"].max()) + 1
    g = train["rating"].mean()
    y = test["rating"].values
    cold = ~test["item"].isin(set(train["item"])).values    # films jamais vus -> moyenne globale
    known = pd.concat([train, val])                          # deja vus par l'utilisateur
    res = {}

    # notes
    res["moyenne globale"] = [rmse(y, np.full(len(y), g)), mae(y, np.full(len(y), g))]
    p = item_mean_table(train, n_items)[test["item"].values]
    res["moyenne par film"] = [rmse(y, p), mae(y, p)]
    net, meta = load_net("models/mf")
    p = net(mx.nd.array(test["user"].values), mx.nd.array(test["item"].values)).asnumpy()
    p = np.clip(p, 1, 5)
    p[cold] = g
    res["MF (MXNet)"] = [rmse(y, p), mae(y, p)]
    print(f"\n== TEST : notes (cold start remplace par la moyenne globale : {cold.sum()} notes) ==")
    for k, v in res.items():
        print(f"{k:25s} RMSE {v[0]:.4f}  MAE {v[1]:.4f}")

    # classement
    pop = train.groupby("item").size().reindex(range(n_items), fill_value=0).values
    bnet, _ = load_net("models/bpr")
    rng = np.random.RandomState(0)
    sc = {
        "aleatoire": rng.rand(n_users, n_items),
        "popularite": np.tile(pop, (n_users, 1)),
        "MF notes (MXNet)": score_matrix(net, n_users, n_items),
        "MF BPR (MXNet)": score_matrix(bnet, n_users, n_items),
    }
    rank = {}
    print("\n== TEST : classement top-10 (pertinent = note >= 4) ==")
    for name, s in sc.items():
        r = ranking_metrics(s, known, test, k=10)
        rank[name] = {a: float(b) for a, b in r.items()}
        print(f"{name:25s}", "  ".join(f"{a} {b:.4f}" for a, b in r.items()))

    json.dump({"notes": res, "classement": rank}, open("reports/metrics_test.json", "w"), indent=2)
    print("\necrit : reports/metrics_test.json")


if __name__ == "__main__":
    main()