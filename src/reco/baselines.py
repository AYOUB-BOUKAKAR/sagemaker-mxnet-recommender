"""M3 - baselines : moyenne globale, moyenne par film, popularite, aleatoire."""
import sys
import numpy as np
import pandas as pd
from reco.metrics import rmse, mae, ranking_metrics

D = "data/processed"


def item_mean_table(train, n_items, damp=0):
    # damp > 0 : on tire les films peu notes vers la moyenne globale
    g = train["rating"].mean()
    s = train.groupby("item")["rating"].agg(["sum", "count"])
    m = np.full(n_items, g)
    m[s.index.values] = (s["sum"].values + damp * g) / (s["count"].values + damp)
    return m


def main(split="val"):
    train = pd.read_csv(f"{D}/train.csv")
    target = pd.read_csv(f"{D}/{split}.csv")
    items = pd.read_csv(f"{D}/items.csv")
    n_items = len(items)
    n_users = train["user"].max() + 1
    g = train["rating"].mean()
    rows = {}

    # --- notes (RMSE / MAE) ---
    y = target["rating"].values
    rows["moyenne globale"] = (rmse(y, np.full(len(y), g)), mae(y, np.full(len(y), g)))
    im = item_mean_table(train, n_items)             # film jamais vu -> moyenne globale
    p = im[target["item"].values]
    rows["moyenne par film"] = (rmse(y, p), mae(y, p))
    im10 = item_mean_table(train, n_items, damp=10)
    p = im10[target["item"].values]
    rows["moyenne par film (damp=10)"] = (rmse(y, p), mae(y, p))
    print(f"\n== RMSE / MAE sur {split} ==")
    for k, v in rows.items():
        print(f"{k:30s} RMSE {v[0]:.4f}  MAE {v[1]:.4f}")

    # --- classement (top-10) ---
    pop = train.groupby("item").size().reindex(range(n_items), fill_value=0).values
    rng = np.random.RandomState(0)
    sc = {
        "aleatoire": rng.rand(n_users, n_items),
        "popularite": np.tile(pop, (n_users, 1)),
        "moyenne film (damp=10)": np.tile(im10, (n_users, 1)),
    }
    print(f"\n== Classement top-10 sur {split} (pertinent = note >= 4) ==")
    for name, s in sc.items():
        r = ranking_metrics(s, train, target, k=10)
        print(f"{name:25s}", "  ".join(f"{a} {b:.4f}" for a, b in r.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")