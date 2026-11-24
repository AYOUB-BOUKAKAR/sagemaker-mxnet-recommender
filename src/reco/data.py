"""M1 - chargement de MovieLens 100K + split train/val/test par utilisateur."""
import os
import numpy as np
import pandas as pd

RAW = "data/raw/ml-100k"
OUT = "data/processed"

# meme ordre que dans le fichier u.genre
GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
          "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def load_ratings(path=RAW):
    # u.data : user, item, rating, timestamp (separes par des tabulations, pas d'en-tete)
    df = pd.read_csv(os.path.join(path, "u.data"), sep="\t",
                     names=["user", "item", "rating", "ts"])
    # les ids commencent a 1, on passe a 0 pour les embeddings
    df["user"] = df["user"] - 1
    df["item"] = df["item"] - 1
    return df


def load_items(path=RAW):
    # u.item : 5 premieres colonnes (id, titre, dates, url) puis 19 colonnes de genres
    cols = ["item", "title", "date", "video_date", "url"] + GENRES
    df = pd.read_csv(os.path.join(path, "u.item"), sep="|", header=None,
                     names=cols, encoding="latin-1")   # latin-1, sinon les accents plantent
    df = df[["item", "title"] + GENRES].copy()
    df["item"] = df["item"] - 1
    return df


def split_by_user(df, val_frac=0.1, test_frac=0.1):
    # tri fixe pour avoir toujours le meme resultat
    df = df.sort_values(["user", "ts", "item"]).reset_index(drop=True)

    rank = df.groupby("user").cumcount()                  # position de la note chez l'utilisateur
    total = df.groupby("user")["item"].transform("size")  # nb de notes de l'utilisateur

    # nb de notes en val et en test pour chaque utilisateur (au moins 1)
    n_val = (total * val_frac).round().clip(lower=1)
    n_test = (total * test_frac).round().clip(lower=1)
    n_train = total - n_val - n_test

    # les plus anciennes -> train, ensuite val, les plus recentes -> test
    part = np.where(rank < n_train, "train",
                    np.where(rank < n_train + n_val, "val", "test"))

    train = df[part == "train"]
    val = df[part == "val"]
    test = df[part == "test"]
    return train, val, test


def check_split(df, train, val, test):
    # 1) rien de perdu ni de duplique
    assert len(train) + len(val) + len(test) == len(df)
    # 2) notes entre 1 et 5, pas de valeurs manquantes
    assert df["rating"].between(1, 5).all()
    assert not df.isna().any().any()
    # 3) chaque utilisateur est present dans les 3 parties
    n = df["user"].nunique()
    assert train["user"].nunique() == val["user"].nunique() == test["user"].nunique() == n
    # 4) pour chaque utilisateur : train <= val <= test dans le temps
    last_train = train.groupby("user")["ts"].max()
    first_val = val.groupby("user")["ts"].min()
    last_val = val.groupby("user")["ts"].max()
    first_test = test.groupby("user")["ts"].min()
    assert (last_train <= first_val).all()
    assert (last_val <= first_test).all()
    print("checks OK")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    ratings = load_ratings()
    items = load_items()
    train, val, test = split_by_user(ratings)
    check_split(ratings, train, val, test)

    train.to_csv(os.path.join(OUT, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT, "test.csv"), index=False)
    items.to_csv(os.path.join(OUT, "items.csv"), index=False)

    print("total :", len(ratings), "| users :", ratings["user"].nunique(),
          "| items :", ratings["item"].nunique())
    print("train :", len(train), "| val :", len(val), "| test :", len(test))

    # films qui n'existent pas dans le train (cold start)
    seen = set(train["item"])
    print("films absents du train (val) :", len(set(val["item"]) - seen))
    print("films absents du train (test):", len(set(test["item"]) - seen))