"""M5 - prepare les fichiers dont l'API a besoin a cote des modeles (models/)."""
import pandas as pd

D = "data/processed"
train = pd.read_csv(f"{D}/train.csv")
val = pd.read_csv(f"{D}/val.csv")
train["split"] = "train"
val["split"] = "val"
pd.concat([train, val])[["user", "item", "rating", "split"]].to_csv("models/seen.csv", index=False)
pd.read_csv(f"{D}/items.csv")[["item", "title"]].to_csv("models/items.csv", index=False)
print("ok : models/seen.csv + models/items.csv")