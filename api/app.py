"""M5 - API de recommandation. Contrat SageMaker : GET /ping, POST /invocations, port 8080."""
import json
import os
import numpy as np
import pandas as pd
import mxnet as mx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from reco.model import MF, score_matrix

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")
app = FastAPI(title="reco-mxnet")
S = {}


def load_net(name):
    meta = json.load(open(os.path.join(MODEL_DIR, name + ".json")))
    net = MF(meta["n_users"], meta["n_items"], meta["k"], meta["mean"], 0.0)
    net.load_parameters(os.path.join(MODEL_DIR, name + ".params"), ctx=mx.cpu())
    return net, meta


@app.on_event("startup")
def startup():
    mf, meta = load_net("mf")
    bpr, _ = load_net("bpr")
    n_users, n_items = meta["n_users"], meta["n_items"]
    seen = pd.read_csv(os.path.join(MODEL_DIR, "seen.csv"))
    items = pd.read_csv(os.path.join(MODEL_DIR, "items.csv"))
    S["n_users"] = n_users
    S["n_items"] = n_items
    S["mean"] = meta["mean"]
    S["mf_sc"] = np.clip(score_matrix(mf, n_users, n_items), 1, 5)
    S["bpr_sc"] = score_matrix(bpr, n_users, n_items)
    S["seen"] = seen.groupby("user")["item"].apply(set).to_dict()
    S["train_items"] = set(seen[seen["split"] == "train"]["item"])
    S["pop"] = seen.groupby("item").size().reindex(range(n_items), fill_value=0).values
    S["titles"] = items.set_index("item")["title"].to_dict()


def recommend(user, k):
    if 0 <= user < S["n_users"]:
        s = S["bpr_sc"][user].astype(float).copy()
        s[list(S["seen"].get(user, []))] = -np.inf
        mode = "personnalise"
    else:
        s = S["pop"].astype(float).copy()     # utilisateur inconnu -> films populaires
        mode = "popularite (cold start)"
    top = np.argsort(-s, kind="stable")[:k]
    recs = [{"item": int(i), "title": S["titles"].get(int(i), "?"), "score": round(float(s[i]), 4)} for i in top]
    return {"user": user, "mode": mode, "recommendations": recs}


def predict(user, item):
    cold = not (0 <= user < S["n_users"]) or item not in S["train_items"]
    if cold:
        return {"user": user, "item": item, "rating": round(S["mean"], 3), "mode": "moyenne globale (cold start)"}
    return {"user": user, "item": item, "rating": round(float(S["mf_sc"][user, item]), 3), "mode": "MF"}


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/recommend/{user}")
def recommend_get(user: int, k: int = 10):
    return recommend(user, k)


@app.post("/invocations")
async def invocations(request: Request):
    try:
        body = await request.json()
        task = body.get("task", "recommend")
        if task == "recommend":
            return recommend(int(body["user"]), int(body.get("k", 10)))
        if task == "predict":
            return predict(int(body["user"]), int(body["item"]))
        return JSONResponse({"error": "task inconnue (recommend ou predict)"}, status_code=400)
    except (KeyError, ValueError, TypeError) as e:
        return JSONResponse({"error": "requete invalide: " + str(e)}, status_code=400)