# Systeme de recommandation - MXNet + Amazon SageMaker

Mini-projet Cloud - recommandation de films avec Apache MXNet,
deploiement sur Amazon SageMaker.

## Etat d'avancement
- [ ] M0 environnement Docker
- [ ] M1 donnees (MovieLens 100K) + DATASET.md
- [ ] M2 EDA (notebook)
- [ ] M3 baselines + metriques
- [ ] M4 modele MXNet + evaluation
- [ ] M5 API FastAPI
- [ ] M6 deploiement SageMaker
- [ ] M7 documentation

## Lancer l'environnement (PowerShell)
    docker build -t reco-mxnet .
    docker run --rm reco-mxnet python -c "import mxnet as mx; print(mx.__version__)"

## Notebook
    docker run --rm -it -v "${PWD}:/app" -p 8888:8888 reco-mxnet jupyter lab --ip=0.0.0.0 --no-browser --allow-root
