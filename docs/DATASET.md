# Jeu de donnees

- **Source :** GroupLens, MovieLens 100K - *<https://grouplens.org/datasets/movielens/100k/>*

- **Contenu :** 943 utilisateurs, 1682 films, 100 000 notes de 1 a 5, densite 6,3 %

- **Fichiers produits (data/processed) :** train.csv, val.csv, test.csv (user, item, rating, ts) et items.csv (item, title, 19 genres). Identifiants renumerotes a partir de 0.

- **Regle de split :** chronologique par utilisateur, 80 / 10 / 10, tri par user, ts, item, au moins 1 note en validation et en test pour chaque utilisateur.

- **Resultat :** train 80 024 / val 9 988 / test 9 988

- **Cold start :** 32 films de la validation et 45 films du test n'apparaissent pas dans le train.

- **Limites :** fenetre de 7 mois (sept. 1997 - avril 1998), biais de popularite (a chiffrer dans l'EDA).
