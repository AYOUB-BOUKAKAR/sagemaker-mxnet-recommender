"""Point d'entree : SageMaker lance le conteneur avec l'argument 'serve'."""
import sys
import uvicorn

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "serve":
        sys.exit("commande inconnue: " + sys.argv[1])
    uvicorn.run("api.app:app", host="0.0.0.0", port=8080)