# notebooks/src/paths.py
from pathlib import Path


def encontrar_raiz_repo(marcadores=(".git", "data")) -> Path:
    for pasta in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if all((pasta / m).exists() for m in marcadores):
            return pasta
    raise FileNotFoundError(f"Nao achei {marcadores} subindo a partir de {Path.cwd()}")


RAIZ = encontrar_raiz_repo()
PARQUET = RAIZ / "data/argos_project/parquet_data"
REFGENES = RAIZ / "data/raw/refgenes.tsv"