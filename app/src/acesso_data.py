from pathlib import Path


def acesso_data_banco() -> str:
    """
    Função para acessar a pasta de dados do app (app/data/).
    O caminho é relativo a este arquivo, então funciona tanto rodando da
    raiz do repo quanto de dentro de app/.
    """
    path_banco = Path(__file__).resolve().parents[1] / "data"

    return str(path_banco) + "/"
