import os

from src.paginas import Paginas, MODELOS_CLASSIF
from src.acesso_data import acesso_data_banco


def test_acesso_data_banco():
    # Given
    teste_entrada = "app/data/"

    # When
    teste_saida = acesso_data_banco()

    # Then
    assert teste_saida.endswith(teste_entrada)
    assert os.path.isdir(teste_saida)


def test_busca_dados_cluster():
    # Given
    teste_entrada = (1558, 445, 223)

    # When
    iniciando = Paginas()
    df_train, df_valid, df_test = iniciando.busca_dados_cluster()
    teste_saida = (len(df_train), len(df_valid), len(df_test))

    # Then
    assert teste_entrada == teste_saida


def test_busca_dados_class():
    # Given
    teste_entrada_1 = len(MODELOS_CLASSIF)
    teste_entrada_2 = 223

    # When
    iniciando = Paginas()
    teste_saida_1 = len(iniciando.busca_dados_class()[0])
    teste_saida_2 = len(iniciando.busca_dados_class()[1])

    # Then
    assert teste_entrada_1 == teste_saida_1
    assert teste_entrada_2 == teste_saida_2


def test_busca_dados_regr():
    # Given
    teste_entrada_1 = 1558
    teste_entrada_2 = 223

    # When
    iniciando = Paginas()
    teste_saida_1 = len(iniciando.busca_dados_regr()[0])
    teste_saida_2 = len(iniciando.busca_dados_regr()[1])

    # Then
    assert teste_entrada_1 == teste_saida_1
    assert teste_entrada_2 == teste_saida_2
