from src.paginas import Paginas


def test_metricas_clustering():
    # Given
    teste_entrada = "Métricas Calculadas"

    # When
    iniciando = Paginas()
    teste_saida = iniciando.metricas()

    # Then
    assert teste_entrada == teste_saida
