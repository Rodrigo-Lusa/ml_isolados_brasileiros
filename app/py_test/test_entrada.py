from main import pagina_estrutura, pagina_inicio


def test_estrutura_pagina():
    # Given
    teste_entrada = "Ajustado"

    # When
    teste_saida = pagina_estrutura()

    # Then
    assert teste_entrada == teste_saida


def test_iniciando_sistema():
    # Given
    teste_entrada = "Diga lá loco"

    # When
    teste_saida = pagina_inicio(entrada=teste_entrada)

    # Then
    assert teste_entrada == teste_saida
