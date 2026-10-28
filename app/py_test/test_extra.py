import os

import joblib
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from matplotlib.figure import Figure

import src.paginas as paginas
from src.graficos import (
    grafico_concordancia,
    grafico_corr,
    grafico_metricas,
    grafico_perfil,
    grafico_split,
)
from src.paginas import MODELOS_CLASSIF, PREFIXO_CLASSIF, Paginas, _card_github

# Resultados de classificação de brinquedo: os arquivos reais de
# app/data foram salvos com numpy 2 e não abrem com o numpy 1.23.5 do Docker.
CLASSES = [0, 1, 2] * 5
DF_CLASSIF = pd.DataFrame(
    {
        "Dataset": ["Treino"] * len(CLASSES) + ["Teste"] * len(CLASSES),
        "Classe_Predita": CLASSES * 2,
    }
)
DF_TESTE_CLASSIF = pd.DataFrame(
    {"cluster": CLASSES, **{m: CLASSES for m in MODELOS_CLASSIF}}
)


def _load_classif_falso(caminho: str):
    nome = os.path.basename(caminho)
    if nome == "test_resultados_classif":
        return DF_TESTE_CLASSIF
    if nome.startswith(PREFIXO_CLASSIF):
        return DF_CLASSIF
    return joblib.load(caminho)


@pytest.fixture
def classif_falso(monkeypatch):
    monkeypatch.setattr(paginas, "load", _load_classif_falso)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


# ------------------------------------------------------------------ gráficos
def test_grafico_split():
    # Given
    teste_entrada = {"Treino": 70, "Validação": 20, "Teste": 10}

    # When
    teste_saida = grafico_split(teste_entrada)

    # Then
    barras = teste_saida.axes[0].patches
    assert isinstance(teste_saida, Figure)
    assert len(barras) == 3
    assert sum(b.get_height() for b in barras) == pytest.approx(100)


def test_grafico_corr():
    # Given
    teste_entrada = pd.DataFrame(
        [[1, 0.3], [0.3, 1]], index=["a", "b"], columns=["a", "b"]
    )

    # When
    teste_saida = grafico_corr(teste_entrada, "Correlação")

    # Then
    assert teste_saida.axes[0].get_title() == "Correlação"


def test_grafico_metricas():
    # Given
    teste_entrada = pd.DataFrame(
        {
            "modelo": ["A", "A", "A", "B", "B", "B"],
            "conjunto": ["treino", "validação", "teste"] * 2,
            "r2": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "mae": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        }
    )
    metricas = {"r2": "R²", "mae": "MAE"}

    # When
    teste_saida = grafico_metricas(
        teste_entrada, metricas, "Teste", referencias={"r2": 0.5}
    )

    # Then
    titulos = [ax.get_title() for ax in teste_saida.axes[:2]]
    tracejadas = [
        list(ln.get_ydata())
        for ln in teste_saida.axes[0].lines
        if ln.get_linestyle() == "--"
    ]
    assert titulos == ["R²", "MAE"]
    assert tracejadas == [[0.5, 0.5]]  # linha de referência do r2


def test_grafico_concordancia():
    # Given
    teste_entrada = pd.DataFrame(
        [[1, 0.2], [0.2, 1]], index=["kmeans", "dbscan"],
        columns=["kmeans", "dbscan"],
    )

    # When
    teste_saida = grafico_concordancia(teste_entrada)

    # Then
    assert teste_saida.axes[0].get_title() == "Adjusted Rand Index (treino)"


def test_grafico_perfil():
    # Given
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((30, 3)), columns=["f1", "f2", "d1"])
    labels = pd.Series([0, 1, 2] * 10, index=df.index)

    # When
    teste_saida = grafico_perfil(df, labels, ["f1", "f2", "d1"], n_sep=2)

    # Then
    assert teste_saida.axes[0].get_title() == "Perfil por cluster"
    assert len(teste_saida.axes[0].lines) == 1  # linha que separa as features


# ------------------------------------------------------------------- páginas
def test_busca_dados_class_falso(classif_falso):
    # Given
    teste_entrada_1 = len(MODELOS_CLASSIF)
    teste_entrada_2 = len(DF_TESTE_CLASSIF)

    # When
    datasets, df_test = Paginas().busca_dados_class()

    # Then
    assert teste_entrada_1 == len(datasets)
    assert teste_entrada_2 == len(df_test)


def test_card_github():
    # When
    teste_saida = _card_github()

    # Then
    assert teste_saida.startswith("iVBOR")  # PNG em base64


def test_pagina_pre_processamento():
    assert Paginas().pre_processamento() is None


def test_pagina_clustering():
    assert Paginas().clustering() is None


def test_pagina_classificacao(classif_falso):
    assert Paginas().classificacao() is None


def test_pagina_regressao():
    assert Paginas().regressao() is None
