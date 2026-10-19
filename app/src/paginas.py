import os
import pandas as pd
import streamlit as st
from joblib import load
from pydantic import validate_call

from src.acesso_data import acesso_data_banco
from src.modelos.clustering import Clustering
from src.modelos.classifier import Classifier
from src.modelos.regression import Regression

# Clustering: features e parâmetros escolhidos no notebook 04 (seções 1 e 3)
FEATURES_CLUSTER = [
    "has_carbapenemase",
    "has_mbl",
    "has_esbl",
    "n_bl_adquiridas",
    "has_16s_metilase",
    "carbapenemase_plasmidial",
]
PARAMS_CLUSTER = {
    "K-means": ("kmeans", {"n_clusters": 6, "init": "k-means++", "n_init": 10}),
    "Agglomerative": (
        "agglomerative",
        {"n_clusters": 6, "metric": "euclidean", "linkage": "ward"},
    ),
    # min_samples = 3% do treino (1558), reescalado no conjunto comparado
    "DBSCAN": ("dbscan", {"eps": 0.75, "min_samples": 47}),
}

# Classificação: arquivos df_resultados/classif_<modelo> do notebook 05
PREFIXO_CLASSIF = "classif_"
MODELOS_CLASSIF = {
    "knn": "k-NN",
    "svm": "SVM",
    "rf": "RF",
    "gbm": "GBM",
    "nb": "NB",
    "nn": "NN",
}

# Regressão: colunas de train/test_resultados_regr do notebook 06
ALVO_REGR = "risco_idx"
MODELOS_REGR = [
    "Multiple Linear Regression",
    "Suport Vector Machine",
    "Nayve Bayes",
    "Neural Network",
]


class Paginas:
    @st.cache_data
    @validate_call
    def busca_dados_cluster(_self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Função de busca dos dados de treino, validação e teste
        (os mesmos da clusterização e da classificação).
        Retorna:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            DataFrames dos datasets de treino, validação e teste.
        """
        link = acesso_data_banco() + "df_estruturada/"
        df_train = pd.read_csv(link + "df_train_risco.csv", index_col=0)
        df_valid = pd.read_csv(link + "df_valid_risco.csv", index_col=0)
        df_test = pd.read_csv(link + "df_test_risco.csv", index_col=0)

        return df_train, df_valid, df_test

    @st.cache_data
    @validate_call
    def busca_dados_class(_self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Função de busca dos resultados da classificação.
        Retorna:
            tuple[dict[str, pd.DataFrame], pd.DataFrame]:
            {modelo: DataFrame com Dataset e Classe_Predita} e o DataFrame
            de teste com a classe real e uma coluna por modelo.
        """
        banco = acesso_data_banco()
        link = banco + "df_resultados/"
        datasets = {
            i.removeprefix(PREFIXO_CLASSIF): load(link + i)
            for i in sorted(os.listdir(link))
            if i.startswith(PREFIXO_CLASSIF)
        }

        df_test = load(banco + "test_resultados_classif")

        return datasets, df_test

    @st.cache_data
    @validate_call
    def busca_dados_regr(_self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Função de busca dos resultados da regressão.
        Retorna:
            tuple[pd.DataFrame, pd.DataFrame]:
            DataFrames de treino e teste (alvo + previsão de cada modelo).
        """
        banco = acesso_data_banco()
        cols = [ALVO_REGR, *MODELOS_REGR]
        df_train = load(banco + "train_resultados_regr")[cols].copy()
        df_test = load(banco + "test_resultados_regr")[cols].copy()

        return df_train, df_test

    @st.cache_data
    def __clustering(
        _self,
        conjunto: str,
        df_train: pd.DataFrame,
        df_compara: pd.DataFrame,
    ) -> dict[str, tuple]:
        """
        Função de teste dos modelos de clustering.
        Parametros:
            conjunto: nome do conjunto comparado com o treino.
            df_train: DataFrame de treino.
            df_compara: DataFrame de validação ou teste.
        Retorna:
            dict[str, tuple]: {algoritmo: (gráfico, mensagem)}.
        """
        resultados = {}
        for nome, (modelo, best_param) in PARAMS_CLUSTER.items():
            cluster = Clustering(
                features=FEATURES_CLUSTER, df=df_train, modelo=modelo
            )
            resultados[nome] = cluster.testar_modelo(
                best_param=best_param, df_test=df_compara
            )

        return resultados

    @st.cache_data
    def __classifier(_self) -> None:
        dados, df_test_class = _self.busca_dados_class()

        modelos = list(MODELOS_CLASSIF)
        for linha in (modelos[:3], modelos[3:]):
            col = st.columns((5, 0.5, 5, 0.5, 5))
            for k, modelo in zip((0, 2, 4), linha):
                with col[k]:
                    st.write("=" * 69)
                    st.markdown(
                        "<h4 style='text-align: center; '>"
                        f"{MODELOS_CLASSIF[modelo]}</h4>",
                        unsafe_allow_html=True,
                    )
                    st.write("=" * 69)
                    classificador = Classifier(df=dados[modelo])
                    classificador.testar_hipotese()
                    st.divider()
                    classificador.matrix_confusao(
                        df=df_test_class, obs="cluster", pred=modelo
                    )
            st.divider()

        return None

    @st.cache_data
    def __regressor(_self) -> None:
        df_train_regr, df_test_regr = _self.busca_dados_regr()
        col = st.columns((5, 0.5, 5, 0.5, 5, 0.5, 5))
        for k, v in zip((0, 2, 4, 6), MODELOS_REGR):
            with col[k]:
                regressor = Regression(
                    y_pred_treino=df_train_regr[v], y_pred_teste=df_test_regr[v]
                )
                regressor.testar_hipotese(y_pred=v)
                st.divider()
                regressor.obter_grafico(df_test=df_test_regr, y_true=ALVO_REGR, y_pred=v)
        return None

    @st.cache_data
    def __datasets(_self) -> None:
        df_train, df_valid, df_test = _self.busca_dados_cluster()
        col = st.columns((2, 0.05, 2, 0.05, 2))
        with col[0]:
            st.markdown(
                "<h4 style='text-align: center; '>Dados de Treino</h4>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_train)

        with col[2]:
            st.markdown(
                "<h4 style='text-align: center; '>Dados de Validação</h4>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_valid)

        with col[4]:
            st.markdown(
                "<h4 style='text-align: center; '>Dados de Teste</h4>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_test)

        col = st.columns((2, 10))
        with col[0]:
            metr = pd.DataFrame(
                {
                    "Métrica": ["Treino", "Validação", "Teste"],
                    "Tamanho": [len(df_train), len(df_valid), len(df_test)],
                }
            )
            st.write("Tamanho dos datasets:")
            st.dataframe(metr)
        return None

    def metricas(self) -> str:
        """
        Monta as quatro abas do app.
        Retorna:
            str: Mensagem de sucesso.
        """
        abas = st.tabs(
            [
                "Clustering",
                "Classificação",
                "Regressão",
                "Datasets",
            ]
        )
        with abas[0]:
            df_train, df_valid, df_test = self.busca_dados_cluster()
            col = st.columns((1.2, 3, 0.05, 3, 0.05, 3))
            with col[0]:
                conjunto = st.radio("Comparar o treino com:", ["Validação", "Teste"])
            df_compara = df_valid if conjunto == "Validação" else df_test

            resultados = self.__clustering(
                conjunto=conjunto, df_train=df_train, df_compara=df_compara
            )
            for k, (grafico, msn) in zip((1, 3, 5), resultados.values()):
                with col[k]:
                    st.markdown(f"<p>{msn}</p>", unsafe_allow_html=True)
                    st.pyplot(fig=grafico)

        with abas[1]:
            self.__classifier()

        with abas[2]:
            self.__regressor()

        with abas[3]:
            self.__datasets()

        return "Métricas Calculadas"
