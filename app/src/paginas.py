import base64
import json
import os

import pandas as pd
import streamlit as st
from joblib import load
from pydantic import validate_call

from src.acesso_data import acesso_data_banco
from src.graficos import (
    CORES_SPLIT,
    grafico_concordancia,
    grafico_corr,
    grafico_metricas,
    grafico_perfil,
    grafico_split,
)
from src.modelos.clustering import Clustering
from src.modelos.classifier import Classifier
from src.modelos.regression import Regression

# Clustering: features (final/01) e parâmetros (final/02) vêm de app/data/clustering/
ALGORITMOS_CLUSTER = {
    "K-means": "kmeans",
    "Agglomerative": "agglomerative",
    "DBSCAN": "dbscan",
}
METRICAS_CLUSTER = {
    "silhouette": "Silhouette (↑)",
    "davies_bouldin": "Davies-Bouldin (↓)",
    "calinski_harabasz": "Calinski-Harabasz (↑)",
    "ari_projecao_x_refit": "ARI projeção × refit (↑)",
    "qui2_p": "Qui² p-valor (> 0,05)",
}

# Classificação: arquivos df_resultados/classif_<modelo> do notebook final/03
PREFIXO_CLASSIF = "classif_"
MODELOS_CLASSIF = {
    "knn": "k-NN",
    "svm": "SVM",
    "rf": "RF",
    "gbm": "GBM",
    "nb": "NB",
    "nn": "NN",
}
METRICAS_CLASSIF = {
    "acuracia": "Acurácia",
    "acuracia_balanceada": "Acurácia balanceada",
    "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)",
    "f1_macro": "F1 (macro)",
}

# Regressão: colunas de train/test_resultados_regr do notebook final/04
ALVO_REGR = "risco_idx"
CONTEXTO_REGR = ["Species", "source_type", "Date", "Date_na"]
MODELOS_REGR = [
    "Multiple Linear Regression",
    "Support Vector Machine",
    "Bayesian Ridge",
    "Neural Network",
]
METRICAS_REGR = {
    "spearman": "Spearman (↑)",
    "r2": "R² (↑)",
    "mae": "MAE (↓)",
    "rmse": "RMSE (↓)",
    "mae_dp": "MAE / DP (↓)",
}

URL_PROFESSOR ="https://github.com/Laboratorio-de-Analise-de-Dados/disciplina_python_ml"
URL_REPO = "https://github.com/Rodrigo-Lusa/ml_isolados_brasileiros"

TEXTO_PRE = """
Os genomas bacterianos utilizados neste estudo foram obtidos, em 03/02/2026, a partir
de dois repositórios públicos: BV-BRC v3.58.4 ([BV-BRC](https://www.bv-brc.org/)) e
NCBI Pathogen Detection ([NCBI-Pathogens](https://www.ncbi.nlm.nih.gov/pathogens/)).
No BV-BRC, foram aplicados os filtros de origem geográfica (Brasil), hospedeiro humano
e classificação de qualidade "good", resultando em 4.310 genomas. No NCBI Pathogen
Detection, foram aplicados os filtros de origem geográfica (Brasil) e hospedeiro
humano, resultando em 6.646 genomas. Após a padronização dos metadados (ano de coleta,
espécie e fonte de isolamento) e remoção de duplicatas (mesmo Assembly_id) entre as
duas bases, obteve-se um total de 7.068 isolados bacterianos brasileiros únicos.
"""

FERRAMENTAS = [
    "NCBI Datasets (v18.16.0) - download",
    "Prodigal (v2.6.3) — predição de genes codificadores de proteína",
    "RGI (v6.0.5) — identificação de genes de resistência antimicrobiana, "
    "usando o banco de dados CARD (v4.0.1)",
    "geNomad (v1.12.0) — classificação de contigs em cromossomal ou plasmidial.",
]

TEXTO_SELECAO = """
Filtros com os alertas do ydata-profiling, **só no treino**:
1. **Prevalência**: binária presente em < 5% do treino sai.
2. **Correlação**: enquanto houver *High correlation*, sai a feature com **mais ligações**
   (empate: maior soma das correlações) e o profile é refeito.
"""

CODIGO_RISCO = """def risco_idx(df, w_plasmid=0.5):
    carb = df["has_carbapenemase"] > 0
    esbl = df["has_esbl"] > 0
    ampc = df["has_ampc_adquirida"] > 0
    tipo = np.select(
        [carb & (esbl | ampc),
         carb,
         esbl,
         ampc],
        [1.0, 0.75, 0.5, 0.25], default=0.0)
    pct_adquiridos_plasmidial = df["pct_adquiridos_plasmidial"].fillna(0)
    return df["n_bl_adquiridas"] * tipo * (1 + w_plasmid * pct_adquiridos_plasmidial)
"""

COLA_METRICAS = """
- **Silhouette** (−1 a 1, ↑): o quanto cada ponto está mais perto do próprio cluster do
  que do vizinho.
- **Calinski-Harabasz** (≥ 0, ↑): dispersão entre clusters ÷ dispersão dentro deles.
- **Davies-Bouldin** (≥ 0, ↓): semelhança média de cada cluster com o vizinho mais próximo.
- **ARI** (≈ 0 a 1): concordância entre duas partições. 1 = idênticas, 0 = acaso.
- **Qui²**: proporções dos clusters iguais às do treino (p > 0,05).
"""

TEXTO_CLASSIF = """
- **Real**: cluster do KMeans. **Predito**: classe atribuída pelo classificador
  (Grid Search com CV de 5 folds no treino).
- **Matriz de confusão**: real × predito no teste.
- **Qui² de homogeneidade**: H0 = as classes preditas têm a mesma distribuição no
  treino e no teste (p > 0,05 não rejeita).
"""

TEXTO_REGR_TESTES = """
- **Testes de hipótese**: previsões no treino × previsões no teste. Distribuição (KS),
  média (T de Welch), mediana (Mann-Whitney) e variância (Levene). p > 0,05 = iguais.
- **Gráfico**: observado × predito no teste.
"""

PASSOS_REGR = """
1. Padroniza as numéricas e faz one-hot das categóricas.
2. Descarta colunas quase constantes (VarianceThreshold).
3. Escolhe as k melhores pelo f_regression.
4. Grid Search com CV de 5 folds.
"""


def titulo(texto: str, nivel: int = 4) -> None:
    st.markdown(f"<h{nivel}>{texto}</h{nivel}>", unsafe_allow_html=True)


class Paginas:
    # ------------------------------------------------------------------ dados
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
    def busca_resultados_cluster(_self) -> dict:
        """
        Função de busca dos resultados exportados pelos notebooks final/01 e
        final/02 (pasta app/data/clustering/).
        Retorna:
            dict: seleção (tabela, rodadas, correlações), features,
            descritivas, labels, concordancia, resumo, params e clusters.
        """
        link = acesso_data_banco() + "clustering/"
        with open(link + "features_selecionadas.json") as f:
            selecao = json.load(f)
        with open(link + "params_escolhidos.json") as f:
            params = json.load(f)

        return {
            "selecao": pd.read_csv(link + "selecao_features.csv", index_col=0),
            "rodadas": pd.read_csv(link + "rodadas_correlacao.csv"),
            "corr_antes": pd.read_csv(link + "corr_ydata_antes.csv", index_col=0),
            "corr_depois": pd.read_csv(link + "corr_ydata_depois.csv", index_col=0),
            "features": selecao["features_finais"],
            "descritivas": selecao["descritivas"],
            "labels": pd.read_csv(link + "labels_treino.csv", index_col=0),
            "concordancia": pd.read_csv(link + "concordancia_ari.csv", index_col=0),
            "resumo": pd.read_csv(link + "resumo_metricas.csv", index_col=[0, 1]),
            "params": params,
            "clusters": pd.read_csv(link + "clusters_risco.csv", index_col=0),
        }

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
            DataFrames de treino e teste (contexto, alvo e previsão de cada
            modelo).
        """
        banco = acesso_data_banco()
        cols = [*CONTEXTO_REGR, ALVO_REGR, *MODELOS_REGR]
        df_train = load(banco + "train_resultados_regr")[cols].copy()
        df_test = load(banco + "test_resultados_regr")[cols].copy()

        return df_train, df_test

    @st.cache_data
    def busca_metricas(_self, arquivo: str) -> pd.DataFrame:
        """
        Métricas por modelo e conjunto (treino / validação / teste) exportadas
        pelos notebooks final/03 e final/04 (df_resultados/metricas_*.csv).
        """
        return pd.read_csv(acesso_data_banco() + "df_resultados/" + arquivo)

    @st.cache_data
    def busca_hiperparametros_regr(_self) -> dict:
        """Melhores parâmetros de cada regressor (notebook final/04)."""
        with open(acesso_data_banco() + "df_resultados/regr_hiperparametros.json") as f:
            return json.load(f)

    # ---------------------------------------------------- pré-processamento
    def pre_processamento(self) -> None:
        df_train, df_valid, df_test = self.busca_dados_cluster()

        titulo("Pré-processamento")
        st.markdown(TEXTO_PRE)
        itens = "".join(f"<li>{i}</li>" for i in FERRAMENTAS)
        st.markdown(
            f"<div class='caixa-amarela'><ul>{itens}</ul></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Foram filtrados apenas os genomas de espécies ou gêneros listados pela "
            "OMS (2024) como de nível crítico."
        )

        col = st.columns((3, 0.3, 1.4))
        with col[0]:
            st.image(acesso_data_banco() + "imagens/bubble_plot_criticos.png")
        with col[2]:
            st.markdown(
                "<p style='text-align: center;'>Divisão estratificada por espécie<br>"
                "(espécies com &lt; 20 genomas agrupadas em \"outras\").</p>",
                unsafe_allow_html=True,
            )
            tamanhos = {
                "Treino": len(df_train),
                "Validação": len(df_valid),
                "Teste": len(df_test),
            }
            st.pyplot(_fig_split(tamanhos))

        return None

    # ------------------------------------------------------------ clustering
    @st.cache_data
    def __clustering(
        _self,
        conjunto: str,
        df_train: pd.DataFrame,
        df_compara: pd.DataFrame,
        features: list[str],
        params: dict[str, dict],
    ) -> dict[str, tuple]:
        """
        Função de teste dos modelos de clustering.
        Parametros:
            conjunto: nome do conjunto comparado com o treino.
            df_train: DataFrame de treino.
            df_compara: DataFrame de validação ou teste.
            features: features do clustering (final/01).
            params: melhor parâmetro de cada algoritmo (final/02).
        Retorna:
            dict[str, tuple]: {algoritmo: (gráfico, mensagem)}.
        """
        resultados = {}
        for nome, modelo in ALGORITMOS_CLUSTER.items():
            cluster = Clustering(features=features, df=df_train, modelo=modelo)
            resultados[nome] = cluster.testar_modelo(
                best_param=params[modelo], df_test=df_compara
            )

        return resultados

    def __cluster_datasets(self, dados: tuple[pd.DataFrame, ...]) -> None:
        titulo("1. Datasets")
        col = st.columns(3)
        for c, (nome, cor), df in zip(col, CORES_SPLIT.items(), dados):
            with c:
                st.markdown(
                    f"<div class='card-dataset' style='background: {cor};'>"
                    f"<b>{nome.upper()}</b><br>"
                    f"{df.shape[0]} genomas × {df.shape[1]} colunas"
                    "</div>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"Ver dados de {nome.lower()}"):
                    st.dataframe(df)

        return None

    def __cluster_selecao(self, res: dict) -> None:
        titulo("2. <i>Feature selection</i> com ydata-profiling")
        col = st.columns((3, 2))
        with col[0]:
            st.markdown(TEXTO_SELECAO)
        with col[1]:
            st.write("Remoções por correlação, rodada a rodada:")
            st.dataframe(res["rodadas"], hide_index=True)

        st.write("Decisão para cada uma das 15 candidatas:")
        st.dataframe(
            res["selecao"].style.apply(
                lambda linha: [
                    "background-color: #F4A7A7; color: #000"
                    if linha["decisao"] == "sai"
                    else ""
                ]
                * len(linha),
                axis=1,
            )
        )

        col = st.columns((3, 2))
        with col[0]:
            st.pyplot(_fig_corr(res["corr_antes"], "ydata (auto) — antes: 15 candidatas"))
        with col[1]:
            st.pyplot(_fig_corr(res["corr_depois"], "ydata (auto) — depois"))

        st.write("Features finais:")
        st.code(f"FEATURES_FINAIS = {res['features']}", language="python")

        return None

    def __cluster_todas(self, df_train: pd.DataFrame, res: dict) -> None:
        titulo("3. Clustering com as features selecionadas")
        col = st.columns((1, 1))
        with col[0]:
            st.write("Parâmetros escolhidos (melhor silhouette CV do grid):")
            st.json(res["params"])
            st.write("Nº de genomas por cluster no treino:")
            tamanhos = {
                m: res["labels"][m].value_counts().sort_index().to_dict()
                for m in res["labels"].columns
            }
            st.code(
                "\n".join(f"{m:14s} {t}" for m, t in tamanhos.items()),
                language=None,
            )
        with col[1]:
            st.write("Os 3 algoritmos concordam? E com a espécie?")
            st.pyplot(_fig_concordancia(res["concordancia"]))

        return None

    def __cluster_generaliza(self, dados: tuple[pd.DataFrame, ...], res: dict) -> None:
        df_train, df_valid, df_test = dados
        with st.container(border=True):
            titulo("O modelo generaliza pra validação e pro teste?", 4)
            resumo = res["resumo"]
            st.pyplot(_fig_metricas(
                resumo.reset_index(),
                METRICAS_CLUSTER,
                "Clustering — métricas por algoritmo e conjunto",
                referencias={"silhouette": 0.5, "davies_bouldin": 1, "qui2_p": 0.05},
                formatos={"calinski_harabasz": "%.0f"},
            ))
            col = st.columns((3, 2))
            with col[0]:
                st.dataframe(resumo.round(3))
                st.markdown(
                    "- **Projeção**: rótulos dados pelo modelo do treino.\n"
                    "- **Refit**: modelo novo ajustado só no conjunto."
                )
            with col[1]:
                st.info(COLA_METRICAS)

            titulo("Verificação no conjunto de teste", 4)
            conjunto = st.radio(
                "Comparar o treino com:", ["Teste", "Validação"], horizontal=True
            )
            df_compara = df_valid if conjunto == "Validação" else df_test

            resultados = self.__clustering(
                conjunto=conjunto,
                df_train=df_train,
                df_compara=df_compara,
                features=res["features"],
                params=res["params"],
            )
            col = st.columns(3, gap="large")
            for c, (_, msn) in zip(col, resultados.values()):
                with c:
                    st.markdown(f"<p>{msn}</p>", unsafe_allow_html=True)

        return None

    def __cluster_biologia(self, df_train: pd.DataFrame, res: dict) -> None:
        titulo("4. Biologicamente, faz sentido?")
        st.markdown(
            "Cor = z-score; valor = média da feature no cluster.  \n"
            "Acima da linha preta, features do clustering; abaixo, descritivas."
        )
        clusters = res["clusters"]
        lab = clusters.loc[df_train.index, "cluster"]
        n = lab.value_counts().to_dict()

        col = st.columns((3, 2))
        with col[0]:
            st.pyplot(_fig_perfil(df_train, lab, res["features"], res["descritivas"]))
        with col[1]:
            st.write("Nome de cada cluster (mecanismo presente em ≥ 70% do grupo):")
            st.dataframe(
                clusters.loc[df_train.index]
                .groupby("cluster")["perfil"]
                .first()
                .to_frame()
                .assign(n=pd.Series(n)),
                height=260,
            )

        return None

    def clustering(self) -> None:
        dados = self.busca_dados_cluster()
        res = self.busca_resultados_cluster()

        self.__cluster_datasets(dados)
        st.divider()
        self.__cluster_selecao(res)
        st.divider()
        self.__cluster_todas(dados[0], res)
        self.__cluster_generaliza(dados, res)
        st.divider()
        self.__cluster_biologia(dados[0], res)

        return None

    # --------------------------------------------------------- classificação
    @st.cache_data
    def __classifier(_self) -> None:
        dados, df_test_class = _self.busca_dados_class()

        modelos = list(MODELOS_CLASSIF)
        for linha in (modelos[:3], modelos[3:]):
            col = st.columns((5, 0.5, 5, 0.5, 5))
            for k, modelo in zip((0, 2, 4), linha):
                with col[k]:
                    st.markdown(
                        f"<div class='titulo-modelo'>{MODELOS_CLASSIF[modelo]}</div>",
                        unsafe_allow_html=True,
                    )
                    classificador = Classifier(df=dados[modelo])
                    classificador.testar_hipotese()
                    st.divider()
                    classificador.matrix_confusao(
                        df=df_test_class, obs="cluster", pred=modelo
                    )
            st.divider()

        return None

    def classificacao(self) -> None:
        clusters = self.busca_resultados_cluster()["clusters"]
        nomes = clusters.groupby("cluster")["perfil"].first()

        titulo("Classificação")
        col = st.columns((3, 2))
        with col[0]:
            st.markdown(TEXTO_CLASSIF)
        with col[1]:
            st.write("Classes (clusters do KMeans):")
            st.dataframe(nomes.to_frame(), height=260)
        st.divider()
        titulo("Métricas no treino, na validação e no teste", 4)
        st.pyplot(_fig_metricas(
            self.busca_metricas("metricas_classif.csv"),
            METRICAS_CLASSIF,
            "Classificação — métricas por modelo e conjunto",
            formatos={c: "%.2f" for c in METRICAS_CLASSIF},
        ))
        st.divider()
        self.__classifier()

        return None

    # -------------------------------------------------------------- regressão
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

    def __regr_features(self, df_train: pd.DataFrame) -> None:
        titulo("Novas features")
        titulo("Calculando valor contínuo de risco", 5)
        st.markdown("- `n de bl adquiridas` x `peso do tipo de bl` x `fração plasmidial`")
        st.code(CODIGO_RISCO, language="python")

        cols = ["Species", "source_type", "Date", ALVO_REGR]
        perfil = pd.DataFrame(
            {
                "tipo": df_train[cols].dtypes.astype(str),
                "nulos %": (df_train[cols].isna().mean() * 100).round(1),
                "únicos": df_train[cols].nunique(),
                "top (freq %)": [
                    f"{df_train[c].value_counts().index[0]} "
                    f"({df_train[c].value_counts(normalize=True).iloc[0] * 100:.0f}%)"
                    for c in cols
                ],
            }
        )
        col = st.columns((1, 1))
        with col[0]:
            st.dataframe(df_train[[*cols, "Date_na"]].head(), height=230)
        with col[1]:
            st.dataframe(perfil)

        return None

    def __regr_hiperparametros(self) -> None:
        titulo("Treino - CV e hiperparâmetros", 3)
        st.markdown(PASSOS_REGR)

        hiper = self.busca_hiperparametros_regr()
        col = st.columns(4)
        for c, nome in zip(col, MODELOS_REGR):
            h = hiper[nome]
            linhas = [f"{k} -> {v}" for k, v in h["best_params"].items()]
            linhas += [
                "=" * 36,
                f"Variáveis originais:        {h['n_originais']}",
                f"Após VarianceThreshold:     {h['n_apos_variancia']}",
                f"Selecionadas no modelo:     {h['n_selecionadas']}",
            ]
            with c:
                st.markdown(f"**{nome}** (`{h['modelo']}`)")
                st.code("\n".join(linhas), language=None)

        return None

    def regressao(self) -> None:
        df_train_regr, _ = self.busca_dados_regr()

        self.__regr_features(df_train_regr)
        st.divider()
        self.__regr_hiperparametros()
        st.divider()
        titulo("Métricas no treino, na validação e no teste", 3)
        st.pyplot(_fig_metricas(
            self.busca_metricas("metricas_regr.csv"),
            METRICAS_REGR,
            "Regressão — métricas por modelo e conjunto",
        ))
        st.divider()
        titulo("Treino × teste", 3)
        st.markdown(TEXTO_REGR_TESTES)
        self.__regressor()

        return None

    # ------------------------------------------------------------ navegação
    def metricas(self) -> str:
        """
        Monta a navegação lateral com as quatro páginas do app.
        Retorna:
            str: Mensagem de sucesso.
        """
        paginas = [
            st.Page(
                self.pre_processamento,
                title="Pré-processamento",
                url_path="pre_processamento",
                default=True,
            ),
            st.Page(self.clustering, title="Clustering", url_path="clustering"),
            st.Page(
                self.classificacao, title="Classificação", url_path="classificacao"
            ),
            st.Page(self.regressao, title="Regressão", url_path="regressao"),
        ]
        pagina = st.navigation(paginas, position="sidebar")
        # card salvo em app/data/imagens (o opengraph do GitHub tem limite de
        # acesso) e embutido em base64 para continuar clicável
        st.sidebar.markdown(
            f"<a href='{URL_REPO}' target='_blank'>"
            f"<img class='card-github' alt='Rodrigo-Lusa/ml_isolados_brasileiros' "
            f"src='data:image/png;base64,{_card_github()}'></a>",
            unsafe_allow_html=True,
        )
        pagina.run()

        return "Métricas Calculadas"


@st.cache_data
def _card_github() -> str:
    with open(acesso_data_banco() + "imagens/github_card.png", "rb") as f:
        return base64.b64encode(f.read()).decode()


# Figuras em cache (cache_resource: não precisa serializar a figura)
@st.cache_resource
def _fig_split(tamanhos: dict[str, int]):
    return grafico_split(tamanhos)


@st.cache_resource
def _fig_corr(corr: pd.DataFrame, titulo: str):
    return grafico_corr(corr, titulo)


@st.cache_resource
def _fig_metricas(
    df: pd.DataFrame,
    metricas: dict[str, str],
    titulo: str,
    referencias: dict[str, float] | None = None,
    formatos: dict[str, str] | None = None,
):
    return grafico_metricas(df, metricas, titulo, referencias, formatos)


@st.cache_resource
def _fig_concordancia(concordancia: pd.DataFrame):
    return grafico_concordancia(concordancia)


@st.cache_resource
def _fig_perfil(
    df: pd.DataFrame, labels: pd.Series, features: list[str], descritivas: list[str]
):
    return grafico_perfil(df, labels, features + descritivas, len(features))
