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
    grafico_bivariado,
    grafico_concordancia,
    grafico_perfil,
    grafico_split,
)
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
# features que não entram no clustering, só na descrição dos clusters
DESCRITIVAS = [
    "has_mcr",
    "has_pmqr",
    "has_ampc_adquirida",
    "n_classes_adquiridas",
    "n_genes_adquiridos",
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
CONTEXTO_REGR = ["Species", "source_type", "Date", "Date_na"]
MODELOS_REGR = [
    "Multiple Linear Regression",
    "Suport Vector Machine",
    "Nayve Bayes",
    "Neural Network",
]

# média do silhouette CV bivariado abaixo disso = feature fraca (em vermelho)
SILHOUETTE_BAIXO = 0.7

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

CODIGO_SILHOUETTE = """por_feature.groupby("f")["silhouette_cv"].agg(["mean", "max", "count"])\
.sort_values("mean", ascending=False).round(3)"""

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
**Cola das métricas**
- **Silhouette** (−1 a 1, ↑ melhor): cada genoma está mais perto do próprio
  cluster do que do vizinho? > 0,5 = estrutura clara.
- **Calinski-Harabasz** (≥ 0, ↑ melhor): dispersão entre clusters ÷ dentro dos
  clusters. Cresce com o nº de genomas, então não dá pra comparar treino (1558) com
  validação (445) direto, só modelos no mesmo conjunto.
- **Davies-Bouldin** (≥ 0, ↓ melhor): o quanto cada cluster se parece com o
  vizinho mais próximo. < 1 = clusters bem separados.
- **ARI** (≈ 0 a 1): concordância entre duas partições. 1 = idênticas,
  0 = acaso.
- **Qui² (teste abaixo)**: a proporção de genomas em cada cluster é a mesma do
  treino? p > 0,05 = sim.
"""

TEXTO_CLASSIF = """
**O que está sendo comparado**
- **Real** = o cluster de cada genoma, formado pelo KMeans no Clustering (C0 a C5).
- **Predito** = a classe que cada classificador atribui. Os 6 modelos foram
  treinados no conjunto de treino (Grid Search com CV de 5 folds) e preveem a
  validação e o teste.
- A **matriz de confusão** compara real × predito no conjunto de **teste**.

**Por que o qui-quadrado?** A "Tabela de Contingência (Observada)" conta quantos
genomas o modelo colocou em cada classe na validação e no teste (as frequências
*observadas* das classes **preditas**). O qui-quadrado de homogeneidade
(`chi2_contingency`) testa se essas proporções são iguais nos dois conjuntos: um
modelo estável deve distribuir as classes do mesmo jeito em dados que não viu.
- H0: a distribuição das classes preditas é igual na validação e no teste.
- O **p-valor** sai da estatística χ² comparada com a distribuição χ² de
  (nº de classes preditas − 1) × (nº de conjuntos − 1) graus de liberdade (com as
  6 classes, (6 − 1) × (2 − 1) = 5). Com α = 0,05: p > 0,05 → não rejeita H0
  (homogênea).
"""

TEXTO_REGR_TESTES = """
- **Testes de hipótese** (Kolmogorov-Smirnov, T, Mann-Whitney e Levene): comparam as
  **previsões do modelo no treino** (1558 genomas) com as **previsões do modelo no
  teste** (223). O valor real não entra: a pergunta é se o modelo prevê valores com
  a mesma distribuição (KS), média (T de Welch), mediana (Mann-Whitney) e variância
  (Levene) em dados que não viu. IGUAIS (p > 0,05) = previsões estáveis.
- **Gráfico**: `risco_idx` **observado** × **predito**, só no **teste**, com
  Spearman, R², MAE, RMSE e MAE/DP (o MAPE ficou de fora porque ~20% dos genomas
  têm `risco_idx` = 0).
"""

PASSOS_REGR = """
1. Padroniza as numéricas (Date, Date_na) e faz one-hot das categóricas.
2. Descarta colunas quase constantes (VarianceThreshold).
3. Escolhe as k melhores pelo f_regression.
4. Ajusta a regressão, testando o Grid Search com Cross validation de 5 folds.
5. Imprime os melhores parâmetros, quantas variáveis sobraram e quais foram, e o
   gráfico predito × observado.
"""


def titulo(texto: str, nivel: int = 4) -> None:
    st.markdown(f"<h{nivel}>{texto}</h{nivel}>", unsafe_allow_html=True)


def agrupa_especie(s: pd.Series, minimo: int = 14) -> pd.Series:
    """Espécies com menos de `minimo` genomas viram "outras" (notebook 04)."""
    return s.where(s.map(s.value_counts()) >= minimo, "outras")


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
        Função de busca dos resultados exportados pelos notebooks 03 e 04
        (pasta app/data/clustering/).
        Retorna:
            dict: bivariado, features, labels, concordancia, resumo,
            params e clusters.
        """
        link = acesso_data_banco() + "clustering/"
        with open(link + "features_selecionadas.json") as f:
            features = json.load(f)["features_finais"]
        with open(link + "params_escolhidos.json") as f:
            params = json.load(f)

        return {
            "bivariado": pd.read_csv(link + "bivariado_silhouette.csv"),
            "features": features,
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
    def busca_hiperparametros_regr(_self) -> dict:
        """Melhores parâmetros de cada regressor (notebook 06)."""
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
            cluster = Clustering(features=FEATURES_CLUSTER, df=df_train, modelo=modelo)
            resultados[nome] = cluster.testar_modelo(
                best_param=best_param, df_test=df_compara
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

    def __cluster_bivariado(self, df_train: pd.DataFrame, res: dict) -> None:
        titulo("2. Clustering bivariado para <i>feature selection</i>")
        st.markdown(
            "Para cada par de colunas, roda o grid do KMeans com CV (Silhouette). "
            "Pares binária × binária ficam de fora: só têm 4 pontos distintos no "
            "plano e o silhouette vai a ~1,0 de forma trivial."
        )
        bivariado = res["bivariado"]
        por_feature = pd.concat(
            [
                bivariado[["c1", "silhouette_cv"]].rename(columns={"c1": "f"}),
                bivariado[["c2", "silhouette_cv"]].rename(columns={"c2": "f"}),
            ]
        )
        col = st.columns((2, 3))
        with col[0]:
            st.write("Média do silhouette score por feature.")
            st.code(CODIGO_SILHOUETTE, language="python")
            media = (
                por_feature.groupby("f")["silhouette_cv"]
                .agg(["mean", "max", "count"])
                .sort_values("mean", ascending=False)
            )
            st.dataframe(
                media.style.format({"mean": "{:.3f}", "max": "{:.3f}"}).apply(
                    lambda linha: [
                        "background-color: #F4A7A7; color: #000"
                        if linha["mean"] < SILHOUETTE_BAIXO
                        else ""
                    ]
                    * len(linha),
                    axis=1,
                )
            )
            st.caption(
                f"Em vermelho: silhouette médio baixo (< {SILHOUETTE_BAIXO}) — "
                "essas features não separam bem os genomas em nenhum par."
            )
        with col[1]:
            st.write("Ranking dos pares (treino).")
            st.dataframe(
                bivariado.drop(columns=["c1_binaria", "c2_binaria"]),
                hide_index=True,
                height=420,
            )

        st.write("Gráficos de dispersão (todos os pares, cor = espécie):")
        st.pyplot(_fig_bivariado(df_train, bivariado))

        st.write("Features finais:")
        st.code(f"FEATURES_FINAIS = {res['features']}", language="python")

        return None

    def __cluster_todas(self, df_train: pd.DataFrame, res: dict) -> None:
        titulo("3. Clustering com todas as features")
        col = st.columns((1, 1))
        with col[0]:
            st.write("Parâmetros escolhidos (notebook 04, seção 3):")
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
            titulo("O modelo generaliza pros conjuntos de validação?", 4)
            resumo = res["resumo"].drop(index="teste", level=1)
            col = st.columns((3, 2))
            with col[0]:
                st.dataframe(
                    resumo[
                        [
                            "silhouette_refit",
                            "calinski_harabasz_refit",
                            "davies_bouldin_refit",
                            "ari_projecao_x_refit",
                        ]
                    ].round(3)
                )
                st.markdown(
                    "- **métricas_refit** = conjunto de validação escalonado com o "
                    "scaler do treino, mas clusterização do 0 (mesmos "
                    "hiperparâmetros).\n"
                    "- **ari_projeção** = o modelo treinado no conjunto de treino e "
                    "aplicado na validação dá os mesmos clusters?"
                )
            with col[1]:
                st.info(COLA_METRICAS)

            titulo("Verificação no conjunto de teste", 4)
            st.markdown(
                "**Ideia** — ao aplicar o mesmo clustering nos dados de treino e teste "
                "(escalonados), a distribuição dos clusters deve ser igual, indicando "
                "que generalizou."
            )
            conjunto = st.radio(
                "Comparar o treino com:", ["Teste", "Validação"], horizontal=True
            )
            df_compara = df_valid if conjunto == "Validação" else df_test

            resultados = self.__clustering(
                conjunto=conjunto, df_train=df_train, df_compara=df_compara
            )
            col = st.columns(3, gap="large")
            for c, (_, msn) in zip(col, resultados.values()):
                with c:
                    st.markdown(f"<p>{msn}</p>", unsafe_allow_html=True)

        return None

    def __cluster_biologia(self, df_train: pd.DataFrame, res: dict) -> None:
        titulo("4. Biologicamente, faz sentido?")
        st.markdown(
            "Perfil de cada cluster do KMeans no treino: para cada cluster, verificar "
            "as features.  \n"
            "Cor = z-score, pra ver o quão distante da média está.  \n"
            "Valor da célula = média da feature no grupo. Como algumas são binárias, "
            "variam de 0 a 1.  \n"
            "Acima da linha preta, as features do clustering; abaixo, as descritivas."
        )
        clusters = res["clusters"]
        lab = clusters.loc[df_train.index, "cluster"]
        n = lab.value_counts().to_dict()

        col = st.columns((3, 2))
        with col[0]:
            st.pyplot(_fig_perfil(df_train, lab))
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
        self.__cluster_bivariado(dados[0], res)
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
def _fig_bivariado(df: pd.DataFrame, bivariado: pd.DataFrame):
    return grafico_bivariado(df, bivariado, agrupa_especie(df["Species"]))


@st.cache_resource
def _fig_concordancia(concordancia: pd.DataFrame):
    return grafico_concordancia(concordancia)


@st.cache_resource
def _fig_perfil(df: pd.DataFrame, labels: pd.Series):
    return grafico_perfil(df, labels, FEATURES_CLUSTER + DESCRITIVAS, len(FEATURES_CLUSTER))
