"""
Classe Clustering — multivariada, com nomes de método espelhando a classe
do professor (07-aula-09-02). Duas adaptações deliberadas em relação ao
original, por ele ser bivariado e nós sermos multivariados:

1. `feature_cols: list[str]` no lugar de `c1`/`c2`.
2. A transformação (scaler) entra como parâmetro de `cross_validation()`,
   não é fixada como StandardScaler dentro da classe — porque eps/min_samples
   do DBSCAN e n_neighbors do Agglomerative mudam de escala conforme a
   transformação (ver conversa).

Em `__testar_*`, mantive a lógica dupla do professor (projeta o modelo do
treino na validação via predict/KNN-1/NearestCentroid, E também treina um
modelo independente do zero em df_test) — só que, diferente do original,
escalono consistentemente nos dois casos: a projeção reusa o scaler do
treino (.transform() só); o ajuste independente treina um scaler novo,
do zero, em df_test — reconciliando "usa o modelo treinado" (qui-quadrado)
com "roda do zero" (a metade usada pro ARI).
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from scipy.stats import chisquare


def silhouette_scorer_func(estimator, X) -> float:
    """Mesmo scorer do professor — usado pelo cross_val_score do sklearn."""
    labels = estimator.fit_predict(X)
    if len(np.unique(labels)) > 1:
        return silhouette_score(X, labels)
    return -1.0


class Clustering:
    """
    Parâmetros
    ----------
    feature_cols : list[str]
        Colunas usadas como entrada do modelo (qualquer nº de colunas —
        no professor é bivariado, c1/c2; aqui é multivariado).
    df : pd.DataFrame
        Base de treino.
    modelo : str
        "kmeans", "dbscan" ou "agglomerative".
    """

    _SCALERS = {"sem_scaling": None, "standard": StandardScaler(), "minmax": MinMaxScaler()}

    # grids calibrados manualmente por transformação (ver conversa) —
    # substitui o __param_grid fixo do professor, que não distinguia escala
    _GRID_DBSCAN_POR_TRANSFORMACAO = {
        "sem_scaling": {"eps": [5, 10, 15], "min_samples": [3, 5, 7]},
        "standard": {"eps": [1, 1.5, 2], "min_samples": [3, 5, 7]},
        "minmax": {"eps": [0.18, 0.2, 0.25], "min_samples": [3, 5, 7]},
    }
    _N_NEIGHBORS_AGG_POR_TRANSFORMACAO = {"sem_scaling": 5, "standard": 5, "minmax": 6}

    def __init__(self, feature_cols: list, df: pd.DataFrame, modelo: str) -> None:
        self.feature_cols = feature_cols
        self.df = df
        self.modelo = modelo

        # preenchidos por cross_validation
        self.transformacao = None
        self.scaler = None
        self.melhores_parametros = None
        self.labels_ = None
        self.metricas_ = None
        self.tabela_grid_ = None

    # ------------------------------------------------------------------ #
    def __ajustar_scaler(self, df_alvo: pd.DataFrame, transformacao: str):
        """Fita um scaler NOVO em df_alvo (nunca reusa um já ajustado)."""
        scaler = self._SCALERS[transformacao]
        X = df_alvo[self.feature_cols].values
        if scaler is not None:
            scaler = type(scaler)()
            X = scaler.fit_transform(X)
        return X, scaler

    def __calcular_metricas(self, X_scaled: np.ndarray, pred: np.ndarray, obs: np.ndarray = None) -> tuple:
        """
        Espelha __calcular_metricas do professor. `obs` é opcional aqui
        (o professor sempre tinha "classe" verdadeira; nós não temos —
        ARI/AMI só saem se obs for passado explicitamente, ex: comparando
        contra outro algoritmo, nunca contra um rótulo verdadeiro que
        não existe no nosso caso).
        """
        ss = chs = dbs = ari = ami = np.nan
        try:
            mask = pred != -1 if -1 in pred else np.ones_like(pred, dtype=bool)
            if len(set(pred[mask])) >= 2:
                ss = round(silhouette_score(X_scaled[mask], pred[mask]), 3)
                chs = round(calinski_harabasz_score(X_scaled[mask], pred[mask]), 3)
                dbs = round(davies_bouldin_score(X_scaled[mask], pred[mask]), 3)
            if obs is not None:
                ari = round(adjusted_rand_score(obs, pred), 3)
                ami = round(adjusted_mutual_info_score(obs, pred), 3)
        except Exception:
            pass
        return (ss, chs, dbs, ari, ami)

    def _grafico_scatter(self, hue, X_scaled: np.ndarray = None) -> None:
        """
        Adaptado do bivariado do professor (jointplot de c1×c2) pra
        multivariado: projeta em 2 componentes principais (PCA) e plota.
        Chame manualmente com o que quiser em `hue` (cluster_kmeans.labels_,
        metadata_alinhado["WHO_Priority"], etc.) — não é mais chamado
        automaticamente por cross_validation().
        """
        X_scaled = X_scaled if X_scaled is not None else self.__ajustar_scaler(self.df, self.transformacao)[0]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hue, palette="tab20")
        plt.title("Clusters projetados em PCA (adaptado — professor usava c1×c2 direto)")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.show()

    def _grafico_box(self, cluster_labels: np.ndarray, colunas: list = None) -> None:
        """Generaliza o boxplot de c1/c2 do professor pra todas as feature_cols.
        Chame manualmente (ex: cluster_kmeans._grafico_box(cluster_kmeans.labels_))."""
        colunas = colunas or self.feature_cols
        n_cols = 4
        n_linhas = -(-len(colunas) // n_cols)
        _, axes = plt.subplots(n_linhas, n_cols, figsize=(n_cols * 4, n_linhas * 3.5))
        axes = np.array(axes).flatten()

        plt.suptitle("Diferença das classes clusterizadas")
        for ax, col in zip(axes, colunas):
            sns.boxplot(x=cluster_labels, y=self.df[col], ax=ax)
            ax.set(title=f"{col} para Clusters", ylabel=col, xlabel="")
        for ax in axes[len(colunas):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def __param_grid(self, transformacao: str) -> dict:
        if self.modelo == "kmeans":
            return {"n_clusters": [2, 3, 4, 5, 6], "init": ["k-means++", "random"], "n_init": ["auto", 10, 20]}
        elif self.modelo == "dbscan":
            return self._GRID_DBSCAN_POR_TRANSFORMACAO[transformacao]
        elif self.modelo == "agglomerative":
            return {"n_clusters": [2, 3, 4, 5, 6], "linkage": ["ward", "complete", "average"]}

    # ------------------------------------------------------------------ #
    def cross_validation(self, transformacao: str, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
        """
        Grid search com K-Fold + cross_val_score (silhueta média entre folds
        = mean_score/std_score, usada pra ESCOLHER hiperparâmetro robusto).
        Além disso, ajusta cada combinação uma vez no dataset inteiro pra
        calcular as 3 métricas intrínsecas "reais" (silhouette, calinski_harabasz,
        davies_bouldin) — ficam como colunas na própria tabela, uma linha por
        combinação. Não plota nada automaticamente: usa cluster_x.labels_
        (do vencedor) ou pega os rótulos de qualquer linha específica que
        quiser, e chama _grafico_scatter/_grafico_box manualmente.
        """
        X_scaled, scaler = self.__ajustar_scaler(self.df, transformacao)
        self.transformacao = transformacao
        self.scaler = scaler

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        param_grid = self.__param_grid(transformacao)
        chaves = list(param_grid.keys())

        melhor_score = -1.0
        melhores_parametros = {}
        resultados = []
        labels_do_vencedor = None

        for combo in itertools.product(*param_grid.values()):
            parametros = dict(zip(chaves, combo))
            linha = dict(parametros)

            if self.modelo == "kmeans":
                modelo_cv = KMeans(**parametros, random_state=random_state)
            elif self.modelo == "dbscan":
                modelo_cv = DBSCAN(**parametros, n_jobs=-1)
            elif self.modelo == "agglomerative":
                n_neighbors = self._N_NEIGHBORS_AGG_POR_TRANSFORMACAO[transformacao]
                conectividade = kneighbors_graph(X_scaled, n_neighbors=n_neighbors, include_self=False)
                modelo_cv = AgglomerativeClustering(connectivity=conectividade, **parametros)

            try:
                scores = cross_val_score(
                    estimator=modelo_cv, X=X_scaled, cv=kf, scoring=silhouette_scorer_func, n_jobs=-1,
                )
                linha["mean_score"] = np.mean(scores)
                linha["std_score"] = np.std(scores)
            except Exception as e:
                linha["mean_score"] = np.nan
                linha["std_score"] = np.nan
                linha["erro"] = str(e)
                resultados.append(linha)
                continue

            # ajuste único no dataset inteiro — pra métricas intrínsecas por linha
            try:
                if self.modelo == "kmeans":
                    modelo_completo = KMeans(**parametros, random_state=random_state)
                elif self.modelo == "dbscan":
                    modelo_completo = DBSCAN(**parametros, n_jobs=-1)
                elif self.modelo == "agglomerative":
                    modelo_completo = AgglomerativeClustering(connectivity=conectividade, **parametros)

                labels = modelo_completo.fit_predict(X_scaled)
                ss, chs, dbs, _, _ = self.__calcular_metricas(X_scaled, labels)
                linha["silhouette"] = ss
                linha["calinski_harabasz"] = chs
                linha["davies_bouldin"] = dbs
            except Exception:
                labels = None
                linha["silhouette"] = linha["calinski_harabasz"] = linha["davies_bouldin"] = np.nan

            resultados.append(linha)

            if linha["mean_score"] > melhor_score:
                melhor_score = float(linha["mean_score"])
                melhores_parametros = parametros
                labels_do_vencedor = labels

        self.melhores_parametros = melhores_parametros
        self.labels_ = labels_do_vencedor
        if labels_do_vencedor is not None:
            ss, chs, dbs, _, _ = self.__calcular_metricas(X_scaled, labels_do_vencedor)
            self.metricas_ = {"silhouette": ss, "calinski_harabasz": chs, "davies_bouldin": dbs}

        self.tabela_grid_ = pd.DataFrame(resultados)
        return self.tabela_grid_

    # ------------------------------------------------------------------ #
    # Testes — mesma dupla lógica do professor (projeção + independente),
    # escalonamento consistente nos dois casos (ver nota no topo do arquivo)
    # ------------------------------------------------------------------ #
    def __testar_kmeans(self, best_param: dict, df_test: pd.DataFrame) -> bool:
        X_train = self.scaler.transform(self.df[self.feature_cols].values) if self.scaler is not None else self.df[self.feature_cols].values
        kmeans_train = KMeans(**best_param).fit(X_train)
        labels_train = kmeans_train.labels_

        # projeção: reusa o scaler do treino, .transform() só
        X_test_projetado = self.scaler.transform(df_test[self.feature_cols].values) if self.scaler is not None else df_test[self.feature_cols].values
        labels_val_predicted = kmeans_train.predict(X_test_projetado)

        # independente: scaler novo, do zero, em df_test
        X_test_independente, _ = self.__ajustar_scaler(df_test, self.transformacao)
        kmeans_val_direct = KMeans(**best_param).fit(X_test_independente)
        labels_val_direct = kmeans_val_direct.labels_

        return self.__comparar_treino_validacao(
            labels_train, labels_val_predicted, labels_val_direct,
            n_esperado=best_param.get("n_clusters"), nome="K-MEANS",
        )

    def __testar_dbscan(self, best_param: dict, df_test: pd.DataFrame) -> bool:
        X_train = self.scaler.transform(self.df[self.feature_cols].values) if self.scaler is not None else self.df[self.feature_cols].values
        dbscan_train = DBSCAN(**best_param).fit(X_train)
        labels_train = dbscan_train.labels_

        # projeção via KNN(k=1) nos rótulos do treino (igual ao professor)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, labels_train)
        X_test_projetado = self.scaler.transform(df_test[self.feature_cols].values) if self.scaler is not None else df_test[self.feature_cols].values
        labels_val_predicted = knn.predict(X_test_projetado)

        X_test_independente, _ = self.__ajustar_scaler(df_test, self.transformacao)
        dbscan_val_direct = DBSCAN(**best_param).fit(X_test_independente)
        labels_val_direct = dbscan_val_direct.labels_

        return self.__comparar_treino_validacao(
            labels_train, labels_val_predicted, labels_val_direct, nome="DBSCAN",
        )

    def __testar_agg(self, best_param: dict, df_test: pd.DataFrame) -> bool:
        X_train = self.scaler.transform(self.df[self.feature_cols].values) if self.scaler is not None else self.df[self.feature_cols].values
        n_neighbors = self._N_NEIGHBORS_AGG_POR_TRANSFORMACAO[self.transformacao]
        conectividade = kneighbors_graph(X_train, n_neighbors=n_neighbors, include_self=False)
        agg_train = AgglomerativeClustering(connectivity=conectividade, **best_param)
        labels_train = agg_train.fit_predict(X_train)

        # projeção via centróide mais próximo (igual ao professor)
        clf_centroid = NearestCentroid()
        clf_centroid.fit(X_train, labels_train)
        X_test_projetado = self.scaler.transform(df_test[self.feature_cols].values) if self.scaler is not None else df_test[self.feature_cols].values
        labels_val_predicted = clf_centroid.predict(X_test_projetado)

        X_test_independente, _ = self.__ajustar_scaler(df_test, self.transformacao)
        conectividade_test = kneighbors_graph(X_test_independente, n_neighbors=n_neighbors, include_self=False)
        agg_val_direct = AgglomerativeClustering(connectivity=conectividade_test, **best_param)
        labels_val_direct = agg_val_direct.fit_predict(X_test_independente)

        return self.__comparar_treino_validacao(
            labels_train, labels_val_predicted, labels_val_direct, nome="AGGLOMERATIVE CLUSTERING",
        )

    def __comparar_treino_validacao(self, labels_train, labels_val_predicted, labels_val_direct, n_esperado=None, nome="") -> bool:
        """Qui-quadrado (treino vs. projeção) + ARI (projeção vs. independente) — comum aos 3 __testar_*."""
        all_clusters = np.unique(labels_train) if n_esperado is None else range(n_esperado)

        train_counts = pd.Series(labels_train).value_counts().reindex(all_clusters, fill_value=0).sort_index()
        train_proportions = train_counts / len(labels_train)
        expected_val_counts = train_proportions * len(labels_val_predicted)

        observed_val_counts = pd.Series(labels_val_predicted).value_counts().reindex(all_clusters, fill_value=0).sort_index()

        chi2_stat, p_value = chisquare(f_obs=observed_val_counts, f_exp=expected_val_counts)
        ari_score = adjusted_rand_score(labels_val_predicted, labels_val_direct)

        print("=" * 60)
        print(f" RESULTADOS DA COMPARAÇÃO DE CLUSTERS ({nome})")
        print("=" * 60)
        print(f"Clusters no Treino: {np.unique(labels_train)}")
        print(f"Frequências Esperadas na Validação: {expected_val_counts.values.round(2)}")
        print(f"Frequências Observadas na Validação: {observed_val_counts.values}")
        print("-" * 60)
        print(f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}")
        print(f"p-valor: {p_value:.4f}")
        print("-" * 60)

        alpha = 0.05
        if p_value > alpha:
            print("Conclusão: Não rejeitamos H0. A distribuição dos clusters na validação É IGUAL à do treino (p > 0.05).")
            retorno = True
        else:
            print("Conclusão: Rejeitamos H0. A distribuição dos clusters na validação É DIFERENTE da do treino (p <= 0.05).")
            retorno = False

        print("-" * 60)
        print(f"Adjusted Rand Index (ARI) entre projeção e ajuste independente: {ari_score:.4f}")
        return retorno

    def testar_modelo(self, best_param: dict, df_test: pd.DataFrame) -> bool:
        """Dispatcher — mesmo nome do professor."""
        assert self.scaler is not None or self.transformacao == "sem_scaling", \
            "chame cross_validation() antes de testar_modelo()"

        if self.modelo == "kmeans":
            return self.__testar_kmeans(best_param=best_param, df_test=df_test)
        elif self.modelo == "dbscan":
            return self.__testar_dbscan(best_param=best_param, df_test=df_test)
        elif self.modelo == "agglomerative":
            return self.__testar_agg(best_param=best_param, df_test=df_test)