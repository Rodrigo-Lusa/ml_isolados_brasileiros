"""
src/clustering.py

Clustering multivariado -- generalização da classe `Clustering` usada na
disciplina Python/ML (ver
https://github.com/Laboratorio-de-Analise-de-Dados/disciplina_python_ml,
dev/script/aulas_teoricas/src/clustering.py). A original é BIVARIADA (2
colunas fixas -- c1/c2 --, pensada pra plot direto em scatter/jointplot);
aqui o construtor aceita uma LISTA de features (N >= 2), porque perfil de
risco (resistência + mobilidade) precisa de várias colunas ao mesmo tempo,
não dá pra reduzir a um par sem perder informação.

- grid search manual
- validação cruzada via Silhouette (`silhouette_scorer_func`) 
- métricas de qualidade interna (Silhouette/Calinski-Harabasz/Davies-Bouldin)
- plots

`testar_modelo` (qui² + ARI treino x teste) foi PORTADA da original quase sem mudança de lógica --
`KMeans.predict`/`NearestCentroid`/`KNeighborsClassifier` aceitam N colunas por `self.df[self.features]`.

Diferenças deliberadas em relação à original:
- Sem ARI/AMI contra uma coluna `classe` em `cross_validation`/`__calcular_metricas` -- aqui não há
  rótulo verdadeiro conhecido (descoberta não supervisionada sobre genoma real, não dataset didático
  com classe já rotulada). `testar_modelo` continua existindo -- ele compara treino x teste do PRÓPRIO
  modelo, não contra rótulo externo, então não depende de ground truth.
- Visualização por PCA 2D (`__grafico_pca`) em vez de scatter direto --
  N-dimensional não dá pra ver num eixo x/y sem projetar primeiro.
- `rotular_clusters`: gera descrição textual automática de cada cluster
  (quais features mais se destacam da média geral) -- não existe na
  original, é o que permite ir de "cluster 0/1/2" pra frases tipo "alto
  em pct_BD_carbapenemase e pct_beta_lactam_plasmidial" (ponto de partida
  pra nomear clinicamente, não um rótulo definitivo).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from scipy.stats import chisquare


def silhouette_scorer_func(estimator, X) -> float:
    """
    Função scorer pra cross_val_score -- Silhouette precisa de pelo
    menos 2 clusters distintos no fold; se o modelo colapsar tudo num
    cluster só (comum em DBSCAN com eps ruim), devolve pontuação baixa
    em vez de estourar exceção.
    """
    labels = estimator.fit_predict(X)
    if len(np.unique(labels)) > 1:
        return silhouette_score(X, labels)
    return -1.0


class Clustering:
    """
    Clustering multivariado com busca de hiperparâmetros guiada por
    Silhouette (cross-validation) -- generalização pra N features da
    classe bivariada usada na disciplina Python/ML.

        Parâmetros
        ----------
        features : list[str]
            Colunas de `df` usadas no clustering (N >= 2).
        df : pd.DataFrame
            1 linha por genoma.
        modelo : str
            "kmeans", "dbscan" ou "agglomerative".
    """

    def __init__(self, features: list, df: pd.DataFrame, modelo: str) -> None:
        self.features = features
        self.df = df
        self.modelo = modelo

    def __param_grid(self) -> dict:
        if self.modelo == "kmeans":
            return {
                "n_clusters": [2, 3, 4, 5, 6, 7, 8],
                "init": ["k-means++", "random"],
                "n_init": [10, 20],
            }
        elif self.modelo == "dbscan":
            return {
                "eps": [0.3, 0.5, 0.7, 1.0, 1.5],
                "min_samples": [3, 5, 10],
                "metric": ["euclidean", "manhattan"],
            }
        elif self.modelo == "agglomerative":
            return {
                "n_clusters": [2, 3, 4, 5, 6, 7, 8],
                "metric": ["euclidean"],
                "linkage": ["ward", "complete", "average"],
            }

    def __calcular_metricas(self, X_scaled: np.ndarray, pred: np.ndarray) -> tuple:
        try:
            ss = round(silhouette_score(X_scaled, pred), 3)
            chs = round(calinski_harabasz_score(X_scaled, pred), 3)
            dbs = round(davies_bouldin_score(X_scaled, pred), 3)
        except Exception:
            ss, chs, dbs = np.nan, np.nan, np.nan
        return ss, chs, dbs

    def __grafico_pca(self, cluster_labels: np.ndarray, X_scaled: np.ndarray):
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        var_explicada = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(
            x=coords[:, 0], y=coords[:, 1],
            hue=[str(c) for c in cluster_labels], palette="tab10", ax=ax,
        )
        ax.set_xlabel(f"PC1 ({var_explicada[0]*100:.0f}% da variância)")
        ax.set_ylabel(f"PC2 ({var_explicada[1]*100:.0f}% da variância)")
        ax.set_title(f"Clusters ({self.modelo}) -- projeção PCA de {len(self.features)} features")
        ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        return fig

    def __grafico_perfil(self, cluster_labels: np.ndarray):
        perfil = self.df[self.features].groupby(cluster_labels).mean()
        perfil_z = (perfil - self.df[self.features].mean()) / self.df[self.features].std()

        fig, ax = plt.subplots(figsize=(max(8, len(self.features) * 1.1), 5))
        sns.heatmap(
            perfil_z.T, cmap="RdBu_r", center=0, annot=perfil.T.round(2), fmt="",
            ax=ax, cbar_kws={"label": "desvio-padrão vs. média geral"},
        )
        ax.set_title("Perfil de cada cluster (cor = z-score; número anotado = valor médio real)")
        ax.set_xlabel("cluster")
        plt.tight_layout()
        return fig

    def cross_validation(
        self, n_splits: int = 5, random_state: int = 42, print_metricas: bool = True
    ):
        """
        Grid search manual + cross_val_score (Silhouette) sobre `features`,
        escalonadas via StandardScaler; treina o melhor modelo no dataset
        completo ao final.

            Parâmetros
            ----------
            n_splits : int
                Nº de folds da validação cruzada (default 5).
            random_state : int
                Semente (default 42).
            print_metricas : bool
                Se True, imprime métricas + mostra os gráficos (PCA e
                perfil por cluster) e devolve None. Se False, devolve os
                resultados sem imprimir/plotar -- uso em loop (ex:
                comparar kmeans/dbscan/agglomerative).

            Retorna
            -------
            None se print_metricas=True.
            (cluster_labels, X_scaled, metricas, df_resultados_grid) se
            print_metricas=False.
        """
        X_raw = self.df[self.features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        param_grid = self.__param_grid()
        keys = list(param_grid.keys())

        best_score, best_params, results = -1.0, {}, []
        for a in param_grid[keys[0]]:
            for b in param_grid[keys[1]]:
                for c in param_grid[keys[2]]:
                    if self.modelo == "kmeans":
                        model = KMeans(n_clusters=a, init=b, n_init=c, random_state=random_state)
                    elif self.modelo == "dbscan":
                        model = DBSCAN(eps=a, min_samples=b, metric=c, n_jobs=-1)
                    elif self.modelo == "agglomerative":
                        model = AgglomerativeClustering(n_clusters=a, metric=b, linkage=c)

                    scores = cross_val_score(
                        estimator=model, X=X_scaled, cv=kf,
                        scoring=silhouette_scorer_func, n_jobs=-1,
                    )
                    mean_score = float(np.mean(scores))
                    results.append({
                        keys[0]: a, keys[1]: b, keys[2]: c,
                        "mean_score": mean_score, "std_score": float(np.std(scores)),
                    })
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {keys[0]: a, keys[1]: b, keys[2]: c}

        if self.modelo == "kmeans":
            best_model = KMeans(**best_params, random_state=random_state)
        elif self.modelo == "dbscan":
            best_model = DBSCAN(**best_params, n_jobs=-1)
        elif self.modelo == "agglomerative":
            best_model = AgglomerativeClustering(**best_params)

        cluster_labels = best_model.fit_predict(X_scaled)
        metricas = self.__calcular_metricas(X_scaled, cluster_labels)
        df_resultados = pd.DataFrame(results)

        if print_metricas:
            print("=== MELHOR CONFIGURAÇÃO ENCONTRADA VIA cross_val_score ===")
            print(f"Parâmetros: {best_params}")
            print(f"Silhouette Score Médio (CV): {best_score:.4f}\n")

            ss, chs, dbs = metricas
            print(f"Silhouette Score (fit completo) -> {ss}")
            print(f"Calinski Harabasz Score -> {chs}")
            print(f"Davies Bouldin Score -> {dbs}")
            n_clusters_final = len(np.unique(cluster_labels))
            print(f"Nº de clusters -> {n_clusters_final}")

            self.__grafico_pca(cluster_labels, X_scaled)
            self.__grafico_perfil(cluster_labels)
            plt.show()
            return None

        return cluster_labels, X_scaled, metricas, df_resultados

    def __testar_kmeans(self, best_param: dict, df_test) -> tuple[bool, float, float]:
        kmeans_train = KMeans(**best_param).fit(self.df[self.features])
        labels_train = kmeans_train.labels_
        labels_val_predicted = kmeans_train.predict(df_test[self.features])

        kmeans_val_direct = KMeans(**best_param).fit(df_test[self.features])
        labels_val_direct = kmeans_val_direct.labels_

        train_counts = pd.Series(labels_train).value_counts().sort_index()
        train_proportions = train_counts / len(self.df)
        expected_val_counts = train_proportions * len(df_test)
        observed_val_counts = (
            pd.Series(labels_val_predicted)
            .value_counts()
            .reindex(range(best_param["n_clusters"]), fill_value=0)
            .sort_index()
        )

        chi2_stat, p_value = chisquare(f_obs=observed_val_counts, f_exp=expected_val_counts)
        ari_score = adjusted_rand_score(labels_val_predicted, labels_val_direct)
        return p_value > 0.05, chi2_stat, p_value, ari_score

    def __testar_dbscan(self, best_param: dict, df_test) -> tuple:
        dbscan_train = DBSCAN(**best_param).fit(self.df[self.features])
        labels_train = dbscan_train.labels_

        # DBSCAN não tem .predict() -- projeta teste pro cluster do vizinho
        # mais próximo no treino (k=1), igual a versão original
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.df[self.features], labels_train)
        labels_val_predicted = knn.predict(df_test[self.features])

        dbscan_val_direct = DBSCAN(**best_param).fit(df_test[self.features])
        labels_val_direct = dbscan_val_direct.labels_

        all_clusters = np.unique(np.concatenate([labels_train, labels_val_predicted]))
        train_counts = pd.Series(labels_train).value_counts().reindex(all_clusters, fill_value=0).sort_index()
        train_proportions = train_counts / len(self.df)
        expected_val_counts = train_proportions * len(df_test)
        observed_val_counts = (
            pd.Series(labels_val_predicted).value_counts().reindex(all_clusters, fill_value=0).sort_index()
        )

        chi2_stat, p_value = chisquare(f_obs=observed_val_counts, f_exp=expected_val_counts)
        ari_score = adjusted_rand_score(labels_val_predicted, labels_val_direct)
        return p_value > 0.05, chi2_stat, p_value, ari_score

    def __testar_agg(self, best_param: dict, df_test) -> tuple:
        agg_train = AgglomerativeClustering(**best_param)
        labels_train = agg_train.fit_predict(self.df[self.features])

        # Agglomerative não tem .predict() nem centroide nativo -- projeta
        # teste pro centroide mais próximo dos clusters do treino, igual a
        # versão original
        clf_centroid = NearestCentroid()
        clf_centroid.fit(self.df[self.features], labels_train)
        labels_val_predicted = clf_centroid.predict(df_test[self.features])

        agg_val_direct = AgglomerativeClustering(**best_param)
        labels_val_direct = agg_val_direct.fit_predict(df_test[self.features])

        all_clusters = np.unique(labels_train)
        train_counts = pd.Series(labels_train).value_counts().reindex(all_clusters, fill_value=0).sort_index()
        train_proportions = train_counts / len(self.df)
        expected_val_counts = train_proportions * len(df_test)
        observed_val_counts = (
            pd.Series(labels_val_predicted).value_counts().reindex(all_clusters, fill_value=0).sort_index()
        )

        chi2_stat, p_value = chisquare(f_obs=observed_val_counts, f_exp=expected_val_counts)
        ari_score = adjusted_rand_score(labels_val_predicted, labels_val_direct)
        return p_value > 0.05, chi2_stat, p_value, ari_score

    def testar_modelo(self, best_param: dict, df_test: pd.DataFrame, print_resultado: bool = True) -> bool:
        """
        Teste de consistência treino x teste (qui² de aderência + ARI) --
        NÃO compara contra rótulo verdadeiro, compara o modelo consigo
        mesmo: a distribuição de clusters que o treino prevê pro teste bate
        com o que realmente se observa no teste?

        Fluxo de 3 etapas que essa função fecha (treino já foi
        `cross_validation`, validação é rodar `cross_validation` nos dados
        de validação com os MESMOS `best_param` e comparar as métricas --
        ver notebook):
            1. Treino -> `cross_validation()` acha `best_param` (grid search
               + Silhouette CV).
            2. Validação -> aplica `best_param` (congelado) no conjunto de
               validação, confere se Silhouette/CH/DB continuam estáveis.
            3. Teste -> ESTA função: qui² (a proporção de cada cluster no
               teste bate com a proporção vista no treino?) + ARI (o modelo
               treinado no treino, aplicado no teste, concorda com um
               modelo novo treinado direto no teste?).

            Parâmetros
            ----------
            best_param : dict
                Melhores hiperparâmetros (saída de `cross_validation`).
            df_test : pd.DataFrame
                Conjunto de teste -- mesmas colunas de `features`.
            print_resultado : bool
                Se True, imprime o relatório formatado (default True).

            Retorna
            -------
            bool -- True se p > 0.05 (distribuição do teste consistente com
            o treino), False caso contrário.
        """
        if self.modelo == "kmeans":
            consistente, chi2_stat, p_value, ari_score = self.__testar_kmeans(best_param, df_test)
        elif self.modelo == "dbscan":
            consistente, chi2_stat, p_value, ari_score = self.__testar_dbscan(best_param, df_test)
        elif self.modelo == "agglomerative":
            consistente, chi2_stat, p_value, ari_score = self.__testar_agg(best_param, df_test)

        if print_resultado:
            print("=" * 55)
            print(f" TESTE DE CONSISTÊNCIA TREINO x TESTE ({self.modelo})")
            print("=" * 55)
            print(f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}")
            print(f"p-valor:                        {p_value:.4f}")
            print("-" * 55)
            if consistente:
                print("Conclusão: Não rejeitamos H0 (p > 0.05).")
                print("A distribuição dos clusters no TESTE é CONSISTENTE com o TREINO.")
            else:
                print("Conclusão: Rejeitamos H0 (p <= 0.05).")
                print("A distribuição dos clusters no TESTE é DIFERENTE do TREINO.")
            print("-" * 55)
            print(f"Adjusted Rand Index (ARI) treino-projetado x teste-direto: {ari_score:.4f}")
            print("(ARI perto de 1.0 = a partição geométrica é praticamente idêntica)")

        return consistente

    def rotular_clusters(self, cluster_labels: np.ndarray, top_n: int = 3) -> dict:
        """
        Descrição textual automática de cada cluster -- as `top_n`
        features com maior |z-score| em relação à média geral do
        dataset, com a direção (alto/baixo) e o valor médio real.
        Ponto de partida pra nomear clinicamente (ex: "resistente a
        carbapenêmico com plasmídeo"), não um rótulo definitivo -- a
        tradução de "pct_BD_carbapenemase alto" pra "difícil tratar e
        perigoso" é julgamento clínico, não algo que o algoritmo decide.

            Retorna
            -------
            dict[cluster_id, str]
        """
        perfil = self.df[self.features].groupby(cluster_labels).mean()
        z = (perfil - self.df[self.features].mean()) / self.df[self.features].std()

        descricoes = {}
        for cluster in perfil.index:
            destaques = z.loc[cluster].abs().sort_values(ascending=False).head(top_n).index
            partes = []
            for feat in destaques:
                direcao = "alto" if z.loc[cluster, feat] > 0 else "baixo"
                partes.append(f"{feat}={direcao} ({perfil.loc[cluster, feat]:.2f})")
            descricoes[cluster] = "; ".join(partes)
        return descricoes
