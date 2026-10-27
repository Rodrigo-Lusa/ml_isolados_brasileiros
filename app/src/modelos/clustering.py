# Métricas de avaliação de clustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Bibliotecas para clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Testagem dos modelos
from scipy.stats import chisquare
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# Bibliotecas para o modelo de clustering
import numpy as np
import pandas as pd

# Geração de gráficos
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42


class Clustering:
    """
    Reproduz o clustering do notebook final/02 com os parâmetros escolhidos:
    ajusta no treino (features escalonadas com o scaler do treino), projeta
    os rótulos num conjunto novo e compara com um refit nesse conjunto.
    """

    def __init__(self, features: list[str], df: pd.DataFrame, modelo: str) -> None:
        self.features = features
        self.df = df
        self.modelo = modelo
        self.scaler = StandardScaler().fit(df[features])
        self.X_train = self.scaler.transform(df[features])

        return None

    def __instanciar(self, best_param: dict, n: int):
        """
        Instancia o modelo. No DBSCAN, min_samples é reescalado para o
        tamanho do conjunto (mesma fração do treino).
        """
        if self.modelo == "kmeans":
            return KMeans(**best_param, random_state=RANDOM_STATE)
        if self.modelo == "dbscan":
            frac = best_param["min_samples"] / len(self.X_train)
            return DBSCAN(**{**best_param, "min_samples": max(3, round(frac * n))})
        return AgglomerativeClustering(**best_param)

    def __projetar(self, fit, X_novo: np.ndarray) -> np.ndarray:
        """
        Atribui cada amostra nova a um cluster do treino: centroide no
        KMeans e no Agglomerative, vizinho mais próximo no DBSCAN.
        """
        if self.modelo == "kmeans":
            return fit.predict(X_novo)
        if self.modelo == "agglomerative":
            return NearestCentroid().fit(self.X_train, fit.labels_).predict(X_novo)
        return (
            KNeighborsClassifier(n_neighbors=1)
            .fit(self.X_train, fit.labels_)
            .predict(X_novo)
        )

    def grafico_pca(self, X_novo: np.ndarray, hue: np.ndarray, titulo: str):
        """
        Projeção PCA (ajustada no treino) do conjunto novo, cor = cluster.
        Jitter pequeno porque as features binárias repetem pontos.
        """
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(self.X_train)
        rng = np.random.default_rng(RANDOM_STATE)
        coords = pca.transform(X_novo) + rng.normal(0, 0.05, (len(X_novo), 2))

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=pd.Series(hue).astype(str).values,
            hue_order=[str(c) for c in sorted(set(hue))],
            s=15,
            alpha=0.7,
            ax=ax,
        )
        ax.set(
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
            title=titulo,
        )
        ax.legend(title="cluster", fontsize=8, loc="best")
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.close(fig)

        return fig

    def testar_modelo(self, best_param: dict, df_test: pd.DataFrame) -> tuple:
        """
        Testa a consistência dos clusters do treino num conjunto novo
        (validação ou teste) com o Qui-Quadrado de Aderência e o
        Índice de Rand Ajustado (ARI) entre projeção e refit.
            Parâmetros:
                best_param: dict
                    Parâmetros escolhidos no notebook final/02.
                df_test: pd.DataFrame
                    Conjunto comparado com o treino (dados brutos).
            Retorna:
                tuple[Figure, str]: gráfico PCA e mensagem HTML.
        """
        X_test = self.scaler.transform(df_test[self.features])

        # Modelo treinado no Treino e rótulos projetados no conjunto novo
        fit = self.__instanciar(best_param, len(self.X_train)).fit(self.X_train)
        labels_train = fit.labels_
        labels_proj = self.__projetar(fit, X_test)

        # Modelo independente ajustado direto no conjunto novo (para o ARI)
        labels_refit = self.__instanciar(best_param, len(X_test)).fit_predict(X_test)

        # Frequências: proporções do treino * N do conjunto novo
        clusters = np.unique(labels_train)
        train_props = pd.Series(labels_train).value_counts().reindex(clusters) / len(
            labels_train
        )
        expected = train_props * len(X_test)
        observed = pd.Series(labels_proj).value_counts().reindex(clusters, fill_value=0)

        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        ari_score = adjusted_rand_score(labels_proj, labels_refit)

        mask = labels_proj != -1
        silhouette = silhouette_score(X_test[mask], labels_proj[mask])

        # --- Exibição dos Resultados ---
        msn_resul = f"<div class='titulo-modelo'>{self.modelo}</div>"
        msn_resul += f"Clusters: {clusters}<br>"
        msn_resul += f"Esperadas: {expected.round(1).values}<br>"
        msn_resul += f"Observadas: {observed.values}<br><br>"
        msn_resul += f"χ² = {chi2_stat:.4f} | p-valor = {p_value:.4f}<br>"
        msn_resul += f"Silhouette = {silhouette:.3f} | ARI = {ari_score:.4f}<br><br>"

        if p_value > 0.05:
            msn_resul += "Qui²: distribuição <b style='color: green;'>IGUAL</b> à do treino.<br>"
        else:
            msn_resul += "Qui²: distribuição <b style='color: red;'>DIFERENTE</b> da do treino.<br>"

        if ari_score > 0.5:
            msn_resul += "ARI: projeção e refit <b style='color: green;'>SEMELHANTES</b>.<br><br>"
        else:
            msn_resul += "ARI: projeção e refit <b style='color: red;'>DIFERENTES</b>.<br><br>"

        grafico = self.grafico_pca(
            X_novo=X_test, hue=labels_proj, titulo=f"{self.modelo} (projetado)"
        )

        return grafico, msn_resul
