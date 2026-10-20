import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_STATE = 42

# Mesmas cores do design (treino / validação / teste)
CORES_SPLIT = {"Treino": "#82B1D3", "Validação": "#8ED3C7", "Teste": "#FFED6F"}


def grafico_split(tamanhos: dict[str, int]):
    """
    Barra empilhada com a proporção de cada conjunto (70/20/10).
        Parâmetros:
            tamanhos: {"Treino": n, "Validação": n, "Teste": n}
    """
    total = sum(tamanhos.values())
    fig, ax = plt.subplots(figsize=(3.2, 4))
    base = 0.0
    for nome, n in tamanhos.items():
        pct = 100 * n / total
        ax.bar(0, pct, bottom=base, color=CORES_SPLIT[nome], width=0.6, label=nome)
        ax.text(0, base + pct / 2, f"{pct:.0f}% (n = {n})", ha="center", va="center")
        base += pct
    ax.set(xlim=(-0.5, 0.5), ylim=(0, 100), xticks=[])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    sns.despine(ax=ax, bottom=True)
    plt.tight_layout()
    plt.close(fig)

    return fig


def grafico_bivariado(
    df: pd.DataFrame, bivariado: pd.DataFrame, especie: pd.Series, n_col: int = 5
):
    """
    Scatter de todos os pares do clustering bivariado (notebook 03, seção 6),
    na ordem do silhouette CV. Jitter nas binárias para enxergar a densidade.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_lin = int(np.ceil(len(bivariado) / n_col))
    fig, axes = plt.subplots(n_lin, n_col, figsize=(4 * n_col, 3.6 * n_lin))
    axes = axes.ravel()
    for ax, (_, p) in zip(axes, bivariado.iterrows()):
        x = df[p.c1] + (rng.normal(0, 0.06, len(df)) if p.c1_binaria else 0)
        y = df[p.c2] + (rng.normal(0, 0.06, len(df)) if p.c2_binaria else 0)
        sns.scatterplot(
            x=x, y=y, hue=especie.values, s=10, alpha=0.6, ax=ax, legend=False
        )
        ax.set(
            xlabel=p.c1,
            ylabel=p.c2,
            title=f"sil CV={p.silhouette_cv:.2f} (k={p.k})",
        )
    for ax in axes[len(bivariado) :]:
        ax.axis("off")

    # uma legenda só, para a figura inteira
    especies = list(dict.fromkeys(especie.values))
    cores = sns.color_palette(n_colors=len(especies))
    marcadores = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=e)
        for e, c in zip(especies, cores)
    ]
    fig.legend(
        handles=marcadores,
        loc="lower center",
        ncol=min(len(especies), 5),
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    plt.tight_layout()
    plt.close(fig)

    return fig


def grafico_concordancia(concordancia: pd.DataFrame):
    """Heatmap do ARI entre os 3 algoritmos e com a espécie (notebook 04, 3.1)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(concordancia, annot=True, fmt=".3f", vmin=0, vmax=1, ax=ax)
    ax.set_title("Adjusted Rand Index (treino)")
    plt.tight_layout()
    plt.close(fig)

    return fig


def grafico_perfil(df: pd.DataFrame, labels: pd.Series, features: list[str], n_sep: int):
    """
    Perfil por cluster (notebook 04, seção 4): cor = z-score da média do
    cluster em relação ao conjunto; número = média da feature no cluster.
    A linha preta separa as features do clustering das descritivas.
    """
    perfil = df.groupby(labels)[features].mean()
    z = (perfil - df[features].mean()) / df[features].std()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        z.T,
        cmap="RdBu_r",
        center=0,
        annot=perfil.T.round(2),
        fmt="",
        ax=ax,
        cbar_kws={"label": "z score"},
    )
    ax.axhline(n_sep, c="k", lw=2)
    ax.set(title="Perfil por cluster", xlabel="cluster")
    plt.tight_layout()
    plt.close(fig)

    return fig
