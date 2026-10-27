import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def grafico_corr(corr: pd.DataFrame, titulo: str):
    """Matriz de correlação do ydata (auto), antes ou depois da seleção (final/01)."""
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(corr), 0.6 * len(corr)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1, ax=ax,
        annot_kws={"fontsize": 7},
    )
    ax.set_title(titulo)
    plt.tight_layout()
    plt.close(fig)

    return fig


def grafico_metricas(
    df: pd.DataFrame,
    metricas: dict[str, str],
    titulo: str,
    referencias: dict[str, float] | None = None,
    formatos: dict[str, str] | None = None,
):
    """
    Métricas por algoritmo (eixo x) e conjunto (cor), um painel por métrica
    porque as escalas são diferentes (mesmo gráfico dos notebooks final/02–04).
        Parâmetros:
            df: formato longo, colunas "modelo", "conjunto" e uma por métrica.
            metricas: {coluna: título do painel}.
            referencias: linha tracejada por métrica (ex.: {"qui2_p": 0.05}).
            formatos: rótulo das barras por métrica (default "%.2f").
    """
    referencias, formatos = referencias or {}, formatos or {}
    cores = {k.lower(): v for k, v in CORES_SPLIT.items()}
    fig, axes = plt.subplots(1, len(metricas), figsize=(4.3 * len(metricas), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (col, rotulo) in zip(axes, metricas.items()):
        sns.barplot(
            data=df, x="modelo", y=col, hue="conjunto", hue_order=list(cores),
            palette=cores, ax=ax,
        )
        for barras in ax.containers:
            ax.bar_label(barras, fmt=formatos.get(col, "%.2f"), fontsize=7)
        if col in referencias:
            ax.axhline(referencias[col], ls="--", c="k", lw=1)
        ax.set(title=rotulo, xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=25)
        ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
        frameon=False,
    )
    fig.suptitle(titulo, y=1.12)
    sns.despine(fig)
    plt.tight_layout()
    plt.close(fig)

    return fig


def grafico_concordancia(concordancia: pd.DataFrame):
    """Heatmap do ARI entre os 3 algoritmos e com a espécie (notebook final/02, 2.1)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(concordancia, annot=True, fmt=".3f", vmin=0, vmax=1, ax=ax)
    ax.set_title("Adjusted Rand Index (treino)")
    plt.tight_layout()
    plt.close(fig)

    return fig


def grafico_perfil(df: pd.DataFrame, labels: pd.Series, features: list[str], n_sep: int):
    """
    Perfil por cluster (notebook final/02, seção 4): cor = z-score da média do
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
