"""
src/graficos.py

Gráfico de métricas por algoritmo e conjunto (treino / validação / teste), usado nos notebooks
de `notebooks/final/`. Um painel por métrica, porque as escalas são diferentes (silhouette vai
de -1 a 1, Calinski-Harabasz chega a milhares).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CONJUNTOS = ["treino", "validação", "teste"]
# mesmas cores do app (app/src/graficos.py)
CORES = {"treino": "#82B1D3", "validação": "#8ED3C7", "teste": "#FFED6F"}


def grafico_metricas(
    df: pd.DataFrame,
    metricas: dict[str, str],
    titulo: str,
    referencias: dict[str, float] | None = None,
    formatos: dict[str, str] | None = None,
    x: str = "modelo",
):
    """
    Barras agrupadas: `x` no eixo, cor = conjunto, um painel por métrica.

        Parâmetros
        ----------
        df : pd.DataFrame
            Formato longo: colunas `x`, "conjunto" e uma coluna por métrica.
        metricas : dict[str, str]
            {coluna: título do painel}.
        titulo : str
            Título da figura.
        referencias : dict[str, float] | None
            Linha tracejada de referência por métrica (ex.: {"qui2_p": 0.05}).
        formatos : dict[str, str] | None
            Formato do rótulo das barras por métrica (default "%.2f").
    """
    referencias, formatos = referencias or {}, formatos or {}
    fig, axes = plt.subplots(1, len(metricas), figsize=(4.3 * len(metricas), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (col, rotulo) in zip(axes, metricas.items()):
        sns.barplot(data=df, x=x, y=col, hue="conjunto", hue_order=CONJUNTOS, palette=CORES, ax=ax)
        for barras in ax.containers:
            ax.bar_label(barras, fmt=formatos.get(col, "%.2f"), fontsize=7)
        if col in referencias:
            ax.axhline(referencias[col], ls="--", c="k", lw=1)
        ax.set(title=rotulo, xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=25)
        ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle(titulo, y=1.12)
    sns.despine(fig)
    plt.tight_layout()

    return fig
