import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


class Classifier:
    """
    Exibe os resultados de um classificador já treinado (notebook final/03).
        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame com as colunas 'Dataset' (Treino/Teste) e
            'Classe_Predita'.

        Métodos
        -------
        testar_hipotese()
            Qui-quadrado da distribuição das classes preditas.
        matrix_confusao(df, obs, pred)
            Matriz de confusão e classification report no teste.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

        return None

    def testar_hipotese(self) -> None:
        """
        Qui-quadrado de homogeneidade: a distribuição das classes preditas
        é igual no treino e no teste?
        """
        tabela_contingencia = pd.crosstab(
            self.df["Dataset"],
            self.df["Classe_Predita"],
        )
        st.write("Tabela de contingência:")
        st.dataframe(tabela_contingencia.T)

        chi2, p_valor, _, _ = chi2_contingency(tabela_contingencia)
        st.write(f"χ² = {chi2:.4f} | p-valor = {p_valor:.4f}")

        if p_valor < 0.05:
            mensagem = "<b style='color: red;'>DIFERENTE</b> no treino e no teste (rejeita H0)."
        else:
            mensagem = "<b style='color: green;'>IGUAL</b> no treino e no teste (não rejeita H0)."
        st.markdown(mensagem, unsafe_allow_html=True)

        return None

    def matrix_confusao(self, df: pd.DataFrame, obs: str, pred: str) -> None:
        """
        Exibe a matriz de confusão e o classification report no teste.
            Parâmetros
            ----------
            df : pd.DataFrame
                DataFrame com a classe observada e uma coluna de previsão
                por modelo.
            obs : str
                Coluna da classe observada.
            pred : str
                Coluna da classe predita pelo modelo.
        """
        y_obs = df[obs]
        y_pred = df[pred]

        # mesma ordem de rótulos na matriz e nos eixos
        rotulos = sorted(set(y_obs) | set(y_pred))
        accuracy = accuracy_score(y_obs, y_pred)
        cm = confusion_matrix(y_obs, y_pred, labels=rotulos)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rotulos)
        disp.plot(cmap=plt.cm.Greens)
        disp.ax_.set_title(f"Accuracy {accuracy:.2f}")
        st.pyplot(disp.figure_)
        plt.close(disp.figure_)
        st.write("Classification report:")
        st.dataframe(
            (pd.DataFrame(classification_report(y_obs, y_pred, output_dict=True)).T)
        )

        return None
