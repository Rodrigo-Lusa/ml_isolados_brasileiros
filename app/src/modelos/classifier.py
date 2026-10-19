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
    Exibe os resultados de um classificador já treinado (notebook 05).
        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame com as colunas 'Dataset' (Validação/Teste) e
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
        Testa a hipótese de que a distribuição das classes preditas na
        validação e no teste é estatisticamente igual (homogênea)
        ou diferente (heterogênea) usando o teste Qui-Quadrado.
        """
        # Tabela de Contingência (Frequências Observadas)
        tabela_contingencia = pd.crosstab(
            self.df["Dataset"],
            self.df["Classe_Predita"],
        )
        st.write("Tabela de Contingência (Observada):")
        st.dataframe(tabela_contingencia.T)
        st.write("-" * 40)

        chi2, p_valor, graus, _ = chi2_contingency(tabela_contingencia)

        st.write(f"Estatística Qui-Quadrado: {chi2:.4f}")
        st.write(f"Graus de Liberdade: {graus}")
        st.write(f"Valor-p: {p_valor:.4f}")

        alpha = 0.05
        st.write("\nConclusão:")
        if p_valor < alpha:
            mensagem = "Rejeitamos H0: "
            mensagem += "A distribuição das classes na validação e no teste é "
            mensagem += "<b style='color: red;'>ESTATISTICAMENTE DIFERENTE "
            mensagem += "(Heterogênea).</b>"
            st.markdown(mensagem, unsafe_allow_html=True)
        else:
            mensagem = "Falhamos em rejeitar H0: "
            mensagem += "A distribuição das classes na validação e no teste é "
            mensagem += "<b style='color: green;'>ESTATISTICAMENTE IGUAL "
            mensagem += "(Homogênea).</b>"
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
        st.write("\nClassification Report:")
        st.dataframe(
            (pd.DataFrame(classification_report(y_obs, y_pred, output_dict=True)).T)
        )

        return None
