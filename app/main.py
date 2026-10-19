import streamlit as st

from src.paginas import Paginas


def pagina_inicio(entrada: str) -> str:
    """
    Função de inicialização da página inicial.
    Parametros:
        entrada: str
    Retorna:
        str: Mensagem de sucesso.
    """
    return entrada


def pagina_estrutura() -> str:
    """
    Função de estruturação da página inicial.
    Parametros:
        None
    Retorna:
        str: Mensagem de sucesso.
    """
    st.set_page_config(page_title="Isolados BR - ML", layout="wide", page_icon="🧬")
    st.markdown(
        """
        <style>
            .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 3rem;
                    padding-right: 3rem;
                }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title = "Risco de resistência em isolados clínicos brasileiros"
    st.markdown(
        f"<br><h3 style='text-align: center; '>{title}</h3>", unsafe_allow_html=True
    )

    return "Ajustado"


if __name__ == "__main__":
    pagina_estrutura()
    pagina = Paginas()
    pagina.metricas()
