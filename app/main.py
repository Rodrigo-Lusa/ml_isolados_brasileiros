import streamlit as st

from src.paginas import Paginas, URL_PROFESSOR

ESTILO = """
<style>
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .banner {
        padding: 0.5rem 2rem 0.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .banner h1 {
        font-weight: 800;
        font-size: 2.6rem;
        line-height: 1.3;
        padding: 0 0 0.6rem 0;
    }
    .banner .pill {
        display: inline-block;
        text-align: left;
        font-size: 1.05rem;
        line-height: 1.6;
        background: linear-gradient(90deg, rgba(123, 127, 178, 0.25), rgba(130, 177, 211, 0.3));
        border-radius: 2rem;
        padding: 0.3rem 1.5rem;
    }
    .banner .pill a { color: inherit; text-decoration: none; }
    .titulo-modelo {
        background: #82B1D3;
        color: #000;
        border-radius: 0.5rem;
        padding: 0.4rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .caixa-amarela {
        background: #FFED6F;
        color: #000;
        display: inline-block;
        padding: 0.4rem 1rem 0 0.4rem;
        margin-bottom: 1rem;
    }
    .card-dataset {
        color: #000;
        display: inline-block;
        text-align: center;
        padding: 0.4rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .card-github {
        width: 100%;
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 6px;
        margin-top: 1rem;
    }
    [data-testid="stSidebarNav"] a span { font-size: 1.1rem; }
</style>
"""


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
    Função de estruturação das páginas: configuração, estilo e o banner
    com o título, o aluno e o professor (igual em todas as páginas).
    Parametros:
        None
    Retorna:
        str: Mensagem de sucesso.
    """
    st.set_page_config(page_title="Isolados BR - ML", layout="wide")
    st.markdown(ESTILO, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="banner">
            <h1>Python para Machine Learning</h1>
            <div class="pill">
                <b>Aluno:</b> Rodrigo A. Lusa - Mestrado<br>
                <b>Professor:</b> Guilherme F. Silveira
                (<a href="{URL_PROFESSOR}" target="_blank">{URL_PROFESSOR}</a>)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return "Ajustado"


if __name__ == "__main__":
    pagina_estrutura()
    pagina = Paginas()
    pagina.metricas()
