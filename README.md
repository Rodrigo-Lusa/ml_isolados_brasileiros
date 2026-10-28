# Python para Machine Learning

**Aluno:** Rodrigo A. Lusa - Mestrado<br>
**Professor:** Guilherme F. Silveira (https://github.com/Laboratorio-de-Analise-de-Dados/disciplina_python_ml)

## Pré-processamento

Os genomas bacterianos utilizados neste estudo foram obtidos, em 03/02/2026, a partir
de dois repositórios públicos: BV-BRC v3.58.4 ([BV-BRC](https://www.bv-brc.org/)) e
NCBI Pathogen Detection ([NCBI-Pathogens](https://www.ncbi.nlm.nih.gov/pathogens/)).
No BV-BRC, foram aplicados os filtros de origem geográfica (Brasil), hospedeiro humano
e classificação de qualidade "good", resultando em 4.310 genomas. No NCBI Pathogen
Detection, foram aplicados os filtros de origem geográfica (Brasil) e hospedeiro
humano, resultando em 6.646 genomas. Após a padronização dos metadados (ano de coleta,
espécie e fonte de isolamento) e remoção de duplicatas (mesmo `Assembly_id`) entre as
duas bases, obteve-se um total de 7.068 isolados bacterianos brasileiros únicos.

- **NCBI Datasets** (v18.16.0) - download
- **Prodigal** (v2.6.3) — predição de genes codificadores de proteína
- **RGI** (v6.0.5) — identificação de genes de resistência antimicrobiana, usando o banco de dados CARD (v4.0.1)
- **geNomad** (v1.12.0) — classificação de contigs em cromossomal ou plasmidial

Foram filtrados apenas os genomas de espécies ou gêneros listados pela OMS (2024) como de
nível crítico.
Divisão estratificada por espécie (espécies com < 20 genomas agrupadas em "outras").

## App

O app do Streamlit traz as etapas de pré-processamento, clustering, classificação e
regressão: [Abrir o app](https://ml-isolados-brasileiros.streamlit.app/)

## Caminhos (pro professor não se perder)

### Notebooks

**Versão final** (`notebooks/final/`)
1. [01_ydata_selecao.ipynb](notebooks/final/01_ydata_selecao.ipynb): seleção de features com ydata-profiling
2. [02_clusterizacao.ipynb](notebooks/final/02_clusterizacao.ipynb): clusterização
3. [03_classificacao.ipynb](notebooks/final/03_classificacao.ipynb): classificação
4. [04_regressao.ipynb](notebooks/final/04_regressao.ipynb): regressão

**v2** (`notebooks/`)
- [00_ajuste_metadados.ipynb](notebooks/00_ajuste_metadados.ipynb)
- [01_EDA_metadados.ipynb](notebooks/01_EDA_metadados.ipynb)
- [02_classes_bl.ipynb](notebooks/02_classes_bl.ipynb)
- [03_feature_selection.ipynb](notebooks/03_feature_selection.ipynb)
- [03_new_make_features.ipynb](notebooks/03_new_make_features.ipynb)
- [04_clusterizacao_risco.ipynb](notebooks/04_clusterizacao_risco.ipynb)
- [05_classificacao.ipynb](notebooks/05_classificacao.ipynb)
- [06_regressao.ipynb](notebooks/06_regressao.ipynb)
- [07_ydata_final.ipynb](notebooks/07_ydata_final.ipynb)
- [explorar_refgenes.ipynb](notebooks/explorar_refgenes.ipynb)

**v1** (`notebooks/old/`)
- [08_v1_old_make_features.ipynb](notebooks/old/08_v1_old_make_features.ipynb)
- [09_v1_make_features.ipynb](notebooks/old/09_v1_make_features.ipynb)
- [10_v1_ydata_estruturacao.ipynb](notebooks/old/10_v1_ydata_estruturacao.ipynb)
- [11_v1_divisao.ipynb](notebooks/old/11_v1_divisao.ipynb)
- [12_v1_clusterizacao_risco.ipynb](notebooks/old/12_v1_clusterizacao_risco.ipynb)

Funções auxiliares usadas pelos notebooks: [notebooks/src/](notebooks/src/)

### Reports (ydata-profiling)

**Versão final** (`reports/final/`), gerados por `final/01_ydata_selecao.ipynb`
- [features_antes.html](reports/final/features_antes.html)
- [features_depois.html](reports/final/features_depois.html)

**v2** (`reports/`), gerados por `07_ydata_final.ipynb`
- [features_antes_estruturacao.html](reports/features_antes_estruturacao.html)
- [features_depois_estruturacao.html](reports/features_depois_estruturacao.html)
- [features_finais.html](reports/features_finais.html)

### App (Streamlit)

- Código: [app/](app/), com ponto de entrada em [app/main.py](app/main.py) e módulos em [app/src/](app/src/)
- Para rodar localmente: `streamlit run app/main.py`

### Testes e cobertura

- Testes: [app/py_test/](app/py_test/)
