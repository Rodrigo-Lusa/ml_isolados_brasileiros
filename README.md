# ml_isolados_brasileiros

Projeto de mestrado: comparação dos resistomas (conjunto de genes de resistência) de isolados clínicos e de genomas bacterianos obtidos de metagenomas ambientais (MAGs), com foco no conceito de "Uma Só Saúde".


## Pré-processamento dos dados

### Obtenção dos genomas
Os genomas bacterianos utilizados neste estudo foram obtidos, em
03/02/2026, a partir de dois repositórios públicos: BV-BRC v3.58.4
(https://www.bv-brc.org) e NCBI Pathogen Detection
(https://www.ncbi.nlm.nih.gov/pathogens). No BV-BRC, foram aplicados os filtros de
origem geográfica (Brasil), hospedeiro humano e classificação de qualidade "good",
resultando em 4.310 genomas. No NCBI Pathogen Detection, foram aplicados os
filtros de origem geográfica (Brasil) e hospedeiro humano, resultando em 6.646
genomas. Após a padronização dos metadados (ano de coleta, espécie e fonte de
isolamento) e remoção de duplicatas (mesmo `Assembly_id`) entre as duas bases, obteve-se um total de
7.068 isolados bacterianos brasileiros únicos.

- **NCBI Datasets** (v18.16.0) - download
- **Prodigal** (v2.6.3) — predição de genes codificadores de proteína
- **RGI** (v6.0.5) — identificação de genes de resistência antimicrobiana, usando o banco de dados CARD (v4.0.1)
- **geNomad** (v1.12.0) — classificação de contigs em cromossomal, plasmidial ou viral, e identificação de elementos genéticos móveis
- **abricate** (v1.4.0) - anotação de genes de virulência com base no banco de dados VFDB (03/04/2026)

### Filtragem de metadados
Em `notebooks/00_ajuste_metadados.ipynb` filtrei e ajustei a tabela de metadados para o final de 7038 genomas, com melhor separação de `source_type` e classificação OMS.

![alt text](data/images/bubble_plot.png)


### Obtenção das matrizes
As matrizes de presença/ausência de genes de resistência e as tabelas de anotação são obtidas com o **ARGOS**, biblioteca própria que criei com o intuito de me ajudar a analisar os resistomas, instalada em modo editável (`pip install -e .`) em um ambiente conda. 
No entanto, tinha entendido errado o conceito de classes e métodos em python. Por isso, os notebooks que importam `argos` usam métodos e funções que não podem ser feitas sem esse ambiente.

## Ideia do projeto

### Clusterização

Objetivo: agrupar os genomas (isolados clínicos e, futuramente, MAGs ambientais) em grupos de risco de resistência, a partir de um conjunto de features derivadas do resistoma, sem usar os genes/alelos específicamente.

### Classificação

Classificar os genomas segundo o alvo definido pela clusterização acima (prever o grupo de risco um genoma pertence, a partir das mesmas features).

### Regressão

Ainda não decidi.
