"""
src/features_amr.py

Classificação de classe de Ambler (A/B/C/D) a partir do catálogo NCBI
AMRFinderPlus (refgenes.tsv) + construção do conjunto final de 9 features
de resistência/mobilidade por genoma, normalizadas pelo total de hits
beta-lactâmicos daquele genoma -- ver as iterações que levaram a este
desenho em testando_features.ipynb (compartimento por grupo -> muito
esparso; score de mecanismos 0-6 -> redundante com mdr_score; proporção
sobre o total geral de AMR -> diluída por classes de droga irrelevantes).
"""

import re

import numpy as np
import pandas as pd

BETA_LACTAM_DRUG_CLASSES = {
    "penam", "cephalosporin", "cephamycin", "carbapenem", "penem", "monobactam",
}

FAMILIAS_COLISTINA = re.compile(
    r"pmr phosphoethanolamine transferase|mcr phosphoethanolamine transferase|"
    r"polymyxin resistance operon|lipid a acyltransferase|alm glyc",
    re.IGNORECASE,
)
FAMILIAS_AMINOGLICOSIDEO = re.compile(r"^(aac|aph|ant)\(|16s rrna methyltransferase", re.IGNORECASE)

# EXCECOES_MANUAIS: famílias do resistoma que não bateram contra o
# catálogo do NCBI e foram checadas manualmente contra a literatura
# (mesma lista de 02_make_features.ipynb)
EXCECOES_MANUAIS = {
    "class a mycobacterium tuberculosis bla beta-lactamase": "A",
    "mor beta-lactamase": "C",
    "blaa beta-lactamase": "A",
}


class AmblerClassifier:
    """
    Classifica um nome de família de gene (amr_gene_family do CARD) em
    classe de Ambler (A/B1/B2/B3/B/C/D), a partir do catálogo NCBI
    AMRFinderPlus Reference Gene Catalog (refgenes.tsv, schema atual --
    colunas snake_case: product_name, gene_family).

        Parâmetros
        ----------
        refgenes_path : str | Path
            Caminho pro refgenes.tsv.

        Métodos
        -------
        classificar(nome_card) -> str | None
            Classe de Ambler pro nome de família do CARD, ou None se
            não reconhecida (casamento por raiz do nome, ignorando
            sufixos -like/-type -- ver _bate).
    """

    def __init__(self, refgenes_path) -> None:
        ref = pd.read_csv(refgenes_path, sep="\t")
        ref["ambler_class"] = ref["product_name"].apply(self._extrai_classe_ambler)
        ref_bl = ref.dropna(subset=["ambler_class", "gene_family"]).copy()
        ref_bl["familia"] = ref_bl["gene_family"].str.strip().str.replace(
            r"^bla", "", regex=True, flags=re.IGNORECASE
        )
        self.familias_por_classe = ref_bl.groupby("ambler_class")["familia"].apply(
            lambda s: sorted(s.unique())
        )

    @staticmethod
    def _extrai_classe_ambler(product_name):
        if pd.isna(product_name):
            return None
        s = str(product_name)
        m = re.search(r"class ([ABCD]) beta-lactamase", s, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # classe B não segue o padrão "class X" -- vem como "subclass
        # B1/B2/B3 metallo-beta-lactamase"
        m = re.search(r"subclass (B[123]) metallo-beta-lactamase", s, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        if re.search(r"metallo-beta-lactamase", s, re.IGNORECASE):
            return "B"
        return None

    @staticmethod
    def _extrai_stem(nome_card):
        m = re.search(r"([\w\-/]+)\s+beta-lactamase", nome_card, re.IGNORECASE)
        stem = m.group(1) if m else nome_card
        return re.sub(r"-(like|type)$", "", stem, flags=re.IGNORECASE)

    @staticmethod
    def _bate(stem, familia):
        stem, familia = stem.lower(), familia.lower()
        if len(familia) < 2:  # descarta '' e siglas de 1 letra
            return False
        return stem == familia or stem.startswith(familia + "-") or familia.startswith(stem + "-")

    def classificar(self, nome_card: str) -> str | None:
        if nome_card.lower() in EXCECOES_MANUAIS:
            return EXCECOES_MANUAIS[nome_card.lower()]
        stem = self._extrai_stem(nome_card)
        for classe in ["A", "B1", "B2", "B3", "B", "C", "D"]:
            lista = self.familias_por_classe.get(classe, [])
            if any(self._bate(stem, f) for f in lista):
                return classe
        return None


def anotar_resistoma(resistome: pd.DataFrame, classifier: AmblerClassifier) -> pd.DataFrame:
    """
    Adiciona as colunas derivadas usadas por build_features -- ambler_class,
    tem_carbapenem, tem_cefalosporina, carbapenemase_confirmada,
    toca_beta_lactam -- devolve cópia, não modifica o `resistome` de entrada.

        Parâmetros
        ----------
        resistome : pd.DataFrame
            dataset.resistome (ou lido direto de resistome.parquet) --
            precisa de amr_gene_family/drug_class como `set` por linha.
        classifier : AmblerClassifier

        Retorna
        -------
        pd.DataFrame
    """
    r = resistome.copy()
    # um hit pode ter mais de 1 valor em amr_gene_family -- pega a
    # primeira classificação válida entre os valores do set
    r["ambler_class"] = r["amr_gene_family"].apply(
        lambda s: next((classifier.classificar(f) for f in s if classifier.classificar(f)), None)
    )
    r["tem_carbapenem"] = r["drug_class"].apply(lambda s: any("carbapenem" in dc.lower() for dc in s))
    r["tem_cefalosporina"] = r["drug_class"].apply(lambda s: any("cephalosporin" in dc.lower() for dc in s))
    r["carbapenemase_confirmada"] = (
        r["ambler_class"].isin(["A", "B", "B1", "B2", "B3", "D"]) & r["tem_carbapenem"]
    )
    r["toca_beta_lactam"] = r["drug_class"].apply(lambda s: bool(s & BETA_LACTAM_DRUG_CLASSES))
    return r


def _tem_familia(gene_family_set, pattern) -> bool:
    return any(pattern.search(f) for f in gene_family_set)


def build_features(resistome_anotado: pd.DataFrame, index) -> pd.DataFrame:
    """
    Constrói as 9 features finais de resistência/mobilidade, 1 linha por
    genoma. Todos os `pct_*`/`n_beta_lactam_plasmidial` são NaN (não 0)
    pra genoma sem nenhum hit beta-lactâmico -- 0/0 é indefinido, não
    "sem resistência daquele tipo".

        Parâmetros
        ----------
        resistome_anotado : pd.DataFrame
            Saída de anotar_resistoma() -- precisa também de `contig_type`
            (chromosome/plasmid/virus/NaN) pra pct_beta_lactam_plasmidial.
        index
            Índice completo de amostras (ex: metadata.index) -- genoma
            sem nenhum hit vira linha de zeros/NaN, não fica de fora.

        Retorna
        -------
        pd.DataFrame com as colunas:
            n_beta_lactam_total       -- contagem, denominador dos pct_*
            pct_esbl                  -- classe A, hidrolisa cefalosporina, não-carbapenemase
            pct_A_serino_carbapenemase -- classe A, carbapenemase confirmada
            pct_BD_carbapenemase      -- classe B (qualquer subclasse) ou D, carbapenemase confirmada
            pct_C_ampc                -- classe C
            pct_nao_hidrolise         -- toca beta-lactâmico mas sem beta-lactamase reconhecida
                                          (PBP modificada, porina, efluxo)
            pct_beta_lactam_plasmidial -- fração dos hits beta-lactâmicos em contig plasmidial
            n_aminoglicosideo         -- contagem bruta (classe de droga diferente, não normalizada)
            n_polimixina              -- contagem bruta (idem)
    """
    r = resistome_anotado

    mask_esbl = (r["ambler_class"] == "A") & r["tem_cefalosporina"] & ~r["carbapenemase_confirmada"]
    mask_A_carbapenemase = (r["ambler_class"] == "A") & r["carbapenemase_confirmada"]
    mask_BD_carbapenemase = r["ambler_class"].isin(["B", "B1", "B2", "B3", "D"]) & r["carbapenemase_confirmada"]
    mask_C_ampc = r["ambler_class"] == "C"
    mask_nao_hidrolise = r["toca_beta_lactam"] & r["ambler_class"].isna()
    mask_aminoglicosideo = r["amr_gene_family"].apply(lambda s: _tem_familia(s, FAMILIAS_AMINOGLICOSIDEO))
    mask_polimixina = r["amr_gene_family"].apply(lambda s: _tem_familia(s, FAMILIAS_COLISTINA))
    mask_beta_lactam_plasmidial = r["toca_beta_lactam"] & (r["contig_type"] == "plasmid")

    def contagem(mask, nome):
        return r[mask].groupby("sample_id").size().reindex(index, fill_value=0).rename(nome)

    n_beta_lactam_total = contagem(r["toca_beta_lactam"], "n_beta_lactam_total")
    n_esbl = contagem(mask_esbl, "n_esbl")
    n_A_serino = contagem(mask_A_carbapenemase, "n_A_serino_carbapenemase")
    n_BD = contagem(mask_BD_carbapenemase, "n_BD_carbapenemase")
    n_C = contagem(mask_C_ampc, "n_C_ampc")
    n_nao_hidrolise = contagem(mask_nao_hidrolise, "n_nao_hidrolise")
    n_beta_lactam_plasmidial = contagem(mask_beta_lactam_plasmidial, "n_beta_lactam_plasmidial")
    n_aminoglicosideo = contagem(mask_aminoglicosideo, "n_aminoglicosideo")
    n_polimixina = contagem(mask_polimixina, "n_polimixina")

    def pct(n_sub):
        return (n_sub / n_beta_lactam_total).replace([np.inf, -np.inf], np.nan)

    return pd.concat([
        n_beta_lactam_total,
        pct(n_esbl).rename("pct_esbl"),
        pct(n_A_serino).rename("pct_A_serino_carbapenemase"),
        pct(n_BD).rename("pct_BD_carbapenemase"),
        pct(n_C).rename("pct_C_ampc"),
        pct(n_nao_hidrolise).rename("pct_nao_hidrolise"),
        pct(n_beta_lactam_plasmidial).rename("pct_beta_lactam_plasmidial"),
        n_aminoglicosideo,
        n_polimixina,
    ], axis=1)
