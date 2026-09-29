"""
src/features_risco.py

Features de RISCO por genoma, construídas só a partir do resistoma ADQUIRIDO.

Por que existe (substitui `features_amr.build_features` para a clusterização):
as features v1 (`pct_*` sobre o total de hits beta-lactâmicos) misturavam genes
intrínsecos de espécie (OXA-51 em A. baumannii, SHV-1/11 em K. pneumoniae,
EC em E. coli, efluxo/porina em todos) com genes adquiridos. O KMeans então
reencontrava a taxonomia (k=2 = Acinetobacter x Enterobacterales), não o risco.

Duas camadas de decisão por gene (hit do RGI):

1. Intrínseco x adquirido: `INTRINSECOS_POR_GENERO` (lista manual, por
   gênero/espécie, com a referência biológica ao lado). Tudo que não é
   intrínseco E pertence a uma família de resistência reconhecidamente
   transferível (`_classe_adquirida`) conta como adquirido. Efluxo/porina/
   reguladores/alvos cromossômicos nunca entram.
2. Função da beta-lactamase: vem do NCBI AMRFinderPlus Reference Gene
   Catalog (refgenes.tsv, `product_name`), curado alelo a alelo:
        - "carbapenem-hydrolyzing"/"metallo-beta-lactamase" -> carbapenemase,

Sem score de pesos: toda feature é presença/ausência ou contagem de um
mecanismo definido.
"""

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Genes intrínsecos (cromossômicos, presentes em ~toda a espécie/gênero)

# chave = gênero ou espécie
# valor = regex sobre `gene_clean`

INTRINSECOS_POR_GENERO = {
    # OXA-51-like (blaOXA-51 família, cromossômica, marcador de espécie) e
    # ADC (AmpC cromossômica, Acinetobacter-derived cephalosporinase);
    # ANT(3'')-IIa cromossômica em A. baumannii (99% do recorte)
    "Acinetobacter": [r"^OXA-51-like$", r"^ADC-", r"^ANT\(3''\)-IIa$"],
    # SHV/OKP/LEN = beta-lactamase cromossômica de K. pneumoniae /
    # quasipneumoniae / variicola; OXY = K. oxytoca/michiganensis;
    # fosA cromossômica. SHV com função ESBL/carbapenemase NÃO é tratada
    # como intrínseca (ver `_eh_intrinseco`).
    "Klebsiella": [r"^SHV-", r"^OKP-", r"^LEN-", r"^OXY-", r"^Fos[Aa][25678](?:\.\d+)?$"],
    # K. aerogenes (ex-Enterobacter aerogenes): AmpC cromossômica
    # (anotada como CMY-190 pelo CARD, 100% do recorte)
    "Klebsiella aerogenes": [r"^CMY-190$", r"^Fos[Aa][25678](?:\.\d+)?$"],
    # EC = AmpC cromossômica de E. coli; Ecol_ampC_BLA = modelo de promotor
    # (variante) da mesma AmpC -- detecção por variante é pouco confiável
    "Escherichia": [r"^EC-", r"^Ecol_ampC_BLA$"],
    # ACT/MIR/CMH = AmpC cromossômica do complexo E. cloacae; fosA cromossômica
    "Enterobacter": [r"^ACT-", r"^MIR-", r"^CMH-", r"^Fos[Aa][25678](?:\.\d+)?$"],
    # SRT/SST = AmpC cromossômica; AAC(6')-Ic cromossômica (100% do recorte);
    # tet(41) cromossômica
    "Serratia": [r"^SRT-", r"^SST-", r"^AAC\(6'\)-Ic$", r"^Fos[Aa][25678](?:\.\d+)?$", r"^tet\(41\)$"],
    # C. freundii: CMY/CFE cromossômica (origem do CMY-2-like); C. koseri: CKO
    "Citrobacter": [r"^CMY-", r"^CFE-", r"^CKO-"],
    "Morganella": [r"^DHA-", r"^MOR-"],
    "Providencia": [r"^AAC\(2'\)-I"],
    # tet(J) cromossômica em Proteus; fosA8 em Proteus (40% do recorte)
    "Proteus": [r"^tet\(J\)$", r"^Fos[Aa]8$"],
    "Yersinia": [r"^Yent_"],
    "Raoultella": [r"^ORN-", r"^PLA-"],
    "Kluyvera": [r"^KLU"],
}

# fosA3/fosA4 são as variantes plasmidiais (adquiridas).

# --------------------------------------------------------------------------
# 2. Famílias adquiridas (fora beta-lactamase), por classe de droga

# (regex em gene_clean, regex em amr_gene_family) -> classe de droga
_CLASSES_ADQUIRIDAS = [
    ("aminoglicosideo_metilase", None, r"16S rRNA methyltransferase"),
    ("aminoglicosideo_ame", None, r"^\{?['\"]?(?:AAC|APH|ANT)\("),
    ("polimixina_mcr", None, r"MCR phosphoethanolamine transferase"),
    ("quinolona_pmqr", r"^(?:Qnr|qnr|QepA|AAC\(6'\)-Ib-cr)", None),
    ("sulfonamida", None, r"sulfonamide resistant sul"),
    ("trimetoprim", None, r"trimethoprim resistant dihydrofolate reductase"),
    ("fenicol", r"^(?:catA1|catA2|catB\d*|cmlA\d*|floR|cmx)$", None),
    ("tetraciclina", r"^tet\((?:A|B|C|D|E|G|H|M|X|39|59)\)$", None),
    ("macrolideo", r"^(?:mph[A-Z]?|Mrx|msrE|ErmB|ErmC|Erm\(42\)|EreA\d?|EreB)$", None),
    ("rifamicina", r"^arr-\d", None),
    ("fosfomicina", r"^Fos[Aa][34]$", None),
]

# classes de droga que entram em `n_classes_adquiridas`
CLASSES_DROGA = {
    "beta_lactamico": ["bl_carbapenemase", "bl_esbl", "bl_ampc", "bl_estreito"],
    "aminoglicosideo": ["aminoglicosideo_metilase", "aminoglicosideo_ame"],
    "polimixina": ["polimixina_mcr"],
    "quinolona": ["quinolona_pmqr"],
    "sulfonamida": ["sulfonamida"],
    "trimetoprim": ["trimetoprim"],
    "fenicol": ["fenicol"],
    "tetraciclina": ["tetraciclina"],
    "macrolideo": ["macrolideo"],
    "rifamicina": ["rifamicina"],
    "fosfomicina": ["fosfomicina"],
}

# famílias RGI do tipo "OXA-X-like" (sem alelo) -> alelo protótipo no refgenes
_OXA_LIKE_PROTOTIPO = re.compile(r"^(OXA-\d+)-like$")


def _funcao_beta_lactamase(product_name: str) -> str | None:
    """product_name do refgenes -> categoria funcional (sem peso)."""
    if not isinstance(product_name, str):
        return None
    p = product_name.lower()
    if "carbapenem-hydrolyzing" in p:
        return "carbapenemase_serina"          # classe A (KPC, GES-5, BKC) ou D (OXA-23/24/48/58/143)
    if "metallo-beta-lactamase" in p:
        return "carbapenemase_mbl"             # classe B (NDM, IMP, VIM, SPM)
    if "extended-spectrum class c" in p or "cephalosporin-hydrolyzing class c" in p or "class c" in p:
        return "ampc"
    if "extended-spectrum" in p:
        return "esbl"
    if "beta-lactamase" in p:
        return "estreito"                      # TEM-1, OXA-1, LAP, SCO, CARB...
    return None


class CatalogoBetaLactamase:
    """
    Mapeia `gene_clean` do RGI (ex: "KPC-2", "OXA-23-like") para a função
    curada no refgenes.tsv (ex: "carbapenemase_serina").
    """

    def __init__(self, refgenes_path) -> None:
        ref = pd.read_csv(refgenes_path, sep="\t", usecols=["allele", "class", "product_name"])
        ref = ref[(ref["class"] == "BETA-LACTAM") & ref["allele"].notna()]
        self.funcao = {
            a.removeprefix("bla"): _funcao_beta_lactamase(p)
            for a, p in zip(ref["allele"], ref["product_name"])
        }

    def classificar(self, gene_clean: str) -> str | None:
        if not isinstance(gene_clean, str):
            return None
        m = _OXA_LIKE_PROTOTIPO.match(gene_clean)
        if m:
            gene_clean = m.group(1)            # "OXA-23-like" -> "OXA-23"
        return self.funcao.get(gene_clean)


def _regex_intrinseco(species: str) -> re.Pattern | None:
    if not isinstance(species, str):
        return None
    padroes = INTRINSECOS_POR_GENERO.get(species) or INTRINSECOS_POR_GENERO.get(species.split()[0])
    return re.compile("|".join(f"(?:{p})" for p in padroes)) if padroes else None


def anotar_resistoma_risco(resistome: pd.DataFrame, species: pd.Series, catalogo: CatalogoBetaLactamase) -> pd.DataFrame:
    """
    1 linha por (sample_id, gene_clean) ÚNICO -- hits repetidos do mesmo
    gene no mesmo genoma (assembly fragmentado) contam 1 vez -- com as
    colunas `categoria` (mecanismo adquirido, ou NaN se não entra) e
    `plasmidial` (algum hit daquele gene caiu em contig plasmidial).

        Parâmetros
        ----------
        resistome : pd.DataFrame
            resistome_recorte.csv (colunas sample_id, gene_clean,
            amr_gene_family, model_type, contig_type).
        species : pd.Series
            Species indexada por sample.
        catalogo : CatalogoBetaLactamase
    """
    r = resistome[resistome["model_type"] == "protein homolog model"].copy()
    r["plasmidial"] = r["contig_type"].eq("plasmid")
    r = (
        r.groupby(["sample_id", "gene_clean"], as_index=False)
        .agg(amr_gene_family=("amr_gene_family", "first"), plasmidial=("plasmidial", "any"))
    )
    r["Species"] = r["sample_id"].map(species)

    # função da beta-lactamase (só pra família com "beta-lactamase" no nome)
    eh_bl = r["amr_gene_family"].str.contains("beta-lactamase", case=False, na=False)
    r["funcao_bl"] = np.where(eh_bl, r["gene_clean"].map(catalogo.classificar), None)

    # categoria adquirida
    cat = pd.Series(pd.NA, index=r.index, dtype="object")
    mapa_bl = {
        "carbapenemase_serina": "bl_carbapenemase", "carbapenemase_mbl": "bl_carbapenemase",
        "esbl": "bl_esbl", "ampc": "bl_ampc", "estreito": "bl_estreito",
    }
    cat[eh_bl] = r.loc[eh_bl, "funcao_bl"].map(mapa_bl)
    for nome, rx_gene, rx_fam in _CLASSES_ADQUIRIDAS:
        m = pd.Series(True, index=r.index)
        if rx_gene:
            m &= r["gene_clean"].str.contains(rx_gene, regex=True, na=False)
        if rx_fam:
            m &= r["amr_gene_family"].str.contains(rx_fam, regex=True, na=False)
        # AAC(6')-Ib-cr também é AME -- quinolona_pmqr vem depois e sobrescreve
        cat[m & ~eh_bl] = nome
    r["categoria"] = cat

    # remove intrínsecos (por gênero/espécie). SHV com função ESBL/carbapenemase
    # em Klebsiella é mantida como adquirida (SHV-12, SHV-5, ... são plasmidiais)
    intr = pd.Series(False, index=r.index)
    for sp, idx in r.groupby("Species").groups.items():
        rx = _regex_intrinseco(sp)
        if rx is None:
            continue
        sub = r.loc[idx]
        m = sub["gene_clean"].str.contains(rx, na=False)
        shv_funcional = sub["gene_clean"].str.startswith("SHV-") & sub["funcao_bl"].isin(
            ["esbl", "carbapenemase_serina", "carbapenemase_mbl"]
        )
        intr.loc[idx] = m & ~shv_funcional
    r["intrinseco"] = intr
    r.loc[r["intrinseco"], "categoria"] = pd.NA
    return r


def build_features_risco(anotado: pd.DataFrame, index) -> pd.DataFrame:
    """
    Features (sem feature selection ainda):
        has_carbapenemase         - >=1 carbapenemase adquirida (A, B ou D)
        has_mbl                   - >=1 metalo-beta-lactamase (classe B)
        has_carbapenemase_serina  - >=1 carbapenemase de serina (KPC/GES/BKC/OXA-23/24/48/58/143)
        n_carbapenemases          - nº de carbapenemases distintas (coprodução)
        has_esbl                  - >=1 ESBL adquirida (CTX-M, SHV/TEM-ESBL, ...)
        has_ampc_adquirida        - >=1 AmpC fora do hospedeiro natural (CMY-2, DHA, ...)
        n_bl_adquiridas           - nº de beta-lactamases adquiridas distintas (todas)
        has_16s_metilase          - armA/rmt (pan-aminoglicosídeo)
        n_ame                     - nº de enzimas modificadoras de aminoglicosídeo adquiridas
        has_mcr                   - mcr (colistina)
        has_pmqr                  - qnr / aac(6')-Ib-cr / qepA
        n_classes_adquiridas      - nº de classes de droga com >=1 determinante adquirido
        n_genes_adquiridos        - nº total de genes adquiridos distintos
        pct_adquiridos_plasmidial - fração dos adquiridos em contig plasmidial
        carbapenemase_plasmidial  - >=1 carbapenemase em contig plasmidial
    """
    a = anotado[anotado["categoria"].notna()].copy()
    a["classe"] = a["categoria"].map(
        {c: k for k, v in CLASSES_DROGA.items() for c in v}
    )
    g = a.groupby("sample_id")

    def conta(mask):
        return a[mask].groupby("sample_id").size().reindex(index, fill_value=0)

    carb = a["categoria"] == "bl_carbapenemase"
    feats = pd.DataFrame(index=index)
    feats["has_carbapenemase"] = (conta(carb) > 0).astype(int)
    feats["has_mbl"] = (conta(a["funcao_bl"] == "carbapenemase_mbl") > 0).astype(int)
    feats["has_carbapenemase_serina"] = (conta(a["funcao_bl"] == "carbapenemase_serina") > 0).astype(int)
    feats["n_carbapenemases"] = conta(carb)
    feats["has_esbl"] = (conta(a["categoria"] == "bl_esbl") > 0).astype(int)
    feats["has_ampc_adquirida"] = (conta(a["categoria"] == "bl_ampc") > 0).astype(int)
    feats["n_bl_adquiridas"] = conta(a["classe"] == "beta_lactamico")
    feats["has_16s_metilase"] = (conta(a["categoria"] == "aminoglicosideo_metilase") > 0).astype(int)
    feats["n_ame"] = conta(a["categoria"] == "aminoglicosideo_ame")
    feats["has_mcr"] = (conta(a["categoria"] == "polimixina_mcr") > 0).astype(int)
    feats["has_pmqr"] = (conta(a["categoria"] == "quinolona_pmqr") > 0).astype(int)
    feats["n_classes_adquiridas"] = g["classe"].nunique().reindex(index, fill_value=0)
    feats["n_genes_adquiridos"] = g.size().reindex(index, fill_value=0)
    feats["pct_adquiridos_plasmidial"] = (
        g["plasmidial"].mean().reindex(index).fillna(0.0)
    )
    feats["carbapenemase_plasmidial"] = (conta(carb & a["plasmidial"]) > 0).astype(int)
    return feats
