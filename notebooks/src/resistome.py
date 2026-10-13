"""
src/resistome.py

Columns added by `annotate_resistome`:

    level              isolate | bin | scaffolds | unbinned
    group              Isolate | MAF1 | VAN2 | avian | ...
    compartment        Clinical isolates | MAF | VAN | VBN | avian | swine
    bin_id             metagenome hit located in a MAG (scaffolds/unbinned rows
                       are mapped through the contig name)
    host_species/genus isolate species, or the bin's GTDB taxonomy
    drug_category      one drug category per hit (see DRUG_CATEGORIES)
    high_confidence    protein homolog model, identity >= 80%, ref. coverage >= 80%
    bl_function        beta-lactamase function from the NCBI Reference Gene
                       Catalog (same catalog as ml_isolados_brasileiros)
    acquired_category  resistance mechanism from a transferable gene family
    intrinsic          gene is intrinsic to the host genus/species (NA when the
                       host is unknown)
    origin             intrinsic | acquired | intrinsic-type (host unknown) |
                       core (efflux/regulator/target) | variant/overexpression model |
                       beta-lactamase (unclassified)
    contig_length, contig_coverage, coverage_percentile
                       metagenome contigs only (SPAdes header)
    card_plasmid_share plasmid vs chromosome context of the gene in NCBI genomes
                       (0 = always chromosomal, 1 = always plasmid),
                       from CARD Resistomes & Variants (`prevalence.parquet`)

The intrinsic-gene lists and the beta-lactamase catalog come from
`ml_isolados_brasileiros/notebooks/src/features_risco.py`. The lists here
(`EXTRA_INTRINSIC`, `EXTRA_ACQUIRED`) only extend them to species and genera
that the clinical-Critical analysis did not cover (P. aeruginosa, S. aureus,
C. difficile, Gram-positives and the environmental genera of the MAGs).
"""

import ast
import re

import numpy as np
import pandas as pd

from .paths import PARQUET, REFGENES
from .features_risco import _CLASSES_ADQUIRIDAS, INTRINSECOS_POR_GENERO, CatalogoBetaLactamase

# --------------------------------------------------------------------------
# 1. Drug category (one per hit, used for stacked bars)

DRUG_CATEGORIES = [
    ("beta-lactam", {"penicillin beta-lactam", "cephalosporin", "carbapenem", "monobactam", "cephamycin", "penem"}),
    ("aminoglycoside", {"aminoglycoside antibiotic"}),
    ("tetracycline", {"tetracycline antibiotic", "glycylcycline"}),
    ("glycopeptide (vancomycin)", {"glycopeptide antibiotic"}),
    ("MLS (macrolide/lincosamide/streptogramin)", {"macrolide antibiotic", "lincosamide antibiotic",
                                                  "streptogramin antibiotic", "streptogramin A antibiotic",
                                                  "streptogramin B antibiotic", "pleuromutilin antibiotic"}),
    ("quinolone", {"fluoroquinolone antibiotic"}),
    ("sulfonamide", {"sulfonamide antibiotic"}),
    ("trimethoprim", {"diaminopyrimidine antibiotic"}),
    ("phenicol", {"phenicol antibiotic"}),
    ("polymyxin/peptide", {"peptide antibiotic"}),
    ("fosfomycin", {"phosphonic acid antibiotic"}),
    ("rifamycin", {"rifamycin antibiotic"}),
    ("antiseptic", {"disinfecting agents and antiseptics"}),
]
DRUG_CATEGORY_ORDER = [c for c, _ in DRUG_CATEGORIES] + ["multidrug efflux", "other"]


def drug_category(drug_classes, mechanisms) -> str:
    """Efflux pump with >= 2 classes -> 'multidrug efflux'; otherwise the first match in DRUG_CATEGORIES."""
    classes = set(drug_classes)
    if "antibiotic efflux" in mechanisms and len(classes) >= 2:
        return "multidrug efflux"
    for name, members in DRUG_CATEGORIES:
        if classes & members:
            return name
    return "other"


# --------------------------------------------------------------------------
# 2. Intrinsic genes: ml_isolados_brasileiros lists + extensions

# key = genus or species, value = regexes on `gene_clean`
EXTRA_INTRINSIC = {
    # OXA-50 family (blaOXA-50/PoxB) and PDC (chromosomal AmpC) are present in every
    # P. aeruginosa; aph(3')-IIb, catB7 and fosA are chromosomal (100% of PR isolates)
    "Pseudomonas aeruginosa": [r"^OXA-50-like$", r"^PDC-", r"^APH\(3'\)-IIb$", r"^Paer_catB7$", r"^FosA$"],
    # tet(38) and fosB are chromosomal in S. aureus (mecA is acquired, SCCmec)
    "Staphylococcus aureus": [r"^tet\(38\)$", r"^Saur_FosB$"],
    # CDD-1/2: chromosomal class D beta-lactamase of C. difficile
    "Clostridioides": [r"^CDD-"],
    # aac(6')-Ii and msrC: chromosomal in E. faecium
    "Enterococcus faecium": [r"^AAC\(6'\)-Ii$", r"^msrC$"],
    # catA4 was described from the P. mirabilis chromosome (100% of PR isolates)
    "Proteus": [r"^catA4$"],
    # aac(6')-Iaa: cryptic chromosomal gene of S. enterica
    "Salmonella": [r"^AAC\(6'\)-Iaa$"],
    # L1 (MBL) and L2 (class A) beta-lactamases, aph(3')-IIc, aac(6')-Iz
    "Stenotrophomonas": [r"^L1", r"^L2", r"^APH\(3'\)-IIc$", r"^AAC\(6'\)-Iz$"],
    # OXA-114/OXA-243 family: chromosomal class D of Achromobacter
    "Achromobacter": [r"^OXA-243-like$", r"^OXA-114"],
    # Brucella (GTDB) includes the former Ochrobactrum; OXA-919-like is chromosomal
    "Brucella": [r"^OXA-919-like$"],
    # BJP-1: chromosomal subclass B3 MBL described in Bradyrhizobium (same family as Tardiphaga)
    "Tardiphaga": [r"^BJP-"],
    "Bradyrhizobium": [r"^BJP-"],
    # PenA: chromosomal class A of Burkholderia
    "Burkholderia": [r"^Pen[AB]-"],
}


def _intrinsic_regex(species, genus) -> re.Pattern | None:
    patterns = []
    if isinstance(species, str):
        patterns += INTRINSECOS_POR_GENERO.get(species, []) + EXTRA_INTRINSIC.get(species, [])
    if isinstance(genus, str) and not (isinstance(species, str) and species in INTRINSECOS_POR_GENERO):
        patterns += INTRINSECOS_POR_GENERO.get(genus, [])
    if isinstance(genus, str):
        patterns += EXTRA_INTRINSIC.get(genus, [])
    return re.compile("|".join(f"(?:{p})" for p in patterns)) if patterns else None


_ANY_INTRINSIC = re.compile("|".join(
    f"(?:{p})" for d in (INTRINSECOS_POR_GENERO, EXTRA_INTRINSIC) for ps in d.values() for p in ps
))


def intrinsic_hosts(gene_clean: str) -> str:
    """Genera/species in which the gene is listed as intrinsic ('' if none)."""
    hosts = [h for d in (INTRINSECOS_POR_GENERO, EXTRA_INTRINSIC) for h, ps in d.items()
             if any(re.search(p, gene_clean) for p in ps)]
    return ", ".join(sorted(set(hosts)))


# --------------------------------------------------------------------------
# 3. Acquired mechanisms: ml_isolados_brasileiros categories (English) + extensions

ACQUIRED_LABEL = {
    "bl_carbapenemase": "beta-lactam: carbapenemase",
    "bl_esbl": "beta-lactam: ESBL",
    "bl_ampc": "beta-lactam: AmpC",
    "bl_estreito": "beta-lactam: narrow-spectrum",
    "aminoglicosideo_metilase": "aminoglycoside: 16S rRNA methyltransferase",
    "aminoglicosideo_ame": "aminoglycoside: modifying enzyme",
    "polimixina_mcr": "polymyxin: mcr",
    "quinolona_pmqr": "quinolone: PMQR",
    "sulfonamida": "sulfonamide",
    "trimetoprim": "trimethoprim",
    "fenicol": "phenicol",
    "tetraciclina": "tetracycline",
    "macrolideo": "MLS",
    "rifamicina": "rifamycin",
    "fosfomicina": "fosfomycin",
}

BL_FUNCTION_LABEL = {
    "carbapenemase_serina": "carbapenemase (serine)",
    "carbapenemase_mbl": "carbapenemase (MBL)",
    "esbl": "ESBL",
    "ampc": "AmpC",
    "estreito": "narrow-spectrum",
}
_BL_TO_ACQUIRED = {"carbapenemase_serina": "bl_carbapenemase", "carbapenemase_mbl": "bl_carbapenemase",
                   "esbl": "bl_esbl", "ampc": "bl_ampc", "estreito": "bl_estreito"}

# Transposon/plasmid families common in Gram-positives and animal metagenomes,
# absent from the Gram-negative clinical list. Applied only where the ml
# list gave no category (regex on gene_clean).
EXTRA_ACQUIRED = [
    ("beta-lactam: PBP2a (mecA/mecC)", r"^mec[AC]$"),
    ("MLS", r"^(?:Erm[A-Z]{1,2}|Erm\(\d+\)|msr[AD]|mef[AE]|lnu[A-G]|lsa[CE]|vga[ABE]|vat[A-E]|mphC)$"),
    ("tetracycline", r"^tet\((?:K|L|O|S|W|Q|T|32|33|36|40|44)\)$|^tet[AB]\(P\)$"),
    ("phenicol", r"^(?:cmx|fexA|fexB|optrA|cfr(?:\(\w\))?|catA\d+|catB\d+|catP|catQ|catS)$"),
    ("aminoglycoside: modifying enzyme", r"^(?:aad\(6\)|aadA\d*|aadD|aadE|ANT\(\d)"),
    ("streptothricin", r"^SAT-\d$"),
    ("sulfonamide", r"^sul[1-4]$"),
    ("trimethoprim", r"^dfr[A-Z]\d*$"),
    ("antiseptic: qacE/qacEdelta1 (class 1 integron)", r"^qacE(?:delta1)?$"),
]


EXTRA_INTRINSIC.update({
    # V. cholerae: MBL cromossômica varG e catB9 cromossômica
    "Vibrio cholerae": [r"^varG$", r"^catB9$"],
    # E. faecalis: ABC-F cromossômico lsa(A) (resistência intrínseca a clindamicina/estreptogramina)
    "Enterococcus faecalis": [r"^lsa\(A\)$"],
    # L. monocytogenes: fosfomicina transferase cromossômica
    "Listeria": [r"^[Ff]os[Xx]$"],
})

EXTRA_INTRINSIC["Enterococcus faecium"] += [r"^eat\(?A\)?v$"]          # ABC-F cromossômico
EXTRA_INTRINSIC.update({
    "Aeromonas": [r"^cphA", r"^AQU-", r"^CepH", r"^CAV-"],             # MBL/AmpC cromossômicas (variam por espécie)
    "Elizabethkingia": [r"^BlaB", r"^GOB-", r"^CME-"],                  # MBL B1/B3 e classe A cromossômicas
    "Vibrio parahaemolyticus": [r"^CARB-"],                             # classe A cromossômica
})


def _acquired_category(gene: str, families: list, is_bl: bool, bl_function_raw) -> str | None:
    gene = gene if isinstance(gene, str) else ""
    if is_bl:
        cat = _BL_TO_ACQUIRED.get(bl_function_raw)
        return ACQUIRED_LABEL.get(cat)
    cat = None
    for name, rx_gene, rx_fam in _CLASSES_ADQUIRIDAS:              # same order as ml: later wins
        ok = True
        if rx_gene:
            ok &= bool(re.search(rx_gene, gene))
        if rx_fam:
            ok &= any(re.search(rx_fam, f) for f in families)
        if ok:
            cat = name
    if cat:
        return ACQUIRED_LABEL[cat]
    for name, rx in EXTRA_ACQUIRED:
        if re.search(rx, gene):
            return name
    return None


# --------------------------------------------------------------------------
# 4. Loading


def load_metadata() -> pd.DataFrame:
    meta = pd.read_parquet(PARQUET / "metadata.parquet")
    meta["level"] = meta["Type"].map({"Isolate": "isolate", "Bin": "bin", "Scaffolds": "scaffolds", "Unbinned": "unbinned"})
    meta["environment"] = meta["Group"].str.replace(r"\d$", "", regex=True)       # MAF1 -> MAF
    meta["compartment"] = np.where(meta["level"] == "isolate", "Clinical isolates", meta["environment"])
    gtdb_genus = meta["gtdb_genus"].str.replace(r"_[A-Z]+$", "", regex=True)
    gtdb_species = meta["gtdb_species"].str.replace(r"_[A-Z]+(?= |$)", "", regex=True)
    meta["host_species"] = np.where(meta["level"] == "isolate", meta["Species"], gtdb_species)
    meta["host_genus"] = np.where(meta["level"] == "isolate", meta["Species"].str.split().str[0], gtdb_genus)
    return meta


def card_prevalence() -> pd.DataFrame:
    """
    Per ARO, from CARD Resistomes & Variants: how often the gene was found on
    plasmids vs chromosomes in NCBI genomes of CARD's pathogen list.

    card_plasmid_share = sum(% plasmid) / (sum(% plasmid) + sum(% chromosome)),
    summed over pathogens. Indicative only: NCBI "chromosome" includes
    integrated copies (Tn/integrons), so acquired genes often score < 0.5.
    """
    p = pd.read_parquet(PARQUET / "prevalence.parquet")
    p = p.groupby(["aro_accession", "pathogen"])[["ncbi_plasmid", "ncbi_chromosome"]].max()
    tot = p.groupby(level=0).sum()
    share = tot["ncbi_plasmid"] / (tot["ncbi_plasmid"] + tot["ncbi_chromosome"])
    out = pd.DataFrame({
        "card_n_pathogens": p.groupby(level=0).size(),
        "card_plasmid_share": share.round(3),
    })
    return out


def _metagenome_contigs() -> pd.DataFrame:
    c = pd.read_parquet(PARQUET / "contigs.parquet", columns=["seq_name", "sample_id"])
    c = c[~c["sample_id"].str.startswith(("GCA_", "GCF_"))].copy()
    c["group"] = c["sample_id"].str.split("_").str[0]
    return c


def parse_list_column(s: pd.Series) -> pd.Series:
    """CSV exports store CARD lists as Python literals ("{'a', \"b''\"}"), parquet as arrays."""
    def parse(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        if isinstance(v, str):
            return sorted(ast.literal_eval(v)) if v.strip() not in ("", "set()") else []
        return list(v)
    return s.apply(parse)


def load_resistome() -> pd.DataFrame:
    res = pd.read_parquet(PARQUET / "resistome.parquet")
    res = res.rename(columns={"Percentage Length of Reference Sequence": "ref_coverage"})
    for col in ["drug_class", "mechanism", "amr_gene_family", "antibiotic"]:
        res[col] = parse_list_column(res[col])
    return res


def annotate_resistome(res: pd.DataFrame | None = None, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    res = load_resistome() if res is None else res.copy()
    meta = load_metadata() if meta is None else meta
    res = res.join(meta[["level", "Group", "environment", "compartment", "host_species", "host_genus"]]
                   .rename(columns={"Group": "group"}), on="sample_id")

    # metagenome hits: locate the contig in a bin (scaffolds/unbinned rows inherit the bin's taxonomy)
    contigs = _metagenome_contigs()
    in_bin = contigs[contigs["sample_id"].str.contains("_bin_")].set_index(["group", "seq_name"])["sample_id"]
    in_bin = in_bin[~in_bin.index.duplicated()]
    is_meta = res["level"] != "isolate"
    key = pd.MultiIndex.from_arrays([res.loc[is_meta, "group"], res.loc[is_meta, "seq_name"]])
    res["bin_id"] = pd.Series(pd.NA, index=res.index, dtype="object")
    res.loc[is_meta, "bin_id"] = in_bin.reindex(key).to_numpy()
    # geNomad ran on bins/unbinned only: scaffolds rows take the contig_type of the same contig
    located = res[res["level"].isin(["bin", "unbinned"])].drop_duplicates(["group", "seq_name"])
    ctype = located.set_index(["group", "seq_name"])["contig_type"]
    is_scaf = res["level"] == "scaffolds"
    key_s = pd.MultiIndex.from_arrays([res.loc[is_scaf, "group"], res.loc[is_scaf, "seq_name"]])
    res.loc[is_scaf, "contig_type"] = ctype.reindex(key_s).to_numpy()

    bin_tax = meta.loc[meta["level"] == "bin", ["host_species", "host_genus"]]
    for col in ["host_species", "host_genus"]:
        res.loc[is_meta, col] = res.loc[is_meta, "bin_id"].map(bin_tax[col]).to_numpy()

    # contig length/coverage from the SPAdes header + percentile of coverage in the sample
    hdr = res["seq_name"].str.extract(r"length_(?P<contig_length>\d+)_cov_(?P<contig_coverage>[\d.]+)")
    res["contig_length"] = pd.to_numeric(hdr["contig_length"])
    res["contig_coverage"] = pd.to_numeric(hdr["contig_coverage"])
    all_cov = contigs.drop_duplicates(["group", "seq_name"]).assign(
        cov=lambda d: pd.to_numeric(d["seq_name"].str.extract(r"cov_([\d.]+)")[0]))
    cov_by_group = {g: np.sort(s.dropna().to_numpy()) for g, s in all_cov.groupby("group")["cov"]}
    res["coverage_percentile"] = [
        100 * np.searchsorted(cov_by_group[g], c) / len(cov_by_group[g]) if g in cov_by_group and pd.notna(c) else np.nan
        for g, c in zip(res["group"], res["contig_coverage"])
    ]

    res = add_mechanisms(res)
    res = res.join(card_prevalence(), on="aro_accession")
    return res


def add_mechanisms(res: pd.DataFrame) -> pd.DataFrame:
    """
    drug_category, high_confidence, bl_function, acquired_category, intrinsic,
    intrinsic_in and origin. Needs `gene_clean`, `model_type`, `Best_Identities`,
    `ref_coverage`, list columns `drug_class`/`mechanism`/`amr_gene_family` and
    the host columns `host_species`/`host_genus` (NA = unknown host).
    """
    res = res.copy()
    # drug category + confidence filter
    res["drug_category"] = [drug_category(d, m) for d, m in zip(res["drug_class"], res["mechanism"])]
    res["high_confidence"] = (
        (res["model_type"] == "protein homolog model") & (res["Best_Identities"] >= 80) & (res["ref_coverage"] >= 80)
    )

    # beta-lactamase function (NCBI catalog) and acquired mechanism
    catalog = CatalogoBetaLactamase(REFGENES)
    is_bl = res["amr_gene_family"].apply(lambda fams: any("beta-lactamase" in f.lower() for f in fams))
    bl_raw = np.where(is_bl, res["gene_clean"].map(catalog.classificar), None)
    res["bl_function"] = pd.Series(bl_raw, index=res.index).map(BL_FUNCTION_LABEL)
    res["acquired_category"] = [
        _acquired_category(g, f, b, r) for g, f, b, r in zip(res["gene_clean"], res["amr_gene_family"], is_bl, bl_raw)
    ]

    # intrinsic (needs a known host); SHV with ESBL/carbapenemase function is never intrinsic
    intrinsic = pd.Series(pd.NA, index=res.index, dtype="boolean")
    for (sp, gen), idx in res.groupby(["host_species", "host_genus"], dropna=False).groups.items():
        if not isinstance(gen, str):
            continue
        rx = _intrinsic_regex(sp, gen)
        sub = res.loc[idx]
        m = sub["gene_clean"].str.contains(rx, na=False) if rx else pd.Series(False, index=idx)
        functional_shv = sub["gene_clean"].str.startswith("SHV-") & sub["bl_function"].isin(
            ["ESBL", "carbapenemase (serine)", "carbapenemase (MBL)"])
        intrinsic.loc[idx] = (m & ~functional_shv).to_numpy()
    res["intrinsic"] = intrinsic
    res["intrinsic_in"] = res["gene_clean"].map(lambda g: intrinsic_hosts(g) if isinstance(g, str) else "")

    host_known = res["host_genus"].notna()
    looks_intrinsic = res["gene_clean"].str.contains(_ANY_INTRINSIC, na=False)
    homolog = res["model_type"] == "protein homolog model"
    res["origin"] = np.select(
        [~homolog,
         host_known & res["intrinsic"].fillna(False).astype(bool),
         ~host_known & looks_intrinsic & (res["acquired_category"].notna() | is_bl),
         res["acquired_category"].notna(),
         is_bl],
        ["variant/overexpression model", "intrinsic", "intrinsic-type (host unknown)", "acquired",
         "beta-lactamase (unclassified)"],
        "core (efflux/regulator/target)")
    return res
