from pathlib import Path

MAJOR_TISSUE_LIST  = ["brain", "kidney", "intestine", "heart", "lung", "blood", "liver", "breast"]
VERSION = "2024-07-01"
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HOMOLOGY_PATH = ROOT_DIR / "resources" / "homologous.csv"

ORGANISM_TO_CENSUS_KEY = {
    "Homo sapiens": "homo_sapiens",
    "Mus musculus": "mus_musculus",
}

ORGANISM_ALIASES = {
    "human": "Homo sapiens",
    "homo sapiens": "Homo sapiens",
    "homo_sapiens": "Homo sapiens",
    "mouse": "Mus musculus",
    "mus musculus": "Mus musculus",
    "mus_musculus": "Mus musculus",
}

CANCER_LIST_PATH = Path(__file__).resolve().with_name("cancer_list.txt")
with CANCER_LIST_PATH.open() as f:
    CANCER_LIST = [line.rstrip('\n') for line in f]

#  build the value filter dict for each tissue
VALUE_FILTER = {
    tissue : f"suspension_type != 'na' and disease == 'normal' and tissue_general == '{tissue}'" for tissue in MAJOR_TISSUE_LIST
}
# build the value filter dict for cells related with other tissues
# since tileDB does not support `not in ` operator, we will just use `!=` to filter out the other tissues
VALUE_FILTER["others"] = f"suspension_type != 'na' and disease == 'normal'"
for tissue in MAJOR_TISSUE_LIST:
    VALUE_FILTER["others"] = f"{VALUE_FILTER['others']} and (tissue_general != '{tissue}')"

VALUE_FILTER['pan-cancer'] = f"suspension_type != 'na'"
cancer_condition = ""
for disease in CANCER_LIST:
    if cancer_condition == "":
        cancer_condition = f"(disease == '{disease}')"
    else:
        cancer_condition = f"{cancer_condition} or (disease == '{disease}')"
VALUE_FILTER['pan-cancer'] = f"(suspension_type != 'na') and ({cancer_condition})"


def normalize_organism(name: str) -> str:
    key = str(name).strip().lower()
    if key not in ORGANISM_ALIASES:
        raise ValueError(
            f"Unsupported organism: {name!r}. "
            "Expected one of: Homo sapiens, Mus musculus, human, mouse."
        )
    return ORGANISM_ALIASES[key]


def organism_output_name(name: str) -> str:
    normalized = normalize_organism(name)
    return "human" if normalized == "Homo sapiens" else "mouse"

if __name__ == "__main__":
    # print(VALUE_FILTER["others"])
    # print(MAJOR_TISSUE_LIST)
    print(VALUE_FILTER['pan-cancer'])
