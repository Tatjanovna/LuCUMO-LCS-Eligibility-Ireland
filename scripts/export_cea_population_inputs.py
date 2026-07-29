#!/usr/bin/env python3
"""Build the deterministic Irish population and smoking input package for the CEA.

Smoking inputs are transferred only from the eligibility project's final processed
outputs. The exporter does not reopen or re-clean raw smoking microdata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
PROCESSED = ROOT / "data_processed"
OUT = ROOT / "export_for_cea"
MIN_AGE = 50
MAX_AGE = 79
PACKAGE_VERSION = "2.0.0"
AGE_GROUPS = tuple(f"{age}-{age + 4}" for age in range(MIN_AGE, MAX_AGE + 1, 5))
SMOKING_STATUSES = ("current", "former", "never")
STATUS_SOURCE_COLUMNS = {
    "current": "smokers",
    "former": "quitters",
    "never": "never_smokers",
}
HISTORY_VARIANTS = {
    "current": (("smokers", "smokers"),),
    "former": (
        ("quitters_all", "quitters_all"),
        ("quitters_excl_10", "quitters_excl_10"),
        ("quitters_excl_15", "quitters_excl_15"),
    ),
}
STALE_FILES = (
    "smoking_history_donors.csv",
    "smoking_history_cleaning_audit.csv",
    "smoking_history_cleaning_summary.json",
)


def require_sources() -> None:
    required = (
        RAW / "projections2057_raw.csv",
        PROCESSED / "projections2057.csv",
        PROCESSED / "cleaned_smoking_data_ag.csv",
        PROCESSED / "pack_year_dist_cleaned.csv",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required source file(s) absent: " + ", ".join(missing))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_source_age(value: str) -> int | None:
    if value == "Under 1 year":
        return 0
    match = re.fullmatch(r"(\d+) years?", value)
    return int(match.group(1)) if match else None


def five_year_group(age: int) -> str:
    lower = age // 5 * 5
    return f"{lower}-{lower + 4}"


def population_export() -> tuple[list[dict], list[dict]]:
    rows = []
    with (RAW / "projections2057_raw.csv").open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            age = parse_source_age(source["Age"])
            if source["Year"] == "2022" and age is not None and MIN_AGE <= age <= MAX_AGE:
                rows.append({
                    "year": 2022,
                    "age": age,
                    "sex": source["Sex"].lower(),
                    "population_count": int(float(source["VALUE"])),
                    "source_dataset": "CSO_population_projections_based_on_Census_2022",
                    "source_scenario": "CSO_Census2022_M2",
                })
    rows.sort(key=lambda row: (row["age"], row["sex"]))
    expected = {(age, sex) for age in range(MIN_AGE, MAX_AGE + 1) for sex in ("female", "male")}
    observed = {(row["age"], row["sex"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Exact age/sex coverage mismatch: missing={sorted(expected - observed)}")

    exact = defaultdict(int)
    for row in rows:
        exact[(five_year_group(row["age"]), row["sex"])] += row["population_count"]
    grouped = {}
    with (PROCESSED / "projections2057.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["year"] == "2022":
                grouped[(row["age_group"], row["gender"].lower())] = int(float(row["population"]))
    validation = []
    for key in sorted(exact):
        expected_total = grouped.get(key)
        discrepancy = exact[key] - expected_total if expected_total is not None else None
        validation.append({
            "age_group": key[0],
            "sex": key[1],
            "exact_age_total": exact[key],
            "grouped_total": expected_total,
            "discrepancy": discrepancy,
        })
        if discrepancy != 0:
            raise ValueError(f"Population validation discrepancy for {key}: {discrepancy}")
    return rows, validation


def smoking_status_export() -> list[dict]:
    """Export exactly current/former/never assignment probabilities.

    The processed eligibility output retains the original Census denominator, so
    its three reported probabilities leave a residual corresponding to not stated.
    Because the CEA must assign exactly one of three statuses, the exporter
    renormalises those three probabilities within each age-sex cell.
    """
    rows = []
    seen = set()
    with (PROCESSED / "cleaned_smoking_data_ag.csv").open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            age_group = source["age_group"].strip()
            if age_group not in AGE_GROUPS or source["gender"] not in ("Female", "Male"):
                continue
            sex = source["gender"].lower()
            source_probabilities = {
                status: float(source[column]) for status, column in STATUS_SOURCE_COLUMNS.items()
            }
            known_total = sum(source_probabilities.values())
            if not 0 < known_total <= 1:
                raise ValueError(f"Invalid processed smoking probabilities for {(age_group, sex)}")
            for status in SMOKING_STATUSES:
                rows.append({
                    "year": 2022,
                    "age_group": age_group,
                    "sex": sex,
                    "smoking_status": status,
                    "source_probability": f"{source_probabilities[status]:.12f}",
                    "assignment_probability": f"{source_probabilities[status] / known_total:.12f}",
                    "source_population_count": int(float(source["population"])),
                    "source_dataset": "data_processed/cleaned_smoking_data_ag.csv",
                    "probability_transformation": "renormalised_across_current_former_never",
                })
            seen.add((age_group, sex))
    expected = {(age_group, sex) for age_group in AGE_GROUPS for sex in ("female", "male")}
    if seen != expected:
        raise ValueError(f"Processed smoking-status coverage mismatch: missing={sorted(expected - seen)}")
    rows.sort(key=lambda row: (
        int(row["age_group"].split("-")[0]), row["sex"],
        SMOKING_STATUSES.index(row["smoking_status"]),
    ))
    return rows


def _history_values(source: dict, suffix: str) -> dict:
    mapping = {
        "mean_log_pack_years": f"mean_ln_pack_years_{suffix}",
        "sd_log_pack_years": f"std_ln_pack_years_{suffix}",
        "mean_pack_years": f"mean_pack_years_{suffix}",
        "sd_pack_years": f"std_pack_years_{suffix}",
    }
    return {target: source[column].strip() for target, column in mapping.items()}


def smoking_history_export() -> list[dict]:
    """Copy the final adjusted age-sex pack-year estimates unchanged."""
    rows = []
    seen = set()
    with (PROCESSED / "pack_year_dist_cleaned.csv").open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            age_group = source["age_group"].strip()
            if age_group not in AGE_GROUPS or source["gender"] not in ("Female", "Male"):
                continue
            sex = source["gender"].lower()
            for status, variants in HISTORY_VARIANTS.items():
                for history_variant, suffix in variants:
                    values = _history_values(source, suffix)
                    if any(value == "" for value in values.values()):
                        raise ValueError(
                            f"Missing final processed history estimate for "
                            f"{(age_group, sex, status, history_variant)}"
                        )
                    rows.append({
                        "age_group": age_group,
                        "sex": sex,
                        "smoking_status": status,
                        "history_variant": history_variant,
                        **values,
                        "source_dataset": "data_processed/pack_year_dist_cleaned.csv",
                        "source_adjustment_status": "final_processed_eligibility_estimate_copied_unchanged",
                    })
            rows.append({
                "age_group": age_group,
                "sex": sex,
                "smoking_status": "never",
                "history_variant": "structural_zero",
                "mean_log_pack_years": "",
                "sd_log_pack_years": "",
                "mean_pack_years": "0",
                "sd_pack_years": "0",
                "source_dataset": "model_rule",
                "source_adjustment_status": "never_smokers_have_zero_pack_years",
            })
            seen.add((age_group, sex))
    expected = {(age_group, sex) for age_group in AGE_GROUPS for sex in ("female", "male")}
    if seen != expected:
        raise ValueError(f"Processed smoking-history coverage mismatch: missing={sorted(expected - seen)}")
    status_order = {"current": 0, "former": 1, "never": 2}
    variant_order = {
        "smokers": 0,
        "quitters_all": 0,
        "quitters_excl_10": 1,
        "quitters_excl_15": 2,
        "structural_zero": 0,
    }
    rows.sort(key=lambda row: (
        int(row["age_group"].split("-")[0]), row["sex"],
        status_order[row["smoking_status"]], variant_order[row["history_variant"]],
    ))
    return rows


def generation_parameters() -> dict:
    return {
        "model_version": PACKAGE_VERSION,
        "population_year": 2022,
        "population_age_range": {"minimum": MIN_AGE, "maximum": MAX_AGE, "inclusive": True},
        "age_groups": list(AGE_GROUPS),
        "smoking_statuses": list(SMOKING_STATUSES),
        "status_assignment": {
            "source": "data_processed/cleaned_smoking_data_ag.csv",
            "method": "sample exactly one of current, former, or never using assignment_probability",
            "not_stated_handling": "renormalise the three processed probabilities within each age-sex cell",
            "daily_and_occasional_handling": "already combined as smokers in the processed eligibility output",
            "ever_smoker_category": False,
        },
        "history_assignment": {
            "source": "data_processed/pack_year_dist_cleaned.csv",
            "matching_keys": ["sex", "five_year_age_group", "smoking_status"],
            "pooling_or_fallback": None,
            "current_default_variant": "smokers",
            "former_default_variant": "quitters_all",
            "former_available_variants": [
                "quitters_all", "quitters_excl_10", "quitters_excl_15"
            ],
            "never_variant": "structural_zero",
            "sampling_distribution": "lognormal using mean_log_pack_years and sd_log_pack_years",
            "processed_values_changed_by_exporter": False,
            "instruction": (
                "Use the matching reported age-sex estimate exactly; fail rather than "
                "pool or substitute a different age or sex cell."
            ),
        },
        "scope_note": (
            "The processed eligibility output provides adjusted pack-year distributions, "
            "not respondent-level joint histories of initiation age, intensity, duration, "
            "or stopping age."
        ),
    }


def validation_targets(smoking_rows: list[dict], history_rows: list[dict]) -> list[dict]:
    output = []
    comparator_groups = {"55-59", "60-64", "65-69", "70-74"}
    for row in smoking_rows:
        if row["age_group"] in comparator_groups:
            output.append({
                "validation_population": "Irish_LHC_comparator_ages_55_74",
                "sex": row["sex"],
                "age_group": row["age_group"],
                "smoking_status": row["smoking_status"],
                "history_variant": "",
                "measure": "assignment_probability",
                "value": row["assignment_probability"],
                "source_dataset": row["source_dataset"],
            })
    default_variants = {
        "current": "smokers",
        "former": "quitters_all",
        "never": "structural_zero",
    }
    for row in history_rows:
        if row["age_group"] not in comparator_groups:
            continue
        if row["history_variant"] != default_variants[row["smoking_status"]]:
            continue
        for measure in (
            "mean_log_pack_years", "sd_log_pack_years",
            "mean_pack_years", "sd_pack_years",
        ):
            if row[measure] == "":
                continue
            output.append({
                "validation_population": "Irish_LHC_comparator_ages_55_74",
                "sex": row["sex"],
                "age_group": row["age_group"],
                "smoking_status": row["smoking_status"],
                "history_variant": row["history_variant"],
                "measure": measure,
                "value": row[measure],
                "source_dataset": row["source_dataset"],
            })
    return output


def git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def source_data_commit() -> str:
    return git_value(
        "log", "-1", "--format=%H", "--",
        "data_raw/projections2057_raw.csv",
        "data_processed/projections2057.csv",
        "data_processed/cleaned_smoking_data_ag.csv",
        "data_processed/pack_year_dist_cleaned.csv",
    )


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(export_details: dict, population_validation: list[dict]) -> dict:
    source_commit = source_data_commit()
    return {
        "package_version": PACKAGE_VERSION,
        "extraction_date": git_value("show", "-s", "--format=%cs", source_commit),
        "source_repository": "Tatjanovna/LuCUMO-LCS-Eligibility-Ireland",
        "source_commit_sha": source_commit,
        "population_year": 2022,
        "population_age_range": {"minimum": MIN_AGE, "maximum": MAX_AGE, "inclusive": True},
        "smoking_statuses": list(SMOKING_STATUSES),
        "category_mappings": {
            "smokers": "current",
            "quitters": "former",
            "never_smokers": "never",
        },
        "source_files": [
            "data_raw/projections2057_raw.csv",
            "data_processed/projections2057.csv",
            "data_processed/cleaned_smoking_data_ag.csv",
            "data_processed/pack_year_dist_cleaned.csv",
        ],
        "source_notebooks": [
            "code/1. cleaning_population_projections.ipynb",
            "code/2. cleaning_smoking_census2022.ipynb",
            "code/4. pack_year_distribution_baseline.ipynb",
        ],
        "transformations": [
            "Population restricted to exact ages 50-79 and male/female",
            "Processed smokers, quitters, and never_smokers mapped to current, former, and never",
            "Three processed smoking-status probabilities renormalised within each age-sex cell so every synthetic individual receives exactly one status",
            "Final processed age-sex pack-year estimates copied unchanged",
            "No raw smoking microdata read, re-cleaned, pooled, or re-estimated",
        ],
        "history_defaults": {
            "current": "smokers",
            "former": "quitters_all",
            "never": "structural_zero",
        },
        "limitations": [
            "The final processed eligibility output supplies adjusted pack-year distributions rather than respondent-level joint smoking histories",
            "The exporter does not provide initiation age, cigarettes per day, duration, stopping age, or years since quitting",
            "No age pooling, sex pooling, donor sampling, or fallback substitution is performed",
        ],
        "population_grouped_validation": population_validation,
        "exports": export_details,
        "checksum_note": (
            "source_metadata.json cannot contain its own stable checksum because that is "
            "self-referential; all other package files are checksummed."
        ),
    }


def readme() -> str:
    return f"""# CEA Irish population and smoking input export, version {PACKAGE_VERSION}

## Purpose

This package transfers the eligibility project's **final processed estimates** to the CEA. It covers exact ages **{MIN_AGE}-{MAX_AGE} inclusive**; age 80 is excluded.

Run from the repository root:

```bash
python scripts/export_cea_population_inputs.py
```

## Smoking status

Every synthetic person must receive exactly one of:

* `current`
* `former`
* `never`

The source is `data_processed/cleaned_smoking_data_ag.csv`. Its `smokers` estimate already combines daily and occasional smoking. The processed `smokers`, `quitters`, and `never_smokers` proportions use the original Census denominator and therefore leave a residual for not stated. The export renormalises these three values within each age-sex group so `assignment_probability` sums to one. No `ever_smoker`, daily/occasional, or not-stated model category is exported.

## Smoking histories

`smoking_history_strata.csv` copies the final age-sex pack-year estimates from `data_processed/pack_year_dist_cleaned.csv` without recalculation:

* current smokers: `smokers`
* former smokers: `quitters_all`, `quitters_excl_10`, and `quitters_excl_15`
* never smokers: structural zero pack-years

The default former-smoker history is `quitters_all`; the other reported variants are retained for eligibility definitions involving cessation windows. The CEA must use the exact matching sex and five-year age group. There is no age pooling, minimum-cell fallback, donor sampling, raw-data cleaning, or cross-sex substitution in this exporter.

These processed outputs provide adjusted **pack-year distributions**. They do not provide respondent-level linked values for initiation age, cigarettes per day, smoking duration, stopping age, or years since quitting.

## Files

* `irish_population_age_sex_2022.csv`: exact-age/sex population counts, ages 50-79.
* `irish_smoking_status_age_sex_2022.csv`: three-status source and assignment probabilities by age group and sex.
* `smoking_history_strata.csv`: final processed pack-year estimates by age group, sex, status, and history variant.
* `smoking_history_generation_parameters.json`: CEA assignment rules.
* `eligibility_model_validation_targets.csv`: age 55-74 status and default-history targets.
* `source_metadata.json`: provenance, transformations, limitations, row counts, and checksums.

The former raw-history cleaning audit files and donor-oriented generation files are no longer part of this package.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("export_for_cea"),
        help="repository-relative output directory (default: export_for_cea)",
    )
    return parser.parse_args()


def main() -> None:
    global OUT
    args = parse_args()
    if args.output_directory.is_absolute():
        raise ValueError("--output-directory must be repository-relative")
    OUT = (ROOT / args.output_directory).resolve()
    if ROOT.resolve() not in OUT.parents:
        raise ValueError("--output-directory must remain inside the repository")
    require_sources()
    OUT.mkdir(exist_ok=True)
    for filename in STALE_FILES:
        (OUT / filename).unlink(missing_ok=True)

    population, population_validation = population_export()
    smoking = smoking_status_export()
    histories = smoking_history_export()
    parameters = generation_parameters()
    validation = validation_targets(smoking, histories)

    write_csv(OUT / "irish_population_age_sex_2022.csv", list(population[0]), population)
    write_csv(OUT / "irish_smoking_status_age_sex_2022.csv", list(smoking[0]), smoking)
    write_csv(OUT / "smoking_history_strata.csv", list(histories[0]), histories)
    write_csv(OUT / "eligibility_model_validation_targets.csv", list(validation[0]), validation)
    (OUT / "smoking_history_generation_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "README.md").write_text(readme(), encoding="utf-8")

    source_map = {
        "irish_population_age_sex_2022.csv": [
            "data_raw/projections2057_raw.csv",
            "data_processed/projections2057.csv",
        ],
        "irish_smoking_status_age_sex_2022.csv": [
            "data_processed/cleaned_smoking_data_ag.csv"
        ],
        "smoking_history_strata.csv": [
            "data_processed/pack_year_dist_cleaned.csv"
        ],
        "smoking_history_generation_parameters.json": [
            "data_processed/cleaned_smoking_data_ag.csv",
            "data_processed/pack_year_dist_cleaned.csv",
        ],
        "eligibility_model_validation_targets.csv": [
            "data_processed/cleaned_smoking_data_ag.csv",
            "data_processed/pack_year_dist_cleaned.csv",
        ],
        "README.md": [],
    }
    details = {}
    for filename, sources in source_map.items():
        path = OUT / filename
        row_count = None
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
        details[filename] = {
            "original_source_files": sources,
            "row_count": row_count,
            "sha256": checksum(path),
        }
    (OUT / "source_metadata.json").write_text(
        json.dumps(metadata(details, population_validation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"CEA export {PACKAGE_VERSION}: {len(population)} population rows "
        f"for ages {MIN_AGE}-{MAX_AGE}"
    )
    print(
        f"Smoking status: {len(smoking)} rows; exactly current/former/never "
        "within each age-sex group"
    )
    print(
        f"Smoking histories: {len(histories)} processed age-sex-status/variant "
        "rows copied unchanged"
    )
    print(
        f"Validation: {len(population_validation)} population cells agree; "
        f"{len(validation)} targets"
    )
    print("Raw smoking microdata are not read; no age pooling or fallback is applied")


if __name__ == "__main__":
    main()
