#!/usr/bin/env python3
"""Build the deterministic Irish population and smoking input package for the CEA.

The CEA exporter reads only final processed eligibility outputs. Aggregate PLCO
smoking-history parameters are built separately by
``scripts/build_plco_smoking_history_outputs.py``; respondent rows are never
placed in the export package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
PROCESSED = ROOT / "data_processed"
OUT = ROOT / "export_for_cea"
MIN_AGE = 50
MAX_AGE = 79
PACKAGE_VERSION = "3.0.0"
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
PLCO_PROCESSED_FILES = (
    "plco_smoking_history_parameters.csv",
    "plco_smoking_history_correlations.csv",
    "plco_smoking_history_validation_targets.csv",
)
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
        *(PROCESSED / filename for filename in PLCO_PROCESSED_FILES),
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required source file(s) absent: "
            + ", ".join(missing)
            + ". Run scripts/build_plco_smoking_history_outputs.py before exporting."
        )


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
    """Export exactly current/former/never assignment probabilities."""
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


def validate_plco_processed_outputs() -> dict[str, int]:
    required_columns = {
        "plco_smoking_history_parameters.csv": {
            "age_group", "sex", "smoking_status", "history_variable", "n", "mean", "sd",
            "p05", "p25", "p50", "p75", "p95", "minimum", "maximum", "plco_role",
            "distribution_recommendation", "source_dataset", "source_processing",
        },
        "plco_smoking_history_correlations.csv": {
            "age_group", "sex", "smoking_status", "variable_1", "variable_2",
            "n_complete", "correlation", "correlation_estimable", "joint_generation_method",
            "source_dataset",
        },
        "plco_smoking_history_validation_targets.csv": {
            "validation_population", "age_group", "sex", "smoking_status",
            "history_variable", "statistic", "value", "n", "source_dataset",
        },
    }
    row_counts = {}
    for filename, columns in required_columns.items():
        with (PROCESSED / filename).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = columns - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Malformed {filename}: missing columns {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"Processed PLCO output is empty: {filename}")
        if set(row["age_group"] for row in rows) - set(AGE_GROUPS):
            raise ValueError(f"Processed PLCO output extends outside ages 50-79: {filename}")
        if set(row["sex"] for row in rows) - {"female", "male"}:
            raise ValueError(f"Processed PLCO output contains unsupported sex values: {filename}")
        row_counts[filename] = len(rows)
    return row_counts


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
        "adjusted_pack_year_assignment": {
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
        },
        "plcom2012_history_assignment": {
            "parameter_source": "data_processed/plco_smoking_history_parameters.csv",
            "correlation_source": "data_processed/plco_smoking_history_correlations.csv",
            "matching_keys": ["sex", "five_year_age_group", "smoking_status"],
            "pooling_or_fallback": None,
            "marginal_method": "empirical quantiles",
            "joint_method": "Gaussian copula using exported within-cell correlations",
            "current_predictors": [
                "cigarettes_per_day", "smoking_duration", "years_since_quitting"
            ],
            "former_predictors": [
                "cigarettes_per_day", "smoking_duration", "years_since_quitting"
            ],
            "chronology_variables": ["age_at_initiation", "age_at_stopping"],
            "current_years_since_quitting": 0,
            "constraints": [
                "age_at_initiation < attained_age",
                "current: smoking_duration = attained_age - age_at_initiation",
                "former: age_at_initiation < age_at_stopping <= attained_age",
                "former: smoking_duration = age_at_stopping - age_at_initiation",
                "former: years_since_quitting = attained_age - age_at_stopping",
                "cigarettes_per_day > 0",
            ],
            "invalid_draw_handling": "reject and redraw within the exact age-sex-status cell",
            "adjusted_history_note": (
                "effective_cigarettes_per_day and adjusted_pack_years reproduce the eligibility "
                "analysis but are validation variables, not PLCOm2012 predictor substitutions."
            ),
            "respondent_rows_exported": False,
        },
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
        *[f"data_processed/{filename}" for filename in PLCO_PROCESSED_FILES],
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
            *[f"data_processed/{filename}" for filename in PLCO_PROCESSED_FILES],
        ],
        "source_notebooks": [
            "code/1. cleaning_population_projections.ipynb",
            "code/2. cleaning_smoking_census2022.ipynb",
            "code/4. pack_year_distribution_baseline.ipynb",
        ],
        "source_processing_scripts": [
            "scripts/build_plco_smoking_history_outputs.py",
            "scripts/export_cea_population_inputs.py",
        ],
        "transformations": [
            "Population restricted to exact ages 50-79 and male/female",
            "Processed smokers, quitters, and never_smokers mapped to current, former, and never",
            "Three processed smoking-status probabilities renormalised within each age-sex cell",
            "Final processed adjusted pack-year estimates copied unchanged",
            "Aggregate PLCO smoking-history marginals and correlations copied unchanged",
            "No respondent rows, identifiers, age pooling, sex pooling, or CEA-side re-estimation",
        ],
        "history_defaults": {
            "current": "smokers",
            "former": "quitters_all",
            "never": "structural_zero",
        },
        "limitations": [
            "Age-sex-status cells reflect the modest Eurobarometer sample used by the eligibility analysis",
            "Some within-cell correlations are not estimable because a variable is constant or fewer than two complete observations are available",
            "The export provides aggregate marginals and correlations, not respondent-level linked histories",
            "No age or sex pooling is performed; the CEA must fail clearly if an exact cell is absent",
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

## Purpose and regeneration

This package transfers the eligibility project's final processed estimates to the CEA. It covers exact ages **{MIN_AGE}-{MAX_AGE} inclusive**; age 80 is excluded.

Run from the repository root:

```bash
python scripts/build_plco_smoking_history_outputs.py
python scripts/export_cea_population_inputs.py
```

The first command reproduces the final smoking-history transformations reported in `code/4. pack_year_distribution_baseline.ipynb` and writes aggregate, non-identifying PLCO-ready processed files. The second command copies only processed estimates into `export_for_cea`.

## Smoking status

Every synthetic person must receive exactly one of `current`, `former`, or `never`. The processed `smokers` category already combines daily and occasional smoking. The three probabilities are renormalised within each age-sex cell; no ever-smoker or not-stated model category is exported.

## PLCO-ready smoking histories

The package now includes age-sex-status-specific aggregate distributions for:

* age at smoking initiation;
* cigarettes per day;
* smoking duration;
* age at stopping for former smokers;
* years since quitting;
* standard and eligibility-adjusted pack-years.

`plco_smoking_history_correlations.csv` supplies the empirical within-cell correlations needed for coherent joint generation. The recommended method is a Gaussian copula with the exported empirical marginals, followed by chronology checks and redraws. The CEA must use the exact matching sex, five-year age group, and current/former status; there is no pooling or fallback.

`effective_cigarettes_per_day` and `adjusted_pack_years` reproduce the eligibility analysis. They are exported for eligibility calibration and validation and must not silently replace the published PLCOm2012 cigarettes-per-day predictor.

## Files

* `irish_population_age_sex_2022.csv`: exact-age/sex population counts, ages 50-79.
* `irish_smoking_status_age_sex_2022.csv`: three-status assignment probabilities.
* `smoking_history_strata.csv`: final adjusted pack-year estimates and cessation variants.
* `plco_smoking_history_parameters.csv`: empirical PLCO-variable marginals by exact age-sex-status cell.
* `plco_smoking_history_correlations.csv`: empirical within-cell correlations for joint generation.
* `plco_smoking_history_validation_targets.csv`: means, SDs, and medians for CEA validation.
* `smoking_history_generation_parameters.json`: assignment and joint-generation rules.
* `eligibility_model_validation_targets.csv`: Irish LHC status and adjusted pack-year targets.
* `source_metadata.json`: provenance, limitations, row counts, and checksums.

No respondent-level smoking histories or identifiers are exported.
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
    plco_counts = validate_plco_processed_outputs()
    parameters = generation_parameters()
    validation = validation_targets(smoking, histories)

    write_csv(OUT / "irish_population_age_sex_2022.csv", list(population[0]), population)
    write_csv(OUT / "irish_smoking_status_age_sex_2022.csv", list(smoking[0]), smoking)
    write_csv(OUT / "smoking_history_strata.csv", list(histories[0]), histories)
    write_csv(OUT / "eligibility_model_validation_targets.csv", list(validation[0]), validation)
    for filename in PLCO_PROCESSED_FILES:
        shutil.copyfile(PROCESSED / filename, OUT / filename)
    (OUT / "smoking_history_generation_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "README.md").write_text(readme(), encoding="utf-8")

    source_map = {
        "irish_population_age_sex_2022.csv": [
            "data_raw/projections2057_raw.csv", "data_processed/projections2057.csv"
        ],
        "irish_smoking_status_age_sex_2022.csv": [
            "data_processed/cleaned_smoking_data_ag.csv"
        ],
        "smoking_history_strata.csv": ["data_processed/pack_year_dist_cleaned.csv"],
        "plco_smoking_history_parameters.csv": [
            "data_processed/plco_smoking_history_parameters.csv"
        ],
        "plco_smoking_history_correlations.csv": [
            "data_processed/plco_smoking_history_correlations.csv"
        ],
        "plco_smoking_history_validation_targets.csv": [
            "data_processed/plco_smoking_history_validation_targets.csv"
        ],
        "smoking_history_generation_parameters.json": [
            "data_processed/cleaned_smoking_data_ag.csv",
            "data_processed/pack_year_dist_cleaned.csv",
            *[f"data_processed/{filename}" for filename in PLCO_PROCESSED_FILES],
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

    print(f"CEA export {PACKAGE_VERSION}: {len(population)} population rows for ages {MIN_AGE}-{MAX_AGE}")
    print(f"Smoking status: {len(smoking)} rows; exactly current/former/never")
    print(f"Adjusted pack-year histories: {len(histories)} age-sex-status/variant rows")
    print("PLCO processed rows: " + ", ".join(f"{name}={count}" for name, count in plco_counts.items()))
    print(f"Validation: {len(population_validation)} population cells agree; {len(validation)} LHC targets")
    print("No respondent smoking histories are exported; no age or sex pooling is applied")


if __name__ == "__main__":
    main()
