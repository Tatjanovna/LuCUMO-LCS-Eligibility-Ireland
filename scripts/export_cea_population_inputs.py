#!/usr/bin/env python3
"""Build the lean Irish population and smoking-history package used by the CEA."""

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
VERSION = "4.1.0"
MIN_AGE, MAX_AGE = 50, 79
AGE_GROUPS = tuple(f"{age}-{age + 4}" for age in range(50, 80, 5))
SEXES = ("female", "male")
STATUSES = ("current", "former", "never")
STATUS_COLUMNS = {
    "current": "smokers",
    "former": "quitters",
    "never": "never_smokers",
}
POOL_FILE = "plco_smoking_history_synthetic_pool.csv"
PARAMETERS_FILE = "smoking_history_generation_parameters.json"
EXPECTED_FILES = {
    "README.md",
    "irish_population_age_sex_2022.csv",
    "irish_smoking_status_age_sex_2022.csv",
    POOL_FILE,
    PARAMETERS_FILE,
    "source_metadata.json",
}


def require_sources():
    required = (
        RAW / "projections2057_raw.csv",
        PROCESSED / "projections2057.csv",
        PROCESSED / "cleaned_smoking_data_ag.csv",
        PROCESSED / POOL_FILE,
        PROCESSED / PARAMETERS_FILE,
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source files: "
            + ", ".join(missing)
            + ". Run scripts/build_smooth_smoking_history_outputs.py first."
        )


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_age(value):
    if value == "Under 1 year":
        return 0
    match = re.fullmatch(r"(\d+) years?", value)
    return int(match.group(1)) if match else None


def age_group(age):
    lower = age // 5 * 5
    return f"{lower}-{lower + 4}"


def population_export():
    rows = []
    with (RAW / "projections2057_raw.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for source in csv.DictReader(handle):
            age = parse_age(source["Age"])
            if (
                source["Year"] == "2022"
                and age is not None
                and MIN_AGE <= age <= MAX_AGE
            ):
                rows.append(
                    {
                        "year": 2022,
                        "age": age,
                        "sex": source["Sex"].lower(),
                        "population_count": int(float(source["VALUE"])),
                        "source_dataset": (
                            "CSO_population_projections_based_on_Census_2022"
                        ),
                        "source_scenario": "CSO_Census2022_M2",
                    }
                )
    rows.sort(key=lambda row: (row["age"], row["sex"]))
    expected = {(age, sex) for age in range(50, 80) for sex in SEXES}
    observed = {(row["age"], row["sex"]) for row in rows}
    if observed != expected:
        raise ValueError(
            f"Exact age-sex coverage mismatch: missing={sorted(expected - observed)}"
        )

    exact = defaultdict(int)
    for row in rows:
        exact[(age_group(row["age"]), row["sex"])] += row["population_count"]
    grouped = {}
    with (PROCESSED / "projections2057.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["year"] == "2022":
                grouped[(row["age_group"], row["gender"].lower())] = int(
                    float(row["population"])
                )
    validation = []
    for key in sorted(exact):
        grouped_total = grouped.get(key)
        discrepancy = exact[key] - grouped_total if grouped_total is not None else None
        validation.append(
            {
                "age_group": key[0],
                "sex": key[1],
                "exact_age_total": exact[key],
                "grouped_total": grouped_total,
                "discrepancy": discrepancy,
            }
        )
        if discrepancy != 0:
            raise ValueError(f"Population validation discrepancy for {key}: {discrepancy}")
    return rows, validation


def smoking_status_export():
    rows, seen = [], set()
    with (PROCESSED / "cleaned_smoking_data_ag.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for source in csv.DictReader(handle):
            group = source["age_group"].strip()
            gender = source["gender"].strip()
            if group not in AGE_GROUPS or gender not in ("Female", "Male"):
                continue
            sex = gender.lower()
            probabilities = {
                status: float(source[column])
                for status, column in STATUS_COLUMNS.items()
            }
            known_total = sum(probabilities.values())
            if not 0 < known_total <= 1:
                raise ValueError(f"Invalid smoking probabilities for {(group, sex)}")
            for status in STATUSES:
                rows.append(
                    {
                        "year": 2022,
                        "age_group": group,
                        "sex": sex,
                        "smoking_status": status,
                        "source_probability": f"{probabilities[status]:.12f}",
                        "assignment_probability": (
                            f"{probabilities[status] / known_total:.12f}"
                        ),
                        "source_population_count": int(float(source["population"])),
                        "source_dataset": (
                            "data_processed/cleaned_smoking_data_ag.csv"
                        ),
                        "probability_transformation": (
                            "renormalised_across_current_former_never"
                        ),
                    }
                )
            seen.add((group, sex))
    expected = {(group, sex) for group in AGE_GROUPS for sex in SEXES}
    if seen != expected:
        raise ValueError(f"Smoking-status coverage mismatch: {sorted(expected - seen)}")
    rows.sort(
        key=lambda row: (
            int(row["age_group"].split("-")[0]),
            row["sex"],
            STATUSES.index(row["smoking_status"]),
        )
    )
    return rows


def load_parameters():
    parameters = json.loads(
        (PROCESSED / PARAMETERS_FILE).read_text(encoding="utf-8")
    )
    if parameters.get("model_family") != (
        "non_bayesian_robust_penalised_spline_location_scale"
    ):
        raise ValueError("Unexpected smoking-history model family")
    assignment = parameters.get("conditional_history_assignment", {})
    if assignment.get("source") != POOL_FILE:
        raise ValueError("Generation parameters do not designate the synthetic pool")
    if assignment.get("matching_keys") != [
        "attained_age",
        "sex",
        "smoking_status",
    ]:
        raise ValueError("Unexpected synthetic-pool matching keys")
    return parameters


def validate_pool(parameters):
    required = {
        "attained_age",
        "age_group",
        "sex",
        "smoking_status",
        "synthetic_draw",
        "age_at_initiation",
        "age_at_stopping",
        "cigarettes_per_day",
        "smoking_duration",
        "years_since_quitting",
        "standard_pack_years",
        "adjusted_pack_years",
        "residual_donor_source",
        "source_dataset",
        "source_processing",
    }
    counts = defaultdict(int)
    row_count = 0
    with (PROCESSED / POOL_FILE).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Synthetic pool is missing columns: {sorted(missing)}")
        for row in reader:
            row_count += 1
            age = int(row["attained_age"])
            cell = (age, row["sex"], row["smoking_status"])
            if age not in range(50, 80):
                raise ValueError(f"Unsupported age in synthetic pool: {age}")
            if cell[1] not in SEXES or cell[2] not in ("current", "former"):
                raise ValueError(f"Unsupported synthetic-pool cell: {cell}")
            if row["age_group"] != age_group(age):
                raise ValueError(f"Synthetic-pool age-group mismatch for age {age}")
            counts[cell] += 1
    expected_cells = {
        (age, sex, status)
        for age in range(50, 80)
        for sex in SEXES
        for status in ("current", "former")
    }
    if set(counts) != expected_cells:
        raise ValueError(
            f"Synthetic-pool coverage mismatch: {sorted(expected_cells - set(counts))}"
        )
    expected_draws = int(
        parameters["synthetic_pool"]["draws_per_exact_age_sex_status_cell"]
    )
    invalid = {cell: n for cell, n in counts.items() if n != expected_draws}
    if invalid:
        raise ValueError(f"Synthetic-pool cell counts are invalid: {invalid}")
    if row_count != int(parameters["synthetic_pool"]["expected_rows"]):
        raise ValueError("Synthetic-pool row count does not match model parameters")
    return row_count


def git_value(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readme():
    return f"""# CEA Irish population and smoking inputs, version {VERSION}

This directory is the complete runtime input package for CEA population generation.

- `irish_population_age_sex_2022.csv` supplies exact-age and sex population counts from Irish Census-based inputs.
- `irish_smoking_status_age_sex_2022.csv` supplies current, former, and never smoking probabilities by five-year age group and sex.
- `plco_smoking_history_synthetic_pool.csv` supplies complete smoking histories conditional on exact age, sex, and current/former status.
- `smoking_history_generation_parameters.json` records source roles, matching rules, assumptions, constraints, and reproducibility settings.
- `source_metadata.json` records provenance, checksums, and population validation.

For current and former smokers, match `attained_age`, `sex`, and `smoking_status`, then sample one complete row. Never smokers receive structural-zero smoking histories.

Regenerate from the repository root:

```bash
python scripts/build_smooth_smoking_history_outputs.py
python scripts/export_cea_population_inputs.py
```

Diagnostics and cleaning audits remain in `data_processed`; they are not runtime CEA inputs. Legacy pack-year strata, marginal distributions, correlations, and validation-only files are not exported.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("export_for_cea")
    )
    args = parser.parse_args()
    if args.output_directory.is_absolute():
        raise ValueError("--output-directory must be repository-relative")
    output = (ROOT / args.output_directory).resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError("--output-directory must remain inside the repository")

    require_sources()
    parameters = load_parameters()
    pool_rows = validate_pool(parameters)
    population, validation = population_export()
    smoking_status = smoking_status_export()

    output.mkdir(parents=True, exist_ok=True)
    for item in output.iterdir():
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    write_csv(output / "irish_population_age_sex_2022.csv", population)
    write_csv(output / "irish_smoking_status_age_sex_2022.csv", smoking_status)
    shutil.copyfile(PROCESSED / POOL_FILE, output / POOL_FILE)
    shutil.copyfile(PROCESSED / PARAMETERS_FILE, output / PARAMETERS_FILE)
    (output / "README.md").write_text(readme(), encoding="utf-8")

    sources = {
        "README.md": [],
        "irish_population_age_sex_2022.csv": [
            "data_raw/projections2057_raw.csv",
            "data_processed/projections2057.csv",
        ],
        "irish_smoking_status_age_sex_2022.csv": [
            "data_processed/cleaned_smoking_data_ag.csv"
        ],
        POOL_FILE: [f"data_processed/{POOL_FILE}"],
        PARAMETERS_FILE: [f"data_processed/{PARAMETERS_FILE}"],
    }
    details = {}
    for filename, source_files in sources.items():
        path = output / filename
        row_count = None
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
        details[filename] = {
            "original_source_files": source_files,
            "row_count": row_count,
            "sha256": checksum(path),
        }

    source_commit = git_value(
        "log",
        "-1",
        "--format=%H",
        "--",
        "data_raw/projections2057_raw.csv",
        "data_raw/eurobarometer.dta",
        "data_processed/cleaned_smoking_data_ag.csv",
        "scripts/build_smooth_smoking_history_outputs.py",
        "scripts/export_cea_population_inputs.py",
    )
    metadata = {
        "package_version": VERSION,
        "extraction_date": git_value("show", "-s", "--format=%cs", source_commit),
        "source_repository": "Tatjanovna/LuCUMO-LCS-Eligibility-Ireland",
        "source_commit_sha": source_commit,
        "population_year": 2022,
        "population_age_range": {"minimum": 50, "maximum": 79, "inclusive": True},
        "smoking_statuses": list(STATUSES),
        "source_separation": {
            "age_sex_population": "Irish Census-based population inputs",
            "smoking_status_by_age_sex": "Irish Census smoking-status outputs",
            "smoking_histories_conditional_on_age_sex_status": (
                "Eurobarometer 2017 smooth model"
            ),
        },
        "preferred_history_input": POOL_FILE,
        "runtime_files": sorted(EXPECTED_FILES),
        "synthetic_pool_rows": pool_rows,
        "population_grouped_validation": validation,
        "exports": details,
        "checksum_note": (
            "source_metadata.json is not self-checksummed because that would be "
            "self-referential"
        ),
    }
    (output / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    actual = {path.name for path in output.iterdir() if path.is_file()}
    if actual != EXPECTED_FILES:
        raise ValueError(
            f"Lean export mismatch: missing={sorted(EXPECTED_FILES - actual)}, "
            f"extra={sorted(actual - EXPECTED_FILES)}"
        )
    print(
        f"CEA export {VERSION}: {len(population)} population rows, "
        f"{len(smoking_status)} status rows, {pool_rows} histories, "
        f"{len(actual)} runtime files"
    )


if __name__ == "__main__":
    main()
