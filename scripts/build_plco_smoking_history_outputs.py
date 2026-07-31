#!/usr/bin/env python3
"""Create aggregate PLCO-ready smoking-history outputs for the CEA.

The processing follows the final 2017 transformations in
``code/4. pack_year_distribution_baseline.ipynb``. Only aggregate parameters
are exported; respondent records and identifiers are never written.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
PROCESSED = ROOT / "data_processed"
AGE_GROUPS = tuple(f"{age}-{age + 4}" for age in range(50, 80, 5))
CURRENT_RAW = "You currently smoke"
FORMER_RAW = "You used to smoke but you have stopped"
STATUS_MAP = {CURRENT_RAW: "current", FORMER_RAW: "former"}
NOTEBOOK = "code/4. pack_year_distribution_baseline.ipynb"
QUANTILES = (("p05", .05), ("p25", .25), ("p50", .50), ("p75", .75), ("p95", .95))

CURRENT_VARIABLES = (
    "age_at_initiation", "cigarettes_per_day", "effective_cigarettes_per_day",
    "smoking_duration", "years_since_quitting", "standard_pack_years",
    "adjusted_pack_years",
)
FORMER_VARIABLES = (
    "age_at_initiation", "age_at_stopping", "cigarettes_per_day",
    "effective_cigarettes_per_day", "smoking_duration", "years_since_quitting",
    "standard_pack_years", "adjusted_pack_years",
)
CORE = {
    "current": ("age_at_initiation", "cigarettes_per_day", "smoking_duration"),
    "former": (
        "age_at_initiation", "age_at_stopping", "cigarettes_per_day",
        "smoking_duration", "years_since_quitting",
    ),
}
ROLE = {
    "age_at_initiation": "derived_history_input",
    "age_at_stopping": "chronology_input",
    "cigarettes_per_day": "plcom2012_predictor",
    "effective_cigarettes_per_day": "eligibility_adjustment_only",
    "smoking_duration": "plcom2012_predictor",
    "years_since_quitting": "plcom2012_predictor",
    "standard_pack_years": "derived_validation_measure",
    "adjusted_pack_years": "eligibility_validation_measure",
}


def _age_group(age: pd.Series) -> pd.Series:
    bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, np.inf]
    labels = [
        "0 - 14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
        "80-84", "85-89", "90 and over",
    ]
    return pd.cut(age, bins=bins, labels=labels, right=False)


def build_history_frame() -> pd.DataFrame:
    path = RAW / "eurobarometer.dta"
    if not path.is_file():
        raise FileNotFoundError(f"Missing notebook source: {path}")
    data = pd.read_stata(path)
    for column in ("age_start", "age_stop", "age_years", "cig_day_current", "cig_day_past"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[data["wave"].eq(2017)].copy()
    data["cig_day"] = data["cig_day_current"].fillna(data["cig_day_past"])

    former = data["sm_status"].eq(FORMER_RAW)
    current = data["sm_status"].eq(CURRENT_RAW)
    data.loc[former & data["age_stop"].isna(), "age_stop"] = data.loc[former, "age_stop"].mean()
    data = data.dropna(subset=["age_years"]).copy()
    former = data["sm_status"].eq(FORMER_RAW)
    current = data["sm_status"].eq(CURRENT_RAW)
    for mask in (former, current):
        data.loc[mask & data["cig_day"].isna(), "cig_day"] = data.loc[mask, "cig_day"].mean()

    data.loc[current, "smoking_duration"] = data.loc[current, "age_years"] - data.loc[current, "age_start"]
    data.loc[former, "smoking_duration"] = data.loc[former, "age_stop"] - data.loc[former, "age_start"]
    data["age_group"] = _age_group(data["age_years"])
    data["years_since_quitting"] = data["age_years"] - data["age_stop"]
    data.loc[current, "years_since_quitting"] = 0.0
    data["effective_cigarettes_per_day"] = data["cig_day"] * (
        1.0 - np.exp(-0.1 * data["smoking_duration"])
    )
    data["standard_pack_years"] = data["smoking_duration"] * data["cig_day"] / 20.0
    data["adjusted_pack_years"] = (
        data["smoking_duration"] * data["effective_cigarettes_per_day"] / 20.0
    ).clip(upper=60.0)

    data = data.loc[
        data["sm_status"].isin(STATUS_MAP)
        & data["age_group"].astype("string").isin(AGE_GROUPS)
        & data["gender"].isin(["Female", "Male"])
    ].copy()
    data["smoking_status"] = data["sm_status"].map(STATUS_MAP)
    data["sex"] = data["gender"].str.lower()
    data["age_group"] = data["age_group"].astype("string")
    data = data.rename(columns={
        "age_start": "age_at_initiation",
        "age_stop": "age_at_stopping",
        "cig_day": "cigarettes_per_day",
    })

    current_required = [
        "age_at_initiation", "cigarettes_per_day", "smoking_duration",
        "years_since_quitting", "standard_pack_years", "adjusted_pack_years",
    ]
    former_required = [*current_required, "age_at_stopping"]
    is_current = data["smoking_status"].eq("current")
    is_former = data["smoking_status"].eq("former")
    current_valid = (
        is_current
        & data[current_required].notna().all(axis=1)
        & data["age_at_initiation"].lt(data["age_years"])
        & data["cigarettes_per_day"].gt(0)
        & data["smoking_duration"].gt(0)
    )
    former_valid = (
        is_former
        & data[former_required].notna().all(axis=1)
        & data["age_at_initiation"].lt(data["age_at_stopping"])
        & data["age_at_stopping"].le(data["age_years"])
        & data["cigarettes_per_day"].gt(0)
        & data["smoking_duration"].gt(0)
        & data["years_since_quitting"].ge(0)
    )
    valid = current_valid | former_valid
    excluded = int((~valid).sum())
    data = data.loc[valid].copy()
    data.attrs["invalid_or_incomplete_excluded"] = excluded
    if data.empty:
        raise ValueError("No valid complete PLCO histories remain")
    return data


def _summary(values: pd.Series) -> dict[str, str | int]:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if values.empty:
        raise ValueError("Cannot summarise an empty variable")
    result: dict[str, str | int] = {
        "n": int(len(values)),
        "mean": f"{values.mean():.12f}",
        "sd": "" if len(values) < 2 else f"{values.std(ddof=1):.12f}",
        "minimum": f"{values.min():.12f}",
        "maximum": f"{values.max():.12f}",
    }
    for name, probability in QUANTILES:
        result[name] = f"{values.quantile(probability, interpolation='linear'):.12f}"
    return result


def _final_adjusted_pack_years() -> dict[tuple[str, str, str], tuple[str, str]]:
    path = PROCESSED / "pack_year_dist_cleaned.csv"
    source = pd.read_csv(path, dtype=str).fillna("")
    result: dict[tuple[str, str, str], tuple[str, str]] = {}
    for _, row in source.iterrows():
        group, gender = str(row["age_group"]).strip(), str(row["gender"]).strip()
        if group not in AGE_GROUPS or gender not in ("Female", "Male"):
            continue
        sex = gender.lower()
        result[(group, sex, "current")] = (
            str(row["mean_pack_years_smokers"]).strip(),
            str(row["std_pack_years_smokers"]).strip(),
        )
        result[(group, sex, "former")] = (
            str(row["mean_pack_years_quitters_all"]).strip(),
            str(row["std_pack_years_quitters_all"]).strip(),
        )
    return result


def parameter_rows(data: pd.DataFrame) -> list[dict]:
    final_pack_years = _final_adjusted_pack_years()
    rows: list[dict] = []
    variables = {"current": CURRENT_VARIABLES, "former": FORMER_VARIABLES}
    for group in AGE_GROUPS:
        for sex in ("female", "male"):
            for status in ("current", "former"):
                cell = data.loc[
                    data["age_group"].eq(group)
                    & data["sex"].eq(sex)
                    & data["smoking_status"].eq(status)
                ]
                if cell.empty:
                    raise ValueError(f"Missing valid exact cell: {(group, sex, status)}")
                for variable in variables[status]:
                    stats = _summary(cell[variable])
                    if variable == "adjusted_pack_years":
                        mean, sd = final_pack_years[(group, sex, status)]
                        if not mean or not sd:
                            raise ValueError(f"Missing final adjusted pack-year estimate: {(group, sex, status)}")
                        stats["mean"], stats["sd"] = mean, sd
                    structural_zero = status == "current" and variable == "years_since_quitting"
                    rows.append({
                        "age_group": group,
                        "sex": sex,
                        "smoking_status": status,
                        "history_variable": variable,
                        **stats,
                        "plco_role": ROLE[variable],
                        "distribution_recommendation": "structural_zero" if structural_zero else "empirical_quantile",
                        "source_dataset": NOTEBOOK,
                        "source_processing": "final_notebook_valid_complete_history_summary",
                    })
    return rows


def correlation_rows(data: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for group in AGE_GROUPS:
        for sex in ("female", "male"):
            for status in ("current", "former"):
                cell = data.loc[
                    data["age_group"].eq(group)
                    & data["sex"].eq(sex)
                    & data["smoking_status"].eq(status)
                ]
                for first, second in itertools.combinations(CORE[status], 2):
                    pair = cell[[first, second]].dropna().astype(float)
                    estimable = (
                        len(pair) >= 2
                        and pair[first].std(ddof=1) > 0
                        and pair[second].std(ddof=1) > 0
                    )
                    rows.append({
                        "age_group": group,
                        "sex": sex,
                        "smoking_status": status,
                        "variable_1": first,
                        "variable_2": second,
                        "n_complete": int(len(pair)),
                        "correlation": f"{pair[first].corr(pair[second]):.12f}" if estimable else "",
                        "correlation_estimable": str(bool(estimable)).lower(),
                        "joint_generation_method": "gaussian_copula_with_empirical_marginals",
                        "source_dataset": NOTEBOOK,
                    })
    return rows


def validation_rows(parameters: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in parameters:
        for statistic in ("mean", "sd", "p50"):
            if row[statistic] == "":
                continue
            rows.append({
                "validation_population": "eligibility_final_processed_ages_50_79",
                "age_group": row["age_group"],
                "sex": row["sex"],
                "smoking_status": row["smoking_status"],
                "history_variable": row["history_variable"],
                "statistic": statistic,
                "value": row[statistic],
                "n": row["n"],
                "source_dataset": row["source_dataset"],
            })
    return rows


def write_outputs(directory: Path) -> tuple[list[Path], int, int]:
    directory.mkdir(parents=True, exist_ok=True)
    data = build_history_frame()
    excluded = int(data.attrs.get("invalid_or_incomplete_excluded", 0))
    parameters = parameter_rows(data)
    correlations = correlation_rows(data)
    validation = validation_rows(parameters)
    outputs = [
        ("plco_smoking_history_parameters.csv", parameters),
        ("plco_smoking_history_correlations.csv", correlations),
        ("plco_smoking_history_validation_targets.csv", validation),
    ]
    paths: list[Path] = []
    for filename, records in outputs:
        path = directory / filename
        pd.DataFrame(records).to_csv(path, index=False, lineterminator="\n")
        paths.append(path)
    return paths, len(data), excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=PROCESSED)
    args = parser.parse_args()
    directory = args.output_directory if args.output_directory.is_absolute() else ROOT / args.output_directory
    paths, retained, excluded = write_outputs(directory.resolve())
    print(
        f"PLCO smoking-history outputs: {retained} valid complete histories retained; "
        f"{excluded} invalid or incomplete histories excluded; ages 50-79"
    )
    for path in paths:
        print(path.relative_to(ROOT) if ROOT in path.parents else path)
    print("No respondent rows or identifiers exported")


if __name__ == "__main__":
    main()
