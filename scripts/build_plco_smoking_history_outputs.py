#!/usr/bin/env python3
"""Build aggregate PLCO-ready smoking-history outputs from the eligibility notebook logic.

This script reproduces the final transformations in
``code/4. pack_year_distribution_baseline.ipynb`` and writes aggregate,
non-identifying age-sex-status summaries. It does not export respondent rows or
identifiers. The existing final adjusted pack-year file remains authoritative
for adjusted pack-year means and standard deviations.
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
MIN_AGE = 50
MAX_AGE = 79
AGE_GROUPS = tuple(f"{age}-{age + 4}" for age in range(MIN_AGE, MAX_AGE + 1, 5))
CURRENT_SOURCE_STATUS = "You currently smoke"
FORMER_SOURCE_STATUS = "You used to smoke but you have stopped"
STATUS_MAP = {CURRENT_SOURCE_STATUS: "current", FORMER_SOURCE_STATUS: "former"}
QUANTILES = (("p05", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95))
CORE_VARIABLES = {
    "current": (
        "age_at_initiation",
        "cigarettes_per_day",
        "smoking_duration",
    ),
    "former": (
        "age_at_initiation",
        "age_at_stopping",
        "cigarettes_per_day",
        "smoking_duration",
        "years_since_quitting",
    ),
}
SUMMARY_VARIABLES = {
    "current": (
        "age_at_initiation",
        "cigarettes_per_day",
        "effective_cigarettes_per_day",
        "smoking_duration",
        "years_since_quitting",
        "standard_pack_years",
        "adjusted_pack_years",
    ),
    "former": (
        "age_at_initiation",
        "age_at_stopping",
        "cigarettes_per_day",
        "effective_cigarettes_per_day",
        "smoking_duration",
        "years_since_quitting",
        "standard_pack_years",
        "adjusted_pack_years",
    ),
}
PLCO_ROLE = {
    "age_at_initiation": "derived_history_input",
    "age_at_stopping": "chronology_input",
    "cigarettes_per_day": "plcom2012_predictor",
    "effective_cigarettes_per_day": "eligibility_adjustment_only",
    "smoking_duration": "plcom2012_predictor",
    "years_since_quitting": "plcom2012_predictor",
    "standard_pack_years": "derived_validation_measure",
    "adjusted_pack_years": "eligibility_validation_measure",
}
SOURCE_NOTEBOOK = "code/4. pack_year_distribution_baseline.ipynb"


def _age_groups(values: pd.Series) -> pd.Series:
    bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, np.inf]
    labels = [
        "0 - 14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
        "80-84", "85-89", "90 and over",
    ]
    return pd.cut(values, bins=bins, labels=labels, right=False)


def build_notebook_history_frame() -> pd.DataFrame:
    """Reproduce the final 2017 smoking-history transformations used in the notebook."""
    source = RAW / "eurobarometer.dta"
    if not source.is_file():
        raise FileNotFoundError(f"Missing eligibility notebook source: {source}")
    frame = pd.read_stata(source)
    for column in ("age_start", "age_years", "age_stop", "cig_day_current", "cig_day_past"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.loc[frame["wave"].eq(2017)].copy()
    frame.loc[frame["sm_status"].eq(CURRENT_SOURCE_STATUS), "length_of_smoking"] = (
        frame.loc[frame["sm_status"].eq(CURRENT_SOURCE_STATUS), "age_years"]
        - frame.loc[frame["sm_status"].eq(CURRENT_SOURCE_STATUS), "age_start"]
    )
    frame.loc[frame["sm_status"].eq(FORMER_SOURCE_STATUS), "length_of_smoking"] = (
        frame.loc[frame["sm_status"].eq(FORMER_SOURCE_STATUS), "age_stop"]
        - frame.loc[frame["sm_status"].eq(FORMER_SOURCE_STATUS), "age_start"]
    )
    frame["cig_day"] = frame["cig_day_current"].fillna(frame["cig_day_past"])

    former = frame["sm_status"].eq(FORMER_SOURCE_STATUS)
    current = frame["sm_status"].eq(CURRENT_SOURCE_STATUS)
    mean_stop = frame.loc[former, "age_stop"].mean()
    frame.loc[former & frame["age_stop"].isna(), "age_stop"] = mean_stop
    frame = frame.dropna(subset=["age_years"]).copy()
    former = frame["sm_status"].eq(FORMER_SOURCE_STATUS)
    current = frame["sm_status"].eq(CURRENT_SOURCE_STATUS)
    for mask in (former, current):
        mean_cigarettes = frame.loc[mask, "cig_day"].mean()
        frame.loc[mask & frame["cig_day"].isna(), "cig_day"] = mean_cigarettes

    frame["age_group"] = _age_groups(frame["age_years"])
    frame["years_since_quit"] = frame["age_years"] - frame["age_stop"]
    frame.loc[current, "years_since_quit"] = 0.0
    frame["effective_cigarettes_per_day"] = frame["cig_day"] * (
        1.0 - np.exp(-0.1 * frame["length_of_smoking"])
    )
    frame["standard_pack_years"] = frame["length_of_smoking"] * frame["cig_day"] / 20.0
    frame["pack_years"] = (
        frame["length_of_smoking"] * frame["effective_cigarettes_per_day"] / 20.0
    ).clip(upper=60.0)

    frame = frame.loc[
        frame["sm_status"].isin(STATUS_MAP)
        & frame["age_group"].astype("string").isin(AGE_GROUPS)
        & frame["gender"].isin(["Male", "Female"])
    ].copy()
    frame["smoking_status"] = frame["sm_status"].map(STATUS_MAP)
    frame["sex"] = frame["gender"].str.lower()
    frame["age_group"] = frame["age_group"].astype("string")
    frame = frame.rename(
        columns={
            "age_start": "age_at_initiation",
            "age_stop": "age_at_stopping",
            "cig_day": "cigarettes_per_day",
            "length_of_smoking": "smoking_duration",
            "years_since_quit": "years_since_quitting",
            "pack_years": "adjusted_pack_years",
        }
    )

    required = [
        "age_at_initiation", "cigarettes_per_day", "smoking_duration",
        "years_since_quitting", "standard_pack_years", "adjusted_pack_years",
    ]
    if frame[required].isna().any().any():
        missing = frame[required].isna().sum()
        raise ValueError(f"Final notebook histories contain missing required values: {missing.to_dict()}")
    former = frame["smoking_status"].eq("former")
    if frame.loc[former, "age_at_stopping"].isna().any():
        raise ValueError("Final former-smoker histories contain missing stopping ages")
    return frame


def _summary(values: pd.Series) -> dict[str, str | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        raise ValueError("Cannot summarise an empty smoking-history variable")
    output: dict[str, str | int] = {
        "n": int(numeric.size),
        "mean": f"{numeric.mean():.12f}",
        "sd": "" if numeric.size < 2 else f"{numeric.std(ddof=1):.12f}",
        "minimum": f"{numeric.min():.12f}",
        "maximum": f"{numeric.max():.12f}",
    }
    for name, probability in QUANTILES:
        output[name] = f"{numeric.quantile(probability, interpolation='linear'):.12f}"
    return output


def _final_pack_year_lookup() -> dict[tuple[str, str, str], dict[str, str]]:
    path = PROCESSED / "pack_year_dist_cleaned.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing final eligibility pack-year output: {path}")
    source = pd.read_csv(path, dtype=str).fillna("")
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for _, row in source.iterrows():
        age_group = str(row["age_group"]).strip()
        gender = str(row["gender"]).strip()
        if age_group not in AGE_GROUPS or gender not in ("Male", "Female"):
            continue
        sex = gender.lower()
        lookup[(age_group, sex, "current")] = {
            "mean": str(row["mean_pack_years_smokers"]).strip(),
            "sd": str(row["std_pack_years_smokers"]).strip(),
        }
        lookup[(age_group, sex, "former")] = {
            "mean": str(row["mean_pack_years_quitters_all"]).strip(),
            "sd": str(row["std_pack_years_quitters_all"]).strip(),
        }
    return lookup


def build_parameter_rows(frame: pd.DataFrame) -> list[dict]:
    final_pack_years = _final_pack_year_lookup()
    rows: list[dict] = []
    for age_group in AGE_GROUPS:
        for sex in ("female", "male"):
            for status in ("current", "former"):
                subset = frame.loc[
                    frame["age_group"].eq(age_group)
                    & frame["sex"].eq(sex)
                    & frame["smoking_status"].eq(status)
                ]
                if subset.empty:
                    raise ValueError(f"Missing final notebook cell: {(age_group, sex, status)}")
                for variable in SUMMARY_VARIABLES[status]:
                    stats = _summary(subset[variable])
                    if variable == "adjusted_pack_years":
                        final = final_pack_years[(age_group, sex, status)]
                        if not final["mean"] or not final["sd"]:
                            raise ValueError(
                                f"Missing final adjusted pack-year estimate for {(age_group, sex, status)}"
                            )
                        stats["mean"] = final["mean"]
                        stats["sd"] = final["sd"]
                    structural = variable == "years_since_quitting" and status == "current"
                    rows.append(
                        {
                            "age_group": age_group,
                            "sex": sex,
                            "smoking_status": status,
                            "history_variable": variable,
                            **stats,
                            "plco_role": PLCO_ROLE[variable],
                            "distribution_recommendation": (
                                "structural_zero" if structural else "empirical_quantile"
                            ),
                            "source_dataset": SOURCE_NOTEBOOK,
                            "source_processing": "final_notebook_transformation_reproduced",
                        }
                    )
    return rows


def build_correlation_rows(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for age_group in AGE_GROUPS:
        for sex in ("female", "male"):
            for status in ("current", "former"):
                subset = frame.loc[
                    frame["age_group"].eq(age_group)
                    & frame["sex"].eq(sex)
                    & frame["smoking_status"].eq(status)
                ]
                for first, second in itertools.combinations(CORE_VARIABLES[status], 2):
                    paired = subset[[first, second]].dropna().astype(float)
                    correlation = ""
                    estimable = (
                        len(paired) >= 2
                        and paired[first].std(ddof=1) > 0
                        and paired[second].std(ddof=1) > 0
                    )
                    if estimable:
                        correlation = f"{paired[first].corr(paired[second]):.12f}"
                    rows.append(
                        {
                            "age_group": age_group,
                            "sex": sex,
                            "smoking_status": status,
                            "variable_1": first,
                            "variable_2": second,
                            "n_complete": int(len(paired)),
                            "correlation": correlation,
                            "correlation_estimable": str(bool(estimable)).lower(),
                            "joint_generation_method": "gaussian_copula_with_empirical_marginals",
                            "source_dataset": SOURCE_NOTEBOOK,
                        }
                    )
    return rows


def build_validation_rows(parameter_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in parameter_rows:
        for statistic in ("mean", "sd", "p50"):
            if row[statistic] == "":
                continue
            rows.append(
                {
                    "validation_population": "eligibility_final_processed_ages_50_79",
                    "age_group": row["age_group"],
                    "sex": row["sex"],
                    "smoking_status": row["smoking_status"],
                    "history_variable": row["history_variable"],
                    "statistic": statistic,
                    "value": row[statistic],
                    "n": row["n"],
                    "source_dataset": row["source_dataset"],
                }
            )
    return rows


def write_outputs(output_directory: Path = PROCESSED) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    frame = build_notebook_history_frame()
    parameters = pd.DataFrame(build_parameter_rows(frame))
    correlations = pd.DataFrame(build_correlation_rows(frame))
    validation = pd.DataFrame(build_validation_rows(parameters.to_dict("records")))
    parameter_path = output_directory / "plco_smoking_history_parameters.csv"
    correlation_path = output_directory / "plco_smoking_history_correlations.csv"
    validation_path = output_directory / "plco_smoking_history_validation_targets.csv"
    parameters.to_csv(parameter_path, index=False, lineterminator="\n")
    correlations.to_csv(correlation_path, index=False, lineterminator="\n")
    validation.to_csv(validation_path, index=False, lineterminator="\n")
    return parameter_path, correlation_path, validation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROCESSED,
        help="directory for aggregate processed outputs (default: data_processed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_directory
    if not output.is_absolute():
        output = ROOT / output
    paths = write_outputs(output.resolve())
    frame = build_notebook_history_frame()
    print(
        f"PLCO smoking-history outputs: {len(frame)} final current/former histories; "
        f"ages {MIN_AGE}-{MAX_AGE}; no respondent rows exported"
    )
    for path in paths:
        print(path.relative_to(ROOT) if ROOT in path.parents else path)


if __name__ == "__main__":
    main()
