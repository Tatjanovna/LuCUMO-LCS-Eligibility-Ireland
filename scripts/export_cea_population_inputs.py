#!/usr/bin/env python3
"""Build the deterministic, self-contained population input package for the CEA.

This script deliberately uses only the Python standard library.  In particular,
the small Stata 118 reader below reads the numeric variables in the repository's
curated Eurobarometer extract without requiring pandas or notebook execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
PROCESSED = ROOT / "data_processed"
OUT = ROOT / "export_for_cea"
MIN_AGE = 50
MAX_AGE = 80
PACKAGE_VERSION = "1.1.0"
MINIMUM_STRATUM_N = 10
KNOWN_STATUSES = ("current_daily", "current_occasional", "former", "never")
ALL_STATUSES = (*KNOWN_STATUSES, "not_stated")
HISTORY_VARIABLES = (
    "age_at_initiation", "age_at_stopping", "years_since_quitting",
    "cigarettes_per_day", "smoking_duration", "pack_years_standard",
)


def require_sources() -> None:
    required = [
        RAW / "projections2057_raw.csv",
        PROCESSED / "projections2057.csv",
        RAW / "restructured_smoking_population_data.csv",
        RAW / "eurobarometer.dta",
    ]
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
                    "population_count": int(source["VALUE"]),
                    "source_dataset": "CSO_population_projections_based_on_Census_2022",
                    "source_scenario": "CSO_Census2022_M2",
                })
    rows.sort(key=lambda row: (row["age"], row["sex"]))
    expected = {(age, sex) for age in range(MIN_AGE, MAX_AGE + 1) for sex in ("female", "male")}
    observed = {(row["age"], row["sex"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Exact age/sex coverage mismatch: missing={sorted(expected-observed)}")

    exact = defaultdict(int)
    for row in rows:
        group = f"{row['age']//5*5}-{row['age']//5*5+4}"
        exact[(group, row["sex"])] += row["population_count"]
    grouped = {}
    with (PROCESSED / "projections2057.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["year"] == "2022":
                grouped[(row["age_group"], row["gender"].lower())] = int(row["population"])
    validation = []
    # Only complete five-year bands can be compared with grouped totals.  With
    # the default maximum of 80, 80-84 is represented by age 80 alone.
    incomplete_group = five_year_group(MAX_AGE) if MAX_AGE % 5 != 4 else None
    for key in sorted(k for k in exact if k[0] != incomplete_group):
        expected_total = grouped.get(key)
        discrepancy = exact[key] - expected_total if expected_total is not None else None
        validation.append({"age_group": key[0], "sex": key[1],
                           "exact_age_total": exact[key], "grouped_total": expected_total,
                           "discrepancy": discrepancy})
        if discrepancy != 0:
            raise ValueError(f"Population validation discrepancy for {key}: {discrepancy}")
    return rows, validation


def smoking_status_export() -> list[dict]:
    # Include every Census source band intersecting the requested exact-age
    # range.  This gives 50-54 through 75-79 for a maximum age of 79, and also
    # 80-84 (as the source band for exact age 80) for a maximum age of 80.
    age_map = {f"{start}-{start+4} years": f"{start}-{start+4}"
               for start in range(MIN_AGE, MAX_AGE + 1, 5)}
    category_map = {
        "Smoke daily": "current_daily",
        "Smoke occasionally": "current_occasional",
        "Dont smoke - gave up": "former",
        "Never smoked": "never",
        "Not stated": "not_stated",
    }
    counts = {}
    with (RAW / "restructured_smoking_population_data.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for source in csv.DictReader(handle):
            age_group = age_map.get(source["Age Group"])
            if (source["Census Year"] == "2022" and source["County"] == "State"
                    and source["Sex"] in ("Male", "Female") and age_group):
                if not source["VALUE"]:
                    raise ValueError(f"Missing State smoking count: {source}")
                key = (age_group, source["Sex"].lower(), category_map[source["Smoking Tobacco Products"]])
                counts[key] = int(float(source["VALUE"]))
    expected = {(group, sex, status) for group in age_map.values()
                for sex in ("female", "male") for status in ALL_STATUSES}
    if set(counts) != expected:
        raise ValueError("Smoking status source does not contain every requested cell")
    rows = []
    for group, sex, status in sorted(counts):
        total = sum(counts[(group, sex, item)] for item in ALL_STATUSES)
        known_total = sum(counts[(group, sex, item)] for item in KNOWN_STATUSES)
        rows.append({
            "year": 2022, "age_group": group, "sex": sex, "smoking_status": status,
            "count": counts[(group, sex, status)],
            "probability_all_people": f"{counts[(group, sex, status)] / total:.12f}",
            "probability_known_status": "" if status == "not_stated" else
                f"{counts[(group, sex, status)] / known_total:.12f}",
            "source_dataset": "Census_2022_smoking_status_State",
        })
    return rows


def stata118_numeric_rows(path: Path) -> list[dict[str, float | int]]:
    """Read numeric columns from a little-endian Stata release-118 file."""
    payload = path.read_bytes()
    def section(name: bytes) -> bytes:
        try:
            return payload.split(b"<" + name + b">", 1)[1].split(b"</" + name + b">", 1)[0]
        except IndexError as exc:
            raise ValueError(f"Unsupported or corrupt Stata file: missing {name!r}") from exc
    byteorder = re.search(rb"<byteorder>(.*?)</byteorder>", payload).group(1)
    release = re.search(rb"<release>(.*?)</release>", payload).group(1)
    if byteorder != b"LSF" or release != b"118":
        raise ValueError("The dependency-free reader supports little-endian Stata 118 only")
    k = struct.unpack("<H", section(b"K"))[0]
    n = struct.unpack("<Q", section(b"N"))[0]
    names_blob = section(b"varnames")
    names = [names_blob[i*129:(i+1)*129].split(b"\0", 1)[0].decode() for i in range(k)]
    types = struct.unpack("<" + "H" * k, section(b"variable_types"))
    formats = {65530: "b", 65529: "h", 65528: "i", 65527: "f", 65526: "d"}
    if any(value not in formats for value in types):
        raise ValueError("String variables are not supported by this minimal Stata reader")
    row_format = "<" + "".join(formats[value] for value in types)
    size = struct.calcsize(row_format)
    data = section(b"data")
    if len(data) != n * size:
        raise ValueError("Unexpected Stata data-section size")
    return [dict(zip(names, struct.unpack_from(row_format, data, index * size))) for index in range(n)]


def valid_number(value, upper=1000) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value < upper


def five_year_group(age: int) -> str:
    lower = age // 5 * 5
    return f"{lower}-{lower+4}"


def broad_group(age: int) -> str:
    return "50-59" if age < 60 else "60-69" if age < 70 else f"70-{MAX_AGE}"


def clean_histories() -> tuple[list[dict], dict]:
    """Return clean histories and an aggregate, non-identifying cleaning audit.

    ``data_raw/eurobarometer.dta`` is the authoritative curated Irish extract.
    Rejections are assigned to the first applicable reason below, making the
    mutually exclusive counts reconcile exactly to the candidate count.
    """
    source_rows = stata118_numeric_rows(RAW / "eurobarometer.dta")
    donors = []
    rejection = defaultdict(int, {reason: 0 for reason in (
        "unknown_smoking_status_code", "unknown_sex_code",
        "non_finite_or_missing_required_value", "age_at_initiation_below_5",
        "age_at_initiation_not_before_attained_age", "cigarettes_per_day_not_positive",
        "cigarettes_per_day_above_80", "missing_or_invalid_stopping_age",
        "age_at_stopping_not_after_initiation", "age_at_stopping_after_attained_age",
        "pack_years_above_200")})
    candidates = 0
    pre_values = defaultdict(list)
    for row in source_rows:
        if not math.isfinite(float(row["wave"])) or round(row["wave"]) != 2017:
            continue
        raw_age = row["age_years"]
        if not isinstance(raw_age, (int, float)) or not math.isfinite(raw_age):
            continue
        age = int(raw_age)
        if not MIN_AGE <= age <= MAX_AGE:
            continue
        candidates += 1
        if row["sm_status"] not in (1, 2):
            rejection["unknown_smoking_status_code"] += 1
            continue
        if row["gender"] not in (1, 2):
            rejection["unknown_sex_code"] += 1
            continue
        status = "current_unspecified" if row["sm_status"] == 1 else "former"
        start = row["age_start"]
        stop = row["age_stop"]
        cpd = row["cig_day_current"] if status == "current_unspecified" else row["cig_day_past"]
        # Stata numeric missing values are large finite sentinels.  Treat them
        # as invalid rather than allowing them to influence maxima.
        if not valid_number(start, 100) or not valid_number(cpd, 500):
            rejection["non_finite_or_missing_required_value"] += 1
            continue
        pre_values["age_at_initiation"].append(float(start))
        pre_values["cigarettes_per_day"].append(float(cpd))
        if start < 5:
            rejection["age_at_initiation_below_5"] += 1
            continue
        if start >= age:
            rejection["age_at_initiation_not_before_attained_age"] += 1
            continue
        if cpd <= 0:
            rejection["cigarettes_per_day_not_positive"] += 1
            continue
        if cpd > 80:
            rejection["cigarettes_per_day_above_80"] += 1
            continue
        start = int(start)
        if status == "current_unspecified":
            stop_value = None
            years_since = 0
            duration = age - start
        else:
            if not valid_number(stop, 100):
                rejection["missing_or_invalid_stopping_age"] += 1
                continue
            stop_value = int(stop)
            pre_values["age_at_stopping"].append(float(stop))
            if stop_value <= start:
                rejection["age_at_stopping_not_after_initiation"] += 1
                continue
            if stop_value > age:
                rejection["age_at_stopping_after_attained_age"] += 1
                continue
            years_since = age - stop_value
            duration = stop_value - start
        # Derived values are deliberately calculated only after chronology has
        # passed; values are never clipped or winsorised.
        pack_years = float(cpd) * duration / 20
        pre_values["smoking_duration"].append(float(duration))
        pre_values["years_since_quitting"].append(float(years_since))
        pre_values["pack_years_standard"].append(pack_years)
        if not all(math.isfinite(value) for value in (duration, years_since, pack_years)):
            rejection["non_finite_or_missing_required_value"] += 1
            continue
        if pack_years > 200:
            rejection["pack_years_above_200"] += 1
            continue
        donors.append({
            "sex": "male" if row["gender"] == 1 else "female",
            "attained_age": age,
            "attained_age_group": five_year_group(age),
            "broad_age_group": broad_group(age),
            "smoking_status": status,
            "age_at_initiation": start,
            "age_at_stopping": stop_value,
            "years_since_quitting": years_since,
            "cigarettes_per_day": float(cpd),
            "smoking_duration": duration,
            "pack_years_standard": pack_years,
            "survey_weight": 1.0,
            "survey_wave": 2017,
            "source_dataset": "Irish_Eurobarometer_curated_extract",
        })
    donors.sort(key=lambda row: (row["smoking_status"], row["sex"], row["attained_age"],
                                 row["age_at_initiation"], row["cigarettes_per_day"]))
    variables = ("age_at_initiation", "age_at_stopping", "cigarettes_per_day",
                 "smoking_duration", "years_since_quitting", "pack_years_standard")
    maxima = lambda values, variable: max((float(r[variable]) for r in values
                                            if r[variable] is not None), default=None)
    report = {
        "authoritative_source": "data_raw/eurobarometer.dta",
        "candidate_records": candidates,
        "retained_records": len(donors),
        "excluded_records": candidates - len(donors),
        "excluded_records_by_reason": dict(sorted(rejection.items())),
        "records_with_initiation_age_below_10": sum(v < 10 for v in pre_values["age_at_initiation"]),
        "records_with_cigarettes_per_day_above_60": sum(v > 60 for v in pre_values["cigarettes_per_day"]),
        "records_with_pack_years_above_100": sum(v > 100 for v in pre_values["pack_years_standard"]),
        "pre_cleaning_maxima": {v: max(pre_values[v], default=None) for v in variables},
        "post_cleaning_maxima": {v: maxima(donors, v) for v in variables},
    }
    return donors, report


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0,
        "p05": percentile(values, .05), "p25": percentile(values, .25),
        "p50": percentile(values, .50), "p75": percentile(values, .75),
        "p95": percentile(values, .95), "minimum": min(values), "maximum": max(values),
    }


def strata_export(donors: list[dict]) -> list[dict]:
    # Export every supported level so consumers can execute the documented fallback hierarchy.
    specs = (
        ("sex_five_year_status", lambda r: (r["smoking_status"], r["sex"], r["attained_age_group"], r["broad_age_group"])),
        ("sex_broad_status", lambda r: (r["smoking_status"], r["sex"], "all", r["broad_age_group"])),
        ("sex_status", lambda r: (r["smoking_status"], r["sex"], "all", "all")),
        ("status", lambda r: (r["smoking_status"], "all", "all", "all")),
    )
    rows = []
    for level, key_function in specs:
        groups = defaultdict(list)
        for donor in donors:
            groups[key_function(donor)].append(donor)
        for key, members in sorted(groups.items()):
            if len(members) < MINIMUM_STRATUM_N:
                continue
            for variable in HISTORY_VARIABLES:
                values = [float(member[variable]) for member in members if member[variable] is not None]
                if len(values) < MINIMUM_STRATUM_N:
                    continue
                stats = summary(values)
                rows.append({
                    "smoking_status": key[0], "sex": key[1], "attained_age_group": key[2],
                    "broad_age_group": key[3], "history_variable": variable,
                    "n_unweighted": len(values), "weighted_n": f"{len(values):.6f}",
                    **{name: f"{value:.6f}" for name, value in stats.items()},
                    "weighting_status": "unweighted_equal_weights",
                    "fallback_level": level,
                })
    return rows


def correlation(rows: list[dict], first: str, second: str) -> float | None:
    pairs = [(float(r[first]), float(r[second])) for r in rows
             if r[first] is not None and r[second] is not None]
    if len(pairs) < MINIMUM_STRATUM_N:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x-mx)*(y-my) for x, y in pairs)
    denominator = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return numerator / denominator if denominator else None


def generation_parameters(donors: list[dict]) -> dict:
    correlations = {}
    for status in ("current_unspecified", "former"):
        subset = [row for row in donors if row["smoking_status"] == status]
        correlations[status] = {
            "n": len(subset),
            "age_at_initiation__smoking_duration": correlation(subset, "age_at_initiation", "smoking_duration"),
            "cigarettes_per_day__pack_years_standard": correlation(subset, "cigarettes_per_day", "pack_years_standard"),
        }
        if status == "former":
            correlations[status]["years_since_quitting__attained_age"] = correlation(
                subset, "years_since_quitting", "attained_age")
    return {
        "model_version": PACKAGE_VERSION, "population_year": 2022,
        "survey_source": "Irish respondents in the repository's curated Eurobarometer extract",
        "survey_wave": 2017, "weight_variable": None,
        "weighting_status": "unweighted_equal_weights",
        "preferred_generation_method": "constraint-aware multivariate generation calibrated to exported strata and correlations",
        "donor_file_exported": False,
        "donor_file_reason": "Eurobarometer respondent-level redistribution rights were not documented in the source repository.",
        "joint_model_warning": "Marginal strata are not a joint model. Generate jointly, calibrate to the empirical correlations, reject logically invalid draws, and validate all exported margins.",
        "fallback_hierarchy": ["sex_x_five_year_age_x_status", "sex_x_broad_age_x_status", "sex_x_status", "status"],
        "minimum_stratum_sample_size": MINIMUM_STRATUM_N,
        "population_age_range": {"minimum": MIN_AGE, "maximum": MAX_AGE, "inclusive": True},
        "age_groups": [f"{age}-{age+4}" for age in range(MIN_AGE, MAX_AGE + 1, 5)],
        "broad_age_groups": ["50-59", "60-69", f"70-{MAX_AGE}"],
        "status_definitions": {"current_unspecified": "Currently smokes; frequency is unavailable in the extract",
                               "former": "Used to smoke but has stopped"},
        "logical_constraints": {
            "all": ["5 <= age_at_initiation < attained_age", "0 < cigarettes_per_day <= 80",
                    "pack_years_standard <= 200", "all required values are finite"],
            "current": ["age_at_stopping is missing", "years_since_quitting = 0"],
            "former": ["age_at_initiation < age_at_stopping <= attained_age",
                       "years_since_quitting >= 0", "cigarettes_per_day > 0"],
        },
        "pack_year_definition": "cigarettes_per_day * smoking_duration / 20; uncapped; no exponential adjustment",
        "missing_data_handling": "Retain unchanged or exclude with one documented reason; no imputation, clipping, or winsorisation",
        "correlations": correlations,
    }


def validation_targets(smoking_rows: list[dict], donors: list[dict]) -> list[dict]:
    output = []
    for row in smoking_rows:
        start = int(row["age_group"].split("-")[0])
        if 55 <= start <= 70:
            output.append({
                "validation_population": "Irish_LHC_comparator_ages_55_74",
                "sex": row["sex"], "age_group": row["age_group"],
                "smoking_status": row["smoking_status"], "measure": "census_probability_all_people",
                "value": row["probability_all_people"], "n": row["count"],
                "weighting_status": "census_count", "source_dataset": row["source_dataset"],
            })
    groups = defaultdict(list)
    for donor in donors:
        if 55 <= donor["attained_age"] <= 74:
            groups[(donor["sex"], donor["attained_age_group"], donor["smoking_status"])].append(donor)
    variables = ("age_at_initiation", "cigarettes_per_day", "age_at_stopping",
                 "years_since_quitting", "smoking_duration", "pack_years_standard")
    for key, members in sorted(groups.items()):
        for variable in variables:
            values = [float(item[variable]) for item in members if item[variable] is not None]
            if values:
                output.append({"validation_population": "Irish_LHC_comparator_ages_55_74",
                    "sex": key[0], "age_group": key[1], "smoking_status": key[2],
                    "measure": f"mean_{variable}", "value": f"{statistics.fmean(values):.6f}",
                    "n": len(values), "weighting_status": "unweighted_equal_weights",
                    "source_dataset": "Irish_Eurobarometer_curated_extract"})
        pack_years = [float(item["pack_years_standard"]) for item in members]
        for threshold in (15, 20):
            output.append({"validation_population": "Irish_LHC_comparator_ages_55_74",
                "sex": key[0], "age_group": key[1], "smoking_status": key[2],
                "measure": f"proportion_above_{threshold}_pack_years_standard",
                "value": f"{sum(value > threshold for value in pack_years)/len(pack_years):.6f}",
                "n": len(pack_years), "weighting_status": "unweighted_equal_weights",
                "source_dataset": "Irish_Eurobarometer_curated_extract"})
    return output


def git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def source_data_commit() -> str:
    """Return the newest commit touching an authoritative input, not generated outputs."""
    return git_value("log", "-1", "--format=%H", "--", "data_raw", "data_processed/projections2057.csv")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(export_details: dict, history_report: dict, population_validation: list[dict]) -> dict:
    source_commit = source_data_commit()
    return {
        # Use the source commit date rather than wall-clock time so identical inputs
        # produce byte-identical exports on a later day.
        "package_version": PACKAGE_VERSION, "extraction_date": git_value("show", "-s", "--format=%cs", source_commit),
        "source_repository": "Tatjanovna/LuCUMO-LCS-Eligibility-Ireland",
        "source_commit_sha": source_commit,
        "source_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "population_year": 2022,
        "population_age_range": {"minimum": MIN_AGE, "maximum": MAX_AGE, "inclusive": True},
        "population_source_description": "2022 baseline of a Census-2022-based CSO population projection series, M2 scenario; not relabelled as a direct Census single-year-age table.",
        "source_table_publisher": {"population": "Central Statistics Office Ireland",
                                   "smoking_status": "Census 2022 / Central Statistics Office Ireland",
                                   "smoking_history": "Eurobarometer"},
        "category_mappings": {"Smoke daily": "current_daily", "Smoke occasionally": "current_occasional",
                              "Dont smoke - gave up": "former", "Never smoked": "never",
                              "Not stated": "not_stated", "You currently smoke": "current_unspecified",
                              "You used to smoke but you have stopped": "former"},
        "transformations": [f"Population restricted to 2022, exact ages {MIN_AGE}-{MAX_AGE}, and male/female",
                            f"Smoking status restricted to State geography and every source age band intersecting {MIN_AGE}-{MAX_AGE}",
                            f"Eurobarometer restricted to wave 2017 and ages {MIN_AGE}-{MAX_AGE}",
                            "Unknown sex/status, non-finite required values, initiation below 5 or not before attained age, intensity outside (0,80], invalid former stopping age/chronology, and derived pack-years above 200 excluded",
                            "Durations, years since quitting, and standard pack-years calculated only after chronology validation; values are never clipped or winsorised"],
        "original_variable_names": {"population": ["Year", "Age", "Sex", "VALUE"],
                                    "smoking_status": ["Census Year", "Sex", "Smoking Tobacco Products", "Age Group", "County", "VALUE"],
                                    "smoking_history": ["wave", "sm_status", "age_start", "age_stop", "cig_day_past", "cig_day_current", "gender", "age_years"]},
        "exported_variable_names": {"population": ["year", "age", "sex", "population_count", "source_dataset", "source_scenario"],
                                    "smoking_history": list(HISTORY_VARIABLES)},
        "source_files": ["data_raw/projections2057_raw.csv", "data_processed/projections2057.csv",
                         "data_raw/restructured_smoking_population_data.csv", "data_raw/eurobarometer.dta"],
        "source_notebooks": ["code/1. cleaning_population_projections.ipynb",
                             "code/2. cleaning_smoking_census2022.ipynb"],
        "missing_data_handling": "Complete-case smoking histories; no imputation, clipping, or mean replacement",
        "survey_weight_handling": {"weight_variable": None, "status": "unweighted_equal_weights",
            "reason": "The original 2017 file contains multiple weight labels, but no codebook or reproducible mapping confidently identifies the appropriate Irish national variable in the curated extract; no weight was guessed."},
        "pack_year_formula": "cigarettes_per_day * smoking_duration / 20; no cap and no exponential intensity adjustment",
        "not_stated_treatment": "Retained as a separate category; never redistributed",
        "fallback_hierarchy": ["sex x five-year age x status", "sex x broad age x status", "sex x status", "status"],
        "sample_size_suppression_rule": f"Do not export history summaries with n < {MINIMUM_STRATUM_N}",
        "history_cleaning_report": history_report,
        "population_grouped_validation": population_validation,
        "assumptions": ["Eurobarometer status does not support daily/occasional separation and is labelled current_unspecified",
                        "Equal weights are used only because an appropriate Irish weight could not be identified confidently"],
        "limitations": ["No Eurobarometer donor microdata are redistributed because repository evidence does not establish redistribution rights",
                        "Marginal summaries and correlations are calibration inputs, not a fitted joint distribution",
                        "When maximum age is 80, the Census 80-84 smoking band is used only as a source band for synthetic persons aged exactly 80",
                        "The minimal Stata reader is intentionally specific to the numeric release-118 curated extract"],
        "exports": export_details,
        "checksum_note": "source_metadata.json cannot contain its own stable checksum because that is self-referential; all other package files are checksummed.",
    }


def readme(history_report: dict) -> str:
    return f"""# CEA Irish population input export, version {PACKAGE_VERSION}

## Purpose and regeneration

This compact package supplies inputs for a synthetic Irish population aged **{MIN_AGE}–{MAX_AGE} inclusive**. From the repository root run:

```bash
python scripts/export_cea_population_inputs.py
```

The committed package uses the default maximum age of 80. To build a 50–79 package instead, use `python scripts/export_cea_population_inputs.py --maximum-age 79 --output-directory export_for_cea_50_79`. Every population, Census smoking-status, Eurobarometer history, strata, parameter, and metadata export follows the selected range; Lung Health Check validation intentionally remains 55–74.

The script is deterministic, uses repository-relative paths and the Python standard library, recreates the package, validates its inputs, and records SHA-256 checksums. No notebook execution is required.

## Files

* `irish_population_age_sex_2022.csv`: exact-age/sex 2022 counts from the M2 CSO projection series baseline.
* `irish_smoking_status_age_sex_2022.csv`: Census State counts and probabilities, retaining five response categories.
* `smoking_history_strata.csv`: disclosure-safe complete-case Eurobarometer summaries with an explicit fallback level.
* `smoking_history_generation_parameters.json`: constraints, hierarchy and empirical correlations for coherent joint generation.
* `eligibility_model_validation_targets.csv`: Census and ordinary-history targets for ages 55–74.
* `smoking_history_cleaning_audit.csv` and `smoking_history_cleaning_summary.json`: aggregate cleaning counts, thresholds and maxima; neither contains respondent records or identifiers.
* `source_metadata.json`: provenance, decisions, limitations, row counts and checksums.

`smoking_history_donors.csv` is intentionally omitted. Although identifiers could be removed, this repository does not document Eurobarometer respondent-level redistribution rights. Do not copy respondent microdata to another repository without confirming the applicable licence. The disclosure-safe strata and generation parameters may be copied.

## Population and validation coverage

The population export contains every exact age from {MIN_AGE} through {MAX_AGE} for both sexes. When the maximum is 80, `80-84` is retained in the smoking-status export only because it is the source band for synthetic people aged exactly 80; the CEA must not create ages 81–84 from it. A maximum of 79 exports complete five-year source bands only. Lung Health Check validation remains restricted to 55–74 for comparability with the pilot and earlier eligibility work.

The population source is the **2022 baseline of a Census-2022-based CSO projection series under M2**, not a direct Census single-year-age table. Exact-age totals are checked against the processed five-year source totals for complete exported bands.

## Smoking status

Daily and occasional smoking remain distinct in the Census export. `not_stated` remains a fifth category, is included in `probability_all_people`, is excluded from the known-status denominator, and is never silently redistributed. The Eurobarometer extract does not support a reliable daily/occasional distinction, so current history records are labelled `current_unspecified`.

## Smoking history

The authoritative history source is `data_raw/eurobarometer.dta`, the repository's curated Irish extract. Wave-2017 respondents aged {MIN_AGE}–{MAX_AGE} were screened, yielding {history_report['candidate_records']} candidates and {history_report['retained_records']} valid histories. Records failing sex/status, finite-value, initiation (at least 5 and before attained age), intensity (greater than 0 and at most 80), former-smoker stopping-age, chronology, or 200-pack-year rules are excluded unchanged with one documented reason—never imputed, clipped, or winsorised. This is cleaning for aggregate calibration only, not complete-history sampling.

The appropriate Irish national survey-weight variable could not be identified confidently: the original file has multiple weight labels, the curated extract has no weight, and no codebook/mapping is stored here. Weights are therefore set conceptually to 1 and every summary is labelled `unweighted_equal_weights`; no weight is guessed.

Generate smoking variables jointly and enforce the JSON constraints. The exported marginal summaries do **not** by themselves constitute a joint model. Empirical correlations are supplied as calibration targets, and the fallback sequence is five-year/sex/status, broad-age/sex/status, sex/status, then status.

## Pack-years and exclusions

Ordinary, uncapped pack-years are:

```text
cigarettes_per_day * smoking_duration / 20
```

The earlier exponential intensity adjustment (constant 0.1) and 60-pack-year cap are removed. The package also excludes lognormal eligibility calculations, Markov projections, future attenuation, quit-window eligibility exclusions, and undocumented cross-sex/age substitutions.

## Limitations

The Irish Eurobarometer sample is modest, complete-case and unweighted. Small cells (`n < {MINIMUM_STRATUM_N}`) are suppressed and consumers must follow the fallback hierarchy. The package provides empirical calibration inputs and constraints, not a validated causal or multivariate risk model. Confirm data licensing and the target model's handling of `not_stated` before use.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-age", type=int, choices=(79, 80), default=80,
                        help="inclusive upper age for all population inputs (default: 80)")
    parser.add_argument("--output-directory", type=Path, default=Path("export_for_cea"),
                        help="repository-relative output directory (default: export_for_cea)")
    return parser.parse_args()


def main() -> None:
    global OUT, MAX_AGE
    args = parse_args()
    MAX_AGE = args.maximum_age
    if args.output_directory.is_absolute():
        raise ValueError("--output-directory must be repository-relative")
    OUT = (ROOT / args.output_directory).resolve()
    if ROOT.resolve() not in OUT.parents:
        raise ValueError("--output-directory must remain inside the repository")
    require_sources()
    OUT.mkdir(exist_ok=True)
    # Remove stale managed donor output: policy deliberately omits respondent-level export.
    (OUT / "smoking_history_donors.csv").unlink(missing_ok=True)
    population, population_validation = population_export()
    smoking = smoking_status_export()
    donors, history_report = clean_histories()
    strata = strata_export(donors)
    parameters = generation_parameters(donors)
    validation = validation_targets(smoking, donors)

    write_csv(OUT / "irish_population_age_sex_2022.csv", list(population[0]), population)
    write_csv(OUT / "irish_smoking_status_age_sex_2022.csv", list(smoking[0]), smoking)
    write_csv(OUT / "smoking_history_strata.csv", list(strata[0]), strata)
    write_csv(OUT / "eligibility_model_validation_targets.csv", list(validation[0]), validation)
    (OUT / "smoking_history_generation_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_rows = ([{"metric": "candidate_records", "reason": "", "value": history_report["candidate_records"]},
                   {"metric": "retained_records", "reason": "", "value": history_report["retained_records"]},
                   {"metric": "excluded_records", "reason": "", "value": history_report["excluded_records"]}]
                  + [{"metric": "excluded_records_by_reason", "reason": reason, "value": count}
                     for reason, count in history_report["excluded_records_by_reason"].items()]
                  + [{"metric": key, "reason": "", "value": history_report[key]} for key in (
                      "records_with_initiation_age_below_10", "records_with_cigarettes_per_day_above_60",
                      "records_with_pack_years_above_100")]
                  + [{"metric": f"{stage}_{variable}", "reason": "", "value": value}
                     for stage in ("pre_cleaning_maxima", "post_cleaning_maxima")
                     for variable, value in history_report[stage].items()])
    write_csv(OUT / "smoking_history_cleaning_audit.csv", ["metric", "reason", "value"], audit_rows)
    (OUT / "smoking_history_cleaning_summary.json").write_text(
        json.dumps(history_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(readme(history_report), encoding="utf-8")

    source_map = {
        "irish_population_age_sex_2022.csv": ["data_raw/projections2057_raw.csv", "data_processed/projections2057.csv"],
        "irish_smoking_status_age_sex_2022.csv": ["data_raw/restructured_smoking_population_data.csv"],
        "smoking_history_strata.csv": ["data_raw/eurobarometer.dta"],
        "smoking_history_generation_parameters.json": ["data_raw/eurobarometer.dta"],
        "smoking_history_cleaning_audit.csv": ["data_raw/eurobarometer.dta"],
        "smoking_history_cleaning_summary.json": ["data_raw/eurobarometer.dta"],
        "eligibility_model_validation_targets.csv": ["data_raw/restructured_smoking_population_data.csv", "data_raw/eurobarometer.dta"],
        "README.md": [],
    }
    details = {}
    for filename, sources in source_map.items():
        path = OUT / filename
        row_count = None
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
        details[filename] = {"original_source_files": sources, "row_count": row_count,
                             "sha256": checksum(path)}
    (OUT / "source_metadata.json").write_text(
        json.dumps(metadata(details, history_report, population_validation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print(f"CEA export {PACKAGE_VERSION}: {len(population)} population rows, {len(smoking)} smoking-status rows")
    print(f"Coverage: exact ages {MIN_AGE}-{MAX_AGE} inclusive; Census bands "
          f"{', '.join(sorted({row['age_group'] for row in smoking}))}")
    print(f"Eurobarometer: {history_report['candidate_records']} candidates, "
          f"{len(donors)} valid complete cases; {len(strata)} disclosure-safe summary rows")
    print(f"Validation: {len(population_validation)} five-year population cells agree; {len(validation)} targets")
    print("Survey weighting: unweighted (appropriate Irish national weight not confidently identifiable)")


if __name__ == "__main__":
    main()
