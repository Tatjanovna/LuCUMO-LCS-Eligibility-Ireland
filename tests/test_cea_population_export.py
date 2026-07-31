import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "export_for_cea"
EXPORT_SCRIPT = ROOT / "scripts" / "export_cea_population_inputs.py"
BUILD_SCRIPT = ROOT / "scripts" / "build_plco_smoking_history_outputs.py"
AGE_GROUPS = {"50-54", "55-59", "60-64", "65-69", "70-74", "75-79"}
PLCO_FILES = {
    "plco_smoking_history_parameters.csv",
    "plco_smoking_history_correlations.csv",
    "plco_smoking_history_validation_targets.csv",
}


def run_build():
    subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def run_export():
    subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def rows(name, directory=OUT):
    with (directory / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_module():
    run_build()
    run_export()


def test_exact_population_age_sex_coverage_and_counts():
    data = rows("irish_population_age_sex_2022.csv")
    assert {int(row["age"]) for row in data} == set(range(50, 80))
    assert all(int(row["age"]) != 80 for row in data)
    by_age = defaultdict(set)
    for row in data:
        by_age[int(row["age"])].add(row["sex"])
        assert math.isfinite(float(row["population_count"]))
        assert int(row["population_count"]) >= 0
    assert all(sexes == {"female", "male"} for sexes in by_age.values())


def test_smoking_status_is_exactly_three_categories_and_normalised():
    data = rows("irish_smoking_status_age_sex_2022.csv")
    groups = defaultdict(list)
    for row in data:
        groups[(row["age_group"], row["sex"])].append(row)
        assert row["source_dataset"] == "data_processed/cleaned_smoking_data_ag.csv"
    assert set(age_group for age_group, _ in groups) == AGE_GROUPS
    assert set(sex for _, sex in groups) == {"female", "male"}
    for group in groups.values():
        assert {row["smoking_status"] for row in group} == {"current", "former", "never"}
        assert abs(sum(float(row["assignment_probability"]) for row in group) - 1) < 2e-11
        assert sum(float(row["source_probability"]) for row in group) <= 1


def test_status_values_are_copied_from_processed_output_then_normalised():
    source = {}
    with (ROOT / "data_processed" / "cleaned_smoking_data_ag.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["age_group"] in AGE_GROUPS:
                source[(row["age_group"], row["gender"].lower())] = row
    column = {"current": "smokers", "former": "quitters", "never": "never_smokers"}
    for row in rows("irish_smoking_status_age_sex_2022.csv"):
        original = source[(row["age_group"], row["sex"])]
        expected_source = float(original[column[row["smoking_status"]]])
        denominator = sum(float(original[name]) for name in ("smokers", "quitters", "never_smokers"))
        assert math.isclose(float(row["source_probability"]), expected_source, abs_tol=1e-12)
        assert math.isclose(
            float(row["assignment_probability"]), expected_source / denominator, abs_tol=1e-12
        )


def test_processed_history_estimates_are_copied_exactly_without_pooling():
    data = rows("smoking_history_strata.csv")
    groups = defaultdict(list)
    for row in data:
        groups[(row["age_group"], row["sex"], row["smoking_status"])].append(row)
    for age_group in AGE_GROUPS:
        for sex in ("female", "male"):
            assert {row["history_variant"] for row in groups[(age_group, sex, "current")]} == {"smokers"}
            assert {row["history_variant"] for row in groups[(age_group, sex, "former")]} == {
                "quitters_all", "quitters_excl_10", "quitters_excl_15"
            }
            never = groups[(age_group, sex, "never")]
            assert len(never) == 1
            assert float(never[0]["mean_pack_years"]) == 0
    parameters = json.loads((OUT / "smoking_history_generation_parameters.json").read_text())
    assert parameters["adjusted_pack_year_assignment"]["pooling_or_fallback"] is None
    assert parameters["plcom2012_history_assignment"]["pooling_or_fallback"] is None
    assert parameters["plcom2012_history_assignment"]["joint_method"].startswith("Gaussian copula")


def test_plco_parameters_cover_exact_age_sex_status_cells_and_predictors():
    data = rows("plco_smoking_history_parameters.csv")
    groups = defaultdict(set)
    for row in data:
        assert row["age_group"] in AGE_GROUPS
        assert row["sex"] in {"female", "male"}
        assert row["smoking_status"] in {"current", "former"}
        assert int(row["n"]) > 0
        groups[(row["age_group"], row["sex"], row["smoking_status"])].add(
            row["history_variable"]
        )
        for field in ("mean", "p05", "p25", "p50", "p75", "p95", "minimum", "maximum"):
            assert math.isfinite(float(row[field]))
    for age_group in AGE_GROUPS:
        for sex in ("female", "male"):
            assert {
                "age_at_initiation", "cigarettes_per_day", "smoking_duration",
                "years_since_quitting", "standard_pack_years", "adjusted_pack_years",
            }.issubset(groups[(age_group, sex, "current")])
            assert {
                "age_at_initiation", "age_at_stopping", "cigarettes_per_day",
                "smoking_duration", "years_since_quitting", "standard_pack_years",
                "adjusted_pack_years",
            }.issubset(groups[(age_group, sex, "former")])
    current_quit = [
        row for row in data
        if row["smoking_status"] == "current" and row["history_variable"] == "years_since_quitting"
    ]
    assert current_quit
    assert all(float(row["mean"]) == 0 and row["distribution_recommendation"] == "structural_zero" for row in current_quit)


def test_adjusted_pack_year_parameters_match_final_processed_file():
    final = {}
    with (ROOT / "data_processed" / "pack_year_dist_cleaned.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            age_group = row["age_group"].strip()
            if age_group in AGE_GROUPS:
                final[(age_group, row["gender"].lower(), "current")] = (
                    row["mean_pack_years_smokers"], row["std_pack_years_smokers"]
                )
                final[(age_group, row["gender"].lower(), "former")] = (
                    row["mean_pack_years_quitters_all"], row["std_pack_years_quitters_all"]
                )
    parameter_rows = rows("plco_smoking_history_parameters.csv")
    for row in parameter_rows:
        if row["history_variable"] != "adjusted_pack_years":
            continue
        expected = final[(row["age_group"], row["sex"], row["smoking_status"])]
        assert row["mean"] == expected[0]
        assert row["sd"] == expected[1]


def test_plco_correlations_are_cell_specific_and_bounded():
    data = rows("plco_smoking_history_correlations.csv")
    cells = {(row["age_group"], row["sex"], row["smoking_status"]) for row in data}
    assert cells == {
        (age_group, sex, status)
        for age_group in AGE_GROUPS
        for sex in ("female", "male")
        for status in ("current", "former")
    }
    for row in data:
        assert int(row["n_complete"]) > 0
        if row["correlation_estimable"] == "true":
            value = float(row["correlation"])
            assert math.isfinite(value)
            assert -1 <= value <= 1
        else:
            assert row["correlation"] == ""
        assert row["joint_generation_method"] == "gaussian_copula_with_empirical_marginals"


def test_plco_processed_files_are_copied_byte_for_byte_to_export():
    for filename in PLCO_FILES:
        assert (ROOT / "data_processed" / filename).read_bytes() == (OUT / filename).read_bytes()


def test_no_respondent_rows_or_identifiers_are_exported():
    forbidden = {"uniqid", "respondent_id", "survey", "name", "email", "address", "phone"}
    for filename in PLCO_FILES:
        with (OUT / filename).open(encoding="utf-8", newline="") as handle:
            fields = set(csv.DictReader(handle).fieldnames or [])
        assert forbidden.isdisjoint(fields)
    export_script = EXPORT_SCRIPT.read_text(encoding="utf-8")
    assert "eurobarometer.dta" not in export_script
    assert "current_daily" not in export_script
    assert "current_occasional" not in export_script
    assert not (OUT / "smoking_history_cleaning_audit.csv").exists()
    assert not (OUT / "smoking_history_cleaning_summary.json").exists()
    assert not (OUT / "smoking_history_donors.csv").exists()


def test_outputs_are_reproducible():
    before_processed = {
        name: (ROOT / "data_processed" / name).read_bytes() for name in PLCO_FILES
    }
    before_export = {path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()}
    run_build()
    run_export()
    after_processed = {
        name: (ROOT / "data_processed" / name).read_bytes() for name in PLCO_FILES
    }
    after_export = {path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()}
    assert before_processed == after_processed
    assert before_export == after_export


def test_metadata_commit_and_checksums():
    metadata = json.loads((OUT / "source_metadata.json").read_text(encoding="utf-8"))
    assert metadata["package_version"] == "3.0.0"
    assert metadata["population_age_range"] == {"minimum": 50, "maximum": 79, "inclusive": True}
    assert metadata["smoking_statuses"] == ["current", "former", "never"]
    for filename in PLCO_FILES:
        assert f"data_processed/{filename}" in metadata["source_files"]
    assert all("eurobarometer.dta" not in source for source in metadata["source_files"])
    commit = metadata["source_commit_sha"]
    assert len(commit) == 40
    subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, check=True)
    for filename, details in metadata["exports"].items():
        digest = hashlib.sha256((OUT / filename).read_bytes()).hexdigest()
        assert details["sha256"] == digest
