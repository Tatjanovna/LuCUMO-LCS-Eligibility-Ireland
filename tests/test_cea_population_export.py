import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
OUT = ROOT / "export_for_cea"
BUILD_SCRIPT = ROOT / "scripts" / "build_smooth_smoking_history_outputs.py"
EXPORT_SCRIPT = ROOT / "scripts" / "export_cea_population_inputs.py"

EXPECTED_EXPORT_FILES = {
    "README.md",
    "irish_population_age_sex_2022.csv",
    "irish_smoking_status_age_sex_2022.csv",
    "plco_smoking_history_synthetic_pool.csv",
    "smoking_history_generation_parameters.json",
    "source_metadata.json",
}
INTERNAL_QA_FILES = {
    "smoking_history_model_diagnostics.csv",
    "smoking_history_cleaning_audit.csv",
    "smoking_history_residual_donor_audit.csv",
}
OBSOLETE_FILES = {
    "smoking_history_strata.csv",
    "eligibility_model_validation_targets.csv",
    "plco_smoking_history_parameters.csv",
    "plco_smoking_history_correlations.csv",
    "plco_smoking_history_validation_targets.csv",
    "smoking_history_model_specification.json",
}
AGE_GROUPS = {"50-54", "55-59", "60-64", "65-69", "70-74", "75-79"}


def run(path: Path) -> None:
    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def rows(filename: str, directory: Path = OUT) -> list[dict[str, str]]:
    with (directory / filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_module() -> None:
    run(BUILD_SCRIPT)
    run(EXPORT_SCRIPT)


def test_export_contains_only_runtime_files() -> None:
    actual = {path.name for path in OUT.iterdir() if path.is_file()}
    assert actual == EXPECTED_EXPORT_FILES
    assert OBSOLETE_FILES.isdisjoint(actual)
    assert INTERNAL_QA_FILES.isdisjoint(actual)


def test_obsolete_scripts_and_processed_compatibility_outputs_are_removed() -> None:
    assert not (ROOT / "scripts" / "build_plco_smoking_history_outputs.py").exists()
    assert not (ROOT / "scripts" / "export_cea_population_inputs_v4.py").exists()
    for filename in {
        "plco_smoking_history_parameters.csv",
        "plco_smoking_history_correlations.csv",
        "plco_smoking_history_validation_targets.csv",
        "smoking_history_model_specification.json",
    }:
        assert not (PROCESSED / filename).exists()


def test_census_population_and_status_are_preserved() -> None:
    population = rows("irish_population_age_sex_2022.csv")
    assert {int(row["age"]) for row in population} == set(range(50, 80))
    by_age: defaultdict[int, set[str]] = defaultdict(set)
    for row in population:
        by_age[int(row["age"])].add(row["sex"])
        assert int(row["population_count"]) >= 0
    assert all(sexes == {"female", "male"} for sexes in by_age.values())

    smoking_status = rows("irish_smoking_status_age_sex_2022.csv")
    cells: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in smoking_status:
        cells[(row["age_group"], row["sex"])].append(row)
        assert row["source_dataset"] == "data_processed/cleaned_smoking_data_ag.csv"
    assert {age_group for age_group, _ in cells} == AGE_GROUPS
    assert {sex for _, sex in cells} == {"female", "male"}
    for cell in cells.values():
        assert {row["smoking_status"] for row in cell} == {
            "current",
            "former",
            "never",
        }
        assert abs(
            sum(float(row["assignment_probability"]) for row in cell) - 1.0
        ) < 2e-11


def test_synthetic_pool_exact_coverage_and_constraints() -> None:
    pool = rows("plco_smoking_history_synthetic_pool.csv")
    counts: defaultdict[tuple[int, str, str], int] = defaultdict(int)

    for row in pool:
        age = int(row["attained_age"])
        sex = row["sex"]
        status = row["smoking_status"]
        counts[(age, sex, status)] += 1

        initiation = float(row["age_at_initiation"])
        cigarettes = float(row["cigarettes_per_day"])
        duration = float(row["smoking_duration"])
        years_since_quitting = float(row["years_since_quitting"])
        standard_pack_years = float(row["standard_pack_years"])
        adjusted_pack_years = float(row["adjusted_pack_years"])

        assert 50 <= age <= 79
        assert sex in {"female", "male"}
        assert status in {"current", "former"}
        assert 5 <= initiation < age
        assert 0 < cigarettes <= 80 + 1e-9
        assert duration > 0
        assert standard_pack_years <= 120 + 1e-8
        assert adjusted_pack_years <= 60 + 1e-8
        assert math.isclose(
            standard_pack_years,
            duration * cigarettes / 20,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

        if status == "current":
            assert row["age_at_stopping"] == ""
            assert math.isclose(duration, age - initiation, abs_tol=1e-9)
            assert years_since_quitting == 0
        else:
            stopping = float(row["age_at_stopping"])
            assert initiation < stopping <= age
            assert math.isclose(duration, stopping - initiation, abs_tol=1e-9)
            assert math.isclose(years_since_quitting, age - stopping, abs_tol=1e-9)

    expected_cells = {
        (age, sex, status)
        for age in range(50, 80)
        for sex in ("female", "male")
        for status in ("current", "former")
    }
    assert set(counts) == expected_cells
    assert set(counts.values()) == {250}


def test_generation_parameters_define_one_authoritative_history_path() -> None:
    parameters = json.loads(
        (OUT / "smoking_history_generation_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    assert parameters["model_version"] == "smooth_history_v1"
    assert parameters["model_family"] == (
        "non_bayesian_robust_penalised_spline_location_scale"
    )
    assert parameters["source_roles"]["smoking_status_prevalence"].startswith(
        "Irish Census"
    )
    assert parameters["source_roles"]["conditional_smoking_histories"] == (
        "Eurobarometer 2017"
    )
    assert parameters["cleaning"]["global_mean_imputation"] is False
    assignment = parameters["conditional_history_assignment"]
    assert assignment["source"] == "plco_smoking_history_synthetic_pool.csv"
    assert assignment["matching_keys"] == [
        "attained_age",
        "sex",
        "smoking_status",
    ]
    assert assignment["method"] == "sample one complete synthetic predictive row"
    assert parameters["joint_generation"]["method"] == (
        "resample complete standardised residual vectors"
    )
    assert parameters["synthetic_pool"]["draws_per_exact_age_sex_status_cell"] == 250
    assert "compatibility_outputs" not in parameters


def test_internal_model_qa_is_retained_but_not_exported() -> None:
    for filename in INTERNAL_QA_FILES:
        assert (PROCESSED / filename).is_file()
        assert not (OUT / filename).exists()

    cleaning = rows("smoking_history_cleaning_audit.csv", PROCESSED)
    assert "retained_for_modelling" in {row["reason"] for row in cleaning}

    donors = rows("smoking_history_residual_donor_audit.csv", PROCESSED)
    assert {(row["sex"], row["smoking_status"]) for row in donors} == {
        (sex, status)
        for sex in ("female", "male")
        for status in ("current", "former")
    }
    assert all(int(row["n_complete_residual_vectors"]) > 0 for row in donors)

    diagnostics = rows("smoking_history_model_diagnostics.csv", PROCESSED)
    assert {(row["smoking_status"], row["history_variable"]) for row in diagnostics} == {
        ("current", "start_t"),
        ("current", "cigs_t"),
        ("former", "start_t"),
        ("former", "cigs_t"),
        ("former", "quit_t"),
    }


def test_outputs_are_reproducible() -> None:
    processed_files = {
        "plco_smoking_history_synthetic_pool.csv",
        "smoking_history_generation_parameters.json",
        *INTERNAL_QA_FILES,
    }
    before_processed = {
        filename: (PROCESSED / filename).read_bytes()
        for filename in processed_files
    }
    before_export = {
        path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()
    }

    run(BUILD_SCRIPT)
    run(EXPORT_SCRIPT)

    after_processed = {
        filename: (PROCESSED / filename).read_bytes()
        for filename in processed_files
    }
    after_export = {
        path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()
    }
    assert before_processed == after_processed
    assert before_export == after_export


def test_metadata_checksums_and_no_identifiers() -> None:
    forbidden = {
        "uniqid",
        "respondent_id",
        "source_row",
        "survey",
        "name",
        "email",
        "address",
        "phone",
    }
    for filename in EXPECTED_EXPORT_FILES:
        path = OUT / filename
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                fields = set(csv.DictReader(handle).fieldnames or [])
            assert forbidden.isdisjoint(fields)

    metadata = json.loads((OUT / "source_metadata.json").read_text(encoding="utf-8"))
    assert metadata["package_version"] == "4.1.0"
    assert metadata["preferred_history_input"] == (
        "plco_smoking_history_synthetic_pool.csv"
    )
    assert set(metadata["runtime_files"]) == EXPECTED_EXPORT_FILES
    assert metadata["synthetic_pool_rows"] == 30000
    assert set(metadata["exports"]) == EXPECTED_EXPORT_FILES - {"source_metadata.json"}

    commit = metadata["source_commit_sha"]
    assert len(commit) == 40
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    for filename, details in metadata["exports"].items():
        assert details["sha256"] == hashlib.sha256(
            (OUT / filename).read_bytes()
        ).hexdigest()
