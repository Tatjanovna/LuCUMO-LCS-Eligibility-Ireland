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
SCRIPT = ROOT / "scripts" / "export_cea_population_inputs.py"
AGE_GROUPS = {"50-54", "55-59", "60-64", "65-69", "70-74", "75-79"}


def run_export():
    subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def rows(name):
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_module():
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
        assert {row["smoking_status"] for row in group} == {
            "current", "former", "never"
        }
        assert abs(sum(float(row["assignment_probability"]) for row in group) - 1) < 2e-11
        assert sum(float(row["source_probability"]) for row in group) <= 1
        assert {row["probability_transformation"] for row in group} == {
            "renormalised_across_current_former_never"
        }


def test_status_values_are_copied_from_processed_output_then_normalised():
    source = {}
    with (ROOT / "data_processed" / "cleaned_smoking_data_ag.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["age_group"] in AGE_GROUPS:
                source[(row["age_group"], row["gender"].lower())] = row
    column = {
        "current": "smokers",
        "former": "quitters",
        "never": "never_smokers",
    }
    for row in rows("irish_smoking_status_age_sex_2022.csv"):
        original = source[(row["age_group"], row["sex"])]
        expected_source = float(original[column[row["smoking_status"]]])
        denominator = sum(
            float(original[name]) for name in ("smokers", "quitters", "never_smokers")
        )
        assert math.isclose(
            float(row["source_probability"]), expected_source, abs_tol=1e-12
        )
        assert math.isclose(
            float(row["assignment_probability"]),
            expected_source / denominator,
            abs_tol=1e-12,
        )


def test_processed_history_estimates_are_copied_exactly_without_pooling():
    data = rows("smoking_history_strata.csv")
    groups = defaultdict(list)
    for row in data:
        groups[(row["age_group"], row["sex"], row["smoking_status"])].append(row)
    for age_group in AGE_GROUPS:
        for sex in ("female", "male"):
            assert {
                row["history_variant"]
                for row in groups[(age_group, sex, "current")]
            } == {"smokers"}
            assert {
                row["history_variant"]
                for row in groups[(age_group, sex, "former")]
            } == {"quitters_all", "quitters_excl_10", "quitters_excl_15"}
            never = groups[(age_group, sex, "never")]
            assert len(never) == 1
            assert never[0]["history_variant"] == "structural_zero"
            assert float(never[0]["mean_pack_years"]) == 0
            assert float(never[0]["sd_pack_years"]) == 0
    parameters = json.loads(
        (OUT / "smoking_history_generation_parameters.json").read_text()
    )
    assert parameters["history_assignment"]["pooling_or_fallback"] is None
    assert parameters["history_assignment"]["processed_values_changed_by_exporter"] is False
    assert parameters["history_assignment"]["former_default_variant"] == "quitters_all"


def test_history_rows_match_processed_pack_year_file_exactly():
    source = {}
    with (ROOT / "data_processed" / "pack_year_dist_cleaned.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["age_group"].strip() in AGE_GROUPS:
                source[(row["age_group"].strip(), row["gender"].lower())] = row
    suffix_by_variant = {
        "smokers": "smokers",
        "quitters_all": "quitters_all",
        "quitters_excl_10": "quitters_excl_10",
        "quitters_excl_15": "quitters_excl_15",
    }
    field_map = {
        "mean_log_pack_years": "mean_ln_pack_years_{}",
        "sd_log_pack_years": "std_ln_pack_years_{}",
        "mean_pack_years": "mean_pack_years_{}",
        "sd_pack_years": "std_pack_years_{}",
    }
    for row in rows("smoking_history_strata.csv"):
        if row["smoking_status"] == "never":
            continue
        original = source[(row["age_group"], row["sex"])]
        suffix = suffix_by_variant[row["history_variant"]]
        for exported, template in field_map.items():
            assert row[exported] == original[template.format(suffix)].strip()


def test_no_raw_smoking_sources_or_obsolete_outputs_are_used():
    script = SCRIPT.read_text(encoding="utf-8")
    assert "eurobarometer.dta" not in script
    assert "restructured_smoking_population_data.csv" not in script
    assert "current_daily" not in script
    assert "current_occasional" not in script
    assert "ever_smoker" in script  # explicitly documented as not a category
    assert not (OUT / "smoking_history_cleaning_audit.csv").exists()
    assert not (OUT / "smoking_history_cleaning_summary.json").exists()
    assert not (OUT / "smoking_history_donors.csv").exists()


def test_outputs_are_reproducible():
    before = {path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()}
    run_export()
    after = {path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()}
    assert before == after


def test_metadata_commit_and_checksums():
    metadata = json.loads((OUT / "source_metadata.json").read_text(encoding="utf-8"))
    assert metadata["population_age_range"] == {
        "minimum": 50,
        "maximum": 79,
        "inclusive": True,
    }
    assert metadata["smoking_statuses"] == ["current", "former", "never"]
    assert "data_processed/cleaned_smoking_data_ag.csv" in metadata["source_files"]
    assert "data_processed/pack_year_dist_cleaned.csv" in metadata["source_files"]
    assert all("eurobarometer.dta" not in source for source in metadata["source_files"])
    commit = metadata["source_commit_sha"]
    assert len(commit) == 40
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    for filename, details in metadata["exports"].items():
        digest = hashlib.sha256((OUT / filename).read_bytes()).hexdigest()
        assert details["sha256"] == digest
