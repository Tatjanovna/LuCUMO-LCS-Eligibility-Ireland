import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "export_for_cea"
SCRIPT = ROOT / "scripts" / "export_cea_population_inputs.py"


def run_export():
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True, capture_output=True, text=True)


def rows(name):
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_module():
    run_export()


def test_exact_population_age_sex_coverage_and_counts():
    data = rows("irish_population_age_sex_2022.csv")
    assert {int(row["age"]) for row in data} == set(range(50, 81))
    by_age = defaultdict(set)
    for row in data:
        by_age[int(row["age"])].add(row["sex"])
        assert math.isfinite(float(row["population_count"]))
        assert int(row["population_count"]) >= 0
    assert all(sexes == {"female", "male"} for sexes in by_age.values())


def test_optional_50_to_79_export_covers_all_and_only_intersecting_bands():
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        output = Path(temporary).relative_to(ROOT)
        subprocess.run([sys.executable, str(SCRIPT), "--maximum-age", "79",
                        "--output-directory", str(output)], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        with (ROOT / output / "irish_population_age_sex_2022.csv").open(newline="") as handle:
            population = list(csv.DictReader(handle))
        with (ROOT / output / "irish_smoking_status_age_sex_2022.csv").open(newline="") as handle:
            smoking = list(csv.DictReader(handle))
        assert {int(row["age"]) for row in population} == set(range(50, 80))
        assert {row["age_group"] for row in smoking} == {
            "50-54", "55-59", "60-64", "65-69", "70-74", "75-79"
        }
        parameters = json.loads((ROOT / output / "smoking_history_generation_parameters.json").read_text())
        assert parameters["population_age_range"] == {"minimum": 50, "maximum": 79, "inclusive": True}


def test_smoking_status_categories_and_probabilities():
    data = rows("irish_smoking_status_age_sex_2022.csv")
    assert {row["age_group"] for row in data} == {
        "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84"
    }
    groups = defaultdict(list)
    for row in data:
        groups[(row["age_group"], row["sex"])].append(row)
    expected = {"current_daily", "current_occasional", "former", "never", "not_stated"}
    assert groups
    for group in groups.values():
        assert {row["smoking_status"] for row in group} == expected
        assert abs(sum(float(row["probability_all_people"]) for row in group) - 1) < 1e-9
        known = [row for row in group if row["smoking_status"] != "not_stated"]
        assert abs(sum(float(row["probability_known_status"]) for row in known) - 1) < 1e-9
        assert next(row for row in group if row["smoking_status"] == "not_stated")["probability_known_status"] == ""


def test_no_donor_microdata_or_identifiers_are_exported():
    assert not (OUT / "smoking_history_donors.csv").exists()
    forbidden = {"respondent_id", "uniqid", "name"}
    for path in OUT.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            assert forbidden.isdisjoint(set(csv.DictReader(handle).fieldnames or []))


def test_clean_history_logic_and_standard_pack_years():
    # Import the deterministic cleaner: private donors remain in memory and are never exported.
    sys.path.insert(0, str(SCRIPT.parent))
    import export_cea_population_inputs as exporter
    donors, report = exporter.clean_histories()
    assert donors
    for row in donors:
        assert row["sex"] in {"female", "male"}
        assert row["smoking_status"] in {"current_unspecified", "former"}
        assert 5 <= row["age_at_initiation"] < row["attained_age"]
        assert 0 < row["cigarettes_per_day"] <= 80
        assert row["years_since_quitting"] >= 0
        if row["smoking_status"] == "former":
            assert row["age_at_initiation"] < row["age_at_stopping"] <= row["attained_age"]
            assert row["smoking_duration"] == row["age_at_stopping"] - row["age_at_initiation"]
        else:
            assert row["age_at_stopping"] is None
            assert row["years_since_quitting"] == 0
            assert row["smoking_duration"] == row["attained_age"] - row["age_at_initiation"]
        assert math.isclose(row["pack_years_standard"], row["cigarettes_per_day"] * row["smoking_duration"] / 20)
        assert row["pack_years_standard"] <= 200
        assert all(math.isfinite(float(value)) for value in row.values()
                   if isinstance(value, (int, float)))
    assert report["candidate_records"] == report["retained_records"] + report["excluded_records"]
    assert report["excluded_records"] == sum(report["excluded_records_by_reason"].values())


def test_aggregate_cleaning_audit_outputs_exist_and_reconcile():
    audit = rows("smoking_history_cleaning_audit.csv")
    summary = json.loads((OUT / "smoking_history_cleaning_summary.json").read_text())
    assert audit
    assert summary["authoritative_source"] == "data_raw/eurobarometer.dta"
    assert summary["candidate_records"] == summary["retained_records"] + summary["excluded_records"]
    assert summary["excluded_records"] == sum(summary["excluded_records_by_reason"].values())
    audit_reasons = {row["reason"]: int(row["value"]) for row in audit
                     if row["metric"] == "excluded_records_by_reason"}
    assert audit_reasons == summary["excluded_records_by_reason"]
    assert set(summary["pre_cleaning_maxima"]) == set(summary["post_cleaning_maxima"])


def test_outputs_are_reproducible():
    before = {path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()}
    run_export()
    after = {path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()}
    assert before == after


def test_metadata_commit_and_checksums():
    metadata = json.loads((OUT / "source_metadata.json").read_text(encoding="utf-8"))
    commit = metadata["source_commit_sha"]
    assert len(commit) == 40
    subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, check=True)
    for filename, details in metadata["exports"].items():
        digest = hashlib.sha256((OUT / filename).read_bytes()).hexdigest()
        assert details["sha256"] == digest
