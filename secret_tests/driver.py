import importlib.util
import datetime
import os
import pandas as pd
import numpy as np
import contextlib
from io import StringIO

@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(StringIO()):
        yield

def test_student_code(solution_path):
    report_dir = os.path.join(os.path.dirname(__file__), "..", "student_workspace")
    report_path = os.path.join(report_dir, "report.txt")
    os.makedirs(report_dir, exist_ok=True)

    spec = importlib.util.spec_from_file_location("student_module", solution_path)
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)
    Analyzer = student_module.VisitAnalyzer

    report_lines = [f"\n=== VisitAnalyzer Test Run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="]

    test_cases = [
        ("Visible", {
            "id": "TC1",
            "desc": "Creating patient visit DataFrame",
            "func": "create_visit_df",
            "input": [[101, "Cardiology", 45, 1500.0], [102, "Neurology", 30, 1200.0]],
            "expected_cols": ["PatientID", "Department", "Duration", "Charges"]
        }),
        ("Visible", {
            "id": "TC2",
            "desc": "Computing total charges per patient",
            "func": "total_charges_per_patient",
            "input": pd.DataFrame([
                [101, "Cardiology", 45, 1500.0],
                [101, "Cardiology", 50, 1800.0]
            ], columns=["PatientID", "Department", "Duration", "Charges"]),
            "expected": pd.DataFrame([[101, 3300.0]], columns=["PatientID", "Total Charges"])
        }),
        ("Visible", {
            "id": "TC3",
            "desc": "Adding cost per minute column",
            "func": "add_cost_per_minute",
            "input": pd.DataFrame([
                [101, "Cardiology", 45, 1500.0],
                [102, "Neurology", 30, 1200.0]
            ], columns=["PatientID", "Department", "Duration", "Charges"]),
            "expected_values": [33.33, 40.0]
        }),
        ("Visible", {
            "id": "TC4",
            "desc": "Filtering frequent visitors correctly",
            "func": "frequent_visitors",
            "input": (
                pd.DataFrame([
                    [201, "Cardiology", 45, 1500.0],
                    [202, "Cardiology", 50, 1800.0],
                    [201, "Cardiology", 40, 1600.0],
                    [203, "Neurology", 30, 1200.0]
                ], columns=["PatientID", "Department", "Duration", "Charges"]),
                1
            ),
            "expected_ids": [201]
        }),
        ("Visible", {
            "id": "TC5",
            "desc": "Calculating department-level average duration",
            "func": "average_duration_per_department",
            "input": pd.DataFrame([
                [101, "Cardiology", 40, 1500.0],
                [102, "Cardiology", 50, 1800.0],
                [103, "Neurology", 30, 1200.0]
            ], columns=["PatientID", "Department", "Duration", "Charges"]),
            "expected_dict": {"Cardiology": 45.0, "Neurology": 30.0}
        }),
        ("Hidden", {
            "id": "HTC1",
            "desc": "Cleaning nulls and sorting by charges",
            "func": "clean_and_sort_visits",
            "input": pd.DataFrame([
                [101, "Cardiology", None, 1500.0],
                [102, "Neurology", 30.0, 1700.0],
                [103, "Ortho", 35.0, 1400.0],
                [104, "Cardiology", None, 1000.0],
                [105, "Ortho", 32.0, 1300.0],
            ], columns=["PatientID", "Department", "Duration", "Charges"]),
            "expected_len": 3,
            "expected_top_charge": 1700.0
        }),
        ("Hidden", {
            "id": "HTC2",
            "desc": "Handling zero duration edge case in cost/min calc",
            "func": "add_cost_per_minute",
            "input": pd.DataFrame([
                [101, "Cardiology", 0, 1500.0],
                [102, "Neurology", 30, 1200.0]
            ], columns=["PatientID", "Department", "Duration", "Charges"]),
            "check_for_inf_or_nan": True
        }),
        ("Hidden", {
            "id": "HTC3",
            "desc": "Correct filtering logic for visit frequency",
            "func": "frequent_visitors",
            "input": (
                pd.DataFrame([
                    [101, "A", 10, 100],
                    [102, "B", 20, 200],
                    [102, "C", 30, 300],
                    [103, "D", 40, 400]
                ], columns=["PatientID", "Department", "Duration", "Charges"]),
                1
            ),
            "expected_ids": [102]
        }),
    ]

    for section, case in test_cases:
        try:
            analyzer = Analyzer()
            method = getattr(analyzer, case["func"])
            with suppress_output():
                if isinstance(case["input"], tuple):
                    result = method(*case["input"])
                else:
                    result = method(case["input"])

            # Logic for each case
            if "expected_cols" in case:
                assert list(result.columns) == case["expected_cols"]
            elif "expected" in case:
                pd.testing.assert_frame_equal(result.reset_index(drop=True), case["expected"].reset_index(drop=True), check_dtype=False)
            elif "expected_values" in case:
                col = result["CostPerMinute"].round(2).tolist()
                assert np.allclose(col, case["expected_values"])
            elif "expected_ids" in case:
                ids = set(result["PatientID"].tolist())
                assert ids == set(case["expected_ids"])
            elif "expected_dict" in case:
                for dept, expected_avg in case["expected_dict"].items():
                    actual = result[result["Department"] == dept]["Average Duration"].values[0]
                    assert abs(actual - expected_avg) < 0.1
            elif "expected_len" in case:
                assert len(result) == case["expected_len"]
                assert not result.isnull().values.any()
                top_charge = result.sort_values(by="Charges", ascending=False).iloc[0]["Charges"]
                assert top_charge == case["expected_top_charge"]
            elif "check_for_inf_or_nan" in case:
                assert not result["CostPerMinute"].isna().any()
                assert not np.isinf(result["CostPerMinute"]).any()

            msg = f"✅ {case['id']}: {case['desc']} passed"
        except Exception as e:
            msg = f"❌ {case['id']}: {case['desc']} failed | Reason: {str(e)}"
        print(msg)
        report_lines.append(msg)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

if __name__ == "__main__":
    solution_file = os.path.join(os.path.dirname(__file__), "..", "student_workspace", "solution.py")
    test_student_code(solution_file)
