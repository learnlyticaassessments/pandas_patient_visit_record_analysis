import importlib.util
import os

report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "report.txt"))
if not os.path.exists(report_path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("")

driver_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "secret_tests", "driver.py"))
solution_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "solution.py"))

spec = importlib.util.spec_from_file_location("driver", driver_path)
driver_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver_module)

if __name__ == "__main__":
    driver_module.test_student_code(solution_path)
