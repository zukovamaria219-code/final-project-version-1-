# tests/test_imports.py
import importlib

MODULES = [
    "src.eda_basic",
    "src.eda_descriptive",
    "src.models.logistic_baseline",
    "src.models.logistic_compare",
    "src.models.check_vif",
    "src.models.random_forest_baseline",
    "src.models.xgboost_baseline",
    "src.models.rq3_test",
    "src.models.logistic_part2_combined",
    "src.models.random_forest_part2",
    "src.models.xgboost_part2",
]

def test_can_import_modules():
    for m in MODULES:
        importlib.import_module(m)
