
def main():
    # -----------------------------
    # 1) EDA
    # -----------------------------
    print("\n[EDA] Running basic EDA...")
    from src.eda_basic import run_eda
    run_eda()

    print("\n[EDA] Running descriptive EDA...")
    from src.eda_descriptive import main as run_eda_descriptive
    run_eda_descriptive()

    # -----------------------------
    # 2) Models
    # -----------------------------
    print("\n[Logistic] Running logistic comparison (simple vs scaled)...")
    from src.models.logistic_compare import main as run_logistic_compare
    run_logistic_compare()

    print("\n[VIF] Checking multicollinearity...")
    from src.models.check_vif import main as run_vif
    run_vif()

    print("\n[Random Forest] Running baseline model...")
    from src.models.random_forest_baseline import evaluate_rf
    evaluate_rf()

    print("\n[XGBoost] Running baseline model...")
    from src.models.xgboost_baseline import main as run_xgb
    run_xgb()

    # -----------------------------
    # 3) RQ3 lift test (optional)
    # -----------------------------
    # Only works if you saved your RQ3 script as a file in src/models/
    try:
        print("\n[RQ3] Running lift test (IMDb-only vs IMDb+Trends)...")
        from src.models.rq3_test import main as run_rq3
        run_rq3()
    except ModuleNotFoundError:
        print("\n[RQ3] Skipped: src/models/rq3_test.py not found.")

    print("\n✅ Done. Check results/ for outputs.")


if __name__ == "__main__":
    main()
