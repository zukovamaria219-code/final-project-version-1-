
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
    print("\n[Logistic] Running simple logistic regression baseline (IMDb+Trends, time split)...")
    from src.models.logistic_baseline import main as run_logistic_baseline
    run_logistic_baseline()
    
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
    # 3) RQ3 lift test
    # -----------------------------
    
    try:
        print("\n[RQ3] Running lift test (IMDb-only vs IMDb+Trends)...")
        from src.models.rq3_test import main as run_rq3
        run_rq3()
    except ModuleNotFoundError:
        print("\n[RQ3] Skipped: src/models/rq3_test.py not found.")

    print("\n✅ Done. Check results/ for outputs.")

    # -----------------------------
    # 4) Part 2 models (IMDb + metadata; larger sample)
    # -----------------------------
    try:
        print("\n[Part 2 Logistic] Running logistic regression (combined features)...")
        from src.models.logistic_part2_combined import main as run_logistic_part2
        run_logistic_part2(random_state=42)
    except ModuleNotFoundError:
        print("\n[Part 2 Logistic] Skipped: src/models/logistic_part2_combined.py not found.")
    except FileNotFoundError as e:
        print(f"\n[Part 2 Logistic] Skipped: missing data file -> {e}")

    # 5) PART 2: Random Forest
    try:
        print("\n[Part 2 Random Forest] Running Random Forest (combined features)...")
        from src.models.random_forest_part2 import run as run_rf_part2
        run_rf_part2(random_state=42)
    except ModuleNotFoundError:
        print("\n[Part 2 Random Forest] Skipped: src/models/random_forest_part2.py not found.")
    except FileNotFoundError as e:
        print(f"\n[Part 2 Random Forest] Skipped: missing data file -> {e}")

    # 6) PART 2: XGBoost
    try:
        print("\n[Part 2 XGBoost] Running XGBoost (combined features)...")
        from src.models.xgboost_part2 import run as run_xgb_part2
        run_xgb_part2(random_state=42)
    except ModuleNotFoundError:
        print("\n[Part 2 XGBoost] Skipped: src/models/xgboost_part2.py not found.")
    except FileNotFoundError as e:
        print(f"\n[Part 2 XGBoost] Skipped: missing data file -> {e}")

if __name__ == "__main__":
    main()
