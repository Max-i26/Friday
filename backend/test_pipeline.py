import os
import pandas as pd
import numpy as np
from backend.ml_pipeline import MLPipeline

def test_pipeline():
    print("Running Pipeline Tests...")
    
    # 1. Create a dummy dataset
    data = {
        "Age": [25, 30, np.nan, 45, 50, 35, 22, 60, np.nan, 28],
        "Salary": [50000, 60000, 75000, 90000, np.nan, 65000, 48000, 110000, 52000, np.nan],
        "Department": ["HR", "IT", "IT", "HR", "Sales", "Sales", "IT", "Sales", "HR", "IT"],
        "Performance_Score": [3, 4, 2, 5, 4, 3, 2, 5, 3, 4]
    }
    df = pd.DataFrame(data)
    test_csv = "test_dataset.csv"
    df.to_csv(test_csv, index=False)
    
    print("[+] Created test_dataset.csv")
    
    try:
        # 2. Test profiling
        print("[*] Profiling dataset...")
        profile = MLPipeline.load_and_profile(test_csv)
        assert profile["num_rows"] == 10
        assert profile["num_cols"] == 4
        print("[+] Profiling metrics computed successfully!")
        
        # 3. Test cleaning
        print("[*] Cleaning dataset...")
        clean_csv = "test_clean_dataset.csv"
        options = {
            "drop_columns": [],
            "imputation": {"numeric": "mean", "categorical": "most_frequent"},
            "scaling": {"strategy": "standard", "columns": ["Age", "Salary"]},
            "encoding": {"strategy": "label", "columns": ["Department"]}
        }
        clean_profile = MLPipeline.clean_dataset(test_csv, clean_csv, options)
        assert clean_profile["num_rows"] == 10
        assert clean_profile["num_cols"] == 4
        # Verify null values are imputed
        for col in clean_profile["columns"]:
            assert col["missing_count"] == 0
        print("[+] Cleaning completed successfully!")
        
        # 4. Test model training (Regression)
        print("[*] Training Regression models...")
        config = {
            "target_column": "Salary",
            "feature_columns": ["Age", "Department", "Performance_Score"],
            "model_type": "linear",
            "task_type": "regression",
            "test_size": 0.2,
            "random_state": 42
        }
        reg_results = MLPipeline.train_and_evaluate(clean_csv, config)
        assert "train_metrics" in reg_results
        assert "test_metrics" in reg_results
        assert "r2" in reg_results["test_metrics"]
        print("[+] Regression model trained & evaluated successfully!")

        # 5. Test model training (Classification)
        print("[*] Training Classification models...")
        config_clf = {
            "target_column": "Department",
            "feature_columns": ["Age", "Salary", "Performance_Score"],
            "model_type": "random_forest",
            "task_type": "classification",
            "test_size": 0.2,
            "random_state": 42,
            "hyperparameters": {"n_estimators": 10}
        }
        clf_results = MLPipeline.train_and_evaluate(clean_csv, config_clf)
        assert "train_metrics" in clf_results
        assert "test_metrics" in clf_results
        assert "accuracy" in clf_results["test_metrics"]
        print("[+] Classification model trained & evaluated successfully!")
        
        print("\n[===] ALL ML PIPELINE TESTS PASSED SUCCESSFULLY! [===]")
        
    finally:
        # Clean up temporary files
        if os.path.exists(test_csv):
            os.remove(test_csv)
        if os.path.exists(clean_csv):
            os.remove(clean_csv)

if __name__ == "__main__":
    test_pipeline()
