import requests
import json
import os

def test_api():
    print("Connecting to local OmniML Brain API at http://127.0.0.1:8000...")
    
    url_upload = "http://127.0.0.1:8000/api/upload"
    url_clean = "http://127.0.0.1:8000/api/clean"
    url_train = "http://127.0.0.1:8000/api/train"
    
    # Bypass proxy for local connection
    no_proxies = {"http": None, "https": None}
    
    # 1. Upload iris.csv
    file_path = os.path.join("DataSets", "iris.csv")
    print(f"[*] Uploading {file_path}...")
    with open(file_path, "rb") as f:
        files = {"file": ("iris.csv", f, "text/csv")}
        res = requests.post(url_upload, files=files, proxies=no_proxies)
        
    if res.status_code != 200:
        print(f"[-] Upload failed: {res.text}")
        return
        
    data = res.json()
    dataset_id = data["dataset_id"]
    print(f"[+] Uploaded successfully! Dataset ID: {dataset_id}")
    print(f"    Rows: {data['profile']['num_rows']}, Columns: {data['profile']['num_cols']}")
    
    # 2. Clean dataset
    print("[*] Testing data cleaning endpoint...")
    clean_payload = {
        "dataset_id": dataset_id,
        "options": {
            "drop_columns": [],
            "imputation": {"numeric": "mean", "categorical": "most_frequent"},
            "scaling": {"strategy": "standard", "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"]},
            "encoding": {"strategy": "label", "columns": ["species"]}
        }
    }
    res_clean = requests.post(url_clean, json=clean_payload, proxies=no_proxies)
    if res_clean.status_code != 200:
        print(f"[-] Cleaning failed: {res_clean.text}")
        return
        
    clean_data = res_clean.json()
    print("[+] Dataset cleaned successfully!")
    
    # 3. Train Decision Tree model
    print("[*] Testing model training endpoint (Decision Tree Classifier on 'species')...")
    train_payload = {
        "dataset_id": dataset_id,
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "model_type": "decision_tree",
        "task_type": "classification",
        "test_size": 0.2,
        "random_state": 42,
        "hyperparameters": {
            "max_depth": 3
        }
    }
    
    res_train = requests.post(url_train, json=train_payload, proxies=no_proxies)
    if res_train.status_code != 200:
        print(f"[-] Training failed: {res_train.text}")
        return
        
    train_data = res_train.json()
    print("[+] Model trained successfully!")
    print(f"    Train Accuracy: {train_data['train_metrics']['accuracy']:.4f}")
    print(f"    Test Accuracy: {train_data['test_metrics']['accuracy']:.4f}")
    print("    Confusion Matrix classes:", train_data["visualization_data"]["confusion_matrix"]["classes"])
    print("    Confusion Matrix matrix:", train_data["visualization_data"]["confusion_matrix"]["matrix"])
    
    print("\n[===] ALL API FLOW TESTS PASSED SUCCESSFULLY! SERVER RUNNING CORRECTLY! [===]")

if __name__ == "__main__":
    test_api()
