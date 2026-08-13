import os
import uuid
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics as sklearn_metrics
def load_csv(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(filepath, encoding='latin1')
        except Exception:
            return pd.read_csv(filepath, encoding='utf-8-sig')

class MLPipeline:
    @staticmethod
    def load_and_profile(filepath: str) -> dict:
        """
        Loads a CSV or Excel dataset and generates a detailed profile:
        preview, columns, types, missing values, descriptive statistics, and correlation matrix.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            df = load_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")

        # Basic shape
        num_rows, num_cols = df.shape

        # Column profiling
        columns_profile = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = float(null_count / num_rows) * 100 if num_rows > 0 else 0.0
            
            # Infer basic data type
            col_type = "categorical"
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 5 and pd.api.types.is_integer_dtype(df[col]):
                    col_type = "categorical"  # Integer codes are often categorical
                else:
                    col_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_type = "datetime"
            elif df[col].nunique() / num_rows < 0.05 and df[col].dtype == object:
                # low cardinality objects are categorical
                col_type = "categorical"
            else:
                col_type = "text"

            columns_profile.append({
                "name": str(col),
                "type": col_type,
                "raw_type": str(df[col].dtype),
                "missing_count": null_count,
                "missing_pct": round(null_pct, 2),
                "unique_count": int(df[col].nunique())
            })

        # Summary statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        stats_numeric = {}
        if numeric_cols:
            desc = df[numeric_cols].describe()
            for col in numeric_cols:
                # Filter out NaNs for clean JSON serialization
                stats_numeric[col] = {
                    "count": int(desc.loc["count", col]) if "count" in desc.index else 0,
                    "mean": float(desc.loc["mean", col]) if "mean" in desc.index and not pd.isna(desc.loc["mean", col]) else None,
                    "std": float(desc.loc["std", col]) if "std" in desc.index and not pd.isna(desc.loc["std", col]) else None,
                    "min": float(desc.loc["min", col]) if "min" in desc.index and not pd.isna(desc.loc["min", col]) else None,
                    "25%": float(desc.loc["25%", col]) if "25%" in desc.index and not pd.isna(desc.loc["25%", col]) else None,
                    "50%": float(desc.loc["50%", col]) if "50%" in desc.index and not pd.isna(desc.loc["50%", col]) else None,
                    "75%": float(desc.loc["75%", col]) if "75%" in desc.index and not pd.isna(desc.loc["75%", col]) else None,
                    "max": float(desc.loc["max", col]) if "max" in desc.index and not pd.isna(desc.loc["max", col]) else None,
                }

        stats_categorical = {}
        for col in categorical_cols:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                top_val = non_null.mode().iloc[0] if not non_null.mode().empty else None
                top_freq = int(non_null.value_counts().iloc[0]) if len(non_null.value_counts()) > 0 else 0
            else:
                top_val = None
                top_freq = 0
            stats_categorical[col] = {
                "count": int(df[col].count()),
                "unique": int(df[col].nunique()),
                "top": str(top_val) if top_val is not None else None,
                "freq": top_freq
            }

        # Correlation Matrix
        correlation_matrix = {}
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr().fillna(0)
            correlation_matrix = {
                "columns": numeric_cols,
                "matrix": corr_df.values.tolist()
            }

        # Preview: convert first 10 rows to lists/dicts, handling NaNs
        preview_df = df.head(15).copy()
        # Replace NaNs with None so it translates to null in JSON
        preview_df = preview_df.replace({np.nan: None, pd.NA: None})
        preview_data = preview_df.to_dict(orient="records")

        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "columns": columns_profile,
            "stats_numeric": stats_numeric,
            "stats_categorical": stats_categorical,
            "correlation_matrix": correlation_matrix,
            "preview": preview_data
        }

    @staticmethod
    def clean_dataset(filepath: str, out_filepath: str, options: dict) -> dict:
        """
        Cleans the dataset and saves the output to out_filepath.
        Options dictionary:
        {
          "drop_columns": ["col1", "col2"],
          "imputation": {"numeric": "mean"|"median"|"most_frequent"|"drop", "categorical": "most_frequent"|"drop"},
          "scaling": {"strategy": "standard"|"minmax"|"none", "columns": ["col3", "col4"]},
          "encoding": {"strategy": "label"|"onehot"|"none", "columns": ["col5", "col6"]}
        }
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            df = load_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format.")

        # 1. Drop Columns
        drop_cols = options.get("drop_columns", [])
        drop_cols = [c for c in drop_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # 2. Imputation
        impute_opts = options.get("imputation", {})
        num_strategy = impute_opts.get("numeric", "mean")
        cat_strategy = impute_opts.get("categorical", "most_frequent")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Handle numeric imputation
        if num_strategy == "drop":
            df = df.dropna(subset=numeric_cols)
        elif num_strategy in ["mean", "median", "most_frequent"]:
            if numeric_cols:
                imp = SimpleImputer(strategy=num_strategy)
                df[numeric_cols] = imp.fit_transform(df[numeric_cols])

        # Handle categorical imputation
        if cat_strategy == "drop":
            df = df.dropna(subset=categorical_cols)
        elif cat_strategy in ["most_frequent"]:
            if categorical_cols:
                imp = SimpleImputer(strategy=cat_strategy)
                # SimpleImputer returns object array, need to make sure we cast back properly
                df[categorical_cols] = pd.DataFrame(imp.fit_transform(df[categorical_cols]), columns=categorical_cols, index=df.index)

        # 3. Encoding
        encode_opts = options.get("encoding", {})
        enc_strategy = encode_opts.get("strategy", "label")
        enc_cols = encode_opts.get("columns", [])
        enc_cols = [c for c in enc_cols if c in df.columns]

        if enc_strategy == "label" and enc_cols:
            for col in enc_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        elif enc_strategy == "onehot" and enc_cols:
            # One-hot encoding using pandas get_dummies for simplicity
            df = pd.get_dummies(df, columns=enc_cols, drop_first=True)

        # Re-evaluate numeric columns after encoding
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 4. Scaling
        scale_opts = options.get("scaling", {})
        scale_strategy = scale_opts.get("strategy", "none")
        scale_cols = scale_opts.get("columns", [])
        scale_cols = [c for c in scale_cols if c in df.columns and c in numeric_cols]

        if scale_strategy != "none" and scale_cols:
            scaler = StandardScaler() if scale_strategy == "standard" else MinMaxScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])

        # Ensure parent directories exist
        dirname = os.path.dirname(out_filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        # Save clean dataframe
        df.to_csv(out_filepath, index=False)

        # Profile the cleaned dataset to show comparison
        return MLPipeline.load_and_profile(out_filepath)

    @staticmethod
    def train_and_evaluate(clean_filepath: str, config: dict) -> dict:
        """
        Trains selected model and evaluates.
        Config dictionary:
        {
          "target_column": "target",
          "feature_columns": ["col1", "col2"],
          "model_type": "linear" | "polynomial" | "decision_tree" | "random_forest",
          "task_type": "regression" | "classification" | "auto",
          "test_size": 0.2,
          "random_state": 42,
          "hyperparameters": {
            "poly_degree": 2,          # for polynomial
            "max_depth": 5,            # for tree/forest
            "n_estimators": 100        # for forest
          }
        }
        """
        if not os.path.exists(clean_filepath):
            raise FileNotFoundError(f"Cleaned dataset not found: {clean_filepath}")

        df = pd.read_csv(clean_filepath)
        
        target_col = config.get("target_column")
        feature_cols = config.get("feature_columns", [])
        
        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        feature_cols = [c for c in feature_cols if c in df.columns and c != target_col]
        if not feature_cols:
            raise ValueError("No valid feature columns provided.")

        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Drop any leftover rows with missing values in features/target
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[feature_cols]
        y = combined[target_col]

        if len(X) < 10:
            raise ValueError("Too few samples (need at least 10) to train the model after cleaning.")

        # Determine Task Type (Regression or Classification)
        task_type = config.get("task_type", "auto")
        if task_type == "auto":
            # If target is numeric and has many unique values, treat as regression. Otherwise, classification.
            unique_ratio = y.nunique() / len(y)
            if pd.api.types.is_numeric_dtype(y) and (y.nunique() > 10 or unique_ratio > 0.05):
                task_type = "regression"
            else:
                task_type = "classification"

        # Safe guard: if features are non-numeric object types, encode them to avoid crashes
        for col in X.columns:
            if X[col].dtype == object or not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Split data
        test_size = float(config.get("test_size", 0.2))
        random_state = int(config.get("random_state", 42))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model_type = config.get("model_type", "linear")
        params = config.get("hyperparameters", {})

        # Build model based on task and type
        model = None
        
        if task_type == "regression":
            if model_type == "linear":
                model = LinearRegression()
            elif model_type == "polynomial":
                degree = int(params.get("poly_degree", 2))
                model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            elif model_type == "decision_tree":
                max_depth = params.get("max_depth", None)
                max_depth = int(max_depth) if max_depth is not None else None
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
            elif model_type == "random_forest":
                n_estimators = int(params.get("n_estimators", 100))
                max_depth = params.get("max_depth", None)
                max_depth = int(max_depth) if max_depth is not None else None
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            else:
                raise ValueError(f"Unknown regression model type: {model_type}")
        else:  # classification
            # For classification, we need to ensure y is encoded as class labels
            y_encoder = LabelEncoder()
            y_train_encoded = y_encoder.fit_transform(y_train.astype(str))
            y_test_encoded = y_encoder.transform(y_test.astype(str))
            classes = [str(c) for c in y_encoder.classes_]

            if model_type == "linear":
                # Logistic regression acts as linear classification
                model = LogisticRegression(max_iter=1000, random_state=random_state)
            elif model_type == "polynomial":
                degree = int(params.get("poly_degree", 2))
                model = make_pipeline(PolynomialFeatures(degree=degree), LogisticRegression(max_iter=1000, random_state=random_state))
            elif model_type == "decision_tree":
                max_depth = params.get("max_depth", None)
                max_depth = int(max_depth) if max_depth is not None else None
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            elif model_type == "random_forest":
                n_estimators = int(params.get("n_estimators", 100))
                max_depth = params.get("max_depth", None)
                max_depth = int(max_depth) if max_depth is not None else None
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            else:
                raise ValueError(f"Unknown classification model type: {model_type}")

        # Train model
        if task_type == "classification":
            model.fit(X_train, y_train_encoded)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        # Generate evaluation reports
        evaluation_results = {
            "task_type": task_type,
            "model_type": model_type,
            "hyperparameters": params,
            "feature_columns": feature_cols,
            "target_column": target_col,
            "train_metrics": {},
            "test_metrics": {},
            "visualization_data": {}
        }

        # Calculate metrics
        if task_type == "regression":
            # Train metrics
            evaluation_results["train_metrics"] = {
                "r2": float(sklearn_metrics.r2_score(y_train, y_train_pred)),
                "mse": float(sklearn_metrics.mean_squared_error(y_train, y_train_pred)),
                "mae": float(sklearn_metrics.mean_absolute_error(y_train, y_train_pred)),
                "rmse": float(np.sqrt(sklearn_metrics.mean_squared_error(y_train, y_train_pred)))
            }
            # Test metrics
            evaluation_results["test_metrics"] = {
                "r2": float(sklearn_metrics.r2_score(y_test, y_test_pred)),
                "mse": float(sklearn_metrics.mean_squared_error(y_test, y_test_pred)),
                "mae": float(sklearn_metrics.mean_absolute_error(y_test, y_test_pred)),
                "rmse": float(np.sqrt(sklearn_metrics.mean_squared_error(y_test, y_test_pred)))
            }

            # Visualizations data for regression
            # Let's take up to 200 data points for plotting actual vs predicted to avoid freezing the UI
            plot_indices = np.random.choice(len(y_test), min(len(y_test), 200), replace=False)
            actual_sample = y_test.iloc[plot_indices].tolist()
            pred_sample = y_test_pred[plot_indices].tolist()
            residuals = (y_test.iloc[plot_indices] - y_test_pred[plot_indices]).tolist()

            evaluation_results["visualization_data"] = {
                "actual_vs_predicted": {
                    "actual": actual_sample,
                    "predicted": pred_sample
                },
                "residuals": {
                    "predicted": pred_sample,
                    "residuals": residuals
                }
            }

        else: # Classification
            # Train metrics
            evaluation_results["train_metrics"] = {
                "accuracy": float(sklearn_metrics.accuracy_score(y_train_encoded, y_train_pred)),
                "precision": float(sklearn_metrics.precision_score(y_train_encoded, y_train_pred, average="weighted", zero_division=0)),
                "recall": float(sklearn_metrics.recall_score(y_train_encoded, y_train_pred, average="weighted", zero_division=0)),
                "f1": float(sklearn_metrics.f1_score(y_train_encoded, y_train_pred, average="weighted", zero_division=0))
            }
            # Test metrics
            evaluation_results["test_metrics"] = {
                "accuracy": float(sklearn_metrics.accuracy_score(y_test_encoded, y_test_pred)),
                "precision": float(sklearn_metrics.precision_score(y_test_encoded, y_test_pred, average="weighted", zero_division=0)),
                "recall": float(sklearn_metrics.recall_score(y_test_encoded, y_test_pred, average="weighted", zero_division=0)),
                "f1": float(sklearn_metrics.f1_score(y_test_encoded, y_test_pred, average="weighted", zero_division=0))
            }

            # Confusion Matrix
            cm = sklearn_metrics.confusion_matrix(y_test_encoded, y_test_pred)
            evaluation_results["visualization_data"] = {
                "confusion_matrix": {
                    "classes": classes,
                    "matrix": cm.tolist()
                }
            }

        # Feature Importance (if tree or forest)
        # Note: Linear/Logistic regression coefs can be treated similarly
        feature_importance = {}
        
        # Helper to get importances
        if model_type in ["decision_tree", "random_forest"]:
            importances = model.feature_importances_
            feature_importance = {feat: float(imp) for feat, imp in zip(feature_cols, importances)}
        elif model_type == "linear":
            if task_type == "regression":
                coefs = model.coef_
                feature_importance = {feat: float(abs(c)) for feat, c in zip(feature_cols, coefs)}
            else: # logistic regression
                # coef_ is shape (n_classes, n_features) or (1, n_features)
                coefs = np.mean(np.abs(model.coef_), axis=0)
                feature_importance = {feat: float(c) for feat, c in zip(feature_cols, coefs)}
        
        if feature_importance:
            # Sort descending
            sorted_fi = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
            evaluation_results["visualization_data"]["feature_importance"] = {
                "features": [item[0] for item in sorted_fi],
                "scores": [item[1] for item in sorted_fi]
            }

        # Decision Tree specific rules export (educational feature)
        if model_type == "decision_tree":
            # Extract underlying tree model
            tree_model = model
            if hasattr(model, 'steps'): # pipeline
                tree_model = model.steps[-1][1]
            try:
                tree_rules = export_text(tree_model, feature_names=feature_cols, max_depth=4)
                evaluation_results["visualization_data"]["tree_rules"] = tree_rules
            except Exception as e:
                evaluation_results["visualization_data"]["tree_rules"] = f"Could not export tree rules: {str(e)}"

        return evaluation_results
