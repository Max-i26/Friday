import os
import json
import google.generativeai as genai

class GeminiBrain:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_model(self):
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured. Please supply an API key.")
        # Using gemini-3.5-flash as the standard brain
        return genai.GenerativeModel("gemini-3.5-flash")

    def analyze_dataset(self, profile: dict) -> dict:
        """
        Sends the dataset profile to Gemini to understand the columns, overall context,
        and recommend target variables, features, and visualizations.
        """
        if not self.is_configured():
            return self._get_fallback_analysis(profile)

        # Build metadata summary to keep prompt tokens reasonable
        columns_summary = []
        for col in profile.get("columns", []):
            col_name = col["name"]
            col_type = col["type"]
            missing_pct = col["missing_pct"]
            unique_count = col["unique_count"]
            
            # Add stats if numeric
            stats_str = ""
            if col_name in profile.get("stats_numeric", {}):
                s = profile["stats_numeric"][col_name]
                stats_str = f", min={s['min']}, max={s['max']}, mean={s['mean']:.2f}" if s['mean'] is not None else ""
            elif col_name in profile.get("stats_categorical", {}):
                s = profile["stats_categorical"][col_name]
                stats_str = f", unique_vals={s['unique']}, top='{s['top']}'"

            columns_summary.append(
                f"- Column '{col_name}' ({col_type}): missing_pct={missing_pct}%, unique_count={unique_count}{stats_str}"
            )

        prompt = f"""
You are an expert Data Scientist and ML Tutor AI. Analyze this dataset's column profile and metadata.
Dataset stats:
- Number of Rows: {profile.get("num_rows")}
- Number of Columns: {profile.get("num_cols")}

Columns:
{chr(10).join(columns_summary)}

Analyze this dataset and return a JSON object with the following structure:
{{
  "dataset_summary": "A 2-3 sentence overview of what this dataset is about, its context, and target audience.",
  "column_insights": [
    {{
      "name": "column_name",
      "insight": "Brief explanation of what this column represents and its significance in data analysis/ML."
    }}
  ],
  "recommended_targets": [
    {{
      "column": "column_name",
      "type": "regression" or "classification",
      "reason": "Why this column is a good target variable for students to predict, and what kind of problem it is."
    }}
  ],
  "recommended_features": ["col1", "col2", "col3"],
  "recommended_plots": [
    {{
      "type": "scatter" or "line" or "bar" or "box" or "histogram",
      "x": "column_x",
      "y": "column_y",
      "title": "Plot Title",
      "explanation": "Why this visualization is interesting and what pattern a student should look for."
    }}
  ]
}}

Ensure the output is valid JSON. Do not include markdown code block characters (like ```json ... ```) in your output, just return the plain string that can be parsed directly.
"""
        try:
            model = self._get_model()
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            return data
        except Exception as e:
            print(f"Gemini API analysis failed: {e}")
            return self._get_fallback_analysis(profile)

    def chat_query(self, profile: dict, history: list, query: str) -> str:
        """
        Interacts with the user, providing data-science Tutoring or Pandas helper tips.
        History is a list of {"role": "user"|"model", "text": "..."}.
        """
        if not self.is_configured():
            return "Gemini API key is not configured. Please input your Gemini API Key in the settings at the top of the interface to chat with the AI Brain!"

        # Prepare context about the dataset
        columns_brief = []
        for col in profile.get("columns", []):
            columns_brief.append(f"'{col['name']}' ({col['type']})")

        dataset_context = f"""
You are an expert Data Science and ML Brain for students.
You are helping the student analyze a dataset with {profile.get('num_rows')} rows and {profile.get('num_cols')} columns.
The columns are: {", ".join(columns_brief)}.

When answering, be educational, encouraging, and clear. Avoid overly complex jargon without explaining it.
If the student asks for Pandas code, write clean code block snippets using ```python.
If they ask for statistics concepts, explain them intuitively.
"""
        # Create a new chat session
        model = self._get_model()
        
        # Format history for google.generativeai
        # Format roles: user -> user, model -> model
        contents = []
        for h in history:
            contents.append({
                "role": "user" if h["role"] == "user" else "model",
                "parts": [h["text"]]
            })

        # Append system instruction inside a system_instruction or as a preamble
        # Since system_instruction is available in genai.GenerativeModel:
        chat = model.start_chat(history=contents)
        
        full_query = f"{dataset_context}\n\nUser Query: {query}"
        try:
            response = chat.send_message(full_query)
            return response.text
        except Exception as e:
            return f"Error talking to Gemini Brain: {str(e)}"

    def explain_model_results(self, eval_results: dict, profile: dict) -> str:
        """
        Generates an educational summary explaining the machine learning training results,
        metrics, feature importances, and potential improvements (e.g. overfitting).
        """
        if not self.is_configured():
            return "Gemini API key is not configured. Please input your Gemini API Key at the top of the dashboard to generate AI explanation reports!"

        task_type = eval_results.get("task_type")
        model_type = eval_results.get("model_type")
        hyperparams = eval_results.get("hyperparameters", {})
        train_metrics = eval_results.get("train_metrics", {})
        test_metrics = eval_results.get("test_metrics", {})
        feat_cols = eval_results.get("feature_columns", [])
        target_col = eval_results.get("target_column")

        # Format metrics string
        metrics_summary = []
        for k in train_metrics.keys():
            train_val = train_metrics[k]
            test_val = test_metrics[k]
            metrics_summary.append(f"- {k.upper()}: Train = {train_val:.4f}, Test = {test_val:.4f}")

        # Format feature importances if any
        fi_summary = ""
        fi = eval_results.get("visualization_data", {}).get("feature_importance", {})
        if fi:
            feats = fi.get("features", [])
            scores = fi.get("scores", [])
            fi_items = [f"'{f}': {s:.4f}" for f, s in zip(feats[:5], scores[:5])]
            fi_summary = f"Top features: {', '.join(fi_items)}"

        prompt = f"""
You are an ML Tutor. Explain these model training results to a student:
- Task: {task_type.upper()}
- Model: {model_type.upper()}
- Target Variable: '{target_col}'
- Features used: {', '.join(feat_cols)}
- Hyperparameters: {json.dumps(hyperparams)}

Performance Metrics:
{chr(10).join(metrics_summary)}

{fi_summary}

Explain:
1. What this model is doing (in simple terms) and how to interpret these results (R2/Accuracy, etc.).
2. Compare Train vs. Test metrics: Is the model overfitting, underfitting, or performing well? Why?
3. What features are most important according to the results?
4. Concrete steps for the student to improve the model (e.g., adding features, scaling, adjusting hyperparameters like max_depth).

Write in a friendly, encouraging, and clear markdown format. Keep it concise but highly educational (approx. 250-350 words).
"""
        try:
            model = self._get_model()
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating report summary: {str(e)}"

    def _get_fallback_analysis(self, profile: dict) -> dict:
        """
        Fallback analysis when Gemini API key is missing or calls fail.
        Provides basic data-driven recommendations.
        """
        cols = profile.get("columns", [])
        num_cols_list = [c["name"] for c in cols if c["type"] == "numeric"]
        cat_cols_list = [c["name"] for c in cols if c["type"] == "categorical"]

        recommended_targets = []
        # Try to find a good target variable
        for c in cols:
            name = c["name"].lower()
            if name in ["target", "label", "class", "price", "revenue", "y", "output", "survived", "median_house_value", "species", "salary"]:
                recommended_targets.append({
                    "column": c["name"],
                    "type": "classification" if c["type"] == "categorical" or c["unique_count"] <= 10 else "regression",
                    "reason": f"Fallback recommendation: The column '{c['name']}' looks like a standard target variable based on its name."
                })
        
        # If no target found, recommend the last numeric or categorical column
        if not recommended_targets and cols:
            last_col = cols[-1]
            recommended_targets.append({
                "column": last_col["name"],
                "type": "classification" if last_col["type"] == "categorical" or last_col["unique_count"] <= 10 else "regression",
                "reason": f"Fallback recommendation: Using the last column '{last_col['name']}' as the target."
            })

        rec_features = [c["name"] for c in cols if c["name"] != recommended_targets[0]["column"]][:6]

        recommended_plots = []
        if len(num_cols_list) >= 2:
            recommended_plots.append({
                "type": "scatter",
                "x": num_cols_list[0],
                "y": num_cols_list[1],
                "title": f"Scatter Plot: {num_cols_list[0]} vs {num_cols_list[1]}",
                "explanation": "Allows visualization of relationship/correlation between two numeric features."
            })
        if len(cat_cols_list) >= 1 and len(num_cols_list) >= 1:
            recommended_plots.append({
                "type": "bar",
                "x": cat_cols_list[0],
                "y": num_cols_list[0],
                "title": f"Bar Chart: Average {num_cols_list[0]} by {cat_cols_list[0]}",
                "explanation": "Compares numeric averages across different categories."
            })
        if num_cols_list:
            recommended_plots.append({
                "type": "histogram",
                "x": num_cols_list[0],
                "y": "",
                "title": f"Distribution of {num_cols_list[0]}",
                "explanation": "Helps you see the shape, center, and spread of the data distribution."
            })

        return {
            "dataset_summary": f"This dataset contains {profile.get('num_rows')} rows and {profile.get('num_cols')} columns. (Configure your Gemini API key for advanced AI summaries!)",
            "column_insights": [{"name": c["name"], "insight": f"Type: {c['type']}, Unique Values: {c['unique_count']}."} for c in cols],
            "recommended_targets": recommended_targets,
            "recommended_features": rec_features,
            "recommended_plots": recommended_plots
        }
