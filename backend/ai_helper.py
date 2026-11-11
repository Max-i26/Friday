import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Load local environment secrets
load_dotenv()

try:

    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class ClaudeBrain:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-5-5"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None
        if self.api_key and anthropic:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Anthropic client: {e}")

    def is_configured(self) -> bool:
        return bool(self.client and self.api_key)

    def _extract_thinking_and_text(self, response) -> Tuple[str, str]:
        thinking = ""
        text = ""
        for block in getattr(response, "content", []):
            if block.type == "thinking":
                thinking += block.thinking
            elif block.type == "text":
                text += block.text
        return thinking.strip(), text.strip()

    def _clean_json(self, raw_text: str) -> str:
        text = raw_text.strip()
        # Remove ```json and ``` if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def analyze_dataset(self, profile: dict) -> dict:
        if not self.is_configured():
            raise ValueError("Anthropic Claude API key is not configured.")

        columns_summary = []
        for col in profile.get("columns", []):
            col_name = col["name"]
            col_type = col["type"]
            missing_pct = col["missing_pct"]
            unique_count = col["unique_count"]
            stats_str = ""
            if col_name in profile.get("stats_numeric", {}):
                s = profile["stats_numeric"][col_name]
                stats_str = f", min={s['min']}, max={s['max']}, mean={s['mean']:.2f}" if s.get('mean') is not None else ""
            elif col_name in profile.get("stats_categorical", {}):
                s = profile["stats_categorical"][col_name]
                stats_str = f", unique_vals={s.get('unique')}, top='{s.get('top')}'"

            columns_summary.append(
                f"- Column '{col_name}' ({col_type}): missing={missing_pct}%, unique={unique_count}{stats_str}"
            )

        prompt = f"""You are an expert Data Scientist and ML Tutor AI.
Analyze this dataset's column profile and metadata thoroughly before providing structured recommendations.

Dataset Stats:
- Total Rows: {profile.get("num_rows")}
- Total Columns: {profile.get("num_cols")}

Columns:
{chr(10).join(columns_summary)}

Return ONLY a valid JSON object matching this schema:
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
"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        thinking, text = self._extract_thinking_and_text(response)
        cleaned = self._clean_json(text)
        data = json.loads(cleaned)
        if thinking:
            data["thinking"] = thinking
        data["provider"] = "Claude Opus 5.5"
        return data

    def chat_query(self, profile: dict, history: list, query: str) -> Dict[str, Any]:
        if not self.is_configured():
            return {
                "response": "Claude API key is not configured. Please input your Claude API Key in the settings at the top of the interface!",
                "thinking": None,
                "provider": "Claude"
            }

        columns_brief = [f"'{c['name']}' ({c['type']})" for c in profile.get("columns", [])]
        system_prompt = f"""You are Friday's AI Data Science & Machine Learning Tutor Brain powered by Claude.
You are tutoring a student analyzing a dataset with {profile.get('num_rows')} rows and {profile.get('num_cols')} columns.
Columns: {', '.join(columns_brief)}.

Guidelines:
1. Think deeply about the statistical implications of student questions (outliers, skewness, collinearity, metric trade-offs).
2. Answer in friendly, intuitive English suitable for students learning data science.
3. If the user asks for code, provide clean, idiomatic Pandas/Scikit-learn snippets in ```python blocks.
4. Keep explanations engaging, clear, and conceptually accurate."""

        messages = []
        for h in history:
            role = "user" if h.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": h.get("text", "")})
        
        messages.append({"role": "user", "content": query})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=messages
        )
        thinking, text = self._extract_thinking_and_text(response)
        return {
            "response": text,
            "thinking": thinking if thinking else None,
            "provider": "Claude Opus 5.5"
        }

    def explain_model_results(self, eval_results: dict, profile: dict) -> Dict[str, Any]:
        if not self.is_configured():
            return {
                "explanation": "Claude API key is not configured. Please input your Claude API Key to generate comprehensive AI reports!",
                "thinking": None,
                "provider": "Claude"
            }

        task_type = eval_results.get("task_type", "unknown")
        model_type = eval_results.get("model_type", "unknown")
        hyperparams = eval_results.get("hyperparameters", {})
        train_metrics = eval_results.get("train_metrics", {})
        test_metrics = eval_results.get("test_metrics", {})
        feat_cols = eval_results.get("feature_columns", [])
        target_col = eval_results.get("target_column")

        metrics_summary = []
        for k in train_metrics.keys():
            train_val = train_metrics[k]
            test_val = test_metrics[k]
            metrics_summary.append(f"- {k.upper()}: Train = {train_val:.4f}, Test = {test_val:.4f}")

        fi_summary = ""
        fi = eval_results.get("visualization_data", {}).get("feature_importance", {})
        if fi:
            feats = fi.get("features", [])
            scores = fi.get("scores", [])
            fi_items = [f"'{f}': {s:.4f}" for f, s in zip(feats[:5], scores[:5])]
            fi_summary = f"Top features: {', '.join(fi_items)}"

        prompt = f"""You are an expert ML Tutor AI. Analyze these machine learning training results for a student:
- Task: {task_type.upper()}
- Model: {model_type.upper()}
- Target Variable: '{target_col}'
- Features used: {', '.join(feat_cols)}
- Hyperparameters: {json.dumps(hyperparams)}

Performance Metrics:
{chr(10).join(metrics_summary)}

{fi_summary}

Please provide an educational audit report covering:
1. What this model is doing in intuitive terms and how to interpret the metrics (R2, MAE, Accuracy, etc.).
2. Model Diagnosis: Compare Train vs. Test metrics. Is the model Overfitting, Underfitting, or Well-Generalized? Explain the specific evidence.
3. Feature Significance: Which features drove the model's predictions?
4. Concrete Actionable Next Steps: 2-3 specific ways the student can improve performance (e.g. feature engineering, hyperparameter tuning, scaling).

Format in friendly, encouraging Markdown with headers and bullet points."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        thinking, text = self._extract_thinking_and_text(response)
        return {
            "explanation": text,
            "thinking": thinking if thinking else None,
            "provider": "Claude Opus 5.5"
        }


class GeminiBrain:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model
        if self.api_key and genai:
            try:
                genai.configure(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to configure Gemini: {e}")

    def is_configured(self) -> bool:
        return bool(self.api_key and genai)

    def _get_model(self):
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured.")
        return genai.GenerativeModel(self.model_name)

    def analyze_dataset(self, profile: dict) -> dict:
        if not self.is_configured():
            return self._get_fallback_analysis(profile)

        columns_summary = []
        for col in profile.get("columns", []):
            col_name = col["name"]
            col_type = col["type"]
            missing_pct = col["missing_pct"]
            unique_count = col["unique_count"]
            stats_str = ""
            if col_name in profile.get("stats_numeric", {}):
                s = profile["stats_numeric"][col_name]
                stats_str = f", min={s['min']}, max={s['max']}, mean={s['mean']:.2f}" if s.get('mean') is not None else ""
            elif col_name in profile.get("stats_categorical", {}):
                s = profile["stats_categorical"][col_name]
                stats_str = f", unique_vals={s.get('unique')}, top='{s.get('top')}'"

            columns_summary.append(
                f"- Column '{col_name}' ({col_type}): missing_pct={missing_pct}%, unique_count={unique_count}{stats_str}"
            )

        prompt = f"""You are an expert Data Scientist and ML Tutor AI. Analyze this dataset's column profile and metadata.
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
Ensure the output is valid JSON.
"""
        try:
            model = self._get_model()
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            data["provider"] = "Gemini 3.5 Flash"
            data["thinking"] = None
            return data
        except Exception as e:
            print(f"Gemini API analysis failed: {e}")
            return self._get_fallback_analysis(profile)

    def chat_query(self, profile: dict, history: list, query: str) -> Dict[str, Any]:
        if not self.is_configured():
            return {
                "response": "Gemini API key is not configured. Please input your Gemini API Key in the settings at the top of the interface!",
                "thinking": None,
                "provider": "Gemini"
            }

        columns_brief = [f"'{col['name']}' ({col['type']})" for col in profile.get("columns", [])]
        dataset_context = f"""You are an expert Data Science and ML Brain for students.
You are helping the student analyze a dataset with {profile.get('num_rows')} rows and {profile.get('num_cols')} columns.
Columns: {", ".join(columns_brief)}.
When answering, be educational, encouraging, and clear."""

        model = self._get_model()
        contents = []
        for h in history:
            contents.append({
                "role": "user" if h.get("role") == "user" else "model",
                "parts": [h.get("text", "")]
            })

        chat = model.start_chat(history=contents)
        full_query = f"{dataset_context}\n\nUser Query: {query}"
        try:
            response = chat.send_message(full_query)
            return {
                "response": response.text,
                "thinking": None,
                "provider": "Gemini 3.5 Flash"
            }
        except Exception as e:
            return {
                "response": f"Error talking to Gemini Brain: {str(e)}",
                "thinking": None,
                "provider": "Gemini"
            }

    def explain_model_results(self, eval_results: dict, profile: dict) -> Dict[str, Any]:
        if not self.is_configured():
            return {
                "explanation": "Gemini API key is not configured. Please configure an API key to generate reports!",
                "thinking": None,
                "provider": "Gemini"
            }

        task_type = eval_results.get("task_type", "")
        model_type = eval_results.get("model_type", "")
        hyperparams = eval_results.get("hyperparameters", {})
        train_metrics = eval_results.get("train_metrics", {})
        test_metrics = eval_results.get("test_metrics", {})
        feat_cols = eval_results.get("feature_columns", [])
        target_col = eval_results.get("target_column")

        metrics_summary = [f"- {k.upper()}: Train = {train_metrics[k]:.4f}, Test = {test_metrics[k]:.4f}" for k in train_metrics.keys()]

        fi_summary = ""
        fi = eval_results.get("visualization_data", {}).get("feature_importance", {})
        if fi:
            feats = fi.get("features", [])
            scores = fi.get("scores", [])
            fi_items = [f"'{f}': {s:.4f}" for f, s in zip(feats[:5], scores[:5])]
            fi_summary = f"Top features: {', '.join(fi_items)}"

        prompt = f"""You are an ML Tutor. Explain these model training results to a student:
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
4. Concrete steps for the student to improve the model.
Write in friendly, encouraging Markdown."""

        try:
            model = self._get_model()
            response = model.generate_content(prompt)
            return {
                "explanation": response.text,
                "thinking": None,
                "provider": "Gemini 3.5 Flash"
            }
        except Exception as e:
            return {
                "explanation": f"Error generating report: {str(e)}",
                "thinking": None,
                "provider": "Gemini"
            }

    def _get_fallback_analysis(self, profile: dict) -> dict:
        cols = profile.get("columns", [])
        num_cols_list = [c["name"] for c in cols if c["type"] == "numeric"]
        cat_cols_list = [c["name"] for c in cols if c["type"] == "categorical"]

        recommended_targets = []
        for c in cols:
            name = c["name"].lower()
            if name in ["target", "label", "class", "price", "revenue", "y", "output", "survived", "median_house_value", "species", "salary"]:
                recommended_targets.append({
                    "column": c["name"],
                    "type": "classification" if c["type"] == "categorical" or c["unique_count"] <= 10 else "regression",
                    "reason": f"Fallback recommendation: The column '{c['name']}' looks like a standard target variable based on its name."
                })
        
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
            "dataset_summary": f"This dataset contains {profile.get('num_rows')} rows and {profile.get('num_cols')} columns. (Configure an AI API key for deep reasoning summaries!)",
            "column_insights": [{"name": c["name"], "insight": f"Type: {c['type']}, Unique Values: {c['unique_count']}."} for c in cols],
            "recommended_targets": recommended_targets,
            "recommended_features": rec_features,
            "recommended_plots": recommended_plots,
            "provider": "Offline Fallback",
            "thinking": None
        }


def get_ai_brain(provider: Optional[str] = None, api_key: Optional[str] = None):
    """
    Factory function returning the active AI Brain (Claude or Gemini).
    Priority:
    1. Explicit provider passed ('claude' or 'gemini')
    2. Environment AI_PROVIDER setting
    3. If ANTHROPIC_API_KEY is available -> Claude
    4. If GEMINI_API_KEY is available -> Gemini
    5. Fallback to GeminiBrain offline handler
    """
    env_provider = os.environ.get("AI_PROVIDER", "").lower()
    selected = (provider or env_provider).lower() if (provider or env_provider) else ""

    if selected == "claude" or (not selected and os.environ.get("ANTHROPIC_API_KEY")):
        return ClaudeBrain(api_key=api_key)
    elif selected == "gemini" or (not selected and os.environ.get("GEMINI_API_KEY")):
        return GeminiBrain(api_key=api_key)
    
    # Default to Claude if key given, else Gemini
    if api_key and api_key.startswith("sk-ant-"):
        return ClaudeBrain(api_key=api_key)
    return GeminiBrain(api_key=api_key)
