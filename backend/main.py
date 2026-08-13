import os
import uuid
import json
import shutil
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Use absolute backend imports assuming app is run from the workspace root
from backend.ml_pipeline import MLPipeline
from backend.ai_helper import GeminiBrain

app = FastAPI(title="No-Code ML Pipeline & Visualization API")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Helper functions to manage dataset paths
def get_dataset_folder(dataset_id: str) -> str:
    folder = os.path.join(DATA_DIR, dataset_id)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="Dataset not found or expired.")
    return folder

def save_profile_metadata(folder: str, profile: dict, filename: str):
    meta_path = os.path.join(folder, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            pass
    meta[filename] = profile
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def get_profile_metadata(folder: str, filename: str) -> Optional[dict]:
    meta_path = os.path.join(folder, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                return meta.get(filename)
        except Exception:
            pass
    return None


# Request schemas
class CleanRequest(BaseModel):
    dataset_id: str
    options: Dict[str, Any]

class AnalyzeRequest(BaseModel):
    dataset_id: str
    api_key: Optional[str] = None

class ChatRequest(BaseModel):
    dataset_id: str
    api_key: Optional[str] = None
    history: List[Dict[str, str]]
    query: str

class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: List[str]
    model_type: str
    task_type: str = "auto"
    test_size: float = 0.2
    random_state: int = 42
    hyperparameters: Dict[str, Any] = {}

class ExplainRequest(BaseModel):
    dataset_id: str
    api_key: Optional[str] = None
    evaluation_results: Dict[str, Any]


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts CSV or Excel files, saves them inside a unique directory,
    and returns initial profiling details.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".csv", ".xls", ".xlsx"]:
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are supported.")

    dataset_id = str(uuid.uuid4())
    folder = os.path.join(DATA_DIR, dataset_id)
    os.makedirs(folder, exist_ok=True)

    raw_filename = f"raw{ext}"
    filepath = os.path.join(folder, raw_filename)

    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        profile = MLPipeline.load_and_profile(filepath)
        save_profile_metadata(folder, profile, "raw")
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "extension": ext,
            "profile": profile
        }
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Failed to profile dataset: {str(e)}")


@app.post("/api/clean")
def clean_dataset(req: CleanRequest):
    """
    Cleans the uploaded raw dataset using selected configuration,
    saves the clean dataset, and returns profile comparison.
    """
    folder = get_dataset_folder(req.dataset_id)
    
    # Locate raw file
    raw_file = None
    for f in os.listdir(folder):
        if f.startswith("raw."):
            raw_file = f
            break
            
    if not raw_file:
        raise HTTPException(status_code=404, detail="Original raw file not found.")

    raw_path = os.path.join(folder, raw_file)
    clean_path = os.path.join(folder, "clean.csv")

    try:
        clean_profile = MLPipeline.clean_dataset(raw_path, clean_path, req.options)
        save_profile_metadata(folder, clean_profile, "clean")
        return {
            "dataset_id": req.dataset_id,
            "clean_profile": clean_profile
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data cleaning failed: {str(e)}")


@app.post("/api/analyze")
def analyze_dataset(req: AnalyzeRequest):
    """
    Uses Gemini API to generate insights, recommendations, and chart suggestions.
    """
    folder = get_dataset_folder(req.dataset_id)
    
    # Check if cleaned version exists, else use raw
    clean_path = os.path.join(folder, "clean.csv")
    profile = None
    if os.path.exists(clean_path):
        profile = get_profile_metadata(folder, "clean")
    else:
        profile = get_profile_metadata(folder, "raw")

    if not profile:
        raise HTTPException(status_code=404, detail="Dataset profile meta could not be found.")

    brain = GeminiBrain(api_key=req.api_key)
    analysis = brain.analyze_dataset(profile)
    return analysis


@app.post("/api/chat")
def chat_query(req: ChatRequest):
    """
    Interactive chat assistance for dataset logic and ML instruction.
    """
    folder = get_dataset_folder(req.dataset_id)
    
    # Check if cleaned version exists, else use raw
    clean_path = os.path.join(folder, "clean.csv")
    profile = None
    if os.path.exists(clean_path):
        profile = get_profile_metadata(folder, "clean")
    else:
        profile = get_profile_metadata(folder, "raw")

    if not profile:
        raise HTTPException(status_code=404, detail="Dataset profile not found.")

    brain = GeminiBrain(api_key=req.api_key)
    response = brain.chat_query(profile, req.history, req.query)
    return {"response": response}


@app.post("/api/train")
def train_model(req: TrainRequest):
    """
    Trains selected machine learning models on target and features.
    """
    folder = get_dataset_folder(req.dataset_id)
    
    # Use clean data if available, otherwise raw
    clean_path = os.path.join(folder, "clean.csv")
    if not os.path.exists(clean_path):
        # If user didn't clean, copy raw to clean to run train
        raw_file = None
        for f in os.listdir(folder):
            if f.startswith("raw."):
                raw_file = f
                break
        if not raw_file:
            raise HTTPException(status_code=404, detail="Dataset file not found.")
        
        # Simple convert raw to clean.csv directly
        raw_path = os.path.join(folder, raw_file)
        if raw_file.endswith(".csv"):
            shutil.copyfile(raw_path, clean_path)
        else:
            df = pd.read_excel(raw_path)
            df.to_csv(clean_path, index=False)
            
        profile = MLPipeline.load_and_profile(clean_path)
        save_profile_metadata(folder, profile, "clean")

    try:
        eval_results = MLPipeline.train_and_evaluate(clean_path, req.dict())
        return eval_results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model training failed: {str(e)}")


@app.post("/api/explain")
def explain_results(req: ExplainRequest):
    """
    Uses Gemini API to interpret model training evaluation metrics.
    """
    folder = get_dataset_folder(req.dataset_id)
    profile = get_profile_metadata(folder, "clean") or get_profile_metadata(folder, "raw")
    
    if not profile:
        raise HTTPException(status_code=404, detail="Dataset profile metadata not found.")

    brain = GeminiBrain(api_key=req.api_key)
    explanation = brain.explain_model_results(req.evaluation_results, profile)
    return {"explanation": explanation}

@app.get("/api/plot-data")
def get_plot_data(dataset_id: str, x: str, y: Optional[str] = None, type: str = "scatter"):
    """
    Retrieves and prepares columns for visual plots.
    Capped at 500 data points.
    """
    import numpy as np
    import pandas as pd
    folder = get_dataset_folder(dataset_id)
    clean_path = os.path.join(folder, "clean.csv")
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
    else:
        raw_file = None
        for f in os.listdir(folder):
            if f.startswith("raw."):
                raw_file = f
                break
        if not raw_file:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        raw_path = os.path.join(folder, raw_file)
        if raw_file.endswith(".csv"):
            try:
                df = pd.read_csv(raw_path)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(raw_path, encoding='latin1')
                except Exception:
                    df = pd.read_csv(raw_path, encoding='utf-8-sig')
        else:
            df = pd.read_excel(raw_path)

    if x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{x}' not found.")
    
    # Drop rows where X is null for clean plotting
    df = df.dropna(subset=[x])

    if type == "histogram":
        # Calculate bins & counts for X (which must be numeric)
        if not pd.api.types.is_numeric_dtype(df[x]):
            raise HTTPException(status_code=400, detail="Histogram requires a numeric column.")
        
        counts, bin_edges = np.histogram(df[x], bins=10)
        bins_labels = []
        for i in range(len(bin_edges) - 1):
            bins_labels.append(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}")
        
        return {
            "counts": counts.tolist(),
            "bins": bins_labels
        }
    
    elif type == "box":
        if not pd.api.types.is_numeric_dtype(df[x]):
            raise HTTPException(status_code=400, detail="Box plot requires a numeric column.")
        
        q1 = float(df[x].quantile(0.25))
        median = float(df[x].quantile(0.50))
        q3 = float(df[x].quantile(0.75))
        min_val = float(df[x].min())
        max_val = float(df[x].max())
        
        return {
            "box_stats": {
                "min": min_val,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": max_val
            }
        }

    # For scatter, line, bar, we need Y column
    if not y or y not in df.columns:
        raise HTTPException(status_code=400, detail=f"Plot type '{type}' requires a valid Y column.")
    
    # Drop rows where Y is null
    df = df.dropna(subset=[y])
    
    # Limit number of points for UI responsiveness
    if len(df) > 500:
        # Sample randomly
        df_plot = df.sample(n=500, random_state=42)
    else:
        df_plot = df

    # Convert to web-safe formats (no NaNs or Infs)
    x_vals = df_plot[x].replace({np.nan: None, np.inf: None, -np.inf: None}).tolist()
    y_vals = df_plot[y].replace({np.nan: None, np.inf: None, -np.inf: None}).tolist()

    return {
        "x_values": x_vals,
        "y_values": y_vals
    }


@app.post("/api/sample/{name}")
def load_sample(name: str):
    """
    Generates standard datasets offline using sklearn or synthetic creators.
    """
    import pandas as pd
    import numpy as np
    dataset_id = str(uuid.uuid4())
    folder = os.path.join(DATA_DIR, dataset_id)
    os.makedirs(folder, exist_ok=True)
    
    filepath = os.path.join(folder, "raw.csv")
    filename = ""

    if name == "iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = [iris.target_names[i] for i in iris.target]
        df.to_csv(filepath, index=False)
        filename = "iris_flowers.csv"
        
    elif name == "housing":
        # Generate clean synthetic Student Study Performance dataset (better regression target)
        np.random.seed(42)
        n_samples = 300
        hours = np.random.uniform(1.0, 10.0, n_samples)
        attendance = np.random.uniform(60.0, 100.0, n_samples)
        sleep = np.random.uniform(4.0, 9.0, n_samples)
        
        # Linear relationship + noise
        score = 15.0 + 4.5 * hours + 0.3 * attendance + 1.2 * sleep + np.random.normal(0, 3.5, n_samples)
        score = np.clip(score, 0, 100) # Keep within 0 - 100
        
        passed = np.where(score >= 60, "Yes", "No")
        
        df = pd.DataFrame({
            "Hours_Studied": np.round(hours, 1),
            "Attendance_Pct": np.round(attendance, 1),
            "Sleep_Hours": np.round(sleep, 1),
            "Exam_Score": np.round(score, 1),
            "Passed": passed
        })
        
        df.to_csv(filepath, index=False)
        filename = "student_exam_grades.csv"
    else:
        raise HTTPException(status_code=400, detail="Invalid sample dataset specified.")

    try:
        profile = MLPipeline.load_and_profile(filepath)
        save_profile_metadata(folder, profile, "raw")
        return {
            "dataset_id": dataset_id,
            "filename": filename,
            "profile": profile
        }
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to profile sample: {str(e)}")


@app.get("/api/download/{dataset_id}/{file_type}")
def download_file(dataset_id: str, file_type: str):
    """
    Download raw or clean dataset.
    """
    folder = get_dataset_folder(dataset_id)
    
    if file_type == "clean":
        filepath = os.path.join(folder, "clean.csv")
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Cleaned file has not been generated yet.")
        return FileResponse(filepath, media_type="text/csv", filename="cleaned_dataset.csv")
    else:
        raw_file = None
        for f in os.listdir(folder):
            if f.startswith("raw."):
                raw_file = f
                break
        if not raw_file:
            raise HTTPException(status_code=404, detail="Raw file not found.")
        filepath = os.path.join(folder, raw_file)
        return FileResponse(filepath, filename=f"raw_dataset{os.path.splitext(raw_file)[1]}")


# Frontend Static Files configuration
# Serve files from the frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

# Ensure directories exist
os.makedirs(frontend_dir, exist_ok=True)
os.makedirs(os.path.join(frontend_dir, "css"), exist_ok=True)
os.makedirs(os.path.join(frontend_dir, "js"), exist_ok=True)

# Mount the static files
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
