from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import io
import base64
import requests
from requests_oauthlib import OAuth1
from typing import Optional
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

app = FastAPI()

# Allow CORS for all origins (adjust as necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and set up templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Model Loading ---
model_name = "skylord/swin-finetuned-food101"
print(f"Loading image processor and model from Hugging Face model: {model_name}")
try:
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

# --- FatSecret API Integration ---
CONSUMER_KEY = "65fe305ada3b4b268366607010b5e336"
CONSUMER_SECRET = "7bcd1b68bba94feab8c8083511001578"

def get_food_nutrition(food: str) -> dict:
    """
    Returns a dict with a fixed subset of nutrients.
    Falls back to "—" if a field is missing.
    """
    auth = OAuth1(
        CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        signature_method="HMAC-SHA1",
        signature_type='query',
    )
    search_url = "https://platform.fatsecret.com/rest/server.api"
    search_params = {
        "method": "foods.search",
        "search_expression": food,
        "format": "json",
        "max_results": 1,
        "include_sub_categories": "true",
        "flag_default_serving": "true",
        "language": "en",
        "region": "US"
    }
    resp = requests.get(search_url, params=search_params, auth=auth)
    if resp.status_code != 200:
        return {"name": food.capitalize(), "error": "Search failed"}

    data = resp.json().get("foods_search", resp.json().get("foods", {}))
    foods = data.get("results", {}).get("food") or data.get("food")
    if not foods:
        return {"name": food.capitalize(), "error": "No data found"}

    item = foods[0] if isinstance(foods, list) else foods
    food_id = item.get("food_id")
    if not food_id:
        return {"name": food.capitalize(), "error": "No food ID"}

    # get detailed nutritional info
    get_params = {
        "method": "food.get.v4",
        "food_id": food_id,
        "format": "json"
    }
    resp2 = requests.get(search_url, params=get_params, auth=auth)
    if resp2.status_code != 200:
        return {"name": food.capitalize(), "error": "Detail fetch failed"}

    details = resp2.json().get("food", {})
    servings = details.get("servings", {}).get("serving")
    if not servings:
        return {"name": food.capitalize(), "error": "No serving info"}

    if isinstance(servings, list):
        default = next((s for s in servings if s.get("is_default") in ["1", 1]), servings[0])
    else:
        default = servings

    def grab(key):
        return default.get(key, "—")

    return {
        "name":        food.capitalize(),
        "serving":     grab("serving_description"),
        "calories":    grab("calories"),
        "total_fat":   grab("fat"),
        "sat_fat":     grab("saturated_fat"),
        "trans_fat":   grab("trans_fat"),
        "cholesterol": grab("cholesterol"),
        "sodium":      grab("sodium"),
        "carbs":       grab("carbohydrate"),
        "fiber":       grab("fiber"),
        "sugars":      grab("sugar"),
        "protein":     grab("protein"),
        "vitamin_d":   grab("vitamin_d"),
        "calcium":     grab("calcium"),
        "iron":        grab("iron"),
        "potassium":   grab("potassium"),
    }

def normalized_label(label: str) -> str:
    return label.replace("_", " ").strip().lower()

# --- Inference ---
def detect_foods(image_bytes: bytes):
    if model is None:
        return ["unknown"]
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        idx = int(torch.argmax(outputs.logits, dim=-1).item())
        label = model.config.id2label.get(idx, "unknown")
        return [label.replace("_", " ")]
    except Exception as e:
        print("Detection error:", e)
        return ["unknown"]

# --- Retraining Endpoint ---
@app.post("/retrain_user", response_class=HTMLResponse)
async def retrain_user(
    request: Request,
    file: Optional[UploadFile] = File(None),
    true_label: str = Form(...),
    predicted_label: str = Form(...),
    image_data: str = Form(...)
):
    if model is None:
        return HTMLResponse("Model not loaded.", status_code=400)

    # load image bytes
    try:
        image_bytes = await file.read() if file else base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
    except Exception as e:
        return HTMLResponse(f"Image error: {e}", status_code=400)

    final = true_label.strip() or predicted_label.strip()
    if not final:
        return HTMLResponse("No label provided.", status_code=400)

    norm = final.lower().strip()
    if not hasattr(model.config, "id2label"):
        return HTMLResponse("No label mapping.", status_code=400)

    mapping = {normalized_label(v): int(k) for k, v in model.config.id2label.items()}
    target_id = mapping.get(norm)
    if target_id is None:
        return HTMLResponse(f"Unknown label: {final}", status_code=400)

    target = torch.tensor([target_id])
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-6)

    try:
        out = model(**inputs)
        loss = loss_fn(out.logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        retrain_msg = f"Retraining successful! Loss: {loss.item():.6f}"
        nutrition_info = get_food_nutrition(final)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": [nutrition_info],
                "predicted_label": predicted_label,
                "image_data": image_data,
                "retrain_message": retrain_msg
            }
        )
    except Exception as e:
        print("Retrain error:", e)
        return HTMLResponse(f"Retrain failed: {e}", status_code=500)

# --- Main Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    label = detect_foods(image_bytes)[0]
    nutrition_info = get_food_nutrition(label)
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": [nutrition_info],
            "predicted_label": label,
            "image_data": image_data
        }
    )