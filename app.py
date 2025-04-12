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
    model.eval()  # Start in evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

# --- FatSecret API Integration ---
CONSUMER_KEY = "65fe305ada3b4b268366607010b5e336"
CONSUMER_SECRET = "7bcd1b68bba94feab8c8083511001578"

def get_food_nutrition(food: str) -> str:
    """
    Uses FatSecret’s API to lookup nutritional details for the given food.
    The food parameter is expected with spaces (e.g., "beef tartare").
    Returns a pipe-separated string: "Food|calories|carbs|protein|fat".
    """
    auth = OAuth1(
        CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        signature_method="HMAC-SHA1",
        signature_type='query',
        realm=""
    )
    # Use food with spaces for better query results.
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
    search_resp = requests.get(search_url, params=search_params, auth=auth)
    if search_resp.status_code != 200:
        return f"{food.capitalize()}: Error retrieving food data."
    try:
        search_data = search_resp.json()
        print("Search Response for", food, ":", search_data)
        fs = search_data.get("foods_search") or search_data.get("foods") or {}
        foods = fs.get("results", {}).get("food") or fs.get("food")
        if not foods:
            return f"{food.capitalize()}: No food data found."
        food_item = foods[0] if isinstance(foods, list) else foods
        food_id = food_item.get("food_id")
        if not food_id:
            return f"{food.capitalize()}: No food ID found."
    except Exception:
        return f"{food.capitalize()}: Error parsing search results."
    
    get_url = "https://platform.fatsecret.com/rest/server.api"
    get_params = {
        "method": "food.get.v4",
        "food_id": food_id,
        "format": "json"
    }
    get_resp = requests.get(get_url, params=get_params, auth=auth)
    if get_resp.status_code != 200:
        return f"{food.capitalize()}: Error retrieving nutritional details."
    try:
        food_details = get_resp.json()
        food_data = food_details.get("food", {})
        servings = food_data.get("servings", {}).get("serving")
        if not servings:
            return f"{food.capitalize()}: No serving data found."
        default_serving = (next((s for s in servings if s.get("is_default") in ["1", 1]), servings[0])
                           if isinstance(servings, list) else servings)
        calories = default_serving.get("calories", "N/A")
        carbohydrate = default_serving.get("carbohydrate", "N/A")
        protein = default_serving.get("protein", "N/A")
        fat = default_serving.get("fat", "N/A")
        # Return food name with spaces for display.
        return f"{food.capitalize()}|{calories}|{carbohydrate}|{protein}|{fat}"
    except Exception:
        return f"{food.capitalize()}: Error parsing nutritional details."

def normalized_label(label: str) -> str:
    """
    Normalizes a label by replacing underscores with spaces and converting to lowercase.
    This is used to compare user input (with spaces) against the model's id2label mapping.
    """
    return label.replace("_", " ").strip().lower()

# --- Inference Function ---
def detect_foods(image_bytes: bytes):
    """
    Performs inference using the Hugging Face Swin‑finetuned Food101 model.
    Expects an RGB image.
    Returns a list with the predicted food label, with underscores replaced by spaces.
    """
    global model
    if model is None:
        return ["unknown"]
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = int(torch.argmax(logits, dim=-1).item())
        if hasattr(model.config, "id2label") and model.config.id2label:
            original_label = model.config.id2label.get(predicted_class, "unknown")
            # Replace underscores with spaces for display and query.
            display_label = original_label.replace("_", " ")
        else:
            display_label = "unknown"
        return [display_label]
    except Exception as e:
        print("Error during detection:", e)
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
    """
    Fine-tunes the model on the image using either the corrected label or the predicted label
    (if the corrected label is left blank). The image is provided via a hidden base64-encoded field.
    After retraining, the index page is re-rendered with a retraining message.
    """
    global model
    if model is None:
        return HTMLResponse(content="Model is not loaded; cannot retrain.", status_code=400)
    
    try:
        if file is not None:
            image_bytes = await file.read()
        else:
            image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
    except Exception as e:
        return HTMLResponse(content=f"Error processing image: {str(e)}", status_code=400)
    
    # Use the corrected label if provided; otherwise, use the predicted label.
    final_label = true_label.strip() if true_label.strip() else predicted_label.strip()
    if not final_label:
        return HTMLResponse(content="No label provided for retraining.", status_code=400)
    
    # Normalize final label (replace spaces with spaces already, so normalization just lowercases)
    normalized_final = final_label.lower().strip()
    
    if hasattr(model.config, "id2label") and model.config.id2label:
        # Build a mapping by normalizing the labels from the model configuration.
        label_to_id = {normalized_label(v): int(k) for k, v in model.config.id2label.items()}
        target_id = label_to_id.get(normalized_final)
        if target_id is None:
            return HTMLResponse(content=f"Unknown label: {final_label}", status_code=400)
    else:
        return HTMLResponse(content="Model configuration does not provide a label mapping.", status_code=400)
    
    target = torch.tensor([target_id])
    
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    
    try:
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        
        retrain_message = f"Retraining completed successfully! Loss: {loss.item():.6f}"
        # For display, use the final_label with spaces.
        nutrition_info = get_food_nutrition(final_label)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": [nutrition_info],
                "predicted_label": predicted_label,
                "image_data": image_data,
                "retrain_message": retrain_message
            }
        )
    except Exception as e:
        print("Error during retraining:", e)
        return HTMLResponse(content=f"Error during retraining: {str(e)}", status_code=500)

# --- Endpoints for Inference ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    detected_food = detect_foods(image_bytes)[0]
    nutrition_info = get_food_nutrition(detected_food)
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": [nutrition_info],
            "predicted_label": detected_food,
            "image_data": image_data
        }
    )