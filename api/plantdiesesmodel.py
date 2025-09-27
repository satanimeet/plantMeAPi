
import tensorflow as tf

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

from PIL import Image
import io
import os
from io import BytesIO
import requests
from tensorflow.keras.models import load_model

# Get model URL from environment variable or use default
MODEL_URL = os.getenv("MODEL_URL")
LOCAL_MODEL_PATH = "plant_model.keras"

# Global model variable for caching
model = None

def download_model(url, local_path):
    """Download model from URL if it doesn't exist locally"""
    if not os.path.exists(local_path):
        print(f"🔄 Downloading model from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Model downloaded successfully to {local_path}")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise
    else:
        print(f"✅ Model already exists at {local_path}")

def initialize_model():
    """Initialize model at API startup"""
    global model
    if model is None:
        if MODEL_URL:
            print("🔄 API Starting: Downloading model from URL...")
            try:
                download_model(MODEL_URL, LOCAL_MODEL_PATH)
                print("🔄 Loading model...")
                model = load_model(LOCAL_MODEL_PATH)
                print("✅ Model loaded successfully from URL!")
                print("💡 Model is now ready and cached in memory!")
            except Exception as e:
                print(f"❌ Error loading model from URL: {e}")
                print("⚠️  Model initialization failed. Some features may not work.")
                model = None
        else:
            print("⚠️  No MODEL_URL provided. Model will be downloaded on first use.")
            print("💡 Set MODEL_URL environment variable for faster startup.")
    return model

def get_model():
    """Get the model, downloading if needed"""
    global model
    if model is None:
        if MODEL_URL:
            print("🔄 Model not loaded, downloading from URL...")
            try:
                download_model(MODEL_URL, LOCAL_MODEL_PATH)
                model = load_model(LOCAL_MODEL_PATH)
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise Exception(f"Failed to load model: {e}")
        else:
            raise Exception("No MODEL_URL provided and no local model available. Please set MODEL_URL environment variable.")
    return model





# Use provided class names
class_names = [
    "AppleScab",
    "Ablackroot",
    "Apple___healthy",
    "Apple___Scab",
    "Blueberry___Healthy",
    "CherryPowdery_mildew",
    "Gblackrot",
    "Grape(BlackMeasles)",
    "Grapehealthy",
    "Orange___Citrus_greening",
    "PEarly_blight",
    "PLate_blight",
    "Peach___Bacterial_spot",
    "Phealthy",
    "Pepper,bellhealthy",
    "Pepper_bellBacterial_spot",
    "Potato___Healthy",
    "Rhealthy",
    "SLeaf_scorch",
    "Shealthy",
    "Soybeanhealthy",
    "SquashPowderymildew",
    "TBacterial_spot",
    "TEarly_blight",
    "TLate_blight",
    "TLeaf_Mold",
    "Thealthy",
    "Tmosaic_virus",
    "Tspiderspot",
    "Ttargetspot",
    "cherry_health",
    "corncommenrust",
    "corngreyleafspot",
    "cornhealthy",
    "cornnorthethleaf",
    "tSeptorialeafspot",
    "tomato_yello_curl_leaf"
]

index_to_class = {i: name for i, name in enumerate(class_names)}
def preprocess_for_your_model(file_bytes):
    # Open as PIL image and convert to RGB
    img = Image.open(BytesIO(file_bytes)).convert("RGB")  # RGB = 3 channels
    
    # Resize to 380x380 (what your model expects)
    img = img.resize((380, 380))
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0  # shape: (380, 380, 3)
     
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 380, 380, 3)
    
    return img_array




def confidence_level(confidence: float) -> str:
    if confidence >= 0.85:
        return "High"
    elif confidence >= 0.60:
        return "Medium"
    else:
        return "Low"









def predict_image(image_array):
    # Get the model (will download from URL if needed)
    loaded_model = get_model()
    
    # Predict
    y_pred = loaded_model.predict(image_array)
    predicted_class_index = np.argmax(y_pred, axis=1)[0]  # get index of highest probability
    predicted_class_name = class_names[predicted_class_index]

    print("Predicted class:", predicted_class_name)
    return predicted_class_name



#file  = "/Users/meetsatani/Desktop/plantapi/api/aee8f52c-50c2-4327-8240-b8767ea8692d___RS_GLSp 9339_90deg.JPG"
#arraysize = prepare_image_for_model(file)

#print(arraysize.shape)