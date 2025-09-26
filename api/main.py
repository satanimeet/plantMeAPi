from fastapi import FastAPI, UploadFile, File, Form,Query,Body
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from .outsideapi import supabase,openai
from tensorflow.keras.models import load_model
from .answer import get_answer,genral_answer
from .receivequery import audio_text_convert, audio_text_convert_file
from .documentvector import extract_text_from_word,chunk_text,create_embedding
import os
from PIL import Image
from io import BytesIO
import cv2
from .plantdiesesmodel import class_names, confidence_level,preprocess_for_your_model,predict_image,get_model,initialize_model
import keras


# Load model lazily to avoid import-time delays
model = None


class_name_mapping = {
    "AppleScab": "Apple - Scab",
    "Ablackroot": "Apple - Black root",
    "Apple___healthy": "Apple - Healthy",
    "Apple___Scab": "Apple - Scab",
    "Blueberry___Healthy": "Blueberry - Healthy",
    "CherryPowdery_mildew": "Cherry - Powdery mildew",
    "Gblackrot": "Grape - Black rot",
    "Grape(BlackMeasles)": "Grape - Black measles",
    "Grapehealthy": "Grape - Healthy",
    "Orange___Citrus_greening": "Orange - Citrus greening",
    "PEarly_blight": "Potato - Early blight",
    "PLate_blight": "Potato - Late blight",
    "Peach___Bacterial_spot": "Peach - Bacterial spot",
    "Phealthy": "Peach - Healthy",
    "Pepper,bellhealthy": "Pepper (Bell) - Healthy",
    "Pepper_bellBacterial_spot": "Pepper (Bell) - Bacterial spot",
    "Potato___Healthy": "Potato - Healthy",
    "Rhealthy": "Raspberry - Healthy",   # guessing 'R'
    "SLeaf_scorch": "Strawberry - Leaf scorch",
    "Shealthy": "Strawberry - Healthy",
    "Soybeanhealthy": "Soybean - Healthy",
    "SquashPowderymildew": "Squash - Powdery mildew",
    "TBacterial_spot": "Tomato - Bacterial spot",
    "TEarly_blight": "Tomato - Early blight",
    "TLate_blight": "Tomato - Late blight",
    "TLeaf_Mold": "Tomato - Leaf mold",
    "Thealthy": "Tomato - Healthy",
    "Tmosaic_virus": "Tomato - Mosaic virus",
    "Tspiderspot": "Tomato - Spider spot",
    "Ttargetspot": "Tomato - Target spot",
    "cherry_health": "Cherry - Healthy",
    "corncommenrust": "Corn - Common rust",
    "corngreyleafspot": "Corn - Grey leaf spot",
    "cornhealthy": "Corn - Healthy",
    "cornnorthethleaf": "Corn - Northern leaf blight",
    "tSeptorialeafspot": "Tomato - Septoria leaf spot",
    "tomato_yello_curl_leaf": "Tomato - Yellow curl leaf virus"
}




# Custom preprocessing for YOUR model (380x380 RGB)

app = FastAPI()

# Initialize model at startup
@app.on_event("startup")
async def startup_event():
    print("🚀 API Starting: Initializing model...")
    try:
        initialize_model()
        print("✅ API Ready: Model initialized successfully!")
    except Exception as e:
        print(f"❌ API Startup Error: {e}")
        raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Plant Disease API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predictscore")
async def predictscore(file: UploadFile = File(...), uid: Optional[str] = Form(None)):
    try:
        file_bytes = await file.read()
        
        # Preprocess for YOUR model (380x380 RGB)
        image_array = preprocess_for_your_model(file_bytes)
        
        # Get model and predict directly
        model = get_model()
        preds = model.predict(image_array)
        class_idx = np.argmax(preds, axis=1)[0]
        class_name = class_names[class_idx]
        confidence = float(preds[0][class_idx])
        
        
        # Store predicted disease in user state if uid is provided
        if uid:
            user_state[uid] = {"disease": class_name, "confidence": confidence}

        if confidence >= 0.6:
            full_name = class_name_mapping.get(class_name, class_name)
            message = f"It's {full_name}. If you need a solution you can ask it."

            return {
                "predicted_class": class_name,
                "full_name": full_name,
                "message": message,
                "stored": uid is not None
            }
        
        else:
            return {
                "status": "low_confidence",
                "message": "It doesn’t look like a plant photo. Can you re-upload it?"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

"""@app.post("/queastionanswer")
async def queastionanswer(uuid,queastion,answer,photo=None,audio=None):
    
    text = ""

    if audio is not None:
        text = audio_text_convert_file(audio)
    else:
        text = queastion

    if photo not None:
        predictd = predict_image(photo)
        confidence = predictd["confidence"]
        predicted_disease = predictd["predicted_disease"]
        accuracy = predictd["accuracy"] * 100

        """


@app.post("/upload-document")
async def upload_document(
    disease_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Step 1: Extract text
        file_bytes = await file.read()
        text = extract_text_from_word(file_bytes)

        # Step 2: Chunk text
        chunks = chunk_text(text)

        # Step 3: Generate embeddings and store in DB
        for chunk in chunks:
            embedding = create_embedding(chunk)

            # Store in Supabase
            supabase.table("documents").insert({
                "disease_name": disease_name,
                "context": chunk,
                "embedding_context": embedding
            }).execute()

        return {"status": "success", "chunks_stored": len(chunks)}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    
    
@app.get("/ask")
async def ask(question: str = Query(..., description="Your question about the disease")):
    answer = genral_answer(question)
    return {"answer": answer}



user_state = {}

@app.post("/queastionAnswer")
async def queastionAnswer(
    queastion: Optional[str] = Form(None),                     
    disease_name: Optional[str] = Form(None),      # optional
    uid: Optional[str] = Form(None),               # optional
    photo: Optional[UploadFile] = File(None),      # optional
    audio: Optional[UploadFile] = File(None)       # optional
    ):  

     
    

    if audio is not None:
        text = audio_text_convert_file(audio)
        
    else:
        text = queastion
        
    
    # Check if text is empty or None
    

    if photo is not None:
        # Process the uploaded photo for disease prediction
        file_bytes = await photo.read()
        
        # Preprocess for the model (380x380 RGB)
        img_batch = preprocess_for_your_model(file_bytes)
        
        # Get model and predict
        model = get_model()
        preds = model.predict(img_batch)
        class_idx = np.argmax(preds, axis=1)[0]
        predicted_disease = class_names[class_idx]
        confidence = float(preds[0][class_idx])

        # save predicted disease in session
        if uid:
            user_state[uid] = {"disease": predicted_disease}

        disease_name = predicted_disease

        if confidence >= 0.6:
            full_name = class_name_mapping.get(predicted_disease, predicted_disease)
            message = f"It's {full_name}. If you need a solution you can ask it."

            return {
                "predicted_class": predicted_disease,
                "full_name": full_name,
                "message": message,
                "stored": uid is not None
            }
        else:
            return {
                "status": "low_confidence",
                "message": "It doesn’t look like a plant photo. Can you re-upload it?"
            }
    else:
        # if no photo, try to reuse stored disease
        if uid and uid in user_state and user_state[uid].get("disease"):
            disease_name = user_state[uid]["disease"]
        else:
            disease_name = None

    if disease_name is None:
        intro = "chatbotintro"

        general_answer = genral_answer(text)
        print(f"General answer generated: '{general_answer}'")

        return {
            "status": "success",
            "disease": None,
            "answer": general_answer
        }

    else:
        answer = get_answer(text,disease_name)
        print(f"Disease-specific answer generated: '{answer}'")
        return {
            "status": "success",
            "disease": disease_name,
            "answer": answer
        }



    

    

    
    

    



    
