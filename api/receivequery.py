from .outsideapi import openai, get_database_client
from .database import insert_data
import uuid
import tempfile
import os
import requests

from fastapi import UploadFile



# for converting audio to text from UploadFile
def audio_text_convert_file(audio_file: UploadFile) -> str:
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            # Read the uploaded file content
            content = audio_file.file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe using Whisper
        with open(temp_file_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"
            )
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return transcript.text.strip()
    
    except Exception as e:
        print(f"Error converting audio to text: {e}")
        return ""

# for converting audio to text from URL
def audio_text_convert(audio_url: str) -> str:
    try:
        # Download the audio from URL
        response = requests.get(audio_url)
        response.raise_for_status()  # raise error if request fails

        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(response.content)
        
        # Transcribe and translate to English using Whisper
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=open(temp_file, "rb"),
            language="en"
        )
        
        # Clean up temp file
        os.remove(temp_file)

        return transcript.text

    except Exception as e:
        print("Error converting audio:", e)

def extract_mfcc(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Plot MFCC
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.show()
    
    return mfcc



def photo_url_convert(file_path: str, user_id: str) -> str:
    file_name = os.path.basename(file_path)
    storage_path = f"user_uploads/{user_id}/{file_name}"


    
    return f"/uploads/{storage_path}"




def add_qa(uid, question, answer, photo_url=None,predicted_disease=None):


    text = f"Q: {question}\nA: {answer}"
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    # Store in PostgreSQL
    insert_data("qa", {
        "uid": uid,
        "question": question,
        "answer": answer,
        "photo_url": photo_url,
        "predicted_disease": predicted_disease,
        "embedding": embedding
    })


def add_document(content, metadata=None):

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=content
    ).data[0].embedding

    # Store in PostgreSQL
    insert_data("documents", {
        "content": content,
        "metadata": metadata or {},
        "embedding": embedding
    })






    




