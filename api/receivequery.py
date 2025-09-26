from supabase import create_client, Client
from .outsideapi import openai
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



def photo_url_convert(file_path: str, user_id: str) -> str:
    file_name = os.path.basename(file_path)
    storage_path = f"user_uploads/{user_id}/{file_name}"


    # Upload original file to Supabase
    result = supabase.storage.from_("images").upload(
        storage_path,
        open(file_path, "rb")
    )

    if result.get("error"):
        print("Upload error:", result["error"])
        return ""

    public_url = supabase.storage.from_("images").get_public_url(storage_path)
    return public_url["publicUrl"]




def add_qa(uid, question, answer, photo_url=None,predicted_disease=None):


    text = f"Q: {question}\nA: {answer}"
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    supabase.table("qa").insert({
        "uid": uid,
        "question": question,
        "answer": answer,
        "photo_url": photo_url,
        "predicted_disease": predicted_disease,
        "embedding": embedding
    }).execute()


def add_document(content, metadata=None):

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=content
    ).data[0].embedding

    supabase.table("documents").insert({
        "content": content,
        "metadata": metadata or {},
        "embedding": embedding
    }).execute()






    




