# main.py
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from io import BytesIO
import docx2txt
import numpy as np
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter  

from openai import OpenAI
from .outsideapi import openai

# Initialize FastAPI


# Initialize OpenAI client (for embeddings)


# Initialize Supabase client


# Table in Supabase: plant_documents
# Columns: id (uuid), disease_name (text), chunk_text (text), embedding (vector)

def extract_text_from_word(file_bytes: bytes):
    """Extract text from Word document"""
    return docx2txt.process(BytesIO(file_bytes))

def chunk_text(text: str):
    """Split text into chunks with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=40,
    separators=["\n", ".", "!", "?"," "]
    ) 
    return text_splitter.split_text(text)

def create_embedding(text: str):
    """Get embedding from OpenAI"""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


