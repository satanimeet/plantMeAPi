from supabase import create_client, Client
import openai

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients with error handling
supabase = None
openai_client = None

def get_supabase_client():
    global supabase
    if supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("⚠️  Warning: Supabase credentials not found. Some features may not work.")
            return None
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Supabase client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Supabase client: {e}")
            return None
    return supabase

def get_openai_client():
    global openai_client
    if openai_client is None:
        if not OPENAI_API_KEY:
            print("⚠️  Warning: OpenAI API key not found. Some features may not work.")
            return None
        try:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            print("✅ OpenAI client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing OpenAI client: {e}")
            return None
    return openai_client

# Initialize clients
supabase = get_supabase_client()
openai = get_openai_client()