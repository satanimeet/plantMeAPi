import openai
import os
from dotenv import load_dotenv
from .database import get_database_engine

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients with error handling
openai_client = None

def get_database_client():
    """Get PostgreSQL database client"""
    engine = get_database_engine()
    if engine:
        print("✅ PostgreSQL database client initialized successfully")
        return engine
    else:
        print("⚠️  Warning: PostgreSQL database not available. Some features may not work.")
        return None

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

# Clients will be initialized when first needed