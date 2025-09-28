import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Supabase configuration from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Fix SUPABASE_KEY if it has an extra = at the beginning
if SUPABASE_KEY and SUPABASE_KEY.startswith('='):
    SUPABASE_KEY = SUPABASE_KEY[1:]

def get_database_engine():
    """Get Supabase REST API client"""
    if SUPABASE_URL and SUPABASE_KEY:
        print("✅ Supabase REST API client initialized successfully")
        return "supabase_rest"
    else:
        print("⚠️  No SUPABASE_URL or SUPABASE_KEY provided.")
        return None

def execute_query(query, params=None):
    """Execute a query using Supabase REST API"""
    try:
        engine = get_database_engine()
        if not engine:
            print("❌ Database engine not available")
            return []
        
        # This function is not used directly, select_data handles the REST API calls
        return []
    except Exception as e:
        print(f"Database query error: {e}")
        return []

def insert_data(table, data):
    """Insert data into a table using Supabase REST API"""
    try:
        engine = get_database_engine()
        if not engine:
            print("❌ Database engine not available")
            return False
        
        # Use Supabase REST API to insert data
        url = f"{SUPABASE_URL}/rest/v1/{table}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Database insert error: {e}")
        return False

def select_data(table, columns="*", where_clause=None, params=None):
    """Select data from a table using Supabase REST API"""
    try:
        engine = get_database_engine()
        if not engine:
            print("❌ Database engine not available")
            return []
        
        # Use Supabase REST API to query data
        url = f"{SUPABASE_URL}/rest/v1/{table}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        # Build query parameters
        query_params = {}
        if columns != "*":
            query_params["select"] = columns
        
        # Convert where_clause to Supabase format
        if where_clause and params:
            # Convert SQL WHERE clause to Supabase format
            # Example: "disease_name = :disease_name" -> "disease_name=eq.{value}"
            for key, value in params.items():
                if f"{key} = :{key}" in where_clause:
                    query_params[key] = f"eq.{value}"
        
        response = requests.get(url, headers=headers, params=query_params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Database select error: {e}")
        return []

# Database connection will be initialized when first needed
