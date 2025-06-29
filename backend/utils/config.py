"""
Configuration management for WhatsApp AI Second Brain Assistant
Loads environment variables and provides centralized config access
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    
    # IBM Granite Configuration
    GRANITE_API_KEY = os.getenv("GRANITE_API_KEY")
    GRANITE_PROJECT_ID = os.getenv("GRANITE_PROJECT_ID")
    GRANITE_API_URL = os.getenv("GRANITE_API_URL", "https://us-south.ml.cloud.ibm.com")
    IBM_MODEL_ID = os.getenv("IBM_MODEL_ID", "ibm/granite-3-8b-instruct")
    
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "secondbrain")
    
    # Application Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # File Configuration
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "./vectorstore"))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB
    
    # Scheduler Configuration
    SCHEDULER_TIMEZONE = os.getenv("SCHEDULER_TIMEZONE", "UTC")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_vars = [
            ("TWILIO_ACCOUNT_SID", cls.TWILIO_ACCOUNT_SID),
            ("TWILIO_AUTH_TOKEN", cls.TWILIO_AUTH_TOKEN),
            ("TWILIO_PHONE_NUMBER", cls.TWILIO_PHONE_NUMBER),
            ("GRANITE_API_KEY", cls.GRANITE_API_KEY),
            ("GRANITE_PROJECT_ID", cls.GRANITE_PROJECT_ID),
            ("MONGO_URI", cls.MONGO_URI),
        ]
        
        missing = [var for var, value in required_vars if not value]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        # Create directories if they don't exist
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_PATH.mkdir(exist_ok=True)
        
        return True

# Global config instance
config = Config()
