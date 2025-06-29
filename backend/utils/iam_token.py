"""
IBM Cloud IAM Token Generator and Manager
Handles authentication with IBM watsonx.ai using API keys and IAM tokens

This module:
1. Fetches IAM tokens using IBM Cloud API keys
2. Caches tokens with automatic expiration handling
3. Refreshes tokens when needed
4. Provides thread-safe token management
"""

import asyncio
import httpx
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from backend.utils.config import config

logger = logging.getLogger(__name__)

class IAMTokenManager:
    """Manages IBM Cloud IAM tokens with automatic refresh"""
    
    def __init__(self):
        self.api_key = config.GRANITE_API_KEY
        self.token_url = "https://iam.cloud.ibm.com/identity/token"
        self._token = None
        self._token_expires_at = None
        self._refresh_lock = asyncio.Lock()
        
    async def get_token(self) -> str:
        """Get valid IAM token, refreshing if necessary"""
        async with self._refresh_lock:
            # Check if we need a new token
            if self._needs_refresh():
                await self._fetch_new_token()
            
            if not self._token:
                raise Exception("Failed to obtain IAM token")
                
            return self._token
    
    def _needs_refresh(self) -> bool:
        """Check if token needs to be refreshed"""
        if not self._token or not self._token_expires_at:
            return True
        
        # Refresh 5 minutes before expiration
        refresh_threshold = self._token_expires_at - timedelta(minutes=5)
        return datetime.now() >= refresh_threshold
    
    async def _fetch_new_token(self) -> None:
        """Fetch new IAM token from IBM Cloud"""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.token_url,
                    headers=headers,
                    data=data
                )
                response.raise_for_status()
                token_data = response.json()
                
                self._token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
                
                self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                logger.info(f"IAM token refreshed, expires at: {self._token_expires_at}")
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch IAM token: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise Exception(f"IAM token fetch failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching IAM token: {e}")
            raise
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers with valid IAM token"""
        token = await self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def invalidate_token(self) -> None:
        """Manually invalidate current token to force refresh"""
        self._token = None
        self._token_expires_at = None
        logger.info("IAM token manually invalidated")

# Global token manager instance
iam_token_manager = IAMTokenManager()

# Utility functions for easy access
async def get_iam_token() -> str:
    """Get valid IAM token"""
    return await iam_token_manager.get_token()

async def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers with valid IAM token"""
    return await iam_token_manager.get_auth_headers()

def invalidate_token() -> None:
    """Invalidate current token to force refresh"""
    iam_token_manager.invalidate_token()

# Test function for development
async def test_iam_token():
    """Test IAM token functionality"""
    try:
        print("🔄 Testing IAM token generation...")
        
        # Test token generation
        token = await get_iam_token()
        print(f"✅ Token generated: {token[:20]}...")
        
        # Test headers
        headers = await get_auth_headers()
        print(f"✅ Headers generated: {headers}")
        
        # Test token refresh
        print("🔄 Testing token invalidation and refresh...")
        invalidate_token()
        new_token = await get_iam_token()
        print(f"✅ New token after invalidation: {new_token[:20]}...")
        
        print("✅ All IAM token tests passed!")
        
    except Exception as e:
        print(f"❌ IAM token test failed: {e}")

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_iam_token())
