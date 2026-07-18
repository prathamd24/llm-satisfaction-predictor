"""
app/auth.py
===========
Firebase Authentication dependency for FastAPI.

This module verifies the Firebase ID token sent in the `Authorization` header.
If the token is valid, it allows the request to proceed.
If it is invalid or missing, it returns a 401 Unauthorized response.
"""

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth, credentials, initialize_app

logger = logging.getLogger(__name__)

# FastAPI security scheme to extract the Bearer token
security = HTTPBearer(auto_error=False)

# Initialize Firebase Admin SDK
# It expects the GOOGLE_APPLICATION_CREDENTIALS environment variable
# to point to a service account JSON file.
try:
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
            "Firebase Auth will fail until this is configured."
        )
    # Initialize the default app using the environment credentials
    initialize_app()
    logger.info("Firebase Admin SDK initialized successfully.")
except ValueError as e:
    # App already initialized (can happen during hot reloads)
    logger.debug(f"Firebase app already initialized: {e}")
except Exception as e:
    logger.error(f"Failed to initialize Firebase Admin SDK: {e}")


def verify_firebase_token(
    cred: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """
    Verify the Firebase ID token in the Authorization header.
    
    Returns:
        dict: The decoded token payload (containing user info).
        
    Raises:
        HTTPException: 401 if token is missing, invalid, or expired.
    """
    if not cred:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = cred.credentials

    try:
        # Verify the ID token using the Firebase Admin SDK
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except ValueError:
        # App not initialized properly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Firebase Auth is not properly configured on the server",
        )
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
