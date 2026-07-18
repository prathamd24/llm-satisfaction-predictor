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

import json

# Initialize Firebase Admin SDK
try:
    firebase_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if firebase_json_str:
        cred_dict = json.loads(firebase_json_str)
        cred = credentials.Certificate(cred_dict)
        initialize_app(cred)
        logger.info("Firebase Admin SDK initialized from FIREBASE_CREDENTIALS_JSON string.")
    else:
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.warning(
                "Neither FIREBASE_CREDENTIALS_JSON nor GOOGLE_APPLICATION_CREDENTIALS is set. "
                "Initializing Firebase with projectId only (this is sufficient for token verification)."
            )
            # Token verification only requires the project ID, not a full service account!
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "llm-satisfaction-predictor")
            initialize_app(options={"projectId": project_id})
            logger.info(f"Firebase Admin SDK initialized with projectId: {project_id}")
        else:
            initialize_app()
            logger.info("Firebase Admin SDK initialized using default credentials.")
except ValueError as e:
    logger.debug(f"Firebase app initialization error (might be already initialized): {e}")
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
    except ValueError as e:
        # App not initialized properly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Firebase Auth is not properly configured on the server: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
