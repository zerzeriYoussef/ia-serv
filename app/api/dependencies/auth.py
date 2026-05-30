"""
Authentication dependencies for FastAPI routes.

Decodes the JWT produced by the auth backend (my-awesome-project/backend).
The token's `sub` claim contains the user-id string.
The token's `role` claim (optional) is used for role-based access control.
"""



from __future__ import annotations



import logging

from dataclasses import dataclass

from typing import Optional



from fastapi import Depends, HTTPException, Request, status

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from jose import ExpiredSignatureError, JWTError, jwt



from app.core.config import settings



logger = logging.getLogger(__name__)







_bearer_scheme = HTTPBearer(auto_error=False)





@dataclass

class CurrentUser:

    """Lightweight user context extracted from the JWT."""



    user_id: str

    role: str = "user"



    @property

    def is_admin(self) -> bool:

        return self.role == "admin"













def _decode_token(token: str) -> CurrentUser:

    """
    Validate and decode a JWT access-token.

    Raises:
        HTTPException 401  – token is missing, malformed, or expired.
    """

    try:

        payload = jwt.decode(

            token,

            settings.SECRET_KEY,

            algorithms=[settings.ALGORITHM],

        )

    except ExpiredSignatureError:

        logger.warning("JWT validation failed: token has expired")

        raise HTTPException(

            status_code=status.HTTP_401_UNAUTHORIZED,

            detail="Token has expired",

            headers={"WWW-Authenticate": "Bearer"},

        )

    except JWTError as exc:

        logger.warning("JWT validation failed: %s", exc)

        raise HTTPException(

            status_code=status.HTTP_401_UNAUTHORIZED,

            detail="Invalid authentication token",

            headers={"WWW-Authenticate": "Bearer"},

        )



    user_id: Optional[str] = payload.get("sub")

    if not user_id:

        logger.warning("JWT missing 'sub' claim")

        raise HTTPException(

            status_code=status.HTTP_401_UNAUTHORIZED,

            detail="Token is missing subject claim",

            headers={"WWW-Authenticate": "Bearer"},

        )



    role: str = payload.get("role", "user")

    return CurrentUser(user_id=str(user_id), role=role)













async def get_current_user(

    request: Request,

    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),

) -> Optional[CurrentUser]:

    """
    Soft auth dependency – returns a ``CurrentUser`` when a valid Bearer token
    is present, or ``None`` when the Authorization header is absent entirely.

    Use this when the endpoint works for both anonymous and authenticated users.
    """

    if credentials is not None:

        return _decode_token(credentials.credentials)



    token = request.query_params.get("access_token")

    if token:

        return _decode_token(token)



    return None





async def require_auth(

    request: Request,

    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),

) -> CurrentUser:

    """
    Hard auth dependency – raises **401** when no valid token is provided.

    Inject as ``Depends(require_auth)`` on any route that must be authenticated.
    Returns a ``CurrentUser`` with at least ``user_id`` and ``role``.
    """

    if credentials is not None:

        token = credentials.credentials

    else:

        token = request.query_params.get("access_token")



    if not token:

        logger.warning("Auth required but no Authorization header or access_token provided")

        raise HTTPException(

            status_code=status.HTTP_401_UNAUTHORIZED,

            detail="Authentication required",

            headers={"WWW-Authenticate": "Bearer"},

        )

    return _decode_token(token)





async def require_admin(

    current_user: CurrentUser = Depends(require_auth),

) -> CurrentUser:

    """
    Admin-only dependency – raises **403** when the authenticated user does
    not have the ``admin`` role.

    Always chains ``require_auth`` so a 401 is raised before the 403 check.
    """

    if not current_user.is_admin:

        logger.warning(

            "Admin access denied for user_id=%s (role=%s)",

            current_user.user_id,

            current_user.role,

        )

        raise HTTPException(

            status_code=status.HTTP_403_FORBIDDEN,

            detail="Admin privileges required",

        )

    return current_user

