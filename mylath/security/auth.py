# mylath/security/auth.py
from functools import wraps
import jwt
import time
from typing import Optional, Dict, Any
from flask import request, jsonify


class AuthManager:
    """Authentication and authorization for MyLath API"""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def generate_token(self, user_id: str, permissions: List[str], 
                      expires_in: int = 3600) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': time.time() + expires_in,
            'iat': time.time()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def require_auth(self, required_permission: str = None):
        """Decorator for requiring authentication"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid authorization header'}), 401
                
                token = auth_header.split(' ')[1]
                payload = self.verify_token(token)
                
                if not payload:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                if required_permission and required_permission not in payload.get('permissions', []):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Add user info to request context
                request.user = payload
                return func(*args, **kwargs)
            return wrapper
        return decorator
