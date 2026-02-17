"""
API Middleware for Gestro
Author: Rakin Mohammed Rafeeq
Description: Custom middleware for logging, rate limiting, and error handling
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs all API requests and responses
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"Status: {response.status_code} Time: {process_time:.2f}ms"
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        response.headers["X-Server-Time"] = datetime.now().isoformat()
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware
    """
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.request_counts = {
            ip: [(t, c) for t, c in requests if current_time - t < self.window_seconds]
            for ip, requests in self.request_counts.items()
        }
        
        # Check rate limit
        if client_ip in self.request_counts:
            request_count = len(self.request_counts[client_ip])
            if request_count >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Maximum {self.max_requests} requests per {self.window_seconds} seconds"
                    }
                )
        
        # Record request
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        self.request_counts[client_ip].append((current_time, 1))
        
        # Process request
        response = await call_next(request)
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware
    """
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            )
