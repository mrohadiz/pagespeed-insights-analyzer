"""
Utilities Module

This module provides utility functions and classes for the PageSpeed Analyzer application,
including progress tracking, URL validation, logging, and error handling.
"""

import os
import json
import logging
import re
import xml.etree.ElementTree as ET
import requests
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, Type, Tuple
from datetime import datetime
from functools import wraps
from requests.exceptions import RequestException, Timeout, HTTPError, ConnectionError


# Type variable for generic functions
T = TypeVar('T')


# -------------------- Custom Exceptions --------------------

class PageSpeedAnalyzerError(Exception):
    """Base exception for PageSpeed Analyzer errors."""
    pass


class SitemapError(PageSpeedAnalyzerError):
    """Exception raised for sitemap parsing or fetching errors."""
    pass


class APIError(PageSpeedAnalyzerError):
    """Exception raised for API-related errors."""
    pass


class ProgressTrackingError(PageSpeedAnalyzerError):
    """Exception raised for progress tracking errors."""
    pass


# -------------------- Progress Tracking --------------------

class ProgressState:
    """Enum-like class for progress states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETE = "complete"
    ERROR = "error"


class Progress:
    """
    Class for managing analysis progress state.
    
    Attributes:
        current: Current progress count
        total: Total items to process
        status: Current status of processing
        errors: Number of errors encountered
        message: Optional message (e.g., error details)
        report: Optional path to the generated report
    """
    
    def __init__(
        self, 
        current: int = 0, 
        total: int = 0, 
        status: str = ProgressState.INITIALIZING,
        errors: int = 0,
        message: Optional[str] = None,
        report: Optional[str] = None
    ):
        self.current = current
        self.total = total
        self.status = status
        self.errors = errors
        self.message = message
        self.report = report
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary for serialization."""
        result = {
            'current': self.current,
            'total': self.total,
            'status': self.status,
            'errors': self.errors
        }
        
        if self.message:
            result['message'] = self.message
            
        if self.report:
            result['report'] = self.report
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Progress':
        """Create a Progress instance from a dictionary."""
        return cls(
            current=data.get('current', 0),
            total=data.get('total', 0),
            status=data.get('status', ProgressState.INITIALIZING),
            errors=data.get('errors', 0),
            message=data.get('message'),
            report=data.get('report')
        )
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if progress is complete."""
        return self.status == ProgressState.COMPLETE or (
            self.current >= self.total and self.status != ProgressState.ERROR
        )
    
    @property
    def has_error(self) -> bool:
        """Check if progress has error."""
        return self.status == ProgressState.ERROR


def initialize_progress(progress_file: str, total: int) -> None:
    """
    Initialize the progress tracking file.
    
    Args:
        progress_file: Path to the progress file
        total: Total number of items to process
        
    Raises:
        ProgressTrackingError: If progress file cannot be created
    """
    try:
        progress = Progress(current=0, total=total, status=ProgressState.INITIALIZING)
        with open(progress_file, 'w') as pf:
            json.dump(progress.to_dict(), pf)
    except Exception as e:
        msg = f"Failed to initialize progress file: {e}"
        logging.error(msg)
        raise ProgressTrackingError(msg) from e


def update_progress(
    progress_file: str, 
    current: Optional[int] = None,
    total: Optional[int] = None,
    status: Optional[str] = None,
    errors: Optional[int] = None,
    message: Optional[str] = None,
    report: Optional[str] = None
) -> None:
    """
    Update the progress tracking file.
    
    Args:
        progress_file: Path to the progress file
        current: Current progress count (if None, keep existing value)
        total: Total items (if None, keep existing value)
        status: Progress status (if None, keep existing value)
        errors: Error count (if None, keep existing value)
        message: Optional message (e.g., error details)
        report: Path to the generated report
        
    Raises:
        ProgressTrackingError: If progress file cannot be updated
    """
    try:
        # Read current progress
        progress = read_progress(progress_file)
        
        # Update values if provided
        if current is not None:
            progress.current = current
        if total is not None:
            progress.total = total
        if status is not None:
            progress.status = status
        if errors is not None:
            progress.errors = errors
        if message is not None:
            progress.message = message
        if report is not None:
            progress.report = report
        
        # Write updated progress
        with open(progress_file, 'w') as pf:
            json.dump(progress.to_dict(), pf)
    except Exception as e:
        # Log but don't raise (non-critical)
        logging.error(f"Failed to update progress file: {e}")


def read_progress(progress_file: str) -> Progress:
    """
    Read progress from tracking file.
    
    Args:
        progress_file: Path to the progress file
        
    Returns:
        Progress object with current state
        
    Raises:
        ProgressTrackingError: If progress file cannot be read
    """
    try:
        if not os.path.exists(progress_file):
            return Progress()
            
        with open(progress_file) as pf:
            data = json.load(pf)
        return Progress.from_dict(data)
    except Exception as e:
        msg = f"Failed to read progress file: {e}"
        logging.error(msg)
        raise ProgressTrackingError(msg) from e


# -------------------- URL Validation & Parsing --------------------

def validate_sitemap_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted and could be a sitemap URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
    """
    if not url:
        return False
        
    # Basic URL format validation
    url_pattern = re.compile(
        r'^(https?://)' # http:// or https://
        r'([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+' # domain
        r'[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?' # tld
        r'(/[a-zA-Z0-9_-]+)*' # path
        r'(/[a-zA-Z0-9_-]+\.(xml|txt))$' # sitemap file
    )
    
    return bool(url_pattern.match(url))


def extract_domain_from_url(url: str) -> str:
    """
    Extract the domain name from a URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        Domain name without www prefix
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain
    except Exception as e:
        logging.warning(f"Failed to extract domain from {url}: {e}")
        return "unknown"


def is_valid_xml(xml_content: str) -> bool:
    """
    Check if the provided content is valid XML.
    
    Args:
        xml_content: XML content to validate
        
    Returns:
        True if valid XML, False otherwise
    """
    try:
        ET.fromstring(xml_content)
        return True
    except Exception:
        return False


def is_valid_sitemap(xml_content: str) -> bool:
    """
    Check if the provided XML content is a valid sitemap.
    
    Args:
        xml_content: XML content to validate
        
    Returns:
        True if valid sitemap, False otherwise
    """
    try:
        root = ET.fromstring(xml_content)
        
        # Check for sitemap namespace
        ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Try with namespace first
        loc_elements = root.findall('.//sm:loc', ns)
        
        # If not found, try without namespace
        if not loc_elements:
            loc_elements = root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            
        return len(loc_elements) > 0
    except Exception:
        return False


# -------------------- Logging Setup --------------------

def configure_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file (if None, log to console only)
        max_size: Maximum size of log file before rotation in bytes (default: 10 MB)
        backup_count: Number of backup log files to keep (default: 3)
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            logging.info(f"Logging to {log_file} (level: {logging.getLevelName(log_level)})")
        except Exception as e:
            logging.warning(f"Failed to set up file logging: {e}")


# -------------------- Error Handling --------------------

def format_error(error: Exception) -> str:
    """
    Format an exception into a readable error message.
    
    Args:
        error: Exception to format
        
    Returns:
        Formatted error message
    """
    if isinstance(error, HTTPError):
        status_code = getattr(error.response, 'status_code', None)
        if status_code:
            return f"HTTP Error {status_code}: {str(error)}"
    
    return f"{error.__class__.__name__}: {str(error)}"


def safe_request(
    url: str, 
    method: str = 'GET', 
    timeout: int = 30, 
    **kwargs
) -> requests.Response:
    """
    Make a safe HTTP request with proper error handling.
    
    Args:
        url: URL to request
        method: HTTP method (default: GET)
        timeout: Request timeout in seconds (default: 30)
        **kwargs: Additional arguments to pass to requests
        
    Returns:
        Response object
        
    Raises:
        SitemapError: For sitemap-related errors
        APIError: For API-related errors
        RequestException: For general request errors
    """
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response
    except HTTPError as e:
        status_code = getattr(e.response, 'status_code', None)
        if 'sitemap' in url.lower():
            if status_code == 404:
                raise SitemapError(f"Sitemap not found (404): {url}")
            elif status_code == 403:
                raise SitemapError(f"Access denied to sitemap (403): {url}")
            else:
                raise SitemapError(f"HTTP error fetching sitemap: {e}")
        else:
            raise APIError(f"HTTP error: {e}")
    except Timeout:
        if 'sitemap' in url.lower():
            raise SitemapError(f"Timeout fetching sitemap: {url}")
        else:
            raise APIError(f"API request timeout: {url}")
    except ConnectionError:
        raise RequestException(f"Connection error: {url}")
    except Exception as e:
        raise RequestException(f"Request failed: {e}")


def retry(
    max_retries: int = 3,
    retry_delay: int = 1,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    backoff_factor: float = 2.0
) -> Callable:
    """
    Decorator for retrying a function on exception.
    
    Args:
        max_retries: Maximum number of retries (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1)
        exceptions: Exception types to catch and retry (default: Exception)
        backoff_factor: Factor to increase delay on each retry (default: 2.0)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logging.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                    
                    if attempt < max_retries - 1:
                        logging.info(f"Retrying in {delay} seconds...")
                        import time
                        time.sleep(delay)
                        delay *= backoff_factor
            

