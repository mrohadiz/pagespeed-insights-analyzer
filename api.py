"""
PageSpeed API Client Module

This module contains functions for interacting with the Google PageSpeed Insights API,
including fetching data, handling rate limits, and processing API responses.
"""

import logging
import random
import time
import requests
import streamlit as st
import backoff
import functools
import xml.etree.ElementTree as ET
from typing import Dict, Any, Union, Optional, List, Callable
from datetime import timedelta
from requests.exceptions import RequestException, Timeout, HTTPError, ConnectionError

# API Constants
PAGESPEED_API_BASE_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
API_REQUEST_TIMEOUT = 30  # seconds
API_RETRY_DELAY = 2  # seconds
API_RATE_LIMIT = 20  # maximum requests per minute (adjusted for PageSpeed API limits)
API_RATE_PERIOD = 60  # period in seconds for rate limiting
MAX_RETRY_ATTEMPTS = 3  # maximum number of retries for API requests
CACHE_TTL_HOURS = 24  # cache TTL for API responses
BACKOFF_FACTOR = 2  # factor for exponential backoff


def rate_limited(max_per_minute: int) -> Callable:
    """
    Rate limiting decorator that limits function calls.
    
    Args:
        max_per_minute: Maximum number of allowed calls per minute
        
    Returns:
        Decorated function with rate limiting
    """
    min_interval = 60.0 / max_per_minute
    
    def decorator(func: Callable) -> Callable:
        last_time_called = [0.0]
        
        @functools.wraps(func)
        def rate_limited_function(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
                
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        
        return rate_limited_function
    
    return decorator


def get_error_result(url: str) -> Dict[str, Any]:
    """
    Create an error result dictionary for a URL.
    
    Args:
        url: The URL that failed analysis
        
    Returns:
        Dictionary with error indicators for all metrics
    """
    return {
        'url': url,
        'performance_score': 'Error',
        'seo_score': 'Error',
        'first-contentful-paint': 'Error',
        'largest-contentful-paint': 'Error',
        'speed-index': 'Error',
        'interactive': 'Error',
        'total-blocking-time': 'Error',
        'cumulative-layout-shift': 'Error'
    }


def validate_api_key(api_key: str) -> bool:
    """
    Validates if the provided API key is properly formatted and works with the API.
    
    Args:
        api_key: Google API key to validate
        
    Returns:
        bool: True if the API key is valid, False otherwise
    """
    # Basic validation - API keys are typically ~39 characters
    if not api_key or len(api_key) < 20:
        return False
    
    # Test the API with a simple request
    try:
        test_url = f"{PAGESPEED_API_BASE_URL}?url=https://example.com&key={api_key}&strategy=mobile&category=performance"
        response = requests.get(test_url, timeout=API_REQUEST_TIMEOUT)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"API key validation error: {e}")
        return False


@backoff.on_exception(
    backoff.expo,
    (ConnectionError, Timeout, RequestException),
    max_tries=MAX_RETRY_ATTEMPTS,
    factor=BACKOFF_FACTOR,
    jitter=backoff.full_jitter
)
def _fetch_pagespeed_data(url: str, api_key: str, strategy: str) -> Dict[str, Any]:
    """
    Internal function to fetch PageSpeed data with retry logic.
    
    Args:
        url: The web page URL to analyze
        api_key: Google API key
        strategy: Either 'mobile' or 'desktop'
        
    Returns:
        Raw PageSpeed API response
    """
    api_url = (
        f"{PAGESPEED_API_BASE_URL}?"
        f"url={url}&key={api_key}&strategy={strategy}&category=performance&category=seo"
    )
    
    # Add a small random delay to prevent concurrent requests hitting rate limits
    time.sleep(random.uniform(0.1, 0.5))
    
    resp = requests.get(api_url, timeout=API_REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@rate_limited(API_RATE_LIMIT)
@st.cache_data(ttl=timedelta(hours=CACHE_TTL_HOURS), show_spinner=False)
def get_pagespeed_data(url: str, api_key: str, strategy: str = "mobile", max_retries: int = MAX_RETRY_ATTEMPTS) -> Dict[str, Union[str, float]]:
    """
    Fetches PageSpeed Insights data for a given URL with caching and retry logic.
    
    This function retrieves performance metrics from Google PageSpeed Insights API,
    processes the response, and extracts key metrics for analysis. The function
    implements caching to reduce API calls and rate limiting to avoid quota issues.
    
    Args:
        url: The web page URL to analyze
        api_key: Google API key for PageSpeed Insights API
        strategy: Either 'mobile' or 'desktop' (default: 'mobile')
        max_retries: Maximum number of retry attempts if request fails (default: 3)
        
    Returns:
        Dictionary containing the following PageSpeed metrics:
          - url: The URL that was analyzed
          - performance_score: Overall performance score (0-100)
          - seo_score: SEO score (0-100)
          - first-contentful-paint: Time to first contentful paint in seconds
          - largest-contentful-paint: Time to largest contentful paint in seconds
          - speed-index: Speed index in seconds
          - interactive: Time to interactive in seconds
          - total-blocking-time: Total blocking time in milliseconds
          - cumulative-layout-shift: Cumulative layout shift score
          
        If an error occurs, metrics will contain the string 'Error'
    """
    try:
        logging.info(f"Fetching PageSpeed data for {url}")
        
        # Use backoff decorator for retry logic
        data = _fetch_pagespeed_data(url, api_key, strategy)
        
        # Extract and validate the Lighthouse result
        if not data or 'lighthouseResult' not in data:
            logging.error(f"Invalid API response for {url}: Missing lighthouseResult")
            return get_error_result(url)
            
        # Extract the main sections from the response
        lh = data.get('lighthouseResult', {})
        categories = lh.get('categories', {})
        audits = lh.get('audits', {})
        
        # Validate essential data is present
        if not categories or not audits:
            logging.error(f"Incomplete API response for {url}: Missing categories or audits")
            return get_error_result(url)
            
        # Process and extract performance metrics
        try:
            # Get performance and SEO scores
            perf_score = categories.get('performance', {}).get('score', 0)
            seo_score = categories.get('seo', {}).get('score', 0)
            
            # Convert to percentage (0-100 scale)
            perf_score = round(perf_score * 100, 1) if isinstance(perf_score, (int, float)) else 'Error'
            seo_score = round(seo_score * 100, 1) if isinstance(seo_score, (int, float)) else 'Error'
            
            # Extract timing metrics and ensure they are valid
            result = {
                'url': url,
                'performance_score': perf_score,
                'seo_score': seo_score
            }
            
            # Process audit metrics - convert ms to seconds where appropriate
            timing_metrics = {
                'first-contentful-paint': lambda v: round(v / 1000, 2),  # ms to seconds
                'largest-contentful-paint': lambda v: round(v / 1000, 2),  # ms to seconds
                'speed-index': lambda v: round(v / 1000, 2),  # ms to seconds
                'interactive': lambda v: round(v / 1000, 2),  # ms to seconds
                'total-blocking-time': lambda v: round(v, 0),  # keep as ms
                'cumulative-layout-shift': lambda v: round(v, 3)  # unitless
            }
            
            # Extract each metric, applying the appropriate transformation
            for metric, transform_func in timing_metrics.items():
                try:
                    # Get numeric value from the audit
                    value = audits.get(metric, {}).get('numericValue')
                    
                    if value is not None and isinstance(value, (int, float)):
                        result[metric] = transform_func(value)
                    else:
                        result[metric] = 'Error'
                        logging.warning(f"Invalid {metric} value for {url}: {value}")
                except Exception as e:
                    result[metric] = 'Error'
                    logging.warning(f"Error processing {metric} for {url}: {e}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error parsing PageSpeed data for {url}: {e}")
            return get_error_result(url)
            
    except Exception as e:
        logging.error(f"Failed to get PageSpeed data for {url}: {e}")
        return get_error_result(url)


@backoff.on_exception(
    backoff.expo,
    (ConnectionError, Timeout, RequestException),
    max_tries=MAX_RETRY_ATTEMPTS,
    factor=BACKOFF_FACTOR
)
def fetch_url_with_retry(url: str, timeout: int = API_REQUEST_TIMEOUT) -> requests.Response:
    """
    Fetch a URL with retry logic and exponential backoff.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Response object
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def get_urls_from_sitemap(sitemap_url: str, max_urls: Optional[int] = None) -> List[str]:
    """
    Extract URLs from a sitemap XML.
    
    Args:
        sitemap_url: URL of the sitemap
        max_urls: Maximum number of URLs to return (None for all)
        
    Returns:
        List of URLs from the sitemap
        
    Raises:
        Exception: If the sitemap cannot be fetched or parsed
    """
    logging.info(f"Fetching URLs from sitemap: {sitemap_url}")
    
    # Fetch the sitemap with retry logic
    response = fetch_url_with_retry(sitemap_url)
    xml_content = response.text
    
    # Parse XML
    root = ET.fromstring(xml_content)
    
    # Define namespace mappings
    namespaces = {
        'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
        'xhtml': 'http://www.w3.org/1999/xhtml'
    }
    
    urls = []
    
    # Try with standard sitemap namespace first
    loc_elements = root.findall('.//sm:loc', namespaces)
    
    # If not found, try without namespace
    if not loc_elements:
        # Try with direct namespace URL in path
        loc_elements = root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
        
        # If still not found, try without any namespace
        if not loc_elements:
            loc_elements = root.findall('.//loc')
    
    # Extract URLs from loc elements
    for loc in loc_elements:
        urls.append(loc.text.strip())
        # Apply limit if specified
        if max_urls and len(urls) >= max_urls:
            break
    
    logging.info(f"Found {len(urls)} URLs in sitemap{' (limited to ' + str(max_urls) + ')' if max_urls else ''}")
    return urls

