"""
User Interface Module

This module handles all Streamlit UI components and interactions for the PageSpeed Analyzer application,
including forms, progress tracking, and report display.
"""

import os
import time
import streamlit as st
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from urllib.parse import urlparse

from utils import (
    Progress, 
    ProgressState, 
    read_progress, 
    validate_sitemap_url, 
    extract_domain_from_url,
    format_error
)


# -------------------- UI Constants --------------------

# App information
APP_TITLE = "📊 PageSpeed Insights Analyzer"
APP_DESCRIPTION = """
Analyze the performance of your website using Google PageSpeed Insights API.
Upload a sitemap to analyze multiple pages at once.
"""

# Form default values
DEFAULT_SITEMAP_URL = "https://jagobahasa.com/post-sitemap.xml"
DEFAULT_URL_LIMIT = 100
DEFAULT_STRATEGY = "mobile"

# CSS styles
CSS = """
<style>
.block-container {max-width: 1200px}
.report-download {margin-top: 1rem; padding: 0.5rem; border-radius: 0.3rem; border: 1px solid #ccc}
.error-box {background-color: #ffebee; padding: 1rem; border-radius: 0.3rem; margin: 1rem 0}
.info-box {background-color: #e3f2fd; padding: 1rem; border-radius: 0.3rem; margin: 1rem 0}
.success-box {background-color: #e8f5e9; padding: 1rem; border-radius: 0.3rem; margin: 1rem 0}
</style>
"""


# -------------------- Session State Helpers --------------------

def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False
    
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
    
    if "progress_file" not in st.session_state:
        st.session_state.progress_file = None
    
    if "report_file" not in st.session_state:
        st.session_state.report_file = None
    
    if "api_key_validated" not in st.session_state:
        st.session_state.api_key_validated = False


def reset_session_state():
    """Reset session state for a new analysis."""
    st.session_state.form_submitted = False
    st.session_state.analysis_running = False
    st.session_state.progress_file = None
    st.session_state.report_file = None
    st.session_state.api_key_validated = False


# -------------------- UI Setup --------------------

def setup_page():
    """Configure page settings and apply custom styles."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom CSS
    st.markdown(CSS, unsafe_allow_html=True)
    
    # Display app title and description
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)


# -------------------- Form Components --------------------

def render_input_form() -> Dict[str, Any]:
    """
    Render the input form for PageSpeed Analyzer.
    
    Returns:
        Dictionary of form values if submitted, empty dict otherwise
    """
    with st.form("pagespeed_form"):
        st.subheader("Analysis Settings")
        
        # Sitemap URL input with validation hint
        sitemap_url = st.text_input(
            "🔗 Sitemap URL",
            value=DEFAULT_SITEMAP_URL,
            help="URL to your sitemap.xml file. Example: https://example.com/sitemap.xml"
        )
        
        # API Key input (from environment or user input)
        default_api_key = os.environ.get('PAGESPEED_API_KEY', '')
        api_key_help = "Google API Key for PageSpeed Insights. Get one from Google Cloud Console."
        
        if default_api_key:
            api_key = st.text_input(
                "🔑 Google API Key", 
                type='password', 
                value=default_api_key,
                help=f"{api_key_help} (Detected from environment)"
            )
        else:
            api_key = st.text_input(
                "🔑 Google API Key", 
                type='password',
                help=api_key_help
            )
        
        # URL limit settings
        col1, col2 = st.columns(2)
        with col1:
            use_limit = st.checkbox("Limit number of URLs", value=True)
        
        with col2:
            url_limit = st.number_input(
                "Maximum URLs to analyze",
                min_value=1,
                max_value=500,
                value=DEFAULT_URL_LIMIT,
                disabled=not use_limit
            )
        
        # Strategy selection
        strategy = st.selectbox(
            "Analysis Strategy",
            options=["mobile", "desktop"],
            index=0,
            help="Device to simulate during analysis"
        )
        
        # Background processing option
        bg_process = st.checkbox(
            "Run in Background",
            value=False,
            help="Run analysis in background to keep UI responsive"
        )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Start Analysis")
        
        if submitted:
            # Collect form values
            return {
                "sitemap_url": sitemap_url,
                "api_key": api_key,
                "url_limit": url_limit if use_limit else None,
                "strategy": strategy,
                "bg_process": bg_process
            }
        
        return {}


def validate_form_input(form_data: Dict[str, Any]) -> List[str]:
    """
    Validate form inputs and return any error messages.
    
    Args:
        form_data: Dictionary of form values
        
    Returns:
        List of error messages, empty if no errors
    """
    errors = []
    
    # Check sitemap URL
    if not form_data.get("sitemap_url"):
        errors.append("Sitemap URL is required")
    elif not validate_sitemap_url(form_data.get("sitemap_url", "")):
        errors.append("Invalid sitemap URL format. Must be a URL ending with .xml or .txt")
    
    # Check API key
    if not form_data.get("api_key"):
        errors.append("Google API Key is required")
    elif len(form_data.get("api_key", "")) < 20:
        errors.append("API Key appears to be invalid (too short)")
    
    # Check URL limit
    if form_data.get("url_limit") is not None and form_data.get("url_limit") < 1:
        errors.append("URL limit must be at least 1")
    
    return errors


# -------------------- Progress Tracking UI --------------------

def display_progress(progress_file: str) -> Tuple[bool, str]:
    """
    Display progress of analysis using a progress bar and status message.
    
    Args:
        progress_file: Path to the progress tracking file
        
    Returns:
        Tuple of (is_complete, report_path)
    """
    # Create UI placeholders
    progress_bar = st.progress(0)
    status_message = st.empty()
    error_message = st.empty()
    
    report_path = None
    is_complete = False
    was_error_shown = False
    
    # Poll progress file and update UI
    try:
        while True:
            if os.path.exists(progress_file):
                try:
                    # Read progress
                    progress = read_progress(progress_file)
                    
                    # Update progress bar
                    if progress.total > 0:
                        progress_bar.progress(min(progress.current / progress.total, 1.0))
                    
                    # Update status message based on state
                    if progress.status == ProgressState.INITIALIZING:
                        status_message.info("Initializing analysis...")
                    elif progress.status == ProgressState.RUNNING:
                        status_message.info(f"Analyzing {progress.current}/{progress.total} URLs...")
                    elif progress.status == ProgressState.RETRYING:
                        status_message.warning(f"Retrying failed URLs ({progress.current}/{progress.total})...")
                    elif progress.status == ProgressState.COMPLETE:
                        status_message.success("Analysis complete!")
                        report_path = progress.report
                        is_complete = True
                        break
                    elif progress.status == ProgressState.ERROR:
                        status_message.error("Analysis failed!")
                        error_message.error(progress.message or "Unknown error occurred")
                        is_complete = True
                        break
                    
                    # Show error count if any
                    if progress.errors > 0 and not was_error_shown:
                        error_message.warning(f"{progress.errors} URLs failed analysis")
                        was_error_shown = True
                    
                    # Check if complete
                    if progress.is_complete:
                        status_message.success("Analysis complete!")
                        report_path = progress.report
                        is_complete = True
                        break
                        
                except Exception as e:
                    status_message.warning(f"Error reading progress: {e}")
                    
            # Sleep to avoid hammering the file system
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        status_message.warning("Progress monitoring interrupted")
    
    return (is_complete, report_path)


# -------------------- Report Display --------------------

def display_reports(base_dir: str, current_report: Optional[str] = None):
    """
    Display available reports and download buttons.
    
    Args:
        base_dir: Directory containing reports
        current_report: Path to the most recent report (optional)
    """
    st.subheader("Reports")
    
    # Get all Excel reports
    reports = [f for f in os.listdir(base_dir) if f.endswith('.xlsx')]
    
    if not reports:
        st.info("No reports available")
        return
    
    # Sort reports by modification time (newest first)
    reports.sort(key=lambda f: os.path.getmtime(os.path.join(base_dir, f)), reverse=True)
    
    # Highlight current report if specified
    if current_report and os.path.basename(current_report) in reports:
        st.success(f"New report generated: {os.path.basename(current_report)}")
        
        # Create download button for current report
        with open(current_report, 'rb') as f:
            st.download_button(
                "📥 Download Latest Report",
                f.read(),
                os.path.basename(current_report),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        # Display modified time
        mod_time = os.path.getmtime(current_report)
        st.caption(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))}")
    
    # Show all reports with download buttons
    st.subheader("Previous Reports")
    
    for report in reports:
        report_path = os.path.join(base_dir, report)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.text(report)
            mod_time = os.path.getmtime(report_path)
            st.caption(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))}")
            
        with col2:
            with open(report_path, 'rb') as f:
                st.download_button(
                    "📥 Download",
                    f.read(),
                    report,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{report}"
                )


# -------------------- Background Task --------------------

def run_background_task(task_func: Callable, *args, **kwargs):
    """
    Run a function in a background thread.
    
    Args:
        task_func: Function to run in background
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    """
    thread = threading.Thread(
        target=task_func,
        args=args,
        kwargs=kwargs,
        daemon=True
    )
    thread.start()
    return thread


# -------------------- Main UI Flow --------------------

def show_error(message: str):
    """Display an error message in styled container."""
    st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)


def show_info(message: str):
    """Display an info message in styled container."""
    st.markdown(f'<div class="info-box">{message}</div>', unsafe_allow_html=True)


def show_success(message: str):
    """Display a success message in styled container."""
    st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)

