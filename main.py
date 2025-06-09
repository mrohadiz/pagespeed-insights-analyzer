"""
PageSpeed Insights Analyzer

This is the main entry point for the PageSpeed Analyzer application.
It coordinates the interaction between API, data processing, and UI components.

Usage:
    streamlit run main.py
"""

import os
import sys
import logging
import atexit
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

# Import application modules
import api
import data_processing
import utils
import ui

# Constants
BASE_DIR = os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, f"pagespeed_{datetime.now().strftime('%Y%m%d')}.log")
TEMP_DIR = os.path.join(BASE_DIR, "temp")


def setup_application():
    """
    Initialize the application environment.
    
    - Configure logging
    - Create necessary directories
    - Register cleanup functions
    """
    # Make sure necessary directories exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Configure logging
    utils.configure_logging(
        log_level=logging.INFO,
        log_file=LOG_FILE,
        max_size=10 * 1024 * 1024,  # 10 MB
        backup_count=5
    )
    
    # Register cleanup function
    atexit.register(cleanup_temp_files)
    
    logging.info("Application initialized")


def cleanup_temp_files():
    """Clean up temporary files on exit."""
    try:
        # Remove temporary progress files
        for file in os.listdir(TEMP_DIR):
            if file.endswith('.json'):
                file_path = os.path.join(TEMP_DIR, file)
                try:
                    os.remove(file_path)
                    logging.debug(f"Removed temporary file: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove temporary file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")


def get_sitemap_urls(sitemap_url: str, max_urls: Optional[int] = None) -> List[str]:
    """
    Fetch URLs from sitemap and validate.
    
    Args:
        sitemap_url: URL of the sitemap
        max_urls: Maximum number of URLs to fetch
        
    Returns:
        List of URLs from the sitemap
    
    Raises:
        utils.SitemapError: If sitemap cannot be fetched or parsed
    """
    # Validate sitemap URL
    if not utils.validate_sitemap_url(sitemap_url):
        raise utils.SitemapError(f"Invalid sitemap URL format: {sitemap_url}")
    
    # Fetch URLs using the safe_request function
    try:
        response = utils.safe_request(sitemap_url, timeout=api.API_REQUEST_TIMEOUT)
        
        # Check if valid XML
        if not utils.is_valid_xml(response.text):
            raise utils.SitemapError("Invalid XML content in sitemap")
        
        # Check if valid sitemap format
        if not utils.is_valid_sitemap(response.text):
            raise utils.SitemapError("XML file is not a valid sitemap format")
        
        # Parse URLs from sitemap
        return api.get_urls_from_sitemap(sitemap_url, max_urls)
        
    except Exception as e:
        logging.error(f"Failed to get URLs from sitemap: {e}")
        raise utils.SitemapError(f"Failed to get URLs from sitemap: {str(e)}")


def run_analysis(
    sitemap_url: str,
    api_key: str,
    urls: List[str],
    strategy: str,
    progress_file: str,
    report_file: str
) -> bool:
    """
    Run the full PageSpeed analysis process.
    
    Args:
        sitemap_url: URL of the sitemap
        api_key: Google PageSpeed API key
        urls: List of URLs to analyze
        strategy: Analysis strategy ('mobile' or 'desktop')
        progress_file: Path to progress tracking file
        report_file: Path to output report file
        
    Returns:
        True if analysis was successful, False otherwise
    """
    logging.info(f"Starting analysis of {len(urls)} URLs from {sitemap_url}")
    
    # Initialize progress tracking
    total_urls = len(urls)
    utils.initialize_progress(progress_file, total_urls)
    utils.update_progress(progress_file, status=utils.ProgressState.RUNNING)
    
    try:
        # Collect results
        results = []
        error_count = 0
        failed_urls = []
        
        # Process URLs
        for idx, url in enumerate(urls, start=1):
            try:
                logging.info(f"Analyzing URL {idx}/{total_urls}: {url}")
                
                # Get PageSpeed data with rate limiting and caching
                data = api.get_pagespeed_data(url, api_key, strategy=strategy)
                results.append(data)
                
                # Check for errors in results
                if data.get('performance_score') == 'Error':
                    error_count += 1
                    failed_urls.append(url)
                    logging.warning(f"Failed to analyze {url}")
                    
                # Update progress
                utils.update_progress(
                    progress_file,
                    current=idx,
                    errors=error_count,
                    status=utils.ProgressState.RUNNING
                )
                
            except Exception as e:
                error_count += 1
                failed_urls.append(url)
                logging.error(f"Error analyzing {url}: {str(e)}")
                
                # Add error entry to maintain consistent structure
                results.append(api.get_error_result(url))
                
                # Update progress
                utils.update_progress(
                    progress_file,
                    current=idx,
                    errors=error_count,
                    status=utils.ProgressState.RUNNING
                )
        
        # Process the results into DataFrames
        df_results, df_summary, df_criteria, df_top5 = data_processing.process_pagespeed_results(results)
        
        # Generate Excel report
        logging.info("Generating Excel report")
        excel_data = data_processing.export_to_excel({
            'Results': df_results,
            'Summary': df_summary,
            'Criteria': df_criteria,
            'Top5 Slowest': df_top5
        })
        
        # Save Excel file
        if data_processing.save_excel_report(excel_data, report_file):
            logging.info(f"Report saved to: {report_file}")
            
            # Update progress to complete
            utils.update_progress(
                progress_file,
                current=total_urls,
                total=total_urls,
                status=utils.ProgressState.COMPLETE,
                report=report_file
            )
            return True
        else:
            raise RuntimeError("Failed to save Excel report")
            
    except Exception as e:
        error_message = utils.format_error(e)
        logging.error(f"Analysis failed: {error_message}")
        
        # Update progress to error state
        utils.update_progress(
            progress_file,
            status=utils.ProgressState.ERROR,
            message=error_message
        )
        return False


def validate_api_key(api_key: str) -> bool:
    """
    Validate the API key by testing it.
    
    Args:
        api_key: Google PageSpeed API key
        
    Returns:
        True if API key is valid, False otherwise
    """
    try:
        return api.validate_api_key(api_key)
    except Exception as e:
        logging.error(f"API key validation error: {e}")
        return False


def generate_filenames(sitemap_url: str) -> tuple:
    """
    Generate filenames for progress and report.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        Tuple of (progress_file, report_file)
    """
    # Extract domain from URL
    domain = utils.extract_domain_from_url(sitemap_url)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate filenames
    progress_file = os.path.join(TEMP_DIR, f"{domain}_{timestamp}.json")
    report_file = os.path.join(BASE_DIR, f"{domain}_{timestamp}.xlsx")
    
    return progress_file, report_file


def main():
    """Main application entry point."""
    try:
        # Set up application environment
        setup_application()
        
        # Initialize UI
        ui.setup_page()
        ui.init_session_state()
        
        # Display existing reports before form
        ui.display_reports(BASE_DIR)
        
        # Render the input form
        form_data = ui.render_input_form()
        
        # Process form submission
        if form_data:
            # Validate form inputs
            errors = ui.validate_form_input(form_data)
            
            if errors:
                # Display validation errors
                for error in errors:
                    ui.show_error(error)
            else:
                # Form is valid, begin processing
                sitemap_url = form_data["sitemap_url"]
                api_key = form_data["api_key"]
                url_limit = form_data["url_limit"]
                strategy = form_data["strategy"]
                bg_process = form_data["bg_process"]
                
                # Validate API key
                if not validate_api_key(api_key):
                    ui.show_error("Invalid API key or API service unreachable. Please check your key.")
                    return
                
                try:
                    # Fetch URLs from sitemap
                    ui.show_info("Fetching URLs from sitemap...")
                    urls = get_sitemap_urls(sitemap_url, url_limit)
                    
                    if not urls:
                        ui.show_error("No URLs found in the sitemap or sitemap could not be parsed.")
                        return
                    
                    ui.show_info(f"Found {len(urls)} URLs in sitemap.")
                    
                    # Generate filenames
                    progress_file, report_file = generate_filenames(sitemap_url)
                    
                    # Store filenames in session state
                    st.session_state.progress_file = progress_file
                    st.session_state.report_file = report_file
                    st.session_state.analysis_running = True
                    
                    # Run analysis based on mode (background or foreground)
                    if bg_process:
                        # Run in background
                        ui.show_info("Starting analysis in background...")
                        ui.run_background_task(
                            run_analysis,
                            sitemap_url,
                            api_key,
                            urls,
                            strategy,
                            progress_file,
                            report_file
                        )
                        
                        # Display progress
                        is_complete, report_path = ui.display_progress(progress_file)
                        
                        if is_complete and report_path:
                            ui.display_reports(BASE_DIR, report_file)
                    else:
                        # Run in foreground
                        ui.show_info("Starting analysis (this may take some time)...")
                        success = run_analysis(
                            sitemap_url,
                            api_key,
                            urls,
                            strategy,
                            progress_file,
                            report_file
                        )
                        
                        if success:
                            ui.show_success("Analysis complete!")
                            ui.display_reports(BASE_DIR, report_file)
                        else:
                            ui.show_error("Analysis failed. Check logs for details.")
                    
                except utils.SitemapError as e:
                    ui.show_error(f"Sitemap Error: {str(e)}")
                except utils.APIError as e:
                    ui.show_error(f"API Error: {str(e)}")
                except Exception as e:
                    logging.exception("Unexpected error")
                    ui.show_error(f"Unexpected error: {utils.format_error(e)}")
    
    except Exception as e:
        logging.exception("Critical application error")
        st.error(f"Critical error: {utils.format_error(e)}")


if __name__ == "__main__":
    import streamlit as st
    main()

