"""
Data Processing Module

This module handles all data processing operations for PageSpeed Insights data,
including DataFrame creation, statistical analysis, and Excel export functionality.
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from io import BytesIO
from datetime import datetime


# Metric definitions
METRIC_MAP = {
    'performance_score': 'Performance Score',
    'seo_score': 'SEO Score',
    'first-contentful-paint': 'FCP (s)',
    'largest-contentful-paint': 'LCP (s)',
    'speed-index': 'Speed Index (s)',
    'interactive': 'TTI (s)',
    'total-blocking-time': 'TBT (ms)',
    'cumulative-layout-shift': 'CLS'
}


def process_pagespeed_results(results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process raw PageSpeed results into a structured DataFrame and generate analysis data.
    
    Args:
        results: List of PageSpeed data dictionaries
        
    Returns:
        Tuple containing:
        - Results DataFrame
        - Summary DataFrame
        - Criteria analysis DataFrame
        - Top 5 slowest pages DataFrame
    """
    try:
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Create analysis DataFrames
        df_summary = create_summary_dataframe(df_results)
        df_criteria = create_criteria_dataframe(df_results)
        df_top5 = create_top5_slowest_dataframe(df_results)
        
        return df_results, df_summary, df_criteria, df_top5
    except Exception as e:
        logging.error(f"Error processing PageSpeed results: {e}")
        # Return empty DataFrames in case of error
        return (
            pd.DataFrame(), 
            pd.DataFrame(columns=['Metrik', 'Rata-rata', 'Median', '75 persen', 'Terendah', 'Tertinggi']),
            pd.DataFrame(columns=['Kriteria', 'Total', 'Optimal', 'Persentase']),
            pd.DataFrame(columns=['URL', 'Perf Score', 'LCP', 'TTI'])
        )


def create_summary_dataframe(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary DataFrame with statistics for each metric.
    
    Args:
        df_results: DataFrame containing PageSpeed results
        
    Returns:
        Summary DataFrame with statistics for each metric
    """
    try:
        # Validate input DataFrame
        if df_results.empty:
            logging.warning("Empty DataFrame, cannot create summary")
            return pd.DataFrame(columns=['Metrik', 'Rata-rata', 'Median', '75 persen', 'Terendah', 'Tertinggi'])
        
        # Filter out error rows
        valid = df_results[df_results['performance_score'] != 'Error']
        if valid.empty:
            logging.warning("No valid results to create summary")
            return pd.DataFrame(columns=['Metrik', 'Rata-rata', 'Median', '75 persen', 'Terendah', 'Tertinggi'])
        
        # Generate statistics for each metric
        summary = []
        for col, label in METRIC_MAP.items():
            try:
                # Ensure the column exists
                if col not in valid.columns:
                    logging.warning(f"Column {col} not found in results DataFrame")
                    continue
                
                # Convert to numeric values
                vals = pd.to_numeric(valid[col], errors='coerce')
                
                # Calculate statistics
                if not vals.isna().all():
                    summary.append({
                        'Metrik': label,
                        'Rata-rata': round(vals.mean(), 2),
                        'Median': round(vals.median(), 2),
                        '75 persen': round(vals.quantile(0.75), 2),
                        'Terendah': round(vals.min(), 2),
                        'Tertinggi': round(vals.max(), 2)
                    })
                else:
                    raise ValueError(f"No valid numeric data for {col}")
                    
            except Exception as e:
                logging.warning(f"Could not calculate statistics for {label}: {e}")
                summary.append({
                    'Metrik': label,
                    'Rata-rata': 'N/A',
                    'Median': 'N/A',
                    '75 persen': 'N/A',
                    'Terendah': 'N/A',
                    'Tertinggi': 'N/A'
                })
        
        # Create and return the DataFrame
        return pd.DataFrame(summary)
        
    except Exception as e:
        logging.error(f"Failed to create summary DataFrame: {e}")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['Metrik', 'Rata-rata', 'Median', '75 persen', 'Terendah', 'Tertinggi'])


def create_criteria_dataframe(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame with criteria-based analysis of results.
    
    Args:
        df_results: DataFrame containing PageSpeed results
        
    Returns:
        Criteria analysis DataFrame with performance metrics
    """
    try:
        # Validate input DataFrame
        if df_results.empty:
            logging.warning("Empty DataFrame, cannot create criteria analysis")
            return pd.DataFrame(columns=['Kriteria', 'Total', 'Optimal', 'Persentase'])
        
        # Filter out error rows
        valid = df_results[df_results['performance_score'] != 'Error']
        if valid.empty:
            logging.warning("No valid results to create criteria analysis")
            return pd.DataFrame(columns=['Kriteria', 'Total', 'Optimal', 'Persentase'])
        
        # Convert columns to float for comparison
        for col in ['performance_score', 'speed-index']:
            if col in valid.columns:
                valid[col] = pd.to_numeric(valid[col], errors='coerce')
        
        # Calculate criteria counts
        total_valid = len(valid)
        
        # Performance score >= 80
        perf_ok = valid[valid['performance_score'] >= 80]
        perf_count = len(perf_ok)
        perf_pct = f"{round(perf_count/total_valid*100, 1)}%" if total_valid > 0 else "0%"
        
        # Speed index < 4 seconds
        si_ok = valid[valid['speed-index'] < 4]
        si_count = len(si_ok)
        si_pct = f"{round(si_count/total_valid*100, 1)}%" if total_valid > 0 else "0%"
        
        # Both criteria met
        both_ok = valid[(valid['performance_score'] >= 80) & (valid['speed-index'] < 4)]
        both_count = len(both_ok)
        both_pct = f"{round(both_count/total_valid*100, 1)}%" if total_valid > 0 else "0%"
        
        # Create criteria DataFrame
        return pd.DataFrame([
            {'Kriteria': 'Perf ≥80', 'Total': total_valid, 'Optimal': perf_count, 'Persentase': perf_pct},
            {'Kriteria': 'SI<4s', 'Total': total_valid, 'Optimal': si_count, 'Persentase': si_pct},
            {'Kriteria': 'Kedua', 'Total': total_valid, 'Optimal': both_count, 'Persentase': both_pct}
        ])
        
    except Exception as e:
        logging.error(f"Failed to create criteria DataFrame: {e}")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['Kriteria', 'Total', 'Optimal', 'Persentase'])


def create_top5_slowest_dataframe(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame showing the 5 slowest pages based on performance score.
    
    Args:
        df_results: DataFrame containing PageSpeed results
        
    Returns:
        DataFrame with the 5 slowest pages and key metrics
    """
    try:
        # Validate input DataFrame
        if df_results.empty:
            logging.warning("Empty DataFrame, cannot create Top 5 slowest pages")
            return pd.DataFrame(columns=['URL', 'Perf Score', 'LCP', 'TTI'])
        
        # Filter out error rows
        valid = df_results[df_results['performance_score'] != 'Error']
        if valid.empty:
            logging.warning("No valid results to create Top 5 slowest pages")
            return pd.DataFrame(columns=['URL', 'Perf Score', 'LCP', 'TTI'])
            
        # Convert columns to numeric for sorting
        cols_to_convert = ['performance_score', 'largest-contentful-paint', 'interactive']
        for col in cols_to_convert:
            if col in valid.columns:
                valid[col] = pd.to_numeric(valid[col], errors='coerce')
            
        # Check if we have the required columns
        required_cols = ['url', 'performance_score', 'largest-contentful-paint', 'interactive']
        missing_cols = [col for col in required_cols if col not in valid.columns]
        
        if missing_cols:
            logging.warning(f"Missing columns in results DataFrame: {missing_cols}")
            # Create a DataFrame with the correct columns but no data
            return pd.DataFrame(columns=['URL', 'Perf Score', 'LCP', 'TTI'])
            
        # Get 5 slowest pages
        df_low5 = valid.nsmallest(5, 'performance_score')[required_cols].copy()
        
        # Rename columns for better readability
        df_low5.columns = ['URL', 'Perf Score', 'LCP', 'TTI']
        
        return df_low5
        
    except Exception as e:
        logging.error(f"Failed to create Top 5 slowest pages DataFrame: {e}")
        return pd.DataFrame(columns=['URL', 'Perf Score', 'LCP', 'TTI'])


def export_to_excel(report_dfs: Dict[str, pd.DataFrame], include_formatting: bool = True) -> bytes:
    """
    Export multiple DataFrames to an Excel file in memory with formatting.
    
    Args:
        report_dfs: Dictionary mapping sheet names to DataFrames
        include_formatting: Whether to include additional formatting (default: True)
        
    Returns:
        Excel file as bytes
    """
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, df in report_dfs.items():
                # Skip empty DataFrames
                if df.empty:
                    continue
                    
                # Write DataFrame to Excel
                df.to_excel(writer, index=False, sheet_name=sheet_name)
                
                # Apply formatting if requested
                if include_formatting:
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]
                    
                    # Add some formatting
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })
                    
                    # Set column widths
                    for i, col in enumerate(df.columns):
                        max_len = max(
                            df[col].astype(str).map(len).max(),
                            len(str(col))
                        ) + 2
                        worksheet.set_column(i, i, min(max_len, 50))
                    
                    # Apply header format
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                        
                    # Add conditional formatting for performance scores if they exist
                    if 'performance_score' in df.columns:
                        col_idx = df.columns.get_loc('performance_score')
                        worksheet.conditional_format(1, col_idx, len(df)+1, col_idx, {
                            'type': '3_color_scale',
                            'min_color': "#FF0000",
                            'mid_color': "#FFFF00",
                            'max_color': "#00FF00"
                        })
        
        return output.getvalue()
    except Exception as e:
        logging.error(f"Failed to export Excel file: {e}")
        raise


def save_excel_report(excel_data: bytes, filepath: str) -> bool:
    """
    Save Excel data to a file.
    
    Args:
        excel_data: Excel file data as bytes
        filepath: Path where to save the file
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        with open(filepath, 'wb') as f:
            f.write(excel_data)
        logging.info(f"Excel report saved to {filepath}")
        return True
    except Exception as e:
        logging.error(f"Failed to save Excel report to {filepath}: {e}")
        return False


def generate_report_filename(domain: str, base_dir: str) -> str:
    """
    Generate a filename for a PageSpeed report based on domain and date.
    
    Args:
        domain: Website domain
        base_dir: Base directory where to save the file
        
    Returns:
        Full path to the report file
    """
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{domain}_{date_str}.xlsx"
    return os.path.join(base_dir, filename)

