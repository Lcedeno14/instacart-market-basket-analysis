#!/usr/bin/env python3
"""
Full Market Basket Analysis Runner
Runs optimized analysis on all orders and compares results
Outputs everything to a text file for overnight processing
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path

def log_to_file(message, log_file):
    """Write message to both console and log file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    log_file.write(formatted_message + "\n")
    log_file.flush()  # Ensure immediate writing

def run_command_with_logging(command, description, log_file, timeout=None):
    """Run a command and log its output"""
    log_to_file(f"Starting: {description}", log_file)
    log_to_file(f"Command: {command}", log_file)
    log_to_file("-" * 80, log_file)
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                log_to_file(line.rstrip(), log_file)
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            log_to_file(f"✅ Completed successfully: {description}", log_file)
        else:
            log_to_file(f"❌ Failed with return code {return_code}: {description}", log_file)
        
        return return_code == 0
        
    except subprocess.TimeoutExpired:
        log_to_file(f"⏰ Timeout occurred: {description}", log_file)
        process.kill()
        return False
    except Exception as e:
        log_to_file(f"❌ Error running {description}: {str(e)}", log_file)
        return False

def modify_script_for_all_orders():
    """Modify the optimized script to use all orders instead of 100"""
    script_path = "spark_market_basket_optimized.py"
    
    # Read the current script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace the sampling functions with all-orders functions
    content = content.replace(
        "def load_sampled_data_optimized(spark, sample_size=100):",
        "def load_all_data_optimized(spark):"
    )
    content = content.replace(
        "def load_sampled_data_with_prices_optimized(spark, sample_size=100):",
        "def load_all_data_with_prices_optimized(spark):"
    )
    
    # Remove sampling logic and restore full data loading
    content = content.replace(
        'print(f"Loading a sample of {sample_size} orders from database (optimized)...")',
        'print("Loading all orders from database (optimized)...")'
    )
    content = content.replace(
        'print(f"Loading a sample of {sample_size} orders with price data (optimized)...")',
        'print("Loading all orders with price data (optimized)...")'
    )
    
    # Remove the sampling code blocks
    import re
    content = re.sub(
        r'# Sample \d+ unique orders.*?sampled_df = transactions_df\.join\(order_ids, on="order_id", how="inner"\)',
        'sampled_df = transactions_df',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'order_ids = transactions_df\.select\("order_id"\)\.distinct\(\)\.limit\(sample_size\)\s+sampled_df = transactions_df\.join\(order_ids, on="order_id", how="inner"\)',
        '',
        content
    )
    
    content = content.replace(
        'product_count = sampled_df.count()',
        'product_count = transactions_df.count()'
    )
    content = content.replace(
        'print(f"Loaded {product_count} product orders for {sample_size} sampled orders.")',
        'print(f"Loaded {product_count} product orders with price data.")'
    )
    content = content.replace(
        'basket_df = sampled_df.groupBy("order_id").agg(collect_list("product_name").alias("items"))',
        'basket_df = transactions_df.groupBy("order_id").agg(collect_list("product_name").alias("items"))'
    )
    content = content.replace(
        'basket_df = sampled_df.groupBy("order_id").agg(\n        collect_list("product_name").alias("items"),\n        collect_list("price").alias("prices")\n    )',
        'basket_df = transactions_df.groupBy("order_id").agg(\n        collect_list("product_name").alias("items"),\n        collect_list("price").alias("prices")\n    )'
    )
    
    # Update function calls in main()
    content = content.replace(
        'basket_df = load_sampled_data_optimized(spark)',
        'basket_df = load_all_data_optimized(spark)'
    )
    content = content.replace(
        'basket_df_with_prices = load_sampled_data_with_prices_optimized(spark)',
        'basket_df_with_prices = load_all_data_with_prices_optimized(spark)'
    )
    
    # Write the modified script
    with open(script_path, 'w') as f:
        f.write(content)
    
    return True

def main():
    """Main execution function"""
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"full_analysis_log_{timestamp}.txt"
    
    print(f"Starting full market basket analysis...")
    print(f"Log file: {log_filename}")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with open(log_filename, 'w') as log_file:
        log_to_file("=" * 80, log_file)
        log_to_file("FULL MARKET BASKET ANALYSIS - ALL ORDERS", log_file)
        log_to_file("=" * 80, log_file)
        log_to_file(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        log_to_file("", log_file)
        
        # Step 1: Modify script for all orders
        log_to_file("STEP 1: Modifying optimized script for all orders", log_file)
        if modify_script_for_all_orders():
            log_to_file("✅ Script modified successfully", log_file)
        else:
            log_to_file("❌ Failed to modify script", log_file)
            return
        
        # Step 2: Run optimized analysis on all orders
        log_to_file("", log_file)
        log_to_file("STEP 2: Running optimized market basket analysis on ALL orders", log_file)
        log_to_file("This may take several hours...", log_file)
        
        success = run_command_with_logging(
            "python spark_market_basket_optimized.py",
            "Optimized Market Basket Analysis (All Orders)",
            log_file,
            timeout=None  # No timeout for overnight run
        )
        
        if not success:
            log_to_file("❌ Market basket analysis failed", log_file)
            return
        
        # Step 3: Run comparison script
        log_to_file("", log_file)
        log_to_file("STEP 3: Running comparison analysis", log_file)
        
        success = run_command_with_logging(
            "python compare_results.py",
            "Results Comparison",
            log_file
        )
        
        if not success:
            log_to_file("❌ Comparison analysis failed", log_file)
        
        # Final summary
        log_to_file("", log_file)
        log_to_file("=" * 80, log_file)
        log_to_file("FULL ANALYSIS COMPLETED", log_file)
        log_to_file(f"Ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        log_to_file("=" * 80, log_file)
        
        # Create a summary file
        summary_filename = f"analysis_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as summary_file:
            summary_file.write("MARKET BASKET ANALYSIS SUMMARY\n")
            summary_file.write("=" * 40 + "\n")
            summary_file.write(f"Analysis completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            summary_file.write(f"Full log file: {log_filename}\n")
            summary_file.write(f"Database tables updated: market_basket_rules, market_basket_rules_weighted\n")
            summary_file.write("\nNext steps:\n")
            summary_file.write("1. Check the database for stored rules\n")
            summary_file.write("2. Review the comparison results\n")
            summary_file.write("3. Analyze high-lift and high-revenue associations\n")
        
        log_to_file(f"Summary file created: {summary_filename}", log_file)

if __name__ == "__main__":
    main() 