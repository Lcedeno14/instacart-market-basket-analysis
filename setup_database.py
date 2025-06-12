#!/usr/bin/env python3
"""
Database Setup Script using ETL Pipeline
========================================

This script replaces create_db.py with a more robust ETL pipeline approach.
It provides better error handling, data validation, and tracking of processed files.

Usage:
    python setup_database.py

Features:
- Validates data before loading
- Tracks processed files to avoid reprocessing
- Loads data in chunks for large files
- Creates proper database indexes
- Maintains referential integrity
- Detailed logging and error reporting
"""

import os
import sys
from dotenv import load_dotenv
from etl_pipeline import ETLPipeline
from src.utils.logging_config import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging('database_setup')

def main():
    """Main function to set up the database using ETL pipeline"""
    
    # Check for required environment variable
    if "DATABASE_URL" not in os.environ:
        logger.error("DATABASE_URL environment variable must be set for PostgreSQL connection.")
        logger.error("Please create a .env file with your DATABASE_URL")
        sys.exit(1)
    
    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found!")
        logger.error("Please ensure your CSV files are in the 'data' directory")
        sys.exit(1)
    
    # Check for required CSV files
    required_files = [
        'aisles.csv',
        'departments.csv', 
        'products.csv',
        'orders.csv',
        'order_products__train.csv',
        'products_with_price.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required CSV files: {missing_files}")
        logger.error("Please ensure all required files are in the 'data' directory")
        sys.exit(1)
    
    try:
        # Initialize ETL pipeline
        logger.info("Initializing ETL pipeline...")
        pipeline = ETLPipeline(
            source_path=data_dir,
            db_url=os.environ['DATABASE_URL']
        )
        
        # Run the pipeline
        logger.info("Starting database setup...")
        success = pipeline.run_pipeline()
        
        if success:
            logger.info("✅ Database setup completed successfully!")
            logger.info("You can now run your dashboard with: python app.py")
        else:
            logger.error("❌ Database setup failed!")
            logger.error("Check the logs above for details")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error during database setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 