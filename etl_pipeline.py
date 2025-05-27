import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
import os
from datetime import datetime
import hashlib
from typing import Dict, List, Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self, source_path: str, db_url: str):
        self.source_path = source_path
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.processed_files = self._load_processed_files()
        
    def _load_processed_files(self) -> Dict:
        """Load record of processed files from metadata table"""
        try:
            with self.engine.connect() as conn:
                # Create metadata table if it doesn't exist
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS etl_metadata (
                        file_name TEXT PRIMARY KEY,
                        processed_at TIMESTAMP,
                        file_hash TEXT,
                        status TEXT,
                        row_count INTEGER,
                        error_message TEXT
                    )
                """))
                conn.commit()
                
                # Load existing records
                result = conn.execute(text("SELECT * FROM etl_metadata"))
                return {row.file_name: row for row in result}
        except Exception as e:
            logger.error(f"Error loading processed files: {e}")
            return {}
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _validate_data(self, df: pd.DataFrame, table_name: str) -> List[str]:
        """Validate data before loading"""
        errors = []
        
        # Get expected schema from database
        with self.engine.connect() as conn:
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            expected_columns = {row.name: row.type for row in result}
        
        # Check columns
        missing_cols = set(expected_columns.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check data types
        for col, expected_type in expected_columns.items():
            if col in df.columns:
                if 'INT' in expected_type and not pd.api.types.is_integer_dtype(df[col]):
                    errors.append(f"Column {col} should be integer")
                elif 'TEXT' in expected_type and not pd.api.types.is_string_dtype(df[col]):
                    errors.append(f"Column {col} should be string")
        
        # Check for nulls in primary keys
        if table_name == 'products':
            if df['product_id'].isnull().any():
                errors.append("Null values in product_id")
        elif table_name == 'orders':
            if df['order_id'].isnull().any():
                errors.append("Null values in order_id")
        
        return errors
    
    def process_file(self, file_name: str, table_name: str) -> bool:
        """Process a single file and load it into the database"""
        file_path = os.path.join(self.source_path, file_name)
        
        try:
            # Check if file has been processed
            if file_name in self.processed_files:
                current_hash = self._calculate_file_hash(file_path)
                if current_hash == self.processed_files[file_name].file_hash:
                    logger.info(f"File {file_name} already processed and unchanged")
                    return True
            
            # Read and validate data
            df = pd.read_csv(file_path)
            errors = self._validate_data(df, table_name)
            
            if errors:
                logger.error(f"Validation errors in {file_name}: {errors}")
                self._update_metadata(file_name, 'error', len(df), str(errors))
                return False
            
            # Load data in chunks to handle large files
            chunk_size = 10000
            total_rows = 0
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                chunk.to_sql(
                    table_name,
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                total_rows += len(chunk)
            
            # Update metadata
            file_hash = self._calculate_file_hash(file_path)
            self._update_metadata(file_name, 'success', total_rows)
            
            logger.info(f"Successfully processed {file_name} ({total_rows} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            self._update_metadata(file_name, 'error', 0, str(e))
            return False
    
    def _update_metadata(self, file_name: str, status: str, row_count: int, error_message: Optional[str] = None):
        """Update ETL metadata table"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT OR REPLACE INTO etl_metadata 
                        (file_name, processed_at, file_hash, status, row_count, error_message)
                        VALUES (:file_name, :processed_at, :file_hash, :status, :row_count, :error_message)
                    """),
                    {
                        'file_name': file_name,
                        'processed_at': datetime.now(),
                        'file_hash': self._calculate_file_hash(os.path.join(self.source_path, file_name)),
                        'status': status,
                        'row_count': row_count,
                        'error_message': error_message
                    }
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def run_pipeline(self):
        """Run the complete ETL pipeline"""
        # Define processing order to maintain referential integrity
        processing_order = [
            ('departments.csv', 'departments'),
            ('products.csv', 'products'),
            ('orders.csv', 'orders'),
            ('order_products__train.csv', 'order_products')
        ]
        
        success = True
        for file_name, table_name in processing_order:
            if not self.process_file(file_name, table_name):
                success = False
                logger.error(f"Pipeline failed at {file_name}")
                break
        
        return success

if __name__ == "__main__":
    # Example usage
    pipeline = ETLPipeline(
        source_path='data',
        db_url=os.getenv('DATABASE_URL', 'sqlite:///instacart.db')
    )
    pipeline.run_pipeline() 