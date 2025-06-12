from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
import os
from datetime import datetime
import hashlib
from typing import Dict, List, Optional
import json
from src.utils.logging_config import setup_logging, get_logger

# Set up logging for ETL
logger = setup_logging('etl')

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
                # Create metadata table if it doesn't exist (PostgreSQL compatible)
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
        
        # Get expected schema from database (PostgreSQL compatible)
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' 
                AND table_schema = 'public'
            """))
            expected_columns = {row.column_name: row.data_type for row in result}
        
        # Check columns
        missing_cols = set(expected_columns.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check data types (simplified for PostgreSQL)
        for col, expected_type in expected_columns.items():
            if col in df.columns:
                if 'integer' in expected_type.lower() and not pd.api.types.is_integer_dtype(df[col]):
                    errors.append(f"Column {col} should be integer")
                elif 'character' in expected_type.lower() and not pd.api.types.is_string_dtype(df[col]):
                    errors.append(f"Column {col} should be string")
        
        # Check for nulls in primary keys
        if table_name == 'products':
            if df['product_id'].isnull().any():
                errors.append("Null values in product_id")
        elif table_name == 'orders':
            if df['order_id'].isnull().any():
                errors.append("Null values in order_id")
        elif table_name == 'aisles':
            if df['aisle_id'].isnull().any():
                errors.append("Null values in aisle_id")
        
        return errors
    
    def _create_table_schema(self, table_name: str):
        """Create table schema if it doesn't exist"""
        try:
            with self.engine.connect() as conn:
                if table_name == 'aisles':
                    conn.execute(text('''
                        CREATE TABLE IF NOT EXISTS aisles (
                            aisle_id INTEGER PRIMARY KEY,
                            aisle VARCHAR(255) NOT NULL
                        )
                    '''))
                elif table_name == 'departments':
                    conn.execute(text('''
                        CREATE TABLE IF NOT EXISTS departments (
                            department_id INTEGER PRIMARY KEY,
                            department VARCHAR(255) NOT NULL
                        )
                    '''))
                elif table_name == 'products':
                    conn.execute(text('''
                        CREATE TABLE IF NOT EXISTS products (
                            product_id INTEGER PRIMARY KEY,
                            product_name VARCHAR(255) NOT NULL,
                            aisle_id INTEGER,
                            department_id INTEGER,
                            FOREIGN KEY (aisle_id) REFERENCES aisles(aisle_id),
                            FOREIGN KEY (department_id) REFERENCES departments(department_id)
                        )
                    '''))
                elif table_name == 'orders':
                    conn.execute(text('''
                        CREATE TABLE IF NOT EXISTS orders (
                            order_id INTEGER PRIMARY KEY,
                            user_id INTEGER NOT NULL,
                            order_number INTEGER NOT NULL,
                            order_dow INTEGER NOT NULL,
                            order_hour_of_day INTEGER NOT NULL,
                            days_since_prior_order INTEGER
                        )
                    '''))
                elif table_name == 'order_products':
                    conn.execute(text('''
                        CREATE TABLE IF NOT EXISTS order_products (
                            order_id INTEGER,
                            product_id INTEGER,
                            add_to_cart_order INTEGER,
                            reordered INTEGER,
                            PRIMARY KEY (order_id, product_id),
                            FOREIGN KEY (order_id) REFERENCES orders(order_id),
                            FOREIGN KEY (product_id) REFERENCES products(product_id)
                        )
                    '''))
                elif table_name == 'products_with_price':
                    conn.execute(text('''
                        CREATE TABLE IF NOT EXISTS products_with_price (
                            product_id INTEGER PRIMARY KEY,
                            price DECIMAL(10,2) NOT NULL,
                            FOREIGN KEY (product_id) REFERENCES products(product_id)
                        )
                    '''))
                
                conn.commit()
                logger.info(f"Created/verified table schema for {table_name}")
                
        except Exception as e:
            logger.error(f"Error creating table schema for {table_name}: {e}")
    
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
            
            # Create table schema if needed
            self._create_table_schema(table_name)
            
            # Read and clean data
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()
            
            # Clean nulls for critical tables
            if table_name in ['products', 'orders', 'aisles']:
                df = df.dropna()
            
            # Validate data
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
                    if_exists='replace' if i == 0 else 'append',
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
                # PostgreSQL compatible upsert
                conn.execute(
                    text("""
                        INSERT INTO etl_metadata 
                        (file_name, processed_at, file_hash, status, row_count, error_message)
                        VALUES (:file_name, :processed_at, :file_hash, :status, :row_count, :error_message)
                        ON CONFLICT (file_name) 
                        DO UPDATE SET 
                            processed_at = EXCLUDED.processed_at,
                            file_hash = EXCLUDED.file_hash,
                            status = EXCLUDED.status,
                            row_count = EXCLUDED.row_count,
                            error_message = EXCLUDED.error_message
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
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            with self.engine.connect() as conn:
                indexes = [
                    'CREATE INDEX IF NOT EXISTS idx_product_id ON products(product_id)',
                    'CREATE INDEX IF NOT EXISTS idx_department_id ON products(department_id)',
                    'CREATE INDEX IF NOT EXISTS idx_aisle_id ON products(aisle_id)',
                    'CREATE INDEX IF NOT EXISTS idx_order_id ON order_products(order_id)',
                    'CREATE INDEX IF NOT EXISTS idx_product_id_orders ON order_products(product_id)',
                    'CREATE INDEX IF NOT EXISTS idx_order_id_orders ON orders(order_id)',
                    'CREATE INDEX IF NOT EXISTS idx_product_id_products_with_price ON products_with_price(product_id)'
                ]
                
                for index_sql in indexes:
                    conn.execute(text(index_sql))
                
                conn.commit()
                logger.info("Created database indexes for performance")
                
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def run_pipeline(self):
        """Run the complete ETL pipeline"""
        # Define processing order to maintain referential integrity
        processing_order = [
            ('aisles.csv', 'aisles'),
            ('departments.csv', 'departments'),
            ('products.csv', 'products'),
            ('orders.csv', 'orders'),
            ('order_products__train.csv', 'order_products'),
            ('products_with_price.csv', 'products_with_price')
        ]
        
        success = True
        for file_name, table_name in processing_order:
            if not self.process_file(file_name, table_name):
                success = False
                logger.error(f"Pipeline failed at {file_name}")
                break
        
        if success:
            # Create indexes after all data is loaded
            self._create_indexes()
            logger.info("ETL pipeline completed successfully!")
        else:
            logger.error("ETL pipeline failed!")
        
        return success

if __name__ == "__main__":
    # Example usage
    if "DATABASE_URL" not in os.environ:
        raise RuntimeError("DATABASE_URL environment variable must be set for PostgreSQL connection.")
    pipeline = ETLPipeline(
        source_path='data',
        db_url=os.environ['DATABASE_URL']
    )
    pipeline.run_pipeline() 