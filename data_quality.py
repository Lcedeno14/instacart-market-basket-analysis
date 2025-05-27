import pandas as pd
from sqlalchemy import create_engine, text
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_quality():
    """
    Perform data quality checks on the database
    Returns True if all checks pass, False otherwise
    """
    try:
        # Connect to database
        DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instacart.db")
        engine = create_engine(DATABASE_URL)
        
        checks_passed = True
        
        with engine.connect() as conn:
            # 1. Check if all required tables exist
            required_tables = ['orders', 'products', 'departments', 'order_products']
            existing_tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", 
                conn
            )['name'].tolist()
            
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                checks_passed = False
            
            # 2. Check for null values in critical columns
            for table in existing_tables:
                null_check_query = f"""
                SELECT column_name, COUNT(*) as null_count
                FROM (
                    SELECT * FROM {table}
                    WHERE 1=0
                    UNION ALL
                    SELECT * FROM {table}
                    WHERE {' OR '.join(f"{col} IS NULL" for col in 
                        pd.read_sql_query(f"PRAGMA table_info({table})", conn)['name'])}
                )
                GROUP BY column_name
                HAVING null_count > 0
                """
                null_counts = pd.read_sql_query(null_check_query, conn)
                if not null_counts.empty:
                    logger.error(f"Null values found in {table}:")
                    logger.error(null_counts)
                    checks_passed = False
            
            # 3. Check referential integrity
            integrity_checks = [
                # Check if all product_ids in order_products exist in products
                """
                SELECT COUNT(*) as invalid_count
                FROM order_products op
                LEFT JOIN products p ON op.product_id = p.product_id
                WHERE p.product_id IS NULL
                """,
                # Check if all department_ids in products exist in departments
                """
                SELECT COUNT(*) as invalid_count
                FROM products p
                LEFT JOIN departments d ON p.department_id = d.department_id
                WHERE d.department_id IS NULL
                """,
                # Check if all order_ids in order_products exist in orders
                """
                SELECT COUNT(*) as invalid_count
                FROM order_products op
                LEFT JOIN orders o ON op.order_id = o.order_id
                WHERE o.order_id IS NULL
                """
            ]
            
            for i, check in enumerate(integrity_checks):
                result = pd.read_sql_query(check, conn)
                if result['invalid_count'].iloc[0] > 0:
                    logger.error(f"Referential integrity check {i+1} failed")
                    checks_passed = False
            
            # 4. Check data volume
            min_row_counts = {
                'orders': 1000,
                'products': 100,
                'departments': 10,
                'order_products': 5000
            }
            
            for table, min_count in min_row_counts.items():
                if table in existing_tables:
                    count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)['count'].iloc[0]
                    if count < min_count:
                        logger.error(f"Table {table} has only {count} rows, minimum expected is {min_count}")
                        checks_passed = False
            
            # 5. Check for duplicate primary keys
            for table in existing_tables:
                table_info = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
                pk_columns = table_info[table_info['pk'] == 1]['name'].tolist()
                
                if pk_columns:
                    pk_check = f"""
                    SELECT {', '.join(pk_columns)}, COUNT(*) as count
                    FROM {table}
                    GROUP BY {', '.join(pk_columns)}
                    HAVING count > 1
                    """
                    duplicates = pd.read_sql_query(pk_check, conn)
                    if not duplicates.empty:
                        logger.error(f"Duplicate primary keys found in {table}")
                        logger.error(duplicates)
                        checks_passed = False
        
        if checks_passed:
            logger.info("All data quality checks passed!")
        else:
            logger.error("Some data quality checks failed!")
        
        return checks_passed
        
    except Exception as e:
        logger.error(f"Error during data quality check: {str(e)}")
        return False

if __name__ == "__main__":
    check_data_quality() 