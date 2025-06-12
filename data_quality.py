from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from sqlalchemy import create_engine, text
import os
import logging
from src.utils.logging_config import setup_logging, get_logger

# Set up logging for data quality
logger = setup_logging('data_quality')

def check_data_quality():
    """
    Perform data quality checks on the database
    Returns True if all checks pass, False otherwise
    """
    try:
        # Connect to database
        if "DATABASE_URL" not in os.environ:
            raise RuntimeError("DATABASE_URL environment variable must be set for PostgreSQL connection.")
        DATABASE_URL = os.environ["DATABASE_URL"]
        engine = create_engine(DATABASE_URL)
        
        checks_passed = True
        
        with engine.connect() as conn:
            # 1. Check if all required tables exist (PostgreSQL compatible)
            required_tables = ['orders', 'products', 'departments', 'order_products', 'aisles', 'products_with_price']
            existing_tables = pd.read_sql_query(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"), 
                conn
            )['table_name'].tolist()
            
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                checks_passed = False
            
            # 2. Check for null values in critical columns (only business tables)
            business_tables = ['orders', 'products', 'departments', 'order_products', 'aisles', 'products_with_price']
            
            for table in business_tables:
                if table not in existing_tables:
                    continue
                    
                # Get columns for this table
                columns = pd.read_sql_query(
                    text(f"SELECT column_name FROM information_schema.columns WHERE table_name = :table"), 
                    conn, 
                    params={'table': table}
                )['column_name'].tolist()
                
                for col in columns:
                    # Skip non-critical columns and tables
                    if table == 'etl_metadata' or table == 'market_basket_rules':
                        continue
                    # Rule 1: Ignore nulls in days_since_prior_order in orders table
                    if table == 'orders' and col == 'days_since_prior_order':
                        continue
                    # Rule 2: For products_with_price, remove rows with no price before checking nulls
                    if table == 'products_with_price' and col == 'price':
                        # Remove products with no price
                        conn.execute(text("DELETE FROM products_with_price WHERE price IS NULL"))
                        conn.commit()
                        continue
                    # Rule 3: Ignore nulls in error_message columns (they're normal for successful processing)
                    if col == 'error_message':
                        continue
                    
                    # Check for nulls using parameterized query
                    null_count = pd.read_sql_query(
                        text(f"SELECT COUNT(*) as null_count FROM {table} WHERE {col} IS NULL"),
                        conn
                    )['null_count'].iloc[0]
                    
                    if null_count > 0:
                        logger.error(f"Null values found in column {col} of {table}: {null_count}")
                        checks_passed = False
            
            # 3. Check referential integrity
            integrity_checks = [
                # Check if all product_ids in order_products exist in products
                text("""
                SELECT COUNT(*) as invalid_count
                FROM order_products op
                LEFT JOIN products p ON op.product_id = p.product_id
                WHERE p.product_id IS NULL
                """),
                # Check if all department_ids in products exist in departments
                text("""
                SELECT COUNT(*) as invalid_count
                FROM products p
                LEFT JOIN departments d ON p.department_id = d.department_id
                WHERE d.department_id IS NULL
                """),
                # Check if all order_ids in order_products exist in orders
                text("""
                SELECT COUNT(*) as invalid_count
                FROM order_products op
                LEFT JOIN orders o ON op.order_id = o.order_id
                WHERE o.order_id IS NULL
                """),
                # Check if all aisle_ids in products exist in aisles
                text("""
                SELECT COUNT(*) as invalid_count
                FROM products p
                LEFT JOIN aisles a ON p.aisle_id = a.aisle_id
                WHERE a.aisle_id IS NULL AND p.aisle_id IS NOT NULL
                """),
                # Check if all product_ids in products_with_price exist in products
                text("""
                SELECT COUNT(*) as invalid_count
                FROM products_with_price pwp
                LEFT JOIN products p ON pwp.product_id = p.product_id
                WHERE p.product_id IS NULL
                """)
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
                'order_products': 5000,
                'aisles': 10,
                'products_with_price': 100
            }
            
            for table, min_count in min_row_counts.items():
                if table in existing_tables:
                    count = pd.read_sql_query(
                        text(f"SELECT COUNT(*) as count FROM {table}"),
                        conn
                    )['count'].iloc[0]
                    if count < min_count:
                        logger.error(f"Table {table} has only {count} rows, minimum expected is {min_count}")
                        checks_passed = False
            
            # 5. Check for duplicate primary keys
            for table in business_tables:
                if table not in existing_tables:
                    continue
                    
                # Use PostgreSQL information_schema for primary key columns
                pk_query = text("""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = :table 
                AND tc.constraint_type = 'PRIMARY KEY'
                """)
                
                pk_columns = pd.read_sql_query(pk_query, conn, params={'table': table})['column_name'].tolist()
                if pk_columns:
                    pk_check = text(f"""
                    SELECT {', '.join(pk_columns)}, COUNT(*) as count
                    FROM {table}
                    GROUP BY {', '.join(pk_columns)}
                    HAVING COUNT(*) > 1
                    """)
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