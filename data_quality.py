from dotenv import load_dotenv
load_dotenv()
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
        if "DATABASE_URL" not in os.environ:
            raise RuntimeError("DATABASE_URL environment variable must be set for PostgreSQL connection.")
        DATABASE_URL = os.environ["DATABASE_URL"]
        engine = create_engine(DATABASE_URL)
        
        checks_passed = True
        
        with engine.connect() as conn:
            # 1. Check if all required tables exist (PostgreSQL compatible)
            required_tables = ['orders', 'products', 'departments', 'order_products', 'market_basket_rules']
            existing_tables = pd.read_sql_query(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"), 
                conn
            )['table_name'].tolist()
            
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                checks_passed = False
            
            # 2. Check for null values in critical columns
            for table in existing_tables:
                # Get columns for this table
                columns = pd.read_sql_query(
                    text(f"SELECT column_name FROM information_schema.columns WHERE table_name = :table"), 
                    conn, 
                    params={'table': table}
                )['column_name'].tolist()
                
                for col in columns:
                    # Skip market_basket_rules table as it's not critical
                    if table == 'market_basket_rules':
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
                'order_products': 5000
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
            for table in existing_tables:
                # Skip market_basket_rules as it doesn't have a primary key
                if table == 'market_basket_rules':
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

if __name__ == "__main__":
    check_data_quality() 