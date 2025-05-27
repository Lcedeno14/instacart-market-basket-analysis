import os
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import collect_list, col
from dotenv import load_dotenv
import random
import json
from sqlalchemy import create_engine, text
import pandas as pd

# Load environment variables
load_dotenv()
print("Debug: DATABASE_URL (after load_dotenv) =", os.getenv("DATABASE_URL"))

# Get database URL and parse it for Spark JDBC
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Parse DATABASE_URL (postgresql://user:password@host:port/dbname)
# Remove postgresql:// prefix
db_url = DATABASE_URL.replace("postgresql://", "")
# Split into user:pass and host:port/dbname
user_pass, host_port_db = db_url.split("@")
user, password = user_pass.split(":")
host_port, dbname = host_port_db.split("/")
host, port = host_port.split(":")

# Construct JDBC URL and properties
jdbc_url = f"jdbc:postgresql://{host}:{port}/{dbname}"
connection_properties = {
    "user": user,
    "password": password,
    "driver": "org.postgresql.Driver"
}

# Create SQLAlchemy engine for storing results
engine = create_engine(DATABASE_URL)

def create_spark_session():
    """Create and configure Spark session"""
    return SparkSession.builder \
        .appName("Instacart Market Basket Analysis") \
        .config("spark.jars", "postgresql-42.7.2.jar") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "10g") \
        .getOrCreate()

def test_postgres_connection(spark):
    """Test PostgreSQL connection with a simple query"""
    try:
        # Try to read a small sample from order_products
        test_df = spark.read.jdbc(
            url=jdbc_url,
            table="(SELECT * FROM order_products LIMIT 5) as test",
            properties=connection_properties
        )
        print("Successfully connected to PostgreSQL!")
        print("Sample data:")
        test_df.show()
        return True
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {str(e)}")
        return False

def prepare_market_basket_table():
    """Prepare the market_basket_rules table by dropping and recreating it"""
    with engine.begin() as conn:
        # Drop and recreate the table
        conn.execute(text('DROP TABLE IF EXISTS market_basket_rules'))
        conn.execute(text('''
            CREATE TABLE market_basket_rules (
                id SERIAL PRIMARY KEY,
                algorithm TEXT,
                support FLOAT,
                confidence FLOAT,
                lift FLOAT,
                antecedents TEXT,
                consequents TEXT,
                support_param FLOAT,
                confidence_param FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''))
        print("Market basket rules table recreated.")

def run_market_basket_analysis(spark):
    """Run market basket analysis on the entire dataset"""
    try:
        # Ensure table exists
        prepare_market_basket_table()

        # Parameters for grid search with lower thresholds
        support_values = [0.001, 0.002, 0.005, 0.01, 0.02]  # From 0.1% to 2%
        confidence_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # From 10% to 50%

        # Load all transactions
        print("Reading all orders and products from PostgreSQL...")
        transactions_df = spark.read.jdbc(
            url=jdbc_url,
            table="""
                (SELECT op.order_id, p.product_name 
                FROM order_products op 
                JOIN products p ON op.product_id = p.product_id
                ) as transactions
            """,
            properties=connection_properties
        )
        print(f"Loaded {transactions_df.count()} product orders.")
        
        # Create transactions for FP-Growth
        print("Preparing transactions for analysis...")
        transactions = transactions_df \
            .groupBy("order_id") \
            .agg(collect_list("product_name").alias("items"))
        print(f"Created {transactions.count()} unique order transactions.")
        
        # Run grid search with different support and confidence values
        for support in support_values:
            for confidence in confidence_values:
                print(f"\nRunning FP-Growth with support={support}, confidence={confidence}")
                # Run FP-Growth
                fp_growth = FPGrowth(
                    itemsCol="items",
                    minSupport=support,
                    minConfidence=confidence
                )
                
                model = fp_growth.fit(transactions)
                rules = model.associationRules
                
                if rules.count() > 0:
                    # Convert Spark DataFrame to pandas for easier handling
                    rules_pd = rules.toPandas()
                    
                    # Convert frozensets to lists and then to JSON strings
                    rules_pd['antecedents'] = rules_pd['antecedent'].apply(lambda x: json.dumps(list(x)))
                    rules_pd['consequents'] = rules_pd['consequent'].apply(lambda x: json.dumps(list(x)))
                    
                    # Add algorithm and parameter info
                    rules_pd['algorithm'] = 'fp_growth'
                    rules_pd['support_param'] = support
                    rules_pd['confidence_param'] = confidence
                    
                    # Store in database
                    with engine.begin() as conn:
                        for _, row in rules_pd.iterrows():
                            conn.execute(text('''
                                INSERT INTO market_basket_rules 
                                (algorithm, support, confidence, lift, antecedents, consequents, support_param, confidence_param)
                                VALUES (:algorithm, :support, :confidence, :lift, :antecedents, :consequents, :support_param, :confidence_param)
                            '''), {
                                'algorithm': row['algorithm'],
                                'support': float(row['support']),
                                'confidence': float(row['confidence']),
                                'lift': float(row['lift']),
                                'antecedents': row['antecedents'],
                                'consequents': row['consequents'],
                                'support_param': row['support_param'],
                                'confidence_param': row['confidence_param']
                            })
                    
                    print(f"Stored {len(rules_pd)} rules for support={support}, confidence={confidence}")
                else:
                    print(f"No rules found for support={support}, confidence={confidence}")
        
        print("\nMarket basket analysis complete! Results stored in market_basket_rules table.")
        return True
        
    except Exception as e:
        print(f"Error in market basket analysis: {str(e)}")
        return False

def main():
    # Create Spark session
    print("Initializing Spark session...")
    spark = create_spark_session()
    
    try:
        # Test connection
        if not test_postgres_connection(spark):
            return
        
        # Run analysis on entire dataset
        success = run_market_basket_analysis(spark)
        
        if success:
            print("\nAnalysis complete!")
            # Query and show some stats about stored rules
            with engine.connect() as conn:
                stats = pd.read_sql_query("""
                    SELECT algorithm, support_param, confidence_param, COUNT(*) as rule_count
                    FROM market_basket_rules
                    GROUP BY algorithm, support_param, confidence_param
                    ORDER BY algorithm, support_param, confidence_param
                """, conn)
                print("\nStored rules statistics:")
                print(stats)
        
    finally:
        # Stop Spark session
        spark.stop()
        print("\nSpark session stopped.")

if __name__ == "__main__":
    main() 