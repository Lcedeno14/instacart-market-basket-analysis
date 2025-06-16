import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, explode, array_contains
from pyspark.sql.types import StringType, ArrayType, StructType, StructField
from pyspark.ml.fpm import FPGrowth

# Load environment variables
load_dotenv()
print("Debug: DATABASE_URL (after load_dotenv) =", os.getenv("DATABASE_URL"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

db_url = DATABASE_URL.replace("postgresql://", "")
user_pass, host_port_db = db_url.split("@")
user, password = user_pass.split(":")
host_port, dbname = host_port_db.split("/")
host, port = host_port.split(":")

jdbc_url = f"jdbc:postgresql://{host}:{port}/{dbname}"
connection_properties = {
    "user": user,
    "password": password,
    "driver": "org.postgresql.Driver"
}

engine = create_engine(DATABASE_URL)

def create_spark_session():
    return SparkSession.builder \
        .appName("MarketBasketAnalysisSimple") \
        .config("spark.jars", "postgresql-42.7.2.jar") \
        .config("spark.driver.extraClassPath", "postgresql-42.7.2.jar") \
        .config("spark.executor.extraClassPath", "postgresql-42.7.2.jar") \
        .config("spark.ui.port", "4043") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

def prepare_market_basket_table():
    """Prepare market basket rules table"""
    with engine.begin() as conn:
        # Drop existing table
        conn.execute(text("DROP TABLE IF EXISTS market_basket_rules"))
        
        # Create new table without algorithm column
        conn.execute(text("""
            CREATE TABLE market_basket_rules (
                id SERIAL PRIMARY KEY,
                support DECIMAL(10,6) NOT NULL,
                confidence DECIMAL(10,6) NOT NULL,
                lift DECIMAL(10,6) NOT NULL,
                antecedents JSONB NOT NULL,
                consequents JSONB NOT NULL,
                support_param DECIMAL(10,6) NOT NULL,
                confidence_param DECIMAL(10,6) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Create new table for price-weighted rules
        conn.execute(text("DROP TABLE IF EXISTS market_basket_rules_weighted"))
        
        conn.execute(text("""
            CREATE TABLE market_basket_rules_weighted (
                id SERIAL PRIMARY KEY,
                support DECIMAL(10,6) NOT NULL,
                confidence DECIMAL(10,6) NOT NULL,
                lift DECIMAL(10,6) NOT NULL,
                antecedents JSONB NOT NULL,
                consequents JSONB NOT NULL,
                support_param DECIMAL(10,6) NOT NULL,
                confidence_param DECIMAL(10,6) NOT NULL,
                avg_antecedent_price DECIMAL(10,2),
                avg_consequent_price DECIMAL(10,2),
                revenue_potential DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
    print("Market basket rules tables recreated.")

def load_all_data(spark):
    """Load all data from products_with_price table"""
    print("Loading all orders from database (products_with_price)...")
    
    print("Loading all product orders with price data...")
    # Get all products with prices
    transactions_df = spark.read.jdbc(
        url=jdbc_url,
        table="""
            (SELECT op.order_id, p.product_name
             FROM order_products op 
             JOIN products_with_price pwp ON op.product_id = pwp.product_id
             JOIN products p ON op.product_id = p.product_id
            ) as transactions
        """,
        properties=connection_properties
    )
    product_count = transactions_df.count()
    print(f"Loaded {product_count} product orders with price data.")

    print("Aggregating products per order...")
    # Aggregate products per order
    basket_df = transactions_df.groupBy("order_id").agg(collect_list("product_name").alias("items"))
    basket_count = basket_df.count()
    print(f"Aggregated to {basket_count} baskets.")
    
    return basket_df

def load_all_data_with_prices(spark):
    """Load all data with price information from database"""
    print("Loading all orders with price data from database...")
    
    print("Loading all product orders with prices...")
    # Get all products with prices
    transactions_df = spark.read.jdbc(
        url=jdbc_url,
        table="""
            (SELECT op.order_id, p.product_name, pwp.price
             FROM order_products op 
             JOIN products_with_price pwp ON op.product_id = pwp.product_id
             JOIN products p ON op.product_id = p.product_id
            ) as transactions
        """,
        properties=connection_properties
    )
    product_count = transactions_df.count()
    print(f"Loaded {product_count} product orders with prices.")

    print("Aggregating products per order...")
    # Aggregate products per order
    basket_df = transactions_df.groupBy("order_id").agg(
        collect_list("product_name").alias("items"),
        collect_list("price").alias("prices")
    )
    basket_count = basket_df.count()
    print(f"Aggregated to {basket_count} baskets with price data.")
    
    return basket_df

def run_fp_growth(spark, basket_df):
    """Run FP-Growth algorithm using PySpark"""
    print("Running FP-Growth algorithm...")
    
    # Parameters for full dataset
    support_values = [0.001, 0.005, 0.01]
    confidence_values = [0.1, 0.2, 0.3]
    
    total_rules = 0
    
    for support in support_values:
        for confidence in confidence_values:
            print(f"  FP-Growth with support={support}, confidence={confidence}")
            
            try:
                print(f"    Creating FP-Growth model...")
                # Run FP-Growth
                fpGrowth = FPGrowth(itemsCol="items", minSupport=support, minConfidence=confidence)
                
                print(f"    Fitting model to data...")
                model = fpGrowth.fit(basket_df)
                
                print(f"    Getting association rules...")
                # Get association rules
                rules = model.associationRules
                
                print(f"    Counting rules...")
                rules_count = rules.count()
                
                if rules_count == 0:
                    print(f"    No rules found")
                    continue
                
                print(f"    Found {rules_count} association rules")
                
                # Store rules with progress bar
                print(f"    Collecting rules for storage...")
                rules_list = rules.collect()
                print(f"    Storing {len(rules_list)} rules to database...")
                
                for row in tqdm(rules_list, desc=f"    Storing FP-Growth rules"):
                    with engine.begin() as conn:
                        conn.execute(text('''
                            INSERT INTO market_basket_rules 
                            (support, confidence, lift, antecedents, consequents, support_param, confidence_param)
                            VALUES (:support, :confidence, :lift, :antecedents, :consequents, :support_param, :confidence_param)
                        '''), {
                            'support': float(row['support']),
                            'confidence': float(row['confidence']),
                            'lift': float(row['lift']),
                            'antecedents': json.dumps(row['antecedent']),
                            'consequents': json.dumps(row['consequent']),
                            'support_param': support,
                            'confidence_param': confidence
                        })
                
                total_rules += rules_count
                print(f"    Completed support={support}, confidence={confidence}")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                continue
    
    print(f"FP-Growth complete! Total rules stored: {total_rules}")
    return total_rules

def calculate_avg_price(items):
    """Calculate average price for a list of items"""
    try:
        # Get prices for these items from the database
        with engine.connect() as conn:
            # Convert items list to SQL-friendly format
            items_str = "','".join(items)
            query = f"""
                SELECT AVG(pwp.price) as avg_price
                FROM products_with_price pwp
                JOIN products p ON pwp.product_id = p.product_id
                WHERE p.product_name IN ('{items_str}')
            """
            result = conn.execute(text(query))
            avg_price = result.fetchone()[0]
            return avg_price if avg_price is not None else 0.0
    except Exception as e:
        print(f"Error calculating average price: {e}")
        return 0.0

def run_fp_growth_weighted(spark, basket_df):
    """Run FP-Growth algorithm with price weighting for revenue optimization"""
    print("Running FP-Growth algorithm with price weighting...")
    
    # Parameters for full dataset
    support_values = [0.001, 0.005, 0.01]
    confidence_values = [0.1, 0.2, 0.3]
    
    total_rules = 0
    
    for support in support_values:
        for confidence in confidence_values:
            print(f"  FP-Growth with support={support}, confidence={confidence}")
            
            try:
                # Run FP-Growth
                fpGrowth = FPGrowth(itemsCol="items", minSupport=support, minConfidence=confidence)
                model = fpGrowth.fit(basket_df)
                
                # Get association rules
                rules = model.associationRules
                rules_count = rules.count()
                
                if rules_count == 0:
                    print(f"    No rules found")
                    continue
                
                print(f"    Found {rules_count} association rules")
                
                # Store rules with price information
                rules_list = rules.collect()
                for rule in tqdm(rules_list, desc=f"    Storing price-weighted FP-Growth rules"):
                    antecedents = rule.antecedent
                    consequents = rule.consequent
                    support_val = rule.support
                    confidence_val = rule.confidence
                    lift_val = rule.lift
                    
                    # Calculate average prices for antecedents and consequents
                    avg_antecedent_price = calculate_avg_price(antecedents)
                    avg_consequent_price = calculate_avg_price(consequents)
                    
                    # Calculate revenue potential (confidence * avg_consequent_price)
                    revenue_potential = confidence_val * avg_consequent_price
                    
                    # Store rule
                    with engine.begin() as conn:
                        conn.execute(text('''
                            INSERT INTO market_basket_rules_weighted 
                            (support, confidence, lift, antecedents, consequents, support_param, confidence_param, 
                             avg_antecedent_price, avg_consequent_price, revenue_potential)
                            VALUES (:support, :confidence, :lift, :antecedents, :consequents, :support_param, :confidence_param,
                                    :avg_antecedent_price, :avg_consequent_price, :revenue_potential)
                        '''), {
                            'support': float(support_val),
                            'confidence': float(confidence_val),
                            'lift': float(lift_val),
                            'antecedents': json.dumps(antecedents),
                            'consequents': json.dumps(consequents),
                            'support_param': support,
                            'confidence_param': confidence,
                            'avg_antecedent_price': float(avg_antecedent_price),
                            'avg_consequent_price': float(avg_consequent_price),
                            'revenue_potential': float(revenue_potential)
                        })
                
                total_rules += rules_count
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                continue
    
    print(f"Price-weighted FP-Growth complete! Total rules stored: {total_rules}")
    return total_rules

def main():
    """Main function to run market basket analysis"""
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Recreate market basket rules tables
        prepare_market_basket_table()
        
        # Load all data
        basket_df = load_all_data(spark)
        
        print("\n=== Running FP-Growth Analysis ===")
        fp_growth_rules = run_fp_growth(spark, basket_df)
        
        print("\n=== Summary ===")
        print(f"FP-Growth rules: {fp_growth_rules}")
        
    finally:
        # Stop Spark session
        spark.stop()
        print("Spark session stopped.")

if __name__ == "__main__":
    main() 