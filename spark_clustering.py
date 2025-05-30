import os
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (e.g. DATABASE_URL)
load_dotenv

# Parse DATABASE_URL (assumed to be postgresql://user:password@host:port/dbname) for JDBC
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

db_url = DATABASE_URL.replace("postgresql://", "")
user_pass, host_port_db = db_url.split("@")
user, password = user_pass.split(":")
host_port, dbname = host_port_db.split("/")
host, port = host_port.split(":")

jdbc_url = f"jdbc:postgresql://{host}:{port}/{dbname}"
connection_properties = { "user": user, "password": password, "driver": "org.postgresql.Driver" }

# Create SQLAlchemy engine for storing results
engine = create_engine(DATABASE_URL)

def create_spark_session():
    """Create and configure a local Spark session."""
    return SparkSession.builder.appName("Clustering (Offline)").getOrCreate()

def prepare_cluster_table():
    """Drop (if exists) and re-create the user_clusters table."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS user_clusters"))
        conn.execute(text("""
            CREATE TABLE user_clusters (
                user_id INT,
                cluster INT,
                k INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("user_clusters table recreated.")

def run_clustering(spark, k=5):
    """Run KMeans clustering on user features (assumed to be in a table named 'user_features') and insert results into user_clusters."""
    try:
        # (1) Read user features (e.g. from a table named 'user_features') from PostgreSQL.
        # (Adjust the table name and columns as needed.)
        print("Reading user features from PostgreSQL...")
        features_df = spark.read.jdbc(
            url=jdbc_url,
            table="user_features",  # (or your actual features table)
            properties=connection_properties
        )
        print("Features DataFrame loaded.")

        # (2) Assemble features (e.g. "feature1", "feature2", ...) into a vector column (named "features").
        # (Adjust the inputCols as needed.)
        assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
        assembled_df = assembler.transform(features_df)

        # (3) Run KMeans clustering (with k clusters) on the assembled features.
        print(f"Running KMeans (k={k})...")
        kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k)
        model = kmeans.fit(assembled_df)
        clustered_df = model.transform(assembled_df)

        # (4) Convert (user_id, cluster) to pandas and add a "k" column.
        clustered_pd = clustered_df.select("user_id", "cluster").toPandas()
        clustered_pd["k"] = k

        # (5) Insert (or "upsert") the computed clusters into the "user_clusters" table.
        print("Inserting clustering results into user_clusters...")
        with engine.begin() as conn:
            clustered_pd.to_sql("user_clusters", conn, if_exists="append", index=False)

        print(f"Clustering (k={k}) complete. Results stored in user_clusters.")
        return True
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        return False

def main():
    spark = create_spark_session()
    try:
        # (Optional) Re-create (or "prepare") the user_clusters table.
        prepare_cluster_table()

        # (For example, run clustering with k=5.)
        success = run_clustering(spark, k=5)
        if success:
            print("Clustering (offline) job completed successfully.")
            # (Optional) Query and print a sample of stored clusters.
            with engine.connect() as conn:
                sample = pd.read_sql_query("SELECT * FROM user_clusters LIMIT 5", conn)
                print("Sample of stored clusters:")
                print(sample)
    finally:
        spark.stop()
        print("Spark session stopped.")

if __name__ == "__main__":
    main() 