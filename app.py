from dotenv import load_dotenv
load_dotenv()

# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import os
import logging
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from data_quality import check_data_quality
from flask import jsonify
import json
from src.analysis.data_storytelling import DataStorytelling
from src.analysis.data_processor import DataProcessor
import plotly.graph_objects as go
from src.analysis.metrics import BusinessMetrics
from src.visualization.kpi_dashboard import KPIDashboard
import time
from src.tabs.basic_analysis_tab import get_basic_analysis_tab_layout, register_basic_analysis_callbacks
from src.tabs.market_basket_tab import get_market_basket_tab_layout, register_market_basket_callbacks
from src.tabs.price_analysis_tab import get_price_analysis_tab_layout, register_price_analysis_callbacks
from src.tabs.data_stories_tab import get_data_stories_tab_layout
from src.tabs.kpi_dashboard_tab import get_kpi_dashboard_tab_layout
from src.tabs.executive_dashboard_tab import get_executive_dashboard_tab_layout, ExecutiveDashboard
from src.tabs.product_category_performance_tab import get_product_category_performance_tab_layout, ProductCategoryPerformanceDashboard
from src.tabs.customer_behavior_tab import get_customer_behavior_tab_layout, CustomerBehaviorDashboard
from src.utils.logging_config import setup_global_logging, get_logger

# Set up global logging configuration
setup_global_logging()
logger = get_logger('app')

# Initialize the Dash application
# This creates a Flask server under the hood
app = dash.Dash(__name__)
server = app.server  # Expose server variable for Railway

# Use DATABASE_URL from environment (set by Railway)
if "DATABASE_URL" not in os.environ:
    raise RuntimeError("DATABASE_URL environment variable must be set for PostgreSQL connection.")
DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL)

# Perform data quality checks before starting
if not check_data_quality():
    logger.error("Data quality checks failed! Please check the logs for details.")
    # Continue anyway, but log the error

# Day of week mapping
DAYS_OF_WEEK = [
    "All Time", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]
DOW_MAP = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}
REVERSE_DOW_MAP = {v: k for k, v in DOW_MAP.items()}
ORDERED_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_LABELS = [f"{h % 12 if h % 12 != 0 else 12}{'am' if h < 12 else 'pm'}" for h in range(24)]

# Function to load data from database
def load_data():
    try:
        with engine.connect() as conn:
            # Load departments for dropdown
            departments_df = pd.read_sql_query("SELECT * FROM departments", conn)
            
            # Load the merged data we need for visualizations, including order_dow and order_hour_of_day
            query = """
            SELECT 
                op.product_id,
                p.product_name,
                d.department,
                d.department_id,
                a.aisle,
                o.order_dow,
                o.order_hour_of_day,
                o.order_id,
                o.user_id,
                o.order_number,
                o.days_since_prior_order,
                CASE 
                    WHEN op.reordered IS NOT NULL THEN op.reordered 
                    ELSE 0 
                END as reordered
            FROM order_products op
            JOIN products p ON op.product_id = p.product_id
            JOIN departments d ON p.department_id = d.department_id
            JOIN aisles a ON p.aisle_id = a.aisle_id
            JOIN orders o ON op.order_id = o.order_id
            """
            merged_df = pd.read_sql_query(query, conn)
            
            # Map order_dow to day name
            merged_df['day_of_week'] = merged_df['order_dow'].map(DOW_MAP)
            
            logger.info("Data loaded successfully")
            return departments_df, merged_df
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        # Return empty DataFrames with the correct structure
        return pd.DataFrame(columns=['department']), pd.DataFrame()

# Initialize data storyteller and data processor
try:
    departments_df, merged_df = load_data()
    
    # Split merged_df into orders and order_products
    orders_df = merged_df[['order_id', 'user_id', 'order_number', 'order_dow', 
                          'order_hour_of_day', 'days_since_prior_order']].drop_duplicates()
    
    # Get order_products dataframe, handling missing reordered column
    order_products_columns = ['order_id', 'product_id']
    if 'reordered' in merged_df.columns:
        order_products_columns.append('reordered')
    order_products_df = merged_df[order_products_columns].drop_duplicates()
    
    # Get products dataframe
    products_df = merged_df[['product_id', 'product_name', 'department_id']].drop_duplicates()
    
    # Initialize data storyteller and data processor
    storyteller = DataStorytelling(merged_df=merged_df)
    
    # Pass already-merged data to DataProcessor to avoid redundant merging
    data_processor = DataProcessor(merged_df=merged_df)
    
except Exception as e:
    logger.error(f"Failed to load initial data: {str(e)}")
    departments_df = pd.DataFrame(columns=['department'])
    merged_df = pd.DataFrame()
    orders_df = pd.DataFrame()
    products_df = pd.DataFrame()
    order_products_df = pd.DataFrame()
    storyteller = None
    data_processor = None

# Add 'All Departments' option to dropdown
all_departments_option = pd.DataFrame({'department': ['All Departments']})
departments_dropdown_df = pd.concat([all_departments_option, departments_df], ignore_index=True)

# Initialize metrics
metrics = BusinessMetrics(data_processor.merged_df if data_processor else merged_df)
kpi_dashboard = KPIDashboard(metrics)

# Initialize Executive Dashboard with the same data source as KPI Dashboard
executive_dashboard = ExecutiveDashboard(data_processor.merged_df if data_processor else merged_df)

# Instantiate dashboards
product_category_dashboard = ProductCategoryPerformanceDashboard(data_processor.merged_df if data_processor else merged_df)
customer_behavior_dashboard = CustomerBehaviorDashboard(data_processor.merged_df if data_processor else merged_df)

# Define the layout of the application
app.layout = html.Div([
    html.H1('Instacart Market Basket Analysis Dashboard', 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    dcc.Tabs(id='main-tabs', value='tab-executive', children=[
        get_executive_dashboard_tab_layout(executive_dashboard),
        get_product_category_performance_tab_layout(product_category_dashboard),
        get_customer_behavior_tab_layout(customer_behavior_dashboard),
        get_basic_analysis_tab_layout(departments_dropdown_df, DAYS_OF_WEEK),
        get_market_basket_tab_layout(),
        get_price_analysis_tab_layout(data_processor),
        get_data_stories_tab_layout(storyteller),
        get_kpi_dashboard_tab_layout(kpi_dashboard)
    ]),
    html.Div(id='intermediate-data', style={'display': 'none'})
])

# Register callbacks for each tab
register_basic_analysis_callbacks(app, merged_df, DOW_MAP, HOUR_LABELS, logger)
register_market_basket_callbacks(app, merged_df, logger)
register_price_analysis_callbacks(app, data_processor, logger)
# Data Stories and KPI Dashboard handle their own content internally

# Add error handling for callbacks
def safe_callback(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except dash.exceptions.PreventUpdate:
            # Re-raise PreventUpdate without logging it as an error
            raise
        except Exception as e:
            logger.error(f"Error in callback {func.__name__}: {str(e)}", exc_info=True)
            # Return empty figures with error message
            return px.scatter(title=f"Error: {str(e)}")
    return wrapper

@server.route('/health')
def health_check():
    try:
        # Check database connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}, 500

# Import the API blueprint
from api import api

# Register the API blueprint
app.server.register_blueprint(api)

# Add error handlers
@app.server.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Not Found',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.server.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {str(error)}")
    return jsonify({
        'error': 'Internal Server Error',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    logger.info("Starting Instacart Market Basket Analysis Dashboard...")
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8050,
        use_reloader=False
    )
