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
from src.analysis.price_analysis import PriceAnalysis
import plotly.graph_objects as go
from src.analysis.metrics import BusinessMetrics
from src.visualization.kpi_dashboard import KPIDashboard
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This ensures logs go to console
        logging.FileHandler('app.log')  # This will also save logs to a file
    ]
)
logger = logging.getLogger(__name__)

# Set specific logger levels for our modules
logging.getLogger('src.analysis.price_analysis').setLevel(logging.INFO)
logging.getLogger('src.analysis.data_storytelling').setLevel(logging.INFO)

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

# Initialize data storyteller and price analyzer with the correct dataframes
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
    
    # Initialize data storyteller and price analyzer
    storyteller = DataStorytelling(
        orders_df=orders_df,
        products_df=products_df,
        departments_df=departments_df,
        order_products_df=order_products_df
    )
    
    price_analyzer = PriceAnalysis(
        orders_df=orders_df,
        products_df=products_df,
        departments_df=departments_df,
        order_products_df=order_products_df
    )
    
except Exception as e:
    logger.error(f"Failed to load initial data: {str(e)}")
    departments_df = pd.DataFrame(columns=['department'])
    merged_df = pd.DataFrame()
    orders_df = pd.DataFrame()
    products_df = pd.DataFrame()
    order_products_df = pd.DataFrame()
    storyteller = None
    price_analyzer = None

# Add 'All Departments' option to dropdown
all_departments_option = pd.DataFrame({'department': ['All Departments']})
departments_dropdown_df = pd.concat([all_departments_option, departments_df], ignore_index=True)

# Initialize metrics
metrics = BusinessMetrics(price_analyzer.merged_df if price_analyzer else merged_df)
kpi_dashboard = KPIDashboard(metrics)

# Define the layout of the application
app.layout = html.Div([
    html.H1('Instacart Market Basket Analysis Dashboard', 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    dcc.Tabs(id='main-tabs', value='tab-0', children=[
        # Basic Analysis Tab
        dcc.Tab(label='Basic Analysis', value='tab-0', children=[
    # Container for filters and controls
    html.Div([
        # Department Dropdown
        html.Div([
            html.Label('Select Department:'),
            dcc.Dropdown(
                id='department-dropdown',
                options=[{'label': dept, 'value': dept} 
                                for dept in departments_dropdown_df['department'].unique()],
                        value=departments_dropdown_df['department'].iloc[0],
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
        
        # Product Count Slider
        html.Div([
            html.Label('Minimum Product Count:'),
            dcc.Slider(
                id='product-count-slider',
                min=1,
                max=100,
                step=1,
                value=10,
                marks={i: str(i) for i in range(0, 101, 10)},
            )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
                
                # Day of Week Dropdown
                html.Div([
                    html.Label('Day of Week:'),
                    dcc.Dropdown(
                        id='day-dropdown',
                        options=[{'label': day, 'value': day} for day in DAYS_OF_WEEK],
                        value='All Time',
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
            # Container for basic analysis graphs
    html.Div([
        # Top Products Bar Chart
        html.Div([
            dcc.Graph(id='top-products-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        # Department Distribution Pie Chart
        html.Div([
            dcc.Graph(id='department-distribution-chart')
        ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Heatmap
            html.Div([
                dcc.Graph(id='orders-heatmap')
            ], style={'width': '100%', 'display': 'inline-block', 'marginTop': '40px'}),
        ]),
        
        # Market Basket Analysis Tab
        dcc.Tab(label='Market Basket Analysis', value='tab-1', children=[
            html.Div([
                # Header with explanation
                html.Div([
                    html.H2('Market Basket Analysis - Product Association Insights', 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.P('Discover which products customers buy together and identify cross-selling opportunities', 
                          style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
                ]),
                
                # Controls for market basket analysis
                html.Div([
                    html.Div([
                        html.Label('Minimum Support (How often items appear together):', 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Slider(
                            id='support-slider',
                            min=0.001,
                            max=0.02,
                            step=0.001,
                            value=0.001,
                            marks={
                                0.001: '0.1%',
                                0.002: '0.2%',
                                0.005: '0.5%',
                                0.01: '1%',
                                0.02: '2%'
                            },
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        ),
                        html.Div(id='support-explanation', 
                                style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label('Minimum Confidence (How reliable the association is):', 
                                 style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Slider(
                            id='confidence-slider',
                            min=0.1,
                            max=0.5,
                            step=0.1,
                            value=0.1,
                            marks={
                                0.1: '10%',
                                0.2: '20%',
                                0.3: '30%',
                                0.4: '40%',
                                0.5: '50%'
                            },
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        ),
                        html.Div(id='confidence-explanation', 
                                style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '20px'}),
                
                # Metrics Explanation
                html.Div([
                    html.H4('📊 Understanding the Metrics', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.H5('Support', style={'color': '#3498db', 'marginBottom': '5px'}),
                            html.P('How frequently items appear together in transactions. Higher support = more common combinations.', 
                                  style={'fontSize': '14px', 'color': '#7f8c8d'})
                        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),
                        html.Div([
                            html.H5('Confidence', style={'color': '#e74c3c', 'marginBottom': '5px'}),
                            html.P('How reliable the association is. If A is bought, how likely is B also bought?', 
                                  style={'fontSize': '14px', 'color': '#7f8c8d'})
                        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),
                        html.Div([
                            html.H5('Lift', style={'color': '#27ae60', 'marginBottom': '5px'}),
                            html.P('Strength of association. Lift > 1 = positive association, Lift > 2 = strong association.', 
                                  style={'fontSize': '14px', 'color': '#7f8c8d'})
                        ], style={'width': '32%', 'display': 'inline-block', 'marginLeft': '1%'})
                    ])
                ], style={'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '20px'}),
                
                # Key Metrics Summary
                html.Div([
                    html.H3('Key Insights Summary', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div(id='market-basket-summary', 
                            style={'padding': '15px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px'})
                ], style={'marginBottom': '20px'}),
                
                # Main Visualizations
                html.Div([
                    # Top Product Associations
                    html.Div([
                        html.H4('Top Product Associations by Business Impact', 
                               style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        dcc.Graph(id='top-associations-chart')
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    # Association Network
                    html.Div([
                        html.H4('Product Association Network', 
                               style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        dcc.Graph(id='association-network-chart')
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ], style={'marginBottom': '20px'}),
                
                # Department-Level Insights
                html.Div([
                    html.H4('Department-Level Associations', 
                           style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    dcc.Graph(id='department-associations-chart')
                ], style={'marginBottom': '20px'}),
                
                # Business Recommendations
                html.Div([
                    html.H3('Business Recommendations', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div(id='business-recommendations', 
                            style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px'})
                ], style={'marginBottom': '20px'}),
                
                # Actionable Insights
                html.Div([
                    html.H3('Actionable Insights', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.H4('🎯 Cross-Selling Opportunities', style={'color': '#27ae60', 'marginBottom': '10px'}),
                            html.P('Products that customers frequently buy together - perfect for bundling and recommendations', 
                                  style={'color': '#7f8c8d', 'marginBottom': '15px'}),
                            html.Div(id='cross-selling-insights', style={'fontSize': '14px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4('📊 Inventory Planning', style={'color': '#e74c3c', 'marginBottom': '10px'}),
                            html.P('Use association patterns to optimize inventory levels and reduce stockouts', 
                                  style={'color': '#7f8c8d', 'marginBottom': '15px'}),
                            html.Div(id='inventory-insights', style={'fontSize': '14px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
                    ])
                ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                
                # Detailed Rules Table
                html.Div([
                    html.H4('Detailed Association Rules', 
                           style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label('Filter by Minimum Lift:', style={'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='lift-filter',
                            options=[
                                {'label': 'All Rules', 'value': 0},
                                {'label': 'Lift > 1.5', 'value': 1.5},
                                {'label': 'Lift > 2.0', 'value': 2.0},
                                {'label': 'Lift > 3.0', 'value': 3.0}
                            ],
                            value=0,
                            style={'width': '200px', 'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '15px'}),
                    html.Div(id='association-rules-table', 
                            style={'maxHeight': '400px', 'overflowY': 'auto'})
                ])
            ], style={'padding': '20px'})
        ]),
        
        # Price Analysis Tab (only show if price_analyzer is initialized)
        dcc.Tab(label='Price Analysis', value='tab-3', children=[
            html.Div([
                html.Div([
                    html.H2('Price Analysis and Customer Spending Patterns', 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                    dcc.Graph(
                        id='price-analysis-graph',
                        figure=go.Figure()  # Start with empty figure
                    ),
                    html.Div(id='price-analysis-insights', 
                            style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                    html.Div(id='price-analysis-recommendations',
                            style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px', 'marginTop': '20px'})
                ]) if price_analyzer else html.Div("Price analysis is not available. Please check the logs for details.")
            ], style={'padding': '20px'})
        ]),
        
        # Data Stories Tab (only show if storyteller is initialized)
        dcc.Tab(label='Data Stories', value='tab-4', children=[
            html.Div([
                html.Div([
                    html.H2('Customer Journey Insights', 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                    dcc.Graph(figure=storyteller.create_story_visualization('customer_journey') if storyteller else go.Figure()),
                    html.Div([
                        html.Div([
                            html.H3(section['title'], 
                                   style={'color': '#2c3e50', 'marginBottom': '10px'}),
                            html.Ul([
                                html.Li(insight, style={'marginBottom': '5px'})
                                for insight in section['insights']
                            ])
                        ], style={'marginBottom': '20px'})
                        for section in (storyteller.generate_customer_journey_story()['sections'] if storyteller else [])
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                ], style={'marginBottom': '40px'}) if storyteller else html.Div("Customer journey analysis is not available. Please check the logs for details."),
                
                html.Div([
                    html.H2('Seasonal Trends Analysis', 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                    dcc.Graph(figure=storyteller.create_story_visualization('seasonal_trends') if storyteller else go.Figure()),
                    html.Div([
                        html.Div([
                            html.H3(section['title'], 
                                   style={'color': '#2c3e50', 'marginBottom': '10px'}),
                            html.Ul([
                                html.Li(insight, style={'marginBottom': '5px'})
                                for insight in section['insights']
                            ])
                        ], style={'marginBottom': '20px'})
                        for section in (storyteller.generate_seasonal_trends_story()['sections'] if storyteller else [])
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                ], style={'marginBottom': '40px'})
            ], style={'padding': '20px'})
        ]) if storyteller else None,
        
        # KPI Dashboard Tab
        dcc.Tab(label='Business KPIs', value='tab-5', children=[
            kpi_dashboard.create_dashboard_layout()
        ])
    ]),
    
    # Hidden div for storing intermediate data
    html.Div(id='intermediate-data', style={'display': 'none'})
])

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

# Define callback functions
# Callbacks are the heart of Dash's interactivity
# They define how the app responds to user input

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

# Add callback for price analysis tab
