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
from customer_segmentation import (
    calculate_rfm,
    perform_clustering,
    analyze_department_preferences,
    analyze_purchase_patterns,
    create_segmentation_visualization
)
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
        
        # Customer Segmentation Tab
        dcc.Tab(label='Customer Segmentation', value='tab-2', children=[
            html.Div([
                # Controls for customer segmentation
                html.Div([
                    html.Label('Number of Clusters:'),
                    dcc.Slider(
                        id='cluster-slider',
                        min=2,
                        max=8,
                        step=1,
                        value=4,
                        marks={i: str(i) for i in range(2, 9)},
                    ),
                    html.Label('Segmentation Type:'),
                    dcc.RadioItems(
                        id='segmentation-type',
                        options=[
                            {'label': 'RFM Analysis', 'value': 'rfm'},
                            {'label': 'Purchase Patterns', 'value': 'patterns'},
                            {'label': 'Department Preferences', 'value': 'departments'}
                        ],
                        value='rfm',
                        labelStyle={'display': 'inline-block', 'margin': '10px'}
                    )
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                
                # Customer Segmentation Results
                html.Div([
                    dcc.Graph(id='segmentation-graph')
                ], style={'width': '100%', 'marginTop': '20px'})
            ])
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

@app.callback(
    # Output components that will be updated
    [Output('top-products-chart', 'figure'),
     Output('department-distribution-chart', 'figure'),
     Output('orders-heatmap', 'figure')],
    # Input components that will trigger the callback
    [Input('department-dropdown', 'value'),
     Input('product-count-slider', 'value'),
     Input('day-dropdown', 'value')]
)
@safe_callback
def update_graphs(selected_department, min_count, selected_day):
    """
    This callback function updates both graphs when either the department dropdown
    or the product count slider changes.
    
    Parameters:
    - selected_department: The department selected in the dropdown
    - min_count: The minimum product count from the slider
    - selected_day: The selected day from the dropdown
    
    Returns:
    - Two Plotly figure objects for the bar chart and pie chart
    """
    # Filter data based on selected department
    if selected_department == 'All Departments':
        filtered_df = merged_df.copy()
    else:
        filtered_df = merged_df[merged_df['department'] == selected_department]
    
    # Filter by day of week if not All Time
    if selected_day != 'All Time':
        filtered_df = filtered_df[filtered_df['day_of_week'] == selected_day]
        pie_df = merged_df[merged_df['day_of_week'] == selected_day]
    else:
        pie_df = merged_df
    
    # Create top products bar chart
    product_counts = filtered_df['product_name'].value_counts()
    product_counts = product_counts[product_counts >= min_count]
    top_10 = product_counts.head(10)
    bar_df = pd.DataFrame({
        'product': top_10.index,
        'count': top_10.values
    })
    
    bar_fig = px.bar(
        bar_df,
        x='product',
        y='count',
        title=f'Top Products in {selected_department}' + (f' on {selected_day}' if selected_day != 'All Time' else ''),
        labels={'product': 'Product', 'count': 'Count'},
        template='plotly_white'
    )
    bar_fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        margin=dict(b=100)  # Add bottom margin for rotated labels
    )
    
    # Create department distribution pie chart (filtered by day)
    dept_dist = pie_df['department'].value_counts()
    pie_fig = px.pie(
        values=dept_dist.values,
        names=dept_dist.index,
        title='Distribution of Orders Across Departments' + (f' on {selected_day}' if selected_day != 'All Time' else ''),
        template='plotly_white'
    )
    pie_fig.update_layout(height=500)
    
    # Create heatmap of orders by day of week vs hour of day
    heatmap_df = merged_df.copy()
    # Only use the 7 days, not 'All Time'
    heatmap_df = heatmap_df[heatmap_df['day_of_week'].isin(ORDERED_DAYS)]
    pivot = pd.pivot_table(
        heatmap_df,
        index='day_of_week',
        columns='order_hour_of_day',
        values='product_id',
        aggfunc='count',
        fill_value=0
    )
    # Reorder days for display
    pivot = pivot.reindex(ORDERED_DAYS)
    # Relabel columns to hour labels
    pivot.columns = [HOUR_LABELS[h] for h in pivot.columns]
    heatmap_fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="# Orders"),
        aspect="auto",
        color_continuous_scale='YlOrRd',
        title="Order Frequency by Day of Week and Hour of Day"
    )
    heatmap_fig.update_layout(height=500)
    
    return bar_fig, pie_fig, heatmap_fig

# Helper function for market basket analysis
def get_department_associations(rules_df, conn):
    """
    Create department-level associations from product-level rules
    """
    try:
        # Get department information
        dept_df = pd.read_sql_query("SELECT department_id, department FROM departments", conn)
        dept_map = dict(zip(dept_df['department_id'], dept_df['department']))
        
        # Get product-department mapping
        product_dept_df = pd.read_sql_query("SELECT product_id, department_id FROM products", conn)
        product_dept_df = product_dept_df[product_dept_df['product_id'].notna()]
        product_dept_map = dict(zip(product_dept_df['product_id'], product_dept_df['department_id']))
        
        # Create a mapping from product names to departments
        # This is a simplified approach - in practice you'd need to join with product names
        dept_associations = []
        
        # Group rules by lift ranges for better visualization
        lift_ranges = {
            'Very Strong (Lift > 3)': len(rules_df[rules_df['lift'] > 3]),
            'Strong (Lift 2-3)': len(rules_df[(rules_df['lift'] > 2) & (rules_df['lift'] <= 3)]),
            'Moderate (Lift 1.5-2)': len(rules_df[(rules_df['lift'] > 1.5) & (rules_df['lift'] <= 2)]),
            'Weak (Lift 1-1.5)': len(rules_df[(rules_df['lift'] > 1) & (rules_df['lift'] <= 1.5)]),
            'Very Weak (Lift ≤ 1)': len(rules_df[rules_df['lift'] <= 1])
        }
        
        return lift_ranges
        
    except Exception as e:
        logger.error(f"Error in department associations: {str(e)}")
        return {}

def generate_business_insights(rules_df):
    """
    Generate actionable business insights from market basket analysis
    """
    insights = []
    
    if len(rules_df) == 0:
        return ["No association rules found with current parameters"]
    
    # Analyze rule strength distribution
    strong_rules = rules_df[rules_df['lift'] > 2]
    moderate_rules = rules_df[(rules_df['lift'] > 1.5) & (rules_df['lift'] <= 2)]
    
    if len(strong_rules) > 0:
        insights.append(f"🎯 {len(strong_rules)} strong associations found - excellent cross-selling opportunities")
        
        # Find the strongest rule
        strongest_rule = rules_df.loc[rules_df['lift'].idxmax()]
        insights.append(f"💡 Strongest association: {', '.join(strongest_rule['antecedents'])} → {', '.join(strongest_rule['consequents'])} (Lift: {strongest_rule['lift']:.2f})")
    
    if len(moderate_rules) > 0:
        insights.append(f"📈 {len(moderate_rules)} moderate associations - good potential for targeted marketing")
    
    # Analyze confidence levels
    high_confidence_rules = rules_df[rules_df['confidence'] > 0.5]
    if len(high_confidence_rules) > 0:
        insights.append(f"🎯 {len(high_confidence_rules)} high-confidence rules - very reliable for recommendations")
    
    # Analyze support levels
    high_support_rules = rules_df[rules_df['support'] > 0.01]
    if len(high_support_rules) > 0:
        insights.append(f"📊 {len(high_support_rules)} frequently occurring patterns - high-volume opportunities")
    
    # Business recommendations
    if rules_df['lift'].mean() > 2:
        insights.append("🚀 Overall strong associations - consider implementing recommendation engine")
    
    if rules_df['confidence'].mean() > 0.3:
        insights.append("✅ High average confidence - reliable for automated recommendations")
    
    if len(rules_df) > 100:
        insights.append("📋 Rich pattern database - excellent foundation for personalization")
    
    return insights

# Add new callback for market basket analysis
@app.callback(
    [Output('market-basket-summary', 'children'),
     Output('top-associations-chart', 'figure'),
     Output('association-network-chart', 'figure'),
     Output('department-associations-chart', 'figure'),
     Output('business-recommendations', 'children'),
     Output('cross-selling-insights', 'children'),
     Output('inventory-insights', 'children'),
     Output('association-rules-table', 'children'),
     Output('support-explanation', 'children'),
     Output('confidence-explanation', 'children')],
    [Input('support-slider', 'value'),
     Input('confidence-slider', 'value'),
     Input('lift-filter', 'value')]
)
@safe_callback
def update_market_basket(support, confidence, lift_filter):
    """
    Update market basket analysis visualization based on selected parameters
    """
    # Update explanations
    support_explanation = f"Items appear together in {support*100:.1f}% of all transactions"
    confidence_explanation = f"When customers buy the first item, they buy the second item {confidence*100:.0f}% of the time"
    
    # Get rules from database that match the selected parameters
    with engine.connect() as conn:
        rules_df = pd.read_sql_query("""
            SELECT 
                algorithm,
                support,
                confidence,
                lift,
                antecedents,
                consequents,
                support_param,
                confidence_param
            FROM market_basket_rules
            WHERE support_param = %s
            AND confidence_param = %s
            ORDER BY lift DESC
        """, conn, params=(support, confidence))
    
    if rules_df.empty:
        # Return empty figures and messages if no rules found
        empty_fig = px.scatter(
            title='No rules found for selected parameters. Try adjusting support/confidence.'
        ).update_layout(
            height=400,
            template='plotly_white',
            xaxis_title='Support',
            yaxis_title='Confidence'
        )
        
        return (
            html.Div("No association rules found with current parameters. Try lowering the thresholds."),
            empty_fig,
            empty_fig,
            empty_fig,
            html.Div("No recommendations available with current parameters."),
            html.Div("No cross-selling insights available with current parameters."),
            html.Div("No inventory insights available with current parameters."),
            html.Div("No rules to display."),
            support_explanation,
            confidence_explanation
        )
    
    # Parse JSON strings back to lists
    rules_df['antecedents'] = rules_df['antecedents'].apply(json.loads)
    rules_df['consequents'] = rules_df['consequents'].apply(json.loads)
    
    # Create summary metrics
    total_rules = len(rules_df)
    avg_lift = rules_df['lift'].mean()
    max_lift = rules_df['lift'].max()
    high_lift_rules = len(rules_df[rules_df['lift'] > 2])
    
    summary_content = html.Div([
        html.Div([
            html.H4(f"{total_rules:,}", style={'color': '#2c3e50', 'margin': '0'}),
            html.P("Total Rules", style={'color': '#7f8c8d', 'margin': '0', 'fontSize': '12px'})
        ], style={'textAlign': 'center', 'width': '25%', 'display': 'inline-block'}),
        html.Div([
            html.H4(f"{avg_lift:.2f}", style={'color': '#2c3e50', 'margin': '0'}),
            html.P("Avg Lift", style={'color': '#7f8c8d', 'margin': '0', 'fontSize': '12px'})
        ], style={'textAlign': 'center', 'width': '25%', 'display': 'inline-block'}),
        html.Div([
            html.H4(f"{max_lift:.2f}", style={'color': '#2c3e50', 'margin': '0'}),
            html.P("Max Lift", style={'color': '#7f8c8d', 'margin': '0', 'fontSize': '12px'})
        ], style={'textAlign': 'center', 'width': '25%', 'display': 'inline-block'}),
        html.Div([
            html.H4(f"{high_lift_rules}", style={'color': '#2c3e50', 'margin': '0'}),
            html.P("Strong Rules (Lift>2)", style={'color': '#7f8c8d', 'margin': '0', 'fontSize': '12px'})
        ], style={'textAlign': 'center', 'width': '25%', 'display': 'inline-block'})
    ])
    
    # Create top associations chart
    top_rules = rules_df.head(10)
    
    # Create a more readable format for product names
    def format_product_names(products):
        if len(products) == 1:
            return products[0][:30] + "..." if len(products[0]) > 30 else products[0]
        else:
            return f"{len(products)} products"
    
    top_associations_fig = px.bar(
        x=[f"{format_product_names(ant)} → {format_product_names(cons)}" 
           for ant, cons in zip(top_rules['antecedents'], top_rules['consequents'])],
        y=top_rules['lift'],
        title='Top Product Associations by Business Impact (Lift Score)',
        labels={'x': 'Product Association', 'y': 'Lift Score'},
        color=top_rules['confidence'],
        color_continuous_scale='Viridis',
        hover_data={
            'antecedents': [', '.join(ant) for ant in top_rules['antecedents']],
            'consequents': [', '.join(cons) for cons in top_rules['consequents']],
            'support': [f"{s:.4f}" for s in top_rules['support']],
            'confidence': [f"{c:.4f}" for c in top_rules['confidence']]
        }
    )
    top_associations_fig.update_layout(
        height=400,
        template='plotly_white',
        xaxis_tickangle=-45,
        coloraxis_colorbar=dict(title="Confidence"),
        hovermode='closest'
    )
    
    # Create association network chart (simplified version)
    network_fig = px.scatter(
        rules_df,
        x='support',
        y='confidence',
        size='lift',
        color='lift',
        hover_data=['antecedents', 'consequents'],
        title='Association Rules Network (Size = Lift, Color = Lift)',
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
    )
    network_fig.update_layout(
        height=400,
        template='plotly_white',
        coloraxis_colorbar=dict(title="Lift Score")
    )
    
    # Create department associations chart
    try:
        lift_ranges = get_department_associations(rules_df, conn)
        
        if lift_ranges:
            # Create bar chart of lift ranges
            dept_fig = px.bar(
                x=list(lift_ranges.keys()),
                y=list(lift_ranges.values()),
                title='Distribution of Association Rule Strengths',
                labels={'x': 'Lift Range', 'y': 'Number of Rules'},
                color=list(lift_ranges.values()),
                color_continuous_scale='RdYlGn'
            )
            dept_fig.update_layout(
                height=400, 
                template='plotly_white',
                xaxis_tickangle=-45,
                coloraxis_colorbar=dict(title="Count")
            )
        else:
            dept_fig = px.scatter(title='No department associations available').update_layout(height=400, template='plotly_white')
    except:
        dept_fig = px.scatter(title='Department analysis not available').update_layout(height=400, template='plotly_white')
    
    # Create business recommendations
    insights = generate_business_insights(rules_df)
    
    if not insights:
        insights = ["Consider lowering thresholds to find more association patterns"]
    
    recommendations_content = html.Ul([
        html.Li(insight, style={'marginBottom': '8px'}) for insight in insights
    ], style={'margin': '0', 'paddingLeft': '20px'})
    
    # Create cross-selling insights
    strong_associations = rules_df[rules_df['lift'] > 2].head(3)
    cross_selling_content = []
    
    if len(strong_associations) > 0:
        cross_selling_content.append(html.P("Top cross-selling opportunities:", style={'fontWeight': 'bold'}))
        for _, rule in strong_associations.iterrows():
            cross_selling_content.append(html.P(
                f"• {', '.join(rule['antecedents'])} → {', '.join(rule['consequents'])} (Lift: {rule['lift']:.2f})",
                style={'marginLeft': '20px', 'marginBottom': '5px'}
            ))
    else:
        cross_selling_content.append(html.P("No strong cross-selling opportunities found with current thresholds."))
    
    # Create inventory insights
    high_support_rules = rules_df[rules_df['support'] > 0.005].head(3)
    inventory_content = []
    
    if len(high_support_rules) > 0:
        inventory_content.append(html.P("Frequently co-purchased items (stock together):", style={'fontWeight': 'bold'}))
        for _, rule in high_support_rules.iterrows():
            inventory_content.append(html.P(
                f"• {', '.join(rule['antecedents'])} + {', '.join(rule['consequents'])} (Support: {rule['support']:.3f})",
                style={'marginLeft': '20px', 'marginBottom': '5px'}
            ))
    else:
        inventory_content.append(html.P("No high-frequency patterns found with current thresholds."))
    
    # Create detailed rules table
    # Apply lift filter
    filtered_rules = rules_df[rules_df['lift'] >= lift_filter] if lift_filter > 0 else rules_df
    
    table_data = []
    for _, rule in filtered_rules.head(20).iterrows():
        table_data.append(html.Tr([
            html.Td(', '.join(rule['antecedents']), style={'padding': '8px', 'border': '1px solid #ddd'}),
            html.Td(', '.join(rule['consequents']), style={'padding': '8px', 'border': '1px solid #ddd'}),
            html.Td(f"{rule['support']:.4f}", style={'padding': '8px', 'border': '1px solid #ddd'}),
            html.Td(f"{rule['confidence']:.4f}", style={'padding': '8px', 'border': '1px solid #ddd'}),
            html.Td(f"{rule['lift']:.2f}", style={'padding': '8px', 'border': '1px solid #ddd'})
        ]))
    
    table_content = html.Table([
        html.Thead(html.Tr([
            html.Th('Antecedent', style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f8f9fa'}),
            html.Th('Consequent', style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f8f9fa'}),
            html.Th('Support', style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f8f9fa'}),
            html.Th('Confidence', style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f8f9fa'}),
            html.Th('Lift', style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f8f9fa'})
        ])),
        html.Tbody(table_data)
    ], style={'width': '100%', 'borderCollapse': 'collapse'})
    
    return (
        summary_content,
        top_associations_fig,
        network_fig,
        dept_fig,
        recommendations_content,
        cross_selling_content,
        inventory_content,
        table_content,
        support_explanation,
        confidence_explanation
    )

@app.callback(
    Output('segmentation-graph', 'figure'),
    [Input('cluster-slider', 'value'),
     Input('segmentation-type', 'value')]
)
@safe_callback
def update_segmentation(n_clusters, segmentation_type):
    """
    Update customer segmentation visualization based on selected parameters
    """
    try:
        if segmentation_type == 'rfm':
            # Calculate RFM metrics
            rfm_df = calculate_rfm(merged_df)
            # Perform clustering
            rfm_clustered = perform_clustering(rfm_df, n_clusters)
            if len(rfm_clustered) == 0:
                return px.scatter(
                    title='No data available for RFM analysis. Please check your data.'
                ).update_layout(
                    height=600,
                    template='plotly_white'
                )
            return create_segmentation_visualization(rfm_clustered, 'rfm', n_clusters)
            
        elif segmentation_type == 'patterns':
            # Analyze purchase patterns
            patterns = analyze_purchase_patterns(merged_df, n_clusters)
            if len(patterns) == 0:
                return px.scatter(
                    title='No data available for pattern analysis. Please check your data.'
                ).update_layout(
                    height=600,
                    template='plotly_white'
                )
            return create_segmentation_visualization(patterns, 'patterns', n_clusters)
            
        else:  # department preferences
            # Analyze department preferences
            dept_prefs = analyze_department_preferences(merged_df, n_clusters)
            if len(dept_prefs) == 0:
                return px.scatter(
                    title='No data available for department preference analysis. Please check your data.'
                ).update_layout(
                    height=600,
                    template='plotly_white'
                )
            return create_segmentation_visualization(dept_prefs, 'departments', n_clusters)
            
    except Exception as e:
        logger.error(f"Error in segmentation analysis: {str(e)}")
        return px.scatter(
            title=f'Error in segmentation analysis: {str(e)}'
        ).update_layout(
            height=600,
            template='plotly_white'
        )

# Add health check endpoint
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
@app.callback(
    [Output('price-analysis-graph', 'figure'),
     Output('price-analysis-insights', 'children'),
     Output('price-analysis-recommendations', 'children')],
    [Input('main-tabs', 'value')]
)
@safe_callback
def update_price_analysis(tab_value):
    """Update price analysis when tab is clicked."""
    try:
        now = datetime.now().isoformat()
        logger.info(f"[CALLBACK] Price analysis callback triggered at {now} with tab_value: {tab_value}")
        logger.info(f"[CALLBACK] price_analyzer is {'initialized' if price_analyzer else 'None'} at {now}")
        
        # Only update if we're on the price analysis tab
        if tab_value != 'tab-3':
            logger.info(f"[CALLBACK] Not on price analysis tab, skipping update at {now}")
            raise dash.exceptions.PreventUpdate
        
        if not price_analyzer:
            logger.warning(f"[CALLBACK] Price analyzer not initialized at {now}")
            return [go.Figure(), html.Div("Price analysis not available"), html.Div()]
        
        logger.info(f"[CALLBACK] Starting price analysis update at {now}")
        start_time = time.time()
        
        # Get visualization (this should use cache after first call)
        logger.info(f"[CALLBACK] Creating visualization at {now}")
        viz_start = time.time()
        try:
            fig = price_analyzer.create_visualization()
            logger.info(f"[CALLBACK] Visualization created in {time.time() - viz_start:.2f} seconds at {now}")
        except Exception as viz_error:
            logger.error(f"[CALLBACK] Error creating visualization: {repr(viz_error)}", exc_info=True)
            raise
        
        # Get insights (this should use cache after first call)
        logger.info(f"[CALLBACK] Getting price insights at {now}")
        insights_start = time.time()
        try:
            insights_data = price_analyzer.get_price_insights()
            logger.info(f"[CALLBACK] Insights retrieved in {time.time() - insights_start:.2f} seconds at {now}")
        except Exception as insights_error:
            logger.error(f"[CALLBACK] Error getting insights: {repr(insights_error)}", exc_info=True)
            raise
        
        # Create insights HTML
        try:
            insights = html.Div([
                html.Div([
                    html.H3(section['title'], 
                           style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(insight, style={'marginBottom': '5px'})
                        for insight in section['insights']
                    ])
                ], style={'marginBottom': '20px'})
                for section in insights_data['sections']
            ])
        except Exception as html_error:
            logger.error(f"[CALLBACK] Error creating insights HTML: {repr(html_error)}", exc_info=True)
            raise
        
        # Create recommendations HTML
        try:
            recommendations = html.Div([
                html.H3('Recommendations', 
                       style={'color': '#2c3e50', 'marginBottom': '10px'}),
                html.Ul([
                    html.Li(rec, style={'marginBottom': '5px'})
                    for rec in insights_data.get('recommendations', [])
                ])
            ])
        except Exception as html_error:
            logger.error(f"[CALLBACK] Error creating recommendations HTML: {repr(html_error)}", exc_info=True)
            raise
        
        logger.info(f"[CALLBACK] Total price analysis update took {time.time() - start_time:.2f} seconds at {now}")
        
        # Return as a list of three elements
        return [fig, insights, recommendations]
        
    except dash.exceptions.PreventUpdate:
        logger.info(f"[CALLBACK] PreventUpdate raised at {datetime.now().isoformat()}")
        raise
    except Exception as e:
        logger.error(f"[CALLBACK] Error in price analysis update: {repr(e)}", exc_info=True)
        # Return a list of three elements even in case of error
        return [go.Figure(), html.Div(f"Error in price analysis: {repr(e)}"), html.Div()]

# Run the application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run_server(debug=debug, port=port, host='0.0.0.0')
