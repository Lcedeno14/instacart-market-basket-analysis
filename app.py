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
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Load data with error handling
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
    
    # Initialize data storyteller with the correct dataframes
    storyteller = DataStorytelling(
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

# Add 'All Departments' option to dropdown
all_departments_option = pd.DataFrame({'department': ['All Departments']})
departments_dropdown_df = pd.concat([all_departments_option, departments_df], ignore_index=True)

# Define the layout of the application
app.layout = html.Div([
    html.H1('Instacart Market Basket Analysis Dashboard', 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    dcc.Tabs([
        # Basic Analysis Tab
        dcc.Tab(label='Basic Analysis', children=[
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
        dcc.Tab(label='Market Basket Analysis', children=[
            html.Div([
                # Controls for market basket analysis
                html.Div([
                    html.Label('Minimum Support:'),
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
                    ),
                    html.Label('Minimum Confidence:'),
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
                    )
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                
                # Market Basket Analysis Results
                html.Div([
                    dcc.Graph(id='association-rules-graph-main')
                ], style={'width': '100%', 'marginTop': '20px'})
            ])
        ]),
        
        # Customer Segmentation Tab
        dcc.Tab(label='Customer Segmentation', children=[
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
        
        # Price Analysis Tab (only show if storyteller is initialized)
        dcc.Tab(label='Price Analysis', children=[
            html.Div([
                html.Div([
                    html.H2('Price Analysis and Customer Spending Patterns', 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                    dcc.Graph(figure=storyteller.create_story_visualization('price_analysis') if storyteller else go.Figure()),
                    html.Div([
                        html.Div([
                            html.H3(section['title'], 
                                   style={'color': '#2c3e50', 'marginBottom': '10px'}),
                            html.Ul([
                                html.Li(insight, style={'marginBottom': '5px'})
                                for insight in section['insights']
                            ])
                        ], style={'marginBottom': '20px'})
                        for section in (storyteller.generate_price_insights_story()['sections'] if storyteller else [])
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                    
                    html.Div([
                        html.H3('Recommendations', 
                               style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.Ul([
                            html.Li(rec, style={'marginBottom': '5px'})
                            for rec in (storyteller.generate_price_insights_story().get('recommendations', []) if storyteller else [])
                        ])
                    ], style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px', 'marginTop': '20px'})
                ]) if storyteller else html.Div("Price analysis is not available. Please check the logs for details.")
            ], style={'padding': '20px'})
        ]),
        
        # Data Stories Tab (only show if storyteller is initialized)
        dcc.Tab(label='Data Stories', children=[
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
                ], style={'marginBottom': '40px'}),
                
                html.Div([
                    html.H2('Product Association Insights', 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                    html.Div([
                        dcc.Input(
                            id='min-support',
                            type='number',
                            value=0.01,
                            min=0.001,
                            max=0.1,
                            step=0.001,
                            style={'width': '150px', 'marginRight': '10px'}
                        ),
                        dcc.Input(
                            id='min-confidence',
                            type='number',
                            value=0.1,
                            min=0.01,
                            max=0.5,
                            step=0.01,
                            style={'width': '150px'}
                        ),
                        html.Label('Min Support', style={'marginRight': '20px'}),
                        html.Label('Min Confidence')
                    ], style={'marginBottom': '20px', 'textAlign': 'center'}),
                    dcc.Graph(id='association-rules-graph-detail'),
                    html.Div(id='association-insights', 
                            style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                ])
            ], style={'padding': '20px'})
        ]) if storyteller else None
    ]),
    
    # Hidden div for storing intermediate data
    html.Div(id='intermediate-data', style={'display': 'none'})
])

# Add error handling for callbacks
def safe_callback(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback {func.__name__}: {str(e)}")
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

# Add new callback for market basket analysis
@app.callback(
    Output('association-rules-graph-main', 'figure'),
    [Input('support-slider', 'value'),
     Input('confidence-slider', 'value')]
)
@safe_callback
def update_market_basket(support, confidence):
    """
    Update market basket analysis visualization based on selected parameters
    """
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
        # Return empty figure if no rules found
        return px.scatter(
            title='No rules found for selected parameters. Try adjusting support/confidence.'
        ).update_layout(
            height=600,
            template='plotly_white',
            xaxis_title='Support',
            yaxis_title='Confidence'
        )
    
    # Parse JSON strings back to lists
    rules_df['antecedents'] = rules_df['antecedents'].apply(json.loads)
    rules_df['consequents'] = rules_df['consequents'].apply(json.loads)
    
    # Create visualization
    fig = px.scatter(
        rules_df,
        x='support',
        y='confidence',
        size='lift',
        hover_data=['antecedents', 'consequents'],
        title='Product Association Rules'
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        xaxis_title='Support',
        yaxis_title='Confidence'
    )
    
    return fig

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

@app.callback(
    [Output('association-rules-graph-detail', 'figure'),
     Output('association-insights', 'children')],
    [Input('min-support', 'value'),
     Input('min-confidence', 'value')]
)
def update_association_rules(min_support, min_confidence):
    """Update association rules visualization and insights based on thresholds."""
    story = storyteller.generate_product_association_story(min_support, min_confidence)
    fig = storyteller.create_story_visualization('product_associations')
    
    insights = html.Div([
        html.Div([
            html.H3(section['title'], 
                   style={'color': '#2c3e50', 'marginBottom': '10px'}),
            html.Ul([
                html.Li(insight, style={'marginBottom': '5px'})
                for insight in section['insights']
            ])
        ], style={'marginBottom': '20px'})
        for section in story['sections']
    ])
    
    return fig, insights

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

# Run the application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run_server(debug=debug, port=port, host='0.0.0.0')
