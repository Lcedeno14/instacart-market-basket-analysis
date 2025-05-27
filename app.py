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
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instacart.db")
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
                o.order_date
            FROM order_products op
            JOIN products p ON op.product_id = p.product_id
            JOIN departments d ON p.department_id = d.department_id
            JOIN orders o ON op.order_id = o.order_id
            """
            merged_df = pd.read_sql_query(query, conn)
            
            # Map order_dow to day name
            merged_df['day_of_week'] = merged_df['order_dow'].map(DOW_MAP)
            
            # Convert order_date to datetime if it's not already
            if 'order_date' in merged_df.columns:
                merged_df['order_date'] = pd.to_datetime(merged_df['order_date'])
            
            logger.info("Data loaded successfully")
            return departments_df, merged_df
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        # Return empty DataFrames with the correct structure
        return pd.DataFrame(columns=['department']), pd.DataFrame()

# Load data with error handling
try:
    departments_df, merged_df = load_data()
except Exception as e:
    logger.error(f"Failed to load initial data: {str(e)}")
    departments_df = pd.DataFrame(columns=['department'])
    merged_df = pd.DataFrame()

# Add 'All Departments' option to dropdown
all_departments_option = pd.DataFrame({'department': ['All Departments']})
departments_dropdown_df = pd.concat([all_departments_option, departments_df], ignore_index=True)

# Define the layout of the application
app.layout = html.Div([
    # Header
    html.H1('Instacart Market Basket Analysis Dashboard',
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    
    # Tabs for different analyses
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
                        max=0.1,
                        step=0.001,
                        value=0.005,
                        marks={i/100: f'{i}%' for i in range(1, 11)},
                    ),
                    html.Label('Minimum Confidence:'),
                    dcc.Slider(
                        id='confidence-slider',
                        min=0.1,
                        max=1.0,
                        step=0.05,
                        value=0.3,
                        marks={i/10: f'{i*10}%' for i in range(1, 11)},
                    ),
                    html.Label('Analysis Type:'),
                    dcc.RadioItems(
                        id='analysis-type',
                        options=[
                            {'label': 'Product Associations', 'value': 'product'},
                            {'label': 'Department Associations', 'value': 'department'},
                            {'label': 'Price-Weighted Analysis', 'value': 'price'}
                        ],
                        value='product',
                        labelStyle={'display': 'inline-block', 'margin': '10px'}
                    )
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                
                # Market Basket Analysis Results
                html.Div([
                    dcc.Graph(id='association-rules-graph')
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
    Output('association-rules-graph', 'figure'),
    [Input('support-slider', 'value'),
     Input('confidence-slider', 'value'),
     Input('analysis-type', 'value')]
)
@safe_callback
def update_market_basket(support, confidence, analysis_type):
    """
    Update market basket analysis visualization based on selected parameters
    """
    if analysis_type == 'product':
        # Get product associations
        basket = merged_df.groupby(['order_id', 'product_name'])['product_id'].count().unstack().fillna(0)
        basket = (basket > 0).astype(int)
        
        # Generate frequent itemsets
        frequent_itemsets = apriori(basket, min_support=support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        
        # Create visualization
        fig = px.scatter(
            rules,
            x='support',
            y='confidence',
            size='lift',
            hover_data=['antecedents', 'consequents'],
            title='Product Association Rules'
        )
        
    elif analysis_type == 'department':
        # Get department associations
        dept_transactions = merged_df.groupby('order_id')['department'].apply(list)
        te = TransactionEncoder()
        te_ary = te.fit(dept_transactions).transform(dept_transactions)
        dept_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Generate frequent itemsets
        frequent_itemsets = fpgrowth(dept_df, min_support=support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        
        # Create visualization
        fig = px.scatter(
            rules,
            x='support',
            y='confidence',
            size='lift',
            hover_data=['antecedents', 'consequents'],
            title='Department Association Rules'
        )
        
    else:  # price-weighted analysis
        # Get product associations with prices
        basket = merged_df.merge(
            pd.read_sql_query("SELECT product_id, price FROM products", engine),
            on='product_id'
        ).groupby(['order_id', 'product_name'])['price'].mean().unstack().fillna(0)
        basket = (basket > 0).astype(int)
        
        # Generate frequent itemsets
        frequent_itemsets = apriori(basket, min_support=support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        
        # Calculate weighted lift
        rules['weighted_lift'] = rules['lift'] * rules['support']
        
        # Create visualization
        fig = px.scatter(
            rules,
            x='support',
            y='confidence',
            size='weighted_lift',
            hover_data=['antecedents', 'consequents', 'weighted_lift'],
            title='Price-Weighted Association Rules'
        )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        xaxis_title='Support',
        yaxis_title='Confidence'
    )
    
    return fig

def calculate_rfm(df):
    """
    Calculate RFM metrics for each customer
    """
    # Get the most recent date in the dataset
    max_date = df['order_date'].max()
    
    # Calculate RFM metrics
    rfm = df.groupby('user_id').agg({
        'order_date': lambda x: (max_date - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'product_id': 'count'  # Monetary (using product count as proxy)
    }).rename(columns={
        'order_date': 'recency',
        'order_id': 'frequency',
        'product_id': 'monetary'
    })
    
    # Lower recency is better, so we'll invert it
    rfm['recency'] = -rfm['recency']
    
    return rfm

def perform_clustering(rfm_df, n_clusters):
    """
    Perform K-means clustering on RFM data
    """
    # Scale the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm_df

def analyze_department_preferences(df, n_clusters):
    """
    Analyze customer department preferences and cluster customers
    """
    # Calculate department preferences for each customer
    dept_prefs = df.groupby(['user_id', 'department'])['product_id'].count().unstack(fill_value=0)
    
    # Normalize by total purchases
    dept_prefs = dept_prefs.div(dept_prefs.sum(axis=1), axis=0)
    
    # Scale the data
    scaler = StandardScaler()
    dept_scaled = scaler.fit_transform(dept_prefs)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dept_prefs['cluster'] = kmeans.fit_predict(dept_scaled)
    
    return dept_prefs

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
    if segmentation_type == 'rfm':
        # Calculate RFM metrics
        rfm_df = calculate_rfm(merged_df)
        
        # Perform clustering
        rfm_clustered = perform_clustering(rfm_df, n_clusters)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            rfm_clustered,
            x='recency',
            y='frequency',
            z='monetary',
            color='cluster',
            title='Customer Segments (RFM Analysis)',
            labels={
                'recency': 'Recency (days since last order)',
                'frequency': 'Frequency (number of orders)',
                'monetary': 'Monetary (total products purchased)'
            }
        )
        
    elif segmentation_type == 'patterns':
        # Analyze purchase patterns by time of day and day of week
        patterns = merged_df.groupby(['user_id', 'order_hour_of_day', 'day_of_week'])['product_id'].count().unstack().unstack()
        patterns = patterns.fillna(0)
        
        # Scale the data
        scaler = StandardScaler()
        patterns_scaled = scaler.fit_transform(patterns)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        patterns['cluster'] = kmeans.fit_predict(patterns_scaled)
        
        # Create heatmap of average patterns by cluster
        cluster_patterns = patterns.groupby('cluster').mean()
        fig = px.imshow(
            cluster_patterns,
            title='Purchase Patterns by Cluster',
            labels=dict(x='Day of Week', y='Hour of Day', color='Average Orders'),
            aspect='auto'
        )
        
    else:  # department preferences
        # Analyze department preferences
        dept_prefs = analyze_department_preferences(merged_df, n_clusters)
        
        # Create radar chart of average department preferences by cluster
        cluster_means = dept_prefs.groupby('cluster').mean()
        
        # Create radar chart
        fig = px.line_polar(
            cluster_means,
            r=cluster_means.values[0],  # First cluster
            theta=cluster_means.columns[:-1],  # Exclude cluster column
            line_close=True,
            title='Department Preferences by Cluster'
        )
        
        # Add other clusters
        for i in range(1, n_clusters):
            fig.add_trace(px.line_polar(
                cluster_means,
                r=cluster_means.values[i],
                theta=cluster_means.columns[:-1],
                line_close=True
            ).data[0])
    
    fig.update_layout(
        height=600,
        template='plotly_white'
    )
    
    return fig

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
