# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine

# Initialize the Dash application
# This creates a Flask server under the hood
app = dash.Dash(__name__)
server = app.server  # Expose server variable for Railway

# Use DATABASE_URL from environment (set by Railway)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instacart.db")
engine = create_engine(DATABASE_URL)

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
            o.order_hour_of_day
        FROM order_products op
        JOIN products p ON op.product_id = p.product_id
        JOIN departments d ON p.department_id = d.department_id
        JOIN orders o ON op.order_id = o.order_id
        """
        merged_df = pd.read_sql_query(query, conn)
        
        # Map order_dow to day name
        merged_df['day_of_week'] = merged_df['order_dow'].map(DOW_MAP)
    
    return departments_df, merged_df

print("Loading data from database...")
departments_df, merged_df = load_data()

# Add 'All Departments' option to dropdown
all_departments_option = pd.DataFrame({'department': ['All Departments']})
departments_dropdown_df = pd.concat([all_departments_option, departments_df], ignore_index=True)

# Define the layout of the application
# The layout is a tree of components that defines how the app looks
app.layout = html.Div([
    # Header
    html.H1('Instacart Market Basket Analysis Dashboard',
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    
    # Container for filters and controls
    html.Div([
        # Department Dropdown
        html.Div([
            html.Label('Select Department:'),
            dcc.Dropdown(
                id='department-dropdown',
                # Get unique departments for the dropdown options
                options=[{'label': dept, 'value': dept} 
                        for dept in departments_dropdown_df['department'].unique()],
                value=departments_dropdown_df['department'].iloc[0],  # Default value
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
    
    # Container for graphs
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
    
    # Heatmap of orders by day of week vs hour of day
    html.Div([
        dcc.Graph(id='orders-heatmap')
    ], style={'width': '100%', 'display': 'inline-block', 'marginTop': '40px'}),
    
    # Hidden div for storing intermediate data
    # This is a common pattern in Dash for storing data that doesn't need to be displayed
    html.Div(id='intermediate-data', style={'display': 'none'})
])

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

# Run the application
if __name__ == '__main__':
    # debug=True enables hot-reloading and shows detailed error messages
    # This is useful during development but should be set to False in production
    app.run_server(debug=True, port=8050)
