import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Set up logging
logger = logging.getLogger(__name__)

def calculate_rfm(df):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing order data with columns: user_id, order_id, product_id, days_since_prior_order
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with RFM metrics for each customer
    """
    try:
        # Calculate RFM metrics
        rfm = df.groupby('user_id').agg({
            'order_id': 'count',  # Frequency
            'product_id': 'count',  # Monetary (using product count as proxy)
            'days_since_prior_order': 'mean'  # Recency (average days between orders)
        }).rename(columns={
            'order_id': 'frequency',
            'product_id': 'monetary',
            'days_since_prior_order': 'recency'
        })
        
        # Calculate additional metrics
        rfm['avg_order_size'] = rfm['monetary'] / rfm['frequency']
        rfm['order_regularity'] = 1 / (rfm['recency'] + 1)  # Convert to a 0-1 scale where 1 is most regular
        
        return rfm
    except Exception as e:
        logger.error(f"Error calculating RFM metrics: {str(e)}")
        return pd.DataFrame()

def perform_clustering(rfm_df, n_clusters):
    """
    Perform K-means clustering on RFM data
    
    Parameters:
    -----------
    rfm_df : pandas.DataFrame
        DataFrame with RFM metrics
    n_clusters : int
        Number of clusters to create
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cluster assignments and cluster characteristics
    """
    try:
        # Drop rows with NaNs before clustering
        rfm_df = rfm_df.dropna()
        
        if len(rfm_df) == 0:
            logger.warning("No valid data for clustering after dropping NaNs")
            return pd.DataFrame()
            
        # Scale the data
        scaler = StandardScaler()
        features = ['frequency', 'monetary', 'recency', 'avg_order_size', 'order_regularity']
        rfm_scaled = scaler.fit_transform(rfm_df[features])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Calculate cluster characteristics
        cluster_stats = rfm_df.groupby('cluster')[features].agg(['mean', 'std']).round(2)
        rfm_df['cluster_size'] = rfm_df.groupby('cluster')['cluster'].transform('count')
        rfm_df['cluster_label'] = rfm_df.apply(
            lambda x: f"Cluster {x['cluster']} ({x['cluster_size']} customers)", 
            axis=1
        )
        
        return rfm_df
    except Exception as e:
        logger.error(f"Error performing clustering: {str(e)}")
        return pd.DataFrame()

def analyze_department_preferences(df, n_clusters):
    """
    Analyze customer department preferences and cluster customers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing order data with columns: user_id, department, product_id
    n_clusters : int
        Number of clusters to create
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with department preferences and cluster assignments
    """
    try:
        # Calculate department preferences for each customer
        dept_prefs = df.groupby(['user_id', 'department'])['product_id'].count().unstack(fill_value=0)
        
        # Calculate total purchases per customer
        total_purchases = dept_prefs.sum(axis=1)
        
        # Normalize by total purchases and calculate percentage
        dept_prefs_pct = dept_prefs.div(total_purchases, axis=0) * 100
        
        # Scale the data
        scaler = StandardScaler()
        dept_scaled = scaler.fit_transform(dept_prefs_pct)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        dept_prefs_pct['cluster'] = kmeans.fit_predict(dept_scaled)
        
        # Calculate cluster characteristics
        dept_prefs_pct['cluster_size'] = dept_prefs_pct.groupby('cluster')['cluster'].transform('count')
        dept_prefs_pct['cluster_label'] = dept_prefs_pct.apply(
            lambda x: f"Cluster {x['cluster']} ({x['cluster_size']} customers)", 
            axis=1
        )
        
        return dept_prefs_pct
    except Exception as e:
        logger.error(f"Error analyzing department preferences: {str(e)}")
        return pd.DataFrame()

def analyze_purchase_patterns(df, n_clusters, sample_size=1000):
    """
    Analyze customer purchase patterns by time of day and day of week
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing order data with columns: user_id, order_hour_of_day, day_of_week, product_id
    n_clusters : int
        Number of clusters to create
    sample_size : int, optional
        Number of users to sample for analysis (default: 1000)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with purchase patterns and cluster assignments
    """
    try:
        # Sample users to avoid huge DataFrames
        user_sample = df['user_id'].drop_duplicates().sample(
            n=min(sample_size, df['user_id'].nunique()), 
            random_state=42
        )
        
        # Calculate patterns
        patterns = df[df['user_id'].isin(user_sample)].groupby(
            ['user_id', 'order_hour_of_day', 'day_of_week']
        )['product_id'].count().unstack().unstack()
        patterns = patterns.fillna(0)
        
        if len(patterns) == 0:
            logger.warning("No valid data for pattern analysis")
            return pd.DataFrame()
            
        # Calculate total orders per user
        total_orders = patterns.sum(axis=1)
        
        # Normalize by total orders to get percentages
        patterns_pct = patterns.div(total_orders, axis=0) * 100
            
        # Scale the data
        scaler = StandardScaler()
        patterns_scaled = scaler.fit_transform(patterns_pct)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        patterns_pct['cluster'] = kmeans.fit_predict(patterns_scaled)
        
        # Calculate cluster characteristics
        patterns_pct['cluster_size'] = patterns_pct.groupby('cluster')['cluster'].transform('count')
        patterns_pct['cluster_label'] = patterns_pct.apply(
            lambda x: f"Cluster {x['cluster']} ({x['cluster_size']} customers)", 
            axis=1
        )
        
        return patterns_pct
    except Exception as e:
        logger.error(f"Error analyzing purchase patterns: {str(e)}")
        return pd.DataFrame()

def create_segmentation_visualization(df, segmentation_type, n_clusters):
    """
    Create visualization for customer segmentation analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the analysis results
    segmentation_type : str
        Type of segmentation ('rfm', 'patterns', or 'departments')
    n_clusters : int
        Number of clusters used in the analysis
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for visualization
    """
    try:
        if segmentation_type == 'rfm':
            # Create a subplot with 2 rows and 2 columns
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Frequency vs Monetary Value',
                    'Order Regularity vs Average Order Size',
                    'Cluster Distribution',
                    'Cluster Characteristics'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "pie"}, {"type": "bar"}]
                ]
            )
            
            # Plot 1: Frequency vs Monetary
            fig.add_trace(
                go.Scatter(
                    x=df['frequency'],
                    y=df['monetary'],
                    mode='markers',
                    marker=dict(
                        color=df['cluster'],
                        size=8,
                        showscale=True
                    ),
                    text=df['cluster_label'],
                    name='Customers'
                ),
                row=1, col=1
            )
            fig.update_xaxes(title_text='Frequency (Orders)', row=1, col=1)
            fig.update_yaxes(title_text='Monetary Value (Products)', row=1, col=1)
            
            # Plot 2: Order Regularity vs Average Order Size
            fig.add_trace(
                go.Scatter(
                    x=df['order_regularity'],
                    y=df['avg_order_size'],
                    mode='markers',
                    marker=dict(
                        color=df['cluster'],
                        size=8
                    ),
                    text=df['cluster_label'],
                    name='Customers'
                ),
                row=1, col=2
            )
            fig.update_xaxes(title_text='Order Regularity', row=1, col=2)
            fig.update_yaxes(title_text='Average Order Size', row=1, col=2)
            
            # Plot 3: Cluster Distribution
            cluster_sizes = df['cluster'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=[f'Cluster {i}' for i in range(n_clusters)],
                    values=cluster_sizes.values,
                    name='Cluster Distribution'
                ),
                row=2, col=1
            )
            
            # Plot 4: Cluster Characteristics
            cluster_means = df.groupby('cluster')[['frequency', 'monetary', 'order_regularity', 'avg_order_size']].mean()
            for metric in ['frequency', 'monetary', 'order_regularity', 'avg_order_size']:
                fig.add_trace(
                    go.Bar(
                        x=[f'Cluster {i}' for i in range(n_clusters)],
                        y=cluster_means[metric],
                        name=metric.replace('_', ' ').title()
                    ),
                    row=2, col=2
                )
            fig.update_xaxes(title_text='Cluster', row=2, col=2)
            fig.update_yaxes(title_text='Average Value', row=2, col=2)
            
            fig.update_layout(
                height=1000,
                width=1200,
                title_text='Customer Segmentation Analysis (RFM)',
                showlegend=True,
                template='plotly_white'
            )
            
        elif segmentation_type == 'patterns':
            # Create a subplot with 2 rows
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Average Purchase Patterns by Cluster',
                    'Cluster Distribution'
                ),
                specs=[[{"type": "heatmap"}], [{"type": "pie"}]],
                vertical_spacing=0.2
            )
            
            # Plot 1: Heatmap of average patterns by cluster
            cluster_patterns = df.groupby('cluster').mean()
            cluster_patterns = cluster_patterns.drop(['cluster', 'cluster_size'], axis=1)
            
            # Reorder columns to show days in order
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours = [f"{h % 12 if h % 12 != 0 else 12}{'am' if h < 12 else 'pm'}" for h in range(24)]
            cluster_patterns = cluster_patterns.reindex(columns=[f"{day} {hour}" for day in days for hour in hours])
            
            fig.add_trace(
                go.Heatmap(
                    z=cluster_patterns.values,
                    x=cluster_patterns.columns,
                    y=[f'Cluster {i}' for i in range(n_clusters)],
                    colorscale='YlOrRd',
                    colorbar=dict(title='% of Orders')
                ),
                row=1, col=1
            )
            
            # Plot 2: Cluster Distribution
            cluster_sizes = df['cluster'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=[f'Cluster {i}' for i in range(n_clusters)],
                    values=cluster_sizes.values,
                    name='Cluster Distribution'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=1000,
                width=1200,
                title_text='Customer Purchase Patterns Analysis',
                showlegend=True,
                template='plotly_white'
            )
            
        else:  # department preferences
            # Create a subplot with 2 rows
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Department Preferences by Cluster',
                    'Cluster Distribution'
                ),
                specs=[[{"type": "bar"}], [{"type": "pie"}]],
                vertical_spacing=0.2
            )
            
            # Plot 1: Bar chart of department preferences by cluster
            # Drop non-numeric columns before calculating means
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cluster_means = df.groupby('cluster')[numeric_cols].mean()
            
            # Sort departments by average preference across clusters
            # Exclude cluster and cluster_size columns from sorting
            dept_cols = [col for col in cluster_means.columns if col not in ['cluster', 'cluster_size']]
            dept_order = cluster_means[dept_cols].mean().sort_values(ascending=False).index
            
            for i in range(n_clusters):
                fig.add_trace(
                    go.Bar(
                        x=dept_order,
                        y=cluster_means.loc[i, dept_order],
                        name=f'Cluster {i}'
                    ),
                    row=1, col=1
                )
            
            fig.update_xaxes(title_text='Department', row=1, col=1)
            fig.update_yaxes(title_text='% of Purchases', row=1, col=1)
            
            # Plot 2: Cluster Distribution
            cluster_sizes = df['cluster'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=[f'Cluster {i}' for i in range(n_clusters)],
                    values=cluster_sizes.values,
                    name='Cluster Distribution'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=1000,
                width=1200,
                title_text='Customer Department Preferences Analysis',
                showlegend=True,
                template='plotly_white',
                barmode='group'
            )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating segmentation visualization: {str(e)}")
        return px.scatter(title=f"Error creating visualization: {str(e)}") 