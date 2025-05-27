import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import logging

# Set up logging
logger = logging.getLogger(__name__)

def calculate_rfm(df):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing order data with columns: user_id, order_id, product_id
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with RFM metrics for each customer
    """
    try:
        # Calculate RFM metrics (recency will be set to NaN since order_date is not available)
        rfm = df.groupby('user_id').agg({
            'order_id': 'count',  # Frequency
            'product_id': 'count'  # Monetary (using product count as proxy)
        }).rename(columns={
            'order_id': 'frequency',
            'product_id': 'monetary'
        })
        rfm['recency'] = np.nan  # No recency without order_date
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
        DataFrame with cluster assignments
    """
    try:
        # Drop rows with NaNs before clustering
        rfm_df = rfm_df.dropna()
        
        if len(rfm_df) == 0:
            logger.warning("No valid data for clustering after dropping NaNs")
            return pd.DataFrame()
            
        # Scale the data
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_df)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
        
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
        
        # Normalize by total purchases
        dept_prefs = dept_prefs.div(dept_prefs.sum(axis=1), axis=0)
        
        # Scale the data
        scaler = StandardScaler()
        dept_scaled = scaler.fit_transform(dept_prefs)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        dept_prefs['cluster'] = kmeans.fit_predict(dept_scaled)
        
        return dept_prefs
    except Exception as e:
        logger.error(f"Error analyzing department preferences: {str(e)}")
        return pd.DataFrame()

def analyze_purchase_patterns(df, n_clusters, sample_size=500):
    """
    Analyze customer purchase patterns by time of day and day of week
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing order data with columns: user_id, order_hour_of_day, day_of_week, product_id
    n_clusters : int
        Number of clusters to create
    sample_size : int, optional
        Number of users to sample for analysis (default: 500)
        
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
            
        # Scale the data
        scaler = StandardScaler()
        patterns_scaled = scaler.fit_transform(patterns)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        patterns['cluster'] = kmeans.fit_predict(patterns_scaled)
        
        return patterns
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
            fig = px.scatter_3d(
                df,
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
            # Create heatmap of average patterns by cluster
            cluster_patterns = df.groupby('cluster').mean()
            fig = px.imshow(
                cluster_patterns,
                title='Purchase Patterns by Cluster',
                labels=dict(x='Day of Week', y='Hour of Day', color='Average Orders'),
                aspect='auto'
            )
        else:  # department preferences
            # Create radar chart of average department preferences by cluster
            cluster_means = df.groupby('cluster').mean()
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
    except Exception as e:
        logger.error(f"Error creating segmentation visualization: {str(e)}")
        return px.scatter(title=f"Error creating visualization: {str(e)}") 