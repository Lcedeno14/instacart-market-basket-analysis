import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger('customer_behavior')

class CustomerBehaviorDashboard:
    def __init__(self, merged_df: pd.DataFrame):
        """
        Initialize Customer Behavior Dashboard with merged data.
        
        Args:
            merged_df: Merged dataframe containing orders, products, and customer data
        """
        self.merged_df = merged_df
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate all metrics for the dashboard."""
        try:
            if self.merged_df.empty:
                logger.warning("Customer Behavior Dashboard: Empty dataframe provided")
                self.reorder_by_price_quartile = pd.DataFrame()
                self.cumulative_revenue = pd.DataFrame()
                self.days_since_prior = pd.DataFrame()
                self.aov_by_hour_dow = pd.DataFrame()
                self.reorder_vs_order = pd.DataFrame()
                return
            
            logger.info(f"Customer Behavior Dashboard: Processing data with {len(self.merged_df)} rows")
            
            # 1. Reorder Rate by Price Quartile
            if 'price' in self.merged_df.columns and 'reordered' in self.merged_df.columns:
                # Create price quartiles
                self.merged_df['price_quartile'] = pd.qcut(
                    self.merged_df['price'], 
                    q=4, 
                    labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
                )
                
                self.reorder_by_price_quartile = (
                    self.merged_df.groupby('price_quartile')['reordered']
                    .agg(['count', 'sum'])
                    .reset_index()
                )
                self.reorder_by_price_quartile['reorder_rate'] = (
                    self.reorder_by_price_quartile['sum'] / self.reorder_by_price_quartile['count']
                )
            else:
                self.reorder_by_price_quartile = pd.DataFrame()
            
            # 2. Cumulative Revenue per Customer (Top 20)
            if 'price' in self.merged_df.columns:
                customer_revenue = (
                    self.merged_df.groupby('user_id')['price']
                    .sum()
                    .sort_values(ascending=False)
                    .head(20)
                    .reset_index()
                )
                customer_revenue['cumulative_revenue'] = customer_revenue['price'].cumsum()
                customer_revenue['customer_rank'] = range(1, len(customer_revenue) + 1)
                self.cumulative_revenue = customer_revenue
            else:
                self.cumulative_revenue = pd.DataFrame()
            
            # 3. Days Since Prior Order Histogram
            if 'days_since_prior_order' in self.merged_df.columns:
                # Remove null values and filter reasonable range
                days_data = self.merged_df[
                    (self.merged_df['days_since_prior_order'].notna()) & 
                    (self.merged_df['days_since_prior_order'] >= 0) &
                    (self.merged_df['days_since_prior_order'] <= 30)  # Reasonable range
                ]['days_since_prior_order']
                self.days_since_prior = days_data
            else:
                self.days_since_prior = pd.Series()
            
            # 4. AOV by Hour / DOW
            if 'price' in self.merged_df.columns:
                aov_data = (
                    self.merged_df.groupby(['order_hour_of_day', 'order_dow'])
                    .agg({
                        'price': 'sum',
                        'order_id': 'nunique'
                    })
                    .reset_index()
                )
                aov_data['aov'] = aov_data['price'] / aov_data['order_id']
                self.aov_by_hour_dow = aov_data
            else:
                self.aov_by_hour_dow = pd.DataFrame()
            
            # 5. Reorder count vs. order count scatter plot
            if 'reordered' in self.merged_df.columns:
                customer_stats = (
                    self.merged_df.groupby('user_id')
                    .agg({
                        'order_id': 'nunique',
                        'reordered': 'sum'
                    })
                    .reset_index()
                    .rename(columns={'order_id': 'total_orders', 'reordered': 'reordered_items'})
                )
                customer_stats['reorder_rate'] = customer_stats['reordered_items'] / customer_stats['total_orders']
                self.reorder_vs_order = customer_stats
            else:
                self.reorder_vs_order = pd.DataFrame()
                
            logger.info("Customer Behavior Dashboard: Metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating customer behavior metrics: {str(e)}")
            self.reorder_by_price_quartile = pd.DataFrame()
            self.cumulative_revenue = pd.DataFrame()
            self.days_since_prior = pd.Series()
            self.aov_by_hour_dow = pd.DataFrame()
            self.reorder_vs_order = pd.DataFrame()
    
    def create_reorder_by_price_quartile_chart(self):
        """Create reorder rate by price quartile chart."""
        if self.reorder_by_price_quartile.empty:
            return dcc.Graph(figure=go.Figure().add_annotation(
                text="No price data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            ))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=self.reorder_by_price_quartile['price_quartile'],
            y=self.reorder_by_price_quartile['reorder_rate'],
            marker_color=['#e74c3c', '#f39c12', '#f1c40f', '#27ae60'],
            text=[f'{rate:.1%}' for rate in self.reorder_by_price_quartile['reorder_rate']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Reorder Rate by Price Quartile',
            xaxis_title='Price Quartile',
            yaxis_title='Reorder Rate',
            height=400,
            yaxis_tickformat='.1%'
        )
        
        return dcc.Graph(figure=fig)
    
    def create_cumulative_revenue_chart(self):
        """Create cumulative revenue per customer chart."""
        if self.cumulative_revenue.empty:
            return dcc.Graph(figure=go.Figure().add_annotation(
                text="No price data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            ))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.cumulative_revenue['customer_rank'],
            y=self.cumulative_revenue['cumulative_revenue'],
            mode='lines+markers',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8),
            name='Cumulative Revenue'
        ))
        
        fig.update_layout(
            title='Cumulative Revenue per Customer (Top 20)',
            xaxis_title='Customer Rank',
            yaxis_title='Cumulative Revenue ($)',
            height=400
        )
        
        return dcc.Graph(figure=fig)
    
    def create_days_since_prior_histogram(self):
        """Create days since prior order histogram."""
        if self.days_since_prior.empty:
            return dcc.Graph(figure=go.Figure().add_annotation(
                text="No days since prior order data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            ))
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.days_since_prior,
            nbinsx=20,
            marker_color='#9b59b6',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Days Since Prior Order Distribution',
            xaxis_title='Days Since Prior Order',
            yaxis_title='Frequency',
            height=400
        )
        
        return dcc.Graph(figure=fig)
    
    def create_aov_by_hour_dow_heatmap(self):
        """Create AOV by hour and day of week heatmap."""
        if self.aov_by_hour_dow.empty:
            return dcc.Graph(figure=go.Figure().add_annotation(
                text="No price data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            ))
        
        # Pivot data for heatmap
        heatmap_data = self.aov_by_hour_dow.pivot(
            index='order_hour_of_day', 
            columns='order_dow', 
            values='aov'
        ).fillna(0)
        
        # Map day numbers to names
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        heatmap_data.columns = [day_names[i] for i in heatmap_data.columns]
        
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='Blues',
            aspect='auto'
        )
        
        fig.update_layout(
            title='Average Order Value by Hour and Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            height=400
        )
        
        return dcc.Graph(figure=fig)
    
    def create_reorder_vs_order_scatter(self):
        """Create reorder count vs. order count scatter plot."""
        if self.reorder_vs_order.empty:
            return dcc.Graph(figure=go.Figure().add_annotation(
                text="No reorder data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            ))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.reorder_vs_order['total_orders'],
            y=self.reorder_vs_order['reordered_items'],
            mode='markers',
            marker=dict(
                size=8,
                color=self.reorder_vs_order['reorder_rate'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Reorder Rate')
            ),
            text=[f'Customer {uid}<br>Orders: {orders}<br>Reordered: {reordered}<br>Rate: {rate:.1%}' 
                  for uid, orders, reordered, rate in zip(
                      self.reorder_vs_order['user_id'],
                      self.reorder_vs_order['total_orders'],
                      self.reorder_vs_order['reordered_items'],
                      self.reorder_vs_order['reorder_rate']
                  )],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Reorder Count vs. Total Order Count',
            xaxis_title='Total Orders',
            yaxis_title='Reordered Items',
            height=400
        )
        
        return dcc.Graph(figure=fig)
    
    def create_dashboard_layout(self):
        """Create the complete dashboard layout."""
        return html.Div([
            # Header
            html.H2('Customer Behavior & Reordering Analysis', 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
            html.P('Behavioral patterns & customer segmentation insights', 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
            
            # First Row: Reorder Rate and Cumulative Revenue
            html.Div([
                html.Div([
                    html.H4('Reorder Rate by Price Quartile', 
                           style={'textAlign': 'center', 'marginBottom': '15px'}),
                    html.P('Do customers reorder expensive items more or less frequently?', 
                          style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}),
                    self.create_reorder_by_price_quartile_chart()
                ], style={'flex': '1', 'margin': '10px'}),
                
                html.Div([
                    html.H4('Cumulative Revenue per Customer (Top 20)', 
                           style={'textAlign': 'center', 'marginBottom': '15px'}),
                    html.P('Revenue concentration among top customers', 
                          style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}),
                    self.create_cumulative_revenue_chart()
                ], style={'flex': '1', 'margin': '10px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px'}),
            
            # Second Row: Days Since Prior and AOV Heatmap
            html.Div([
                html.Div([
                    html.H4('Days Since Prior Order Distribution', 
                           style={'textAlign': 'center', 'marginBottom': '15px'}),
                    html.P('Customer ordering frequency patterns', 
                          style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}),
                    self.create_days_since_prior_histogram()
                ], style={'flex': '1', 'margin': '10px'}),
                
                html.Div([
                    html.H4('AOV by Hour & Day of Week', 
                           style={'textAlign': 'center', 'marginBottom': '15px'}),
                    html.P('When do customers spend the most?', 
                          style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}),
                    self.create_aov_by_hour_dow_heatmap()
                ], style={'flex': '1', 'margin': '10px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px'}),
            
            # Third Row: Reorder vs Order Scatter
            html.Div([
                html.Div([
                    html.H4('Reorder Count vs. Order Count', 
                           style={'textAlign': 'center', 'marginBottom': '15px'}),
                    html.P('Customer loyalty and reordering behavior', 
                          style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}),
                    self.create_reorder_vs_order_scatter()
                ], style={'width': '100%', 'margin': '10px'})
            ])
        ], style={'padding': '20px'})

def get_customer_behavior_tab_layout(customer_behavior_dashboard=None):
    """Create the layout for the Customer Behavior & Reordering tab"""
    if customer_behavior_dashboard is None:
        return dcc.Tab(
            label='Customer Behavior & Reordering', 
            value='tab-customer-behavior', 
            children=[html.Div(['Data not loaded'])]
        )
    
    return dcc.Tab(
        label='Customer Behavior & Reordering',
        value='tab-customer-behavior',
        children=[customer_behavior_dashboard.create_dashboard_layout()]
    ) 