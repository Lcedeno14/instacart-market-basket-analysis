import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger('executive_dashboard')

class ExecutiveDashboard:
    def __init__(self, merged_df: pd.DataFrame):
        """
        Initialize Executive Dashboard with merged data.
        
        Args:
            merged_df: Merged dataframe containing orders, products, and customer data
        """
        self.merged_df = merged_df
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate all metrics for the dashboard."""
        try:
            if self.merged_df.empty:
                self.kpis = {'total_revenue': 0, 'aov': 0, 'total_orders': 0, 'unique_customers': 0}
                self.weekly_data = pd.DataFrame()
                self.hourly_data = pd.DataFrame()
                self.heatmap_data = pd.DataFrame()
                return
            
            # Calculate KPIs
            self.total_orders = self.merged_df['order_id'].nunique()
            self.unique_customers = self.merged_df['user_id'].nunique()
            
            # Calculate revenue using actual price data only
            if 'price' in self.merged_df.columns and not self.merged_df['price'].isna().all():
                # Use actual price data
                self.total_revenue = self.merged_df['price'].sum()
                logger.info(f"Using actual price data for revenue calculation: ${self.total_revenue:,.2f}")
            else:
                # No price data available - set to 0 and log error
                self.total_revenue = 0
                logger.error("No price data available for revenue calculation. Please ensure price data is loaded.")
            
            self.aov = self.total_revenue / self.total_orders if self.total_orders > 0 else 0
            
            self.kpis = {
                'total_revenue': self.total_revenue,
                'aov': self.aov,
                'total_orders': self.total_orders,
                'unique_customers': self.unique_customers
            }
            
            # Calculate weekly data
            self.weekly_data = self.merged_df.groupby('order_dow').agg({
                'order_id': 'nunique',
                'user_id': 'nunique'
            }).reset_index()
            
            # Map day numbers to names
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            self.weekly_data['day_name'] = self.weekly_data['order_dow'].map(lambda x: day_names[x])
            
            # Calculate weekly revenue using actual prices only
            if 'price' in self.merged_df.columns and not self.merged_df['price'].isna().all():
                weekly_revenue = self.merged_df.groupby('order_dow')['price'].sum().reset_index()
                weekly_revenue = weekly_revenue.rename(columns={'price': 'revenue'})
                self.weekly_data = self.weekly_data.merge(weekly_revenue, on='order_dow')
            else:
                # No price data - set revenue to 0
                self.weekly_data['revenue'] = 0
                logger.error("No price data available for weekly revenue calculation")
            
            # Calculate hourly data
            self.hourly_data = self.merged_df.groupby(['order_dow', 'order_hour_of_day']).agg({
                'order_id': 'nunique',
                'user_id': 'nunique'
            }).reset_index()
            
            # Calculate hourly revenue using actual prices only
            if 'price' in self.merged_df.columns and not self.merged_df['price'].isna().all():
                hourly_revenue = self.merged_df.groupby(['order_dow', 'order_hour_of_day'])['price'].sum().reset_index()
                hourly_revenue = hourly_revenue.rename(columns={'price': 'revenue'})
                self.hourly_data = self.hourly_data.merge(hourly_revenue, on=['order_dow', 'order_hour_of_day'])
            else:
                # No price data - set revenue to 0
                self.hourly_data['revenue'] = 0
                logger.error("No price data available for hourly revenue calculation")
            
            # Map day numbers to names
            self.hourly_data['day_name'] = self.hourly_data['order_dow'].map(lambda x: day_names[x])
            
            # Calculate heatmap data
            self.heatmap_data = self.merged_df.groupby(['order_dow', 'order_hour_of_day']).agg({
                'order_id': 'nunique'
            }).reset_index()
            
            # Calculate heatmap revenue using actual prices only
            if 'price' in self.merged_df.columns and not self.merged_df['price'].isna().all():
                heatmap_revenue = self.merged_df.groupby(['order_dow', 'order_hour_of_day'])['price'].sum().reset_index()
                heatmap_revenue = heatmap_revenue.rename(columns={'price': 'revenue'})
                self.heatmap_data = self.heatmap_data.merge(heatmap_revenue, on=['order_dow', 'order_hour_of_day'])
            else:
                # No price data - set revenue to 0
                self.heatmap_data['revenue'] = 0
                logger.error("No price data available for heatmap revenue calculation")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            self.kpis = {'total_revenue': 0, 'aov': 0, 'total_orders': 0, 'unique_customers': 0}
            self.weekly_data = pd.DataFrame()
            self.hourly_data = pd.DataFrame()
            self.heatmap_data = pd.DataFrame()
    
    def create_kpi_cards(self):
        """Create KPI cards for the dashboard."""
        return [
            # Total Revenue KPI
            html.Div([
                html.Div('', style={'fontSize': '2em', 'textAlign': 'center'}),
                html.H3('Total Revenue', style={'textAlign': 'center', 'margin': '10px 0'}),
                html.H2(f'${self.kpis["total_revenue"]:,.0f}', 
                       style={'textAlign': 'center', 'color': '#27ae60'})
            ], style={'flex': '1', 'padding': '20px', 'margin': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Average Order Value KPI
            html.Div([
                html.Div('', style={'fontSize': '2em', 'textAlign': 'center'}),
                html.H3('Average Order Value', style={'textAlign': 'center', 'margin': '10px 0'}),
                html.H2(f'${self.kpis["aov"]:.2f}', 
                       style={'textAlign': 'center', 'color': '#3498db'})
            ], style={'flex': '1', 'padding': '20px', 'margin': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Total Orders KPI
            html.Div([
                html.Div('', style={'fontSize': '2em', 'textAlign': 'center'}),
                html.H3('Total Orders', style={'textAlign': 'center', 'margin': '10px 0'}),
                html.H2(f'{self.kpis["total_orders"]:,}', 
                       style={'textAlign': 'center', 'color': '#e74c3c'})
            ], style={'flex': '1', 'padding': '20px', 'margin': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Total Unique Customers KPI
            html.Div([
                html.Div('', style={'fontSize': '2em', 'textAlign': 'center'}),
                html.H3('Unique Customers', style={'textAlign': 'center', 'margin': '10px 0'}),
                html.H2(f'{self.kpis["unique_customers"]:,}', 
                       style={'textAlign': 'center', 'color': '#9b59b6'})
            ], style={'flex': '1', 'padding': '20px', 'margin': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ]
    
    def create_weekly_trends_chart(self):
        """Create weekly orders and revenue trends chart."""
        if self.weekly_data.empty:
            return dcc.Graph(figure=go.Figure())
        
        try:
            # Ensure required columns exist
            if 'revenue' not in self.weekly_data.columns:
                logger.error(f"Revenue column missing from weekly_data. Available columns: {self.weekly_data.columns.tolist()}")
                return dcc.Graph(figure=go.Figure())
            
            # Create subplot with shared x-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Weekly Orders Trend', 'Weekly Revenue Trend'),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Orders trend
            fig.add_trace(
                go.Scatter(
                    x=self.weekly_data['day_name'],
                    y=self.weekly_data['order_id'],
                    mode='lines+markers',
                    name='Orders',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Revenue trend
            fig.add_trace(
                go.Scatter(
                    x=self.weekly_data['day_name'],
                    y=self.weekly_data['revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='#27ae60', width=3),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                title_text="Weekly Orders & Revenue Trends",
                title_x=0.5
            )
            
            fig.update_xaxes(title_text="Day of Week", row=2, col=1)
            fig.update_yaxes(title_text="Number of Orders", row=1, col=1)
            fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)
            
            return dcc.Graph(figure=fig)
            
        except Exception as e:
            logger.error(f"Error creating weekly trends chart: {str(e)}")
            return dcc.Graph(figure=go.Figure())
    
    def create_hourly_trends_chart(self):
        """Create hourly orders and revenue by day of week chart."""
        if self.hourly_data.empty:
            return dcc.Graph(figure=go.Figure())
        
        try:
            # Ensure required columns exist
            if 'revenue' not in self.hourly_data.columns:
                logger.error(f"Revenue column missing from hourly_data. Available columns: {self.hourly_data.columns.tolist()}")
                return dcc.Graph(figure=go.Figure())
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Hourly Orders by Day of Week', 'Hourly Revenue by Day of Week'),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Plot lines for each day
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            
            for i, day in enumerate(day_names):
                day_data = self.hourly_data[self.hourly_data['day_name'] == day]
                
                if not day_data.empty:
                    # Orders
                    fig.add_trace(
                        go.Scatter(
                            x=day_data['order_hour_of_day'],
                            y=day_data['order_id'],
                            mode='lines+markers',
                            name=f'{day} Orders',
                            line=dict(color=colors[i], width=2),
                            marker=dict(size=6),
                            showlegend=(i < 4)  # Only show first 4 in legend
                        ),
                        row=1, col=1
                    )
                    
                    # Revenue
                    fig.add_trace(
                        go.Scatter(
                            x=day_data['order_hour_of_day'],
                            y=day_data['revenue'],
                            mode='lines+markers',
                            name=f'{day} Revenue',
                            line=dict(color=colors[i], width=2),
                            marker=dict(size=6),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
            
            fig.update_layout(
                height=500,
                title_text="Hourly Orders & Revenue by Day of Week",
                title_x=0.5
            )
            
            fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
            fig.update_yaxes(title_text="Number of Orders", row=1, col=1)
            fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)
            
            return dcc.Graph(figure=fig)
            
        except Exception as e:
            logger.error(f"Error creating hourly trends chart: {str(e)}")
            return dcc.Graph(figure=go.Figure())
    
    def create_revenue_heatmap(self):
        """Create revenue heatmap."""
        if self.heatmap_data.empty:
            return dcc.Graph(figure=go.Figure())
        
        try:
            # Ensure required columns exist
            if 'revenue' not in self.heatmap_data.columns:
                logger.error(f"Revenue column missing from heatmap_data. Available columns: {self.heatmap_data.columns.tolist()}")
                return dcc.Graph(figure=go.Figure())
            
            # Pivot for heatmap
            heatmap_pivot = self.heatmap_data.pivot(
                index='order_hour_of_day', 
                columns='order_dow', 
                values='revenue'
            ).fillna(0)
            
            # Map day numbers to names
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            heatmap_pivot.columns = [day_names[i] for i in heatmap_pivot.columns]
            
            fig = px.imshow(
                heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                color_continuous_scale='Greens',
                aspect='auto'
            )
            
            fig.update_layout(
                title='Revenue Heatmap (Hour × Day of Week)',
                xaxis_title='Day of Week',
                yaxis_title='Hour of Day',
                height=400
            )
            
            return dcc.Graph(figure=fig)
            
        except Exception as e:
            logger.error(f"Error creating revenue heatmap: {str(e)}")
            return dcc.Graph(figure=go.Figure())
    
    def create_orders_heatmap(self):
        """Create orders heatmap."""
        if self.heatmap_data.empty:
            return dcc.Graph(figure=go.Figure())
        
        try:
            # Pivot for heatmap
            heatmap_pivot = self.heatmap_data.pivot(
                index='order_hour_of_day', 
                columns='order_dow', 
                values='order_id'
            ).fillna(0)
            
            # Map day numbers to names
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            heatmap_pivot.columns = [day_names[i] for i in heatmap_pivot.columns]
            
            fig = px.imshow(
                heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                color_continuous_scale='Reds',
                aspect='auto'
            )
            
            fig.update_layout(
                title='Orders Heatmap (Hour × Day of Week)',
                xaxis_title='Day of Week',
                yaxis_title='Hour of Day',
                height=400
            )
            
            return dcc.Graph(figure=fig)
            
        except Exception as e:
            logger.error(f"Error creating orders heatmap: {str(e)}")
            return dcc.Graph(figure=go.Figure())
    
    def create_dashboard_layout(self):
        """Create the complete dashboard layout."""
        return html.Div([
            # Header
            html.H2('Executive Summary – Orders & Revenue Trends', 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
            
            # KPI Cards Row
            html.Div(
                self.create_kpi_cards(),
                style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '30px'}
            ),
            
            # Weekly Trends Section
            html.Div([
                html.H3('Weekly Orders & Revenue Trends', 
                       style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.P('Are more orders also bringing in more money?', 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}),
                self.create_weekly_trends_chart()
            ], style={'marginBottom': '40px'}),
            
            # Hourly Trends Section
            html.Div([
                html.H3('Hourly Orders & Revenue by Day of Week', 
                       style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.P('Which hours bring high traffic vs. high spending?', 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}),
                self.create_hourly_trends_chart()
            ], style={'marginBottom': '40px'}),
            
            # Heatmaps Section
            html.Div([
                html.H3('🔥 Business Activity Heatmaps', 
                       style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.P('Compare demand vs. earnings patterns across the week', 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}),
                
                # Heatmaps Row
                html.Div([
                    # Revenue Heatmap
                    html.Div([
                        html.H4('Revenue Heatmap (Hour × Day of Week)', 
                               style={'textAlign': 'center', 'marginBottom': '10px'}),
                        self.create_revenue_heatmap()
                    ], style={'flex': '1', 'margin': '10px'}),
                    
                    # Orders Heatmap
                    html.Div([
                        html.H4('Orders Heatmap (Hour × Day of Week)', 
                               style={'textAlign': 'center', 'marginBottom': '10px'}),
                        self.create_orders_heatmap()
                    ], style={'flex': '1', 'margin': '10px'})
                ], style={'display': 'flex', 'flexWrap': 'wrap'})
            ])
        ], style={'padding': '20px'})

def get_executive_dashboard_tab_layout(executive_dashboard=None):
    """Create the layout for the Executive Dashboard tab"""
    
    return dcc.Tab(
        label='Executive Dashboard',
        value='tab-executive',
        children=[
            executive_dashboard.create_dashboard_layout() if executive_dashboard else html.Div("Executive Dashboard is not available. Please check the logs for details.")
        ]
    )

# No callback registration needed for Executive Dashboard
# The ExecutiveDashboard class handles its own data processing internally 