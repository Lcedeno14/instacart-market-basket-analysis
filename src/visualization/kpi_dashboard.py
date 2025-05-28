"""
KPI Dashboard component for displaying business metrics and analytics.
"""

import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd
import numpy as np
from ..analysis.metrics import BusinessMetrics

class KPIDashboard:
    def __init__(self, metrics: BusinessMetrics):
        """
        Initialize KPI Dashboard with business metrics.
        
        Args:
            metrics: BusinessMetrics instance with calculated metrics
        """
        self.metrics = metrics
        self.kpi_data = metrics.generate_kpi_dashboard()
        self.clv_data = metrics.calculate_customer_lifetime_value()
        self.cohort_data = metrics.calculate_cohort_metrics()
    
    def create_kpi_cards(self) -> List[html.Div]:
        """Create KPI cards for the dashboard."""
        cards = []
        
        # Define KPI cards with their titles and values
        kpi_definitions = {
            'total_revenue': {'title': 'Total Revenue', 'format': '${:,.2f}'},
            'total_orders': {'title': 'Total Orders', 'format': '{:,.0f}'},
            'total_customers': {'title': 'Total Customers', 'format': '{:,.0f}'},
            'avg_order_value': {'title': 'Average Order Value', 'format': '${:,.2f}'},
            'repeat_customer_rate': {'title': 'Repeat Customer Rate', 'format': '{:.1%}'},
            'revenue_per_customer': {'title': 'Revenue per Customer', 'format': '${:,.2f}'}
        }
        
        for kpi, definition in kpi_definitions.items():
            value = self.kpi_data[kpi]
            formatted_value = definition['format'].format(value)
            
            card = html.Div(
                className='kpi-card',
                children=[
                    html.H3(definition['title']),
                    html.H2(formatted_value)
                ]
            )
            cards.append(card)
        
        return cards
    
    def create_clv_histogram(self) -> dcc.Graph:
        """Create histogram of Customer Lifetime Value."""
        fig = px.histogram(
            self.clv_data,
            x='clv',
            nbins=50,
            title='Customer Lifetime Value Distribution',
            labels={'clv': 'Customer Lifetime Value ($)'}
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title='Customer Lifetime Value ($)',
            yaxis_title='Number of Customers'
        )
        
        return dcc.Graph(figure=fig)
    
    def create_cohort_heatmap(self) -> dcc.Graph:
        """Create cohort analysis heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=self.cohort_data['retention_rate'].values.reshape(-1, 1),
            x=['Retention Rate'],
            y=self.cohort_data.index,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title='Customer Retention by Cohort',
            xaxis_title='Metric',
            yaxis_title='Cohort'
        )
        
        return dcc.Graph(figure=fig)
    
    def create_metrics_trend(self) -> dcc.Graph:
        """Create trend line for key metrics over time."""
        # Group data by order date and calculate metrics
        daily_metrics = self.metrics.df.groupby('order_dow').agg({
            'price': 'sum',
            'order_id': 'nunique',
            'user_id': 'nunique'
        }).reset_index()
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add revenue line
        fig.add_trace(go.Scatter(
            x=daily_metrics['order_dow'],
            y=daily_metrics['price'],
            name='Revenue',
            line=dict(color='blue')
        ))
        
        # Add orders line
        fig.add_trace(go.Scatter(
            x=daily_metrics['order_dow'],
            y=daily_metrics['order_id'],
            name='Orders',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Daily Revenue and Orders',
            xaxis_title='Day of Week',
            yaxis_title='Revenue ($)',
            yaxis2=dict(
                title='Number of Orders',
                overlaying='y',
                side='right'
            )
        )
        
        return dcc.Graph(figure=fig)
    
    def create_dashboard_layout(self) -> html.Div:
        """Create the complete dashboard layout."""
        return html.Div(
            className='kpi-dashboard',
            children=[
                html.H1('Business Performance Dashboard'),
                
                # KPI Cards
                html.Div(
                    className='kpi-cards-container',
                    children=self.create_kpi_cards()
                ),
                
                # Charts Row 1
                html.Div(
                    className='charts-row',
                    children=[
                        html.Div(
                            className='chart-container',
                            children=self.create_clv_histogram()
                        ),
                        html.Div(
                            className='chart-container',
                            children=self.create_metrics_trend()
                        )
                    ]
                ),
                
                # Charts Row 2
                html.Div(
                    className='charts-row',
                    children=[
                        html.Div(
                            className='chart-container',
                            children=self.create_cohort_heatmap()
                        )
                    ]
                )
            ]
        ) 