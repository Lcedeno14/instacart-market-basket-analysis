import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger('product_category_performance')

class ProductCategoryPerformanceDashboard:
    def __init__(self, merged_df: pd.DataFrame):
        self.merged_df = merged_df
        self._calculate_metrics()

    def _calculate_metrics(self):
        try:
            df = self.merged_df
            logger.info(f"Product Category Dashboard: Processing data with {len(df)} rows and columns: {df.columns.tolist()}")
            
            if df.empty:
                logger.warning("Product Category Dashboard: Empty dataframe provided")
                self.top_products_by_revenue = pd.DataFrame()
                self.top_products_by_orders = pd.DataFrame()
                self.top_reordered_by_revenue = pd.DataFrame()
                self.top_reordered_by_orders = pd.DataFrame()
                self.top_departments_by_revenue = pd.DataFrame()
                self.top_departments_by_orders = pd.DataFrame()
                self.top_aisles_by_revenue = pd.DataFrame()
                self.top_aisles_by_orders = pd.DataFrame()
                self.most_expensive_products = pd.DataFrame()
                return

            # Check if price column exists
            if 'price' not in df.columns:
                logger.error(f"Product Category Dashboard: Price column not found. Available columns: {df.columns.tolist()}")
                # Set all price-dependent metrics to empty DataFrames
                self.top_products_by_revenue = pd.DataFrame()
                self.top_reordered_by_revenue = pd.DataFrame()
                self.top_departments_by_revenue = pd.DataFrame()
                self.top_aisles_by_revenue = pd.DataFrame()
                self.most_expensive_products = pd.DataFrame()
            else:
                logger.info(f"Product Category Dashboard: Price column found with {df['price'].notna().sum()} non-null values")
                # Top 10 Products by Revenue
                self.top_products_by_revenue = (
                    df.groupby(['product_id', 'product_name'], as_index=False)['price'].sum()
                    .sort_values('price', ascending=False).head(10)
                )
                # Top 10 Reordered Products by Revenue
                if 'reordered' in df.columns:
                    reordered_df = df[df['reordered'] == 1]
                    self.top_reordered_by_revenue = (
                        reordered_df.groupby(['product_id', 'product_name'], as_index=False)['price'].sum()
                        .sort_values('price', ascending=False).head(10)
                    )
                else:
                    self.top_reordered_by_revenue = pd.DataFrame()
                # Top 10 Departments by Revenue
                self.top_departments_by_revenue = (
                    df.groupby('department', as_index=False)['price'].sum()
                    .sort_values('price', ascending=False).head(10)
                )
                # Top 10 Aisles by Revenue
                if 'aisle' in df.columns:
                    self.top_aisles_by_revenue = (
                        df.groupby('aisle', as_index=False)['price'].sum()
                        .sort_values('price', ascending=False).head(10)
                    )
                else:
                    self.top_aisles_by_revenue = pd.DataFrame()
                # Most Expensive Purchased Products
                self.most_expensive_products = (
                    df.groupby(['product_id', 'product_name'], as_index=False)['price'].max()
                    .sort_values('price', ascending=False).head(10)
                )

            # These metrics don't depend on price, so calculate them regardless
            # Top 10 Products by Orders
            self.top_products_by_orders = (
                df.groupby(['product_id', 'product_name'], as_index=False)['order_id'].count()
                .rename(columns={'order_id': 'order_count'})
                .sort_values('order_count', ascending=False).head(10)
            )
            # Top 10 Reordered Products by Orders
            if 'reordered' in df.columns:
                reordered_df = df[df['reordered'] == 1]
                self.top_reordered_by_orders = (
                    reordered_df.groupby(['product_id', 'product_name'], as_index=False)['order_id'].count()
                    .rename(columns={'order_id': 'order_count'})
                    .sort_values('order_count', ascending=False).head(10)
                )
            else:
                self.top_reordered_by_orders = pd.DataFrame()
            # Top 10 Departments by Orders
            self.top_departments_by_orders = (
                df.groupby('department', as_index=False)['order_id'].count()
                .rename(columns={'order_id': 'order_count'})
                .sort_values('order_count', ascending=False).head(10)
            )
            # Top 10 Aisles by Orders
            if 'aisle' in df.columns:
                self.top_aisles_by_orders = (
                    df.groupby('aisle', as_index=False)['order_id'].count()
                    .rename(columns={'order_id': 'order_count'})
                    .sort_values('order_count', ascending=False).head(10)
                )
            else:
                self.top_aisles_by_orders = pd.DataFrame()
                
            logger.info("Product Category Dashboard: Metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating product/category metrics: {str(e)}")
            self.top_products_by_revenue = pd.DataFrame()
            self.top_products_by_orders = pd.DataFrame()
            self.top_reordered_by_revenue = pd.DataFrame()
            self.top_reordered_by_orders = pd.DataFrame()
            self.top_departments_by_revenue = pd.DataFrame()
            self.top_departments_by_orders = pd.DataFrame()
            self.top_aisles_by_revenue = pd.DataFrame()
            self.top_aisles_by_orders = pd.DataFrame()
            self.most_expensive_products = pd.DataFrame()

    def create_top_products_charts(self):
        # Top 10 Products by Revenue
        fig1 = go.Figure()
        if not self.top_products_by_revenue.empty:
            fig1.add_trace(go.Bar(
                x=self.top_products_by_revenue['product_name'],
                y=self.top_products_by_revenue['price'],
                marker_color='#27ae60',
                name='Revenue'
            ))
            fig1.update_layout(title='Top 10 Products by Revenue', xaxis_title='Product', yaxis_title='Revenue ($)', height=350)
        else:
            fig1.add_annotation(
                text="No price data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig1.update_layout(title='Top 10 Products by Revenue (No Data)', height=350)
            
        # Top 10 Products by Orders
        fig2 = go.Figure()
        if not self.top_products_by_orders.empty:
            fig2.add_trace(go.Bar(
                x=self.top_products_by_orders['product_name'],
                y=self.top_products_by_orders['order_count'],
                marker_color='#3498db',
                name='Order Count'
            ))
            fig2.update_layout(title='Top 10 Products by Orders', xaxis_title='Product', yaxis_title='Order Count', height=350)
        else:
            fig2.add_annotation(
                text="No order data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig2.update_layout(title='Top 10 Products by Orders (No Data)', height=350)
            
        return html.Div([
            html.H4('Top 10 Products: Revenue vs. Orders'),
            html.Div([
                dcc.Graph(figure=fig1, style={'flex': '1', 'marginRight': '20px'}),
                dcc.Graph(figure=fig2, style={'flex': '1'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div('Insight: Are your top-selling items your top earners?', style={'marginTop': '10px', 'fontStyle': 'italic'})
        ], style={'marginBottom': '40px'})

    def create_top_reordered_products_charts(self):
        fig1 = go.Figure()
        if not self.top_reordered_by_revenue.empty:
            fig1.add_trace(go.Bar(
                x=self.top_reordered_by_revenue['product_name'],
                y=self.top_reordered_by_revenue['price'],
                marker_color='#e67e22',
                name='Revenue'
            ))
            fig1.update_layout(title='Top 10 Reordered Products by Revenue', xaxis_title='Product', yaxis_title='Revenue ($)', height=350)
        fig2 = go.Figure()
        if not self.top_reordered_by_orders.empty:
            fig2.add_trace(go.Bar(
                x=self.top_reordered_by_orders['product_name'],
                y=self.top_reordered_by_orders['order_count'],
                marker_color='#9b59b6',
                name='Order Count'
            ))
            fig2.update_layout(title='Top 10 Reordered Products by Orders', xaxis_title='Product', yaxis_title='Order Count', height=350)
        return html.Div([
            html.H4('Top 10 Reordered Products: Revenue vs. Orders'),
            html.Div([
                dcc.Graph(figure=fig1, style={'flex': '1', 'marginRight': '20px'}),
                dcc.Graph(figure=fig2, style={'flex': '1'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div("Insight: What's sticky and valuable?", style={'marginTop': '10px', 'fontStyle': 'italic'})
        ], style={'marginBottom': '40px'})

    def create_top_departments_charts(self):
        fig1 = go.Figure()
        if not self.top_departments_by_revenue.empty:
            fig1.add_trace(go.Bar(
                x=self.top_departments_by_revenue['department'],
                y=self.top_departments_by_revenue['price'],
                marker_color='#16a085',
                name='Revenue'
            ))
            fig1.update_layout(title='Top 10 Departments by Revenue', xaxis_title='Department', yaxis_title='Revenue ($)', height=350)
        fig2 = go.Figure()
        if not self.top_departments_by_orders.empty:
            fig2.add_trace(go.Bar(
                x=self.top_departments_by_orders['department'],
                y=self.top_departments_by_orders['order_count'],
                marker_color='#2980b9',
                name='Order Count'
            ))
            fig2.update_layout(title='Top 10 Departments by Orders', xaxis_title='Department', yaxis_title='Order Count', height=350)
        return html.Div([
            html.H4('Top 10 Departments: Revenue vs. Orders'),
            html.Div([
                dcc.Graph(figure=fig1, style={'flex': '1', 'marginRight': '20px'}),
                dcc.Graph(figure=fig2, style={'flex': '1'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div('Insight: Is produce high volume but low revenue?', style={'marginTop': '10px', 'fontStyle': 'italic'})
        ], style={'marginBottom': '40px'})

    def create_top_aisles_charts(self):
        fig1 = go.Figure()
        if not self.top_aisles_by_revenue.empty:
            fig1.add_trace(go.Bar(
                x=self.top_aisles_by_revenue['aisle'],
                y=self.top_aisles_by_revenue['price'],
                marker_color='#f39c12',
                name='Revenue'
            ))
            fig1.update_layout(title='Top 10 Aisles by Revenue', xaxis_title='Aisle', yaxis_title='Revenue ($)', height=350)
        fig2 = go.Figure()
        if not self.top_aisles_by_orders.empty:
            fig2.add_trace(go.Bar(
                x=self.top_aisles_by_orders['aisle'],
                y=self.top_aisles_by_orders['order_count'],
                marker_color='#8e44ad',
                name='Order Count'
            ))
            fig2.update_layout(title='Top 10 Aisles by Orders', xaxis_title='Aisle', yaxis_title='Order Count', height=350)
        return html.Div([
            html.H4('Top 10 Aisles: Revenue vs. Orders'),
            html.Div([
                dcc.Graph(figure=fig1, style={'flex': '1', 'marginRight': '20px'}),
                dcc.Graph(figure=fig2, style={'flex': '1'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div('Find hidden gems or underperformers.', style={'marginTop': '10px', 'fontStyle': 'italic'})
        ], style={'marginBottom': '40px'})

    def create_most_expensive_products_chart(self):
        fig = go.Figure()
        if not self.most_expensive_products.empty:
            fig.add_trace(go.Bar(
                x=self.most_expensive_products['product_name'],
                y=self.most_expensive_products['price'],
                marker_color='#c0392b',
                name='Max Price'
            ))
            fig.update_layout(title='Most Expensive Purchased Products', xaxis_title='Product', yaxis_title='Max Price ($)', height=350)
        return html.Div([
            html.H4('Most Expensive Purchased Products'),
            dcc.Graph(figure=fig),
            html.Div('Spot outliers or luxury segments.', style={'marginTop': '10px', 'fontStyle': 'italic'})
        ], style={'marginBottom': '40px'})

    def create_dashboard_layout(self):
        return html.Div([
            html.H2('Product & Category Performance'),
            html.P('Where revenue and order volume diverge—optimize product strategy'),
            self.create_top_products_charts(),
            self.create_top_reordered_products_charts(),
            self.create_top_departments_charts(),
            self.create_top_aisles_charts(),
            self.create_most_expensive_products_chart()
        ], style={'padding': '30px'})

def get_product_category_performance_tab_layout(product_category_dashboard=None):
    if product_category_dashboard is None:
        return dcc.Tab(label="Product & Category Performance", value="tab-product-category", children=[html.Div(['Data not loaded'])])
    return dcc.Tab(
        label="Product & Category Performance",
        value="tab-product-category",
        children=[product_category_dashboard.create_dashboard_layout()]
    ) 