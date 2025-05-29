import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
from sqlalchemy import create_engine, text
import time
from functools import wraps, lru_cache

logger = logging.getLogger(__name__)

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

class PriceAnalysis:
    """Handle price analysis and visualizations separately from data storytelling."""
    
    def __init__(self, orders_df: pd.DataFrame, products_df: pd.DataFrame, 
                 departments_df: pd.DataFrame, order_products_df: pd.DataFrame):
        """
        Initialize with the necessary dataframes.
        
        Parameters:
        -----------
        orders_df : pd.DataFrame
            DataFrame containing order information
        products_df : pd.DataFrame
            DataFrame containing product information
        departments_df : pd.DataFrame
            DataFrame containing department information
        order_products_df : pd.DataFrame
            DataFrame containing order-product relationships
        """
        self.orders_df = orders_df
        self.products_df = products_df
        self.departments_df = departments_df
        self.order_products_df = order_products_df
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
        # Initialize prices_df first
        self.prices_df = self._load_price_data()
        
        # Then prepare merged data
        self.merged_df = self._prepare_data()
        
        # Cache for price elasticity calculations
        self._elasticity_cache = None
        
        # Cache for visualizations
        self._visualization_cache = None
        
        # Cache for insights
        self._insights_cache = None
    
    @timing_decorator
    def _load_price_data(self) -> pd.DataFrame:
        """Load product price data from the database."""
        try:
            with self.engine.connect() as conn:
                # First, check what columns exist in the table
                check_columns = text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'products_with_price'
                """)
                columns_df = pd.read_sql(check_columns, conn)
                available_columns = columns_df['column_name'].tolist()
                
                # Build query based on available columns
                select_columns = ['p.product_id', 'p.product_name', 'd.department_id', 'pw.price']
                query = f"""
                    SELECT 
                        {', '.join(select_columns)}
                    FROM products_with_price pw
                    JOIN products p ON pw.product_id = p.product_id
                    JOIN departments d ON p.department_id = d.department_id
                """
                
                prices_df = pd.read_sql(query, conn)
                logger.info(f"Successfully loaded price data with columns: {prices_df.columns.tolist()}")
                return prices_df
                
        except Exception as e:
            logger.error(f"Error loading price data: {str(e)}")
            return pd.DataFrame(columns=['product_id', 'product_name', 'department_id', 'price'])
    
    @timing_decorator
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare merged dataset for analysis."""
        try:
            # First, ensure we have the basic data
            if self.orders_df.empty or self.products_df.empty or self.departments_df.empty:
                logger.error("Missing required dataframes")
                return pd.DataFrame()
            
            # Merge orders with order_products first
            df = self.order_products_df.merge(
                self.orders_df,
                on='order_id',
                how='inner'
            )
            
            # Then merge with products and departments
            df = df.merge(
                self.products_df,
                on='product_id',
                how='inner'
            ).merge(
                self.departments_df,
                on='department_id',
                how='inner'
            )
            
            # Add price information if available
            if not self.prices_df.empty:
                df = df.merge(
                    self.prices_df[['product_id', 'price']],
                    on='product_id',
                    how='left'
                )
                # Fill missing prices with department average
                dept_avg_prices = df.groupby('department_id')['price'].transform('mean')
                df['price'] = df['price'].fillna(dept_avg_prices)
            
            logger.info(f"Successfully prepared merged data with columns: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return pd.DataFrame()
    
    @timing_decorator
    def get_price_insights(self) -> Dict:
        """Get price analysis insights."""
        # Return cached insights if available
        if self._insights_cache is not None:
            logger.info("Returning cached price insights")
            return self._insights_cache
            
        try:
            if self.prices_df.empty or self.merged_df.empty:
                return {
                    "title": "Price Analysis",
                    "sections": [{
                        "title": "No Data Available",
                        "insights": ["Required data is not available for analysis"]
                    }],
                    "recommendations": []
                }
            
            # Calculate key price metrics
            avg_price_by_dept = self.merged_df.groupby('department', observed=True)['price'].mean().sort_values(ascending=False)
            price_range_by_dept = self.merged_df.groupby('department', observed=True)['price'].agg(['min', 'max'])
            
            # Calculate customer spending patterns
            customer_spending = self.merged_df.groupby('user_id', observed=True).agg({
                'price': ['sum', 'mean', 'count'],
                'order_id': 'nunique'
            }).reset_index()
            customer_spending.columns = ['user_id', 'total_spent', 'avg_item_price', 'items_purchased', 'orders']
            customer_spending['avg_order_value'] = customer_spending['total_spent'] / customer_spending['orders']
            
            # Get price elasticity (using cache)
            price_elasticity = self.get_price_elasticity()
            
            story = {
                "title": "Price Analysis and Customer Spending Patterns",
                "sections": [
                    {
                        "title": "Department Price Analysis",
                        "insights": [
                            f"Highest average price department: {avg_price_by_dept.index[0]} (${avg_price_by_dept.iloc[0]:.2f})",
                            f"Lowest average price department: {avg_price_by_dept.index[-1]} (${avg_price_by_dept.iloc[-1]:.2f})",
                            f"Price range varies significantly across departments, from ${price_range_by_dept['min'].min():.2f} to ${price_range_by_dept['max'].max():.2f}"
                        ]
                    },
                    {
                        "title": "Customer Spending Patterns",
                        "insights": [
                            f"Average customer spends ${customer_spending['total_spent'].mean():.2f} total",
                            f"Average order value: ${customer_spending['avg_order_value'].mean():.2f}",
                            f"Customers purchase {customer_spending['items_purchased'].mean():.1f} items per order on average"
                        ]
                    }
                ],
                "recommendations": [
                    "Implement dynamic pricing for price-sensitive departments",
                    "Create value bundles for high-price departments",
                    "Develop targeted promotions based on customer spending patterns",
                    "Optimize inventory mix based on price elasticity"
                ]
            }
            
            # Add price sensitivity section if available
            if price_elasticity.get('most_elastic') != 'N/A':
                story["sections"].append({
                    "title": "Price Sensitivity",
                    "insights": [
                        f"Most price-sensitive department: {price_elasticity['most_elastic']}",
                        f"Least price-sensitive department: {price_elasticity['least_elastic']}",
                        "Price sensitivity varies significantly across product categories"
                    ]
                })
            
            # Cache the insights
            self._insights_cache = story
            return story
            
        except Exception as e:
            logger.error(f"Error generating price insights: {str(e)}")
            return {
                "title": "Price Analysis",
                "sections": [{
                    "title": "Error in Analysis",
                    "insights": [f"An error occurred while generating insights: {str(e)}"]
                }],
                "recommendations": []
            }
    
    @timing_decorator
    def get_price_elasticity(self) -> Dict:
        """Get price elasticity calculations (using cache if available)."""
        if self._elasticity_cache is not None:
            logger.info("Returning cached price elasticity")
            return self._elasticity_cache
            
        try:
            if self.merged_df.empty or 'price' not in self.merged_df.columns:
                return {'most_elastic': 'N/A', 'least_elastic': 'N/A'}
            
            # Calculate price elasticity by department
            elasticity = {}
            for dept in self.merged_df['department'].unique():
                dept_data = self.merged_df[self.merged_df['department'] == dept]
                
                try:
                    # Use qcut with duplicates='drop' to handle duplicate values
                    price_ranges = pd.qcut(dept_data['price'], q=5, duplicates='drop')
                    # Use observed=True for categorical data
                    demand_by_price = dept_data.groupby(price_ranges, observed=True)['product_id'].count()
                    
                    # Calculate percentage changes
                    price_changes = price_ranges.value_counts().sort_index().pct_change()
                    demand_changes = demand_by_price.pct_change()
                    
                    # Calculate elasticity
                    elasticity[dept] = (demand_changes / price_changes).mean()
                except ValueError as e:
                    logger.warning(f"Could not calculate elasticity for department {dept}: {str(e)}")
                    elasticity[dept] = 0
            
            # Find most and least elastic departments
            if not elasticity:
                return {'most_elastic': 'N/A', 'least_elastic': 'N/A'}
            
            elasticities = pd.Series(elasticity)
            most_elastic = elasticities.idxmax()
            least_elastic = elasticities.idxmin()
            
            result = {
                'most_elastic': most_elastic,
                'least_elastic': least_elastic,
                'elasticities': elasticities.to_dict()
            }
            
            # Cache the result
            self._elasticity_cache = result
            return result
            
        except Exception as e:
            logger.error(f"Error calculating price elasticity: {str(e)}")
            return {'most_elastic': 'N/A', 'least_elastic': 'N/A'}
    
    @timing_decorator
    def create_visualization(self) -> go.Figure:
        """Create visualization for price analysis."""
        # Return cached visualization if available
        if self._visualization_cache is not None:
            logger.info("Returning cached visualization (cache hit)")
            return self._visualization_cache
        else:
            logger.info("No cached visualization found (cache miss)")
            
        try:
            if self.prices_df.empty:
                logger.warning("Empty prices dataframe, returning empty figure")
                return go.Figure()
            
            logger.info("Starting visualization creation...")
            viz_start = time.time()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Average Price by Department',
                    'Price Distribution by Department',
                    'Customer Spending Patterns',
                    # 'Price Elasticity by Department'  # commented out
                )
            )
            
            # 1. Average price by department
            t0 = time.time()
            logger.info("[TIMING] Calculating average prices by department...")
            avg_prices = self.merged_df.groupby('department', observed=True)['price'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(
                    x=avg_prices.index,
                    y=avg_prices.values,
                    name='Average Price'
                ),
                row=1, col=1
            )
            logger.info(f"[TIMING] Average Price by Department: {time.time() - t0:.3f} seconds")
            
            # 2. Price distribution by department
            t0 = time.time()
            logger.info("[TIMING] Creating price distribution box plots...")
            for dept in avg_prices.index[:5]:  # Top 5 departments by average price
                dept_data = self.merged_df[self.merged_df['department'] == dept]['price']
                fig.add_trace(
                    go.Box(
                        y=dept_data,
                        name=dept,
                        boxpoints='outliers'
                    ),
                    row=1, col=2
                )
            logger.info(f"[TIMING] Price Distribution by Department: {time.time() - t0:.3f} seconds")
            
            # 3. Customer spending patterns
            t0 = time.time()
            logger.info("[TIMING] Calculating customer spending patterns...")
            customer_spending = self.merged_df.groupby('user_id', observed=True).agg({
                'price': ['sum', 'mean'],
                'order_id': 'nunique'
            }).reset_index()
            customer_spending.columns = ['user_id', 'total_spent', 'avg_item_price', 'orders']
            
            fig.add_trace(
                go.Scatter(
                    x=customer_spending['orders'],
                    y=customer_spending['total_spent'],
                    mode='markers',
                    marker=dict(
                        size=customer_spending['avg_item_price'],
                        color=customer_spending['avg_item_price'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Customer Spending'
                ),
                row=2, col=1
            )
            logger.info(f"[TIMING] Customer Spending Patterns: {time.time() - t0:.3f} seconds")
            
            # 4. Price elasticity (using cache) -- commented out for performance
            # t0 = time.time()
            # logger.info("[TIMING] Getting price elasticity data...")
            # elasticity = self.get_price_elasticity()
            # if 'elasticities' in elasticity:
            #     elasticities = pd.Series(elasticity['elasticities'])
            #     fig.add_trace(
            #         go.Bar(
            #             x=elasticities.index,
            #             y=elasticities.values,
            #             name='Price Elasticity'
            #         ),
            #         row=2, col=2
            #     )
            # logger.info(f"[TIMING] Price Elasticity by Department: {time.time() - t0:.3f} seconds")

            logger.info("Updating figure layout...")
            fig.update_layout(
                height=1000,
                width=1200,
                title_text='Price Analysis Dashboard',
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text='Department', row=1, col=1)
            fig.update_yaxes(title_text='Average Price ($)', row=1, col=1)
            fig.update_xaxes(title_text='Department', row=1, col=2)
            fig.update_yaxes(title_text='Price ($)', row=1, col=2)
            fig.update_xaxes(title_text='Number of Orders', row=2, col=1)
            fig.update_yaxes(title_text='Total Spent ($)', row=2, col=1)
            # fig.update_xaxes(title_text='Department', row=2, col=2)
            # fig.update_yaxes(title_text='Price Elasticity', row=2, col=2)
            
            # Cache the visualization
            logger.info("Caching visualization...")
            self._visualization_cache = fig
            logger.info(f"Visualization creation took {time.time() - viz_start:.2f} seconds")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price visualization: {str(e)}")
            return go.Figure()

    def clear_cache(self):
        """Clear all caches."""
        logger.info("Clearing all caches")
        self._elasticity_cache = None
        self._visualization_cache = None
        self._insights_cache = None 