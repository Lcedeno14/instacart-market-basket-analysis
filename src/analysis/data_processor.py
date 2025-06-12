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

class DataProcessor:
    """Central data processor for loading, merging, and preparing all application data."""
    
    def __init__(self, orders_df: pd.DataFrame = None, products_df: pd.DataFrame = None, 
                 departments_df: pd.DataFrame = None, order_products_df: pd.DataFrame = None,
                 merged_df: pd.DataFrame = None):
        """
        Initialize with either separate dataframes or already-merged data.
        
        Parameters:
        -----------
        orders_df : pd.DataFrame, optional
            DataFrame containing order information
        products_df : pd.DataFrame, optional
            DataFrame containing product information
        departments_df : pd.DataFrame, optional
            DataFrame containing department information
        order_products_df : pd.DataFrame, optional
            DataFrame containing order-product relationships
        merged_df : pd.DataFrame, optional
            Already-merged DataFrame with all necessary data
        """
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
        if merged_df is not None:
            # Use already-merged data
            self.merged_df = merged_df.copy()
            logger.info("Using pre-merged data for DataProcessor")
        else:
            # Fall back to original behavior for backward compatibility
            self.orders_df = orders_df
            self.products_df = products_df
            self.departments_df = departments_df
            self.order_products_df = order_products_df
            
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
        
        # Add price data if not already present
        if 'price' not in self.merged_df.columns:
            self._add_price_data()
    
    def _add_price_data(self):
        """Add price information to the merged dataframe if not present."""
        try:
            # Load price data
            prices_df = self._load_price_data()
            
            if not prices_df.empty:
                # Merge price data
                self.merged_df = self.merged_df.merge(
                    prices_df[['product_id', 'price']],
                    on='product_id',
                    how='left'
                )
                
                # Fill missing prices with department average
                dept_avg_prices = self.merged_df.groupby('department_id')['price'].transform('mean')
                self.merged_df['price'] = self.merged_df['price'].fillna(dept_avg_prices)
                
                logger.info("Successfully added price data to merged dataframe")
            else:
                logger.warning("No price data available")
                
        except Exception as e:
            logger.error(f"Error adding price data: {str(e)}")
    
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
            price_stats = self.merged_df['price'].describe()
            dept_price_stats = self.merged_df.groupby('department')['price'].agg(['mean', 'std', 'count'])
            
            # Price distribution analysis
            price_quartiles = self.merged_df['price'].quantile([0.25, 0.5, 0.75])
            
            # Department price comparison
            dept_avg_prices = self.merged_df.groupby('department')['price'].mean().sort_values(ascending=False)
            
            # Price range analysis
            price_ranges = {
                'low': self.merged_df[self.merged_df['price'] <= price_quartiles[0.25]],
                'medium': self.merged_df[(self.merged_df['price'] > price_quartiles[0.25]) & (self.merged_df['price'] <= price_quartiles[0.75])],
                'high': self.merged_df[self.merged_df['price'] > price_quartiles[0.75]]
            }
            
            insights = {
                "title": "Price Analysis Insights",
                "sections": [
                    {
                        "title": "Overall Price Statistics",
                        "insights": [
                            f"Average product price: ${price_stats['mean']:.2f}",
                            f"Price range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}",
                            f"Standard deviation: ${price_stats['std']:.2f}",
                            f"Total products analyzed: {price_stats['count']:,.0f}"
                        ]
                    },
                    {
                        "title": "Department Price Analysis",
                        "insights": [
                            f"Highest average price: {dept_avg_prices.index[0]} (${dept_avg_prices.iloc[0]:.2f})",
                            f"Lowest average price: {dept_avg_prices.index[-1]} (${dept_avg_prices.iloc[-1]:.2f})",
                            f"Price variation across departments: ${dept_avg_prices.std():.2f}"
                        ]
                    },
                    {
                        "title": "Price Distribution",
                        "insights": [
                            f"25% of products cost ≤ ${price_quartiles[0.25]:.2f}",
                            f"50% of products cost ≤ ${price_quartiles[0.5]:.2f}",
                            f"75% of products cost ≤ ${price_quartiles[0.75]:.2f}",
                            f"High-value products (>75th percentile): {len(price_ranges['high']):,} items"
                        ]
                    }
                ],
                "recommendations": [
                    "Focus on high-value departments for revenue optimization",
                    "Consider price elasticity for pricing strategy",
                    "Monitor price distribution for market positioning",
                    "Analyze seasonal price variations for inventory planning"
                ]
            }
            
            # Cache the insights
            self._insights_cache = insights
            return insights
            
        except Exception as e:
            logger.error(f"Error generating price insights: {str(e)}")
            return {
                "title": "Price Analysis",
                "sections": [{
                    "title": "Error",
                    "insights": [f"Error generating insights: {str(e)}"]
                }],
                "recommendations": []
            }
    
    @timing_decorator
    def get_price_elasticity(self) -> Dict:
        """Calculate price elasticity of demand."""
        # Return cached elasticity if available
        if self._elasticity_cache is not None:
            logger.info("Returning cached price elasticity")
            return self._elasticity_cache
            
        try:
            if self.merged_df.empty:
                return {"error": "No data available for elasticity calculation"}
            
            # Group by product and calculate price-demand relationship
            product_stats = self.merged_df.groupby('product_id').agg({
                'price': 'mean',
                'order_id': 'count'
            }).reset_index()
            
            # Calculate elasticity using log-log regression
            if len(product_stats) > 1:
                # Log transform for elasticity calculation
                product_stats['log_price'] = np.log(product_stats['price'])
                product_stats['log_quantity'] = np.log(product_stats['order_id'])
                
                # Calculate correlation and elasticity
                correlation = product_stats['log_price'].corr(product_stats['log_quantity'])
                
                # Simple elasticity calculation
                price_change = product_stats['price'].std() / product_stats['price'].mean()
                quantity_change = product_stats['order_id'].std() / product_stats['order_id'].mean()
                
                if price_change > 0:
                    elasticity = (quantity_change / price_change) * -1  # Negative for demand
                else:
                    elasticity = 0
                
                elasticity_analysis = {
                    "price_elasticity": elasticity,
                    "correlation": correlation,
                    "price_volatility": price_change,
                    "demand_volatility": quantity_change,
                    "interpretation": self._interpret_elasticity(elasticity)
                }
            else:
                elasticity_analysis = {"error": "Insufficient data for elasticity calculation"}
            
            # Cache the elasticity analysis
            self._elasticity_cache = elasticity_analysis
            return elasticity_analysis
            
        except Exception as e:
            logger.error(f"Error calculating price elasticity: {str(e)}")
            return {"error": f"Error calculating elasticity: {str(e)}"}
    
    def _interpret_elasticity(self, elasticity: float) -> str:
        """Interpret price elasticity value."""
        if abs(elasticity) < 0.5:
            return "Inelastic demand - price changes have minimal impact on quantity demanded"
        elif abs(elasticity) < 1.5:
            return "Unitary elastic demand - price changes proportionally affect quantity demanded"
        else:
            return "Elastic demand - price changes significantly affect quantity demanded"
    
    @timing_decorator
    def create_visualization(self) -> go.Figure:
        """Create price analysis visualization."""
        # Return cached visualization if available
        if self._visualization_cache is not None:
            logger.info("Returning cached price visualization")
            return self._visualization_cache
            
        try:
            if self.merged_df.empty:
                return go.Figure()
            
            # Create subplot for price analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price Distribution', 'Department Price Comparison', 
                              'Price vs Quantity', 'Price Range Analysis'),
                specs=[[{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "box"}]]
            )
            
            # Price distribution histogram
            fig.add_trace(
                go.Histogram(
                    x=self.merged_df['price'],
                    nbinsx=50,
                    name='Price Distribution',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # Department price comparison
            dept_prices = self.merged_df.groupby('department')['price'].mean().sort_values(ascending=True)
            fig.add_trace(
                go.Bar(
                    x=dept_prices.values,
                    y=dept_prices.index,
                    orientation='h',
                    name='Avg Price by Department',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            # Price vs Quantity scatter
            product_stats = self.merged_df.groupby('product_id').agg({
                'price': 'mean',
                'order_id': 'count'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=product_stats['price'],
                    y=product_stats['order_id'],
                    mode='markers',
                    name='Price vs Quantity',
                    marker=dict(size=8, color='red', opacity=0.6)
                ),
                row=2, col=1
            )
            
            # Price range box plot
            fig.add_trace(
                go.Box(
                    y=self.merged_df['price'],
                    name='Price Range',
                    marker_color='orange'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="Comprehensive Price Analysis",
                title_x=0.5,
                showlegend=False
            )
            
            # Cache the visualization
            self._visualization_cache = fig
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price visualization: {str(e)}")
            return go.Figure()
    
    def clear_cache(self):
        """Clear all cached data."""
        self._elasticity_cache = None
        self._visualization_cache = None
        self._insights_cache = None
        logger.info("Cache cleared")
    
    def get_merged_data(self) -> pd.DataFrame:
        """Get the processed merged dataset."""
        return self.merged_df.copy()
    
    def has_price_data(self) -> bool:
        """Check if price data is available."""
        return 'price' in self.merged_df.columns and not self.merged_df['price'].isna().all() 