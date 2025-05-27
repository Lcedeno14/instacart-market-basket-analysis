import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class DataStorytelling:
    """Generate narrative insights and stories from Instacart data analysis."""
    
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
    
    def generate_customer_journey_story(self) -> Dict:
        """
        Generate insights about customer shopping journey and behavior.
        
        Returns:
        --------
        Dict containing narrative insights and supporting metrics
        """
        try:
            if self.merged_df.empty:
                logger.error("No data available for customer journey analysis")
                return {
                    "title": "Customer Journey Analysis",
                    "sections": [{
                        "title": "No Data Available",
                        "insights": ["Required data is not available for analysis"]
                    }],
                    "recommendations": []
                }
            
            # Map order_dow to day names if not already done
            if 'day_of_week' not in self.merged_df.columns:
                dow_map = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 
                          4: "Thursday", 5: "Friday", 6: "Saturday"}
                self.merged_df['day_of_week'] = self.merged_df['order_dow'].map(dow_map)
            
            # Calculate key metrics
            avg_order_size = self.merged_df.groupby('order_id', observed=True)['product_id'].count().mean()
            avg_days_between_orders = self.merged_df.groupby('user_id', observed=True)['days_since_prior_order'].mean().mean()
            peak_hour = self.merged_df.groupby('order_hour_of_day', observed=True)['order_id'].count().idxmax()
            peak_day = self.merged_df.groupby('day_of_week', observed=True)['order_id'].count().idxmax()
            
            # Calculate department preferences
            dept_preferences = self.merged_df.groupby('department', observed=True)['product_id'].count()
            top_dept = dept_preferences.nlargest(1).index[0]
            top_dept_pct = (dept_preferences[top_dept] / dept_preferences.sum() * 100).round(1)
            
            # Calculate reorder rates if reordered column exists
            reorder_rate = 0
            if 'reordered' in self.merged_df.columns:
                reorder_rate = (self.merged_df.groupby('product_id', observed=True)['reordered'].mean() * 100).mean()
            
            # Add price-based insights if available
            price_insights = []
            if 'price' in self.merged_df.columns:
                avg_order_value = self.merged_df.groupby('order_id', observed=True)['price'].sum().mean()
                price_insights.append(f"Average order value: ${avg_order_value:.2f}")
                
                # Calculate price range by department
                price_range = self.merged_df.groupby('department', observed=True)['price'].agg(['min', 'max', 'mean'])
                highest_price_dept = price_range['mean'].idxmax()
                price_insights.append(
                    f"Highest average price department: {highest_price_dept} "
                    f"(${price_range.loc[highest_price_dept, 'mean']:.2f})"
                )
            
            story = {
                "title": "Understanding the Instacart Customer Journey",
                "sections": [
                    {
                        "title": "Shopping Patterns",
                        "insights": [
                            f"Customers place orders every {avg_days_between_orders:.1f} days on average",
                            f"The average order contains {avg_order_size:.1f} items",
                            f"Peak shopping hours are around {peak_hour}:00, with {peak_day} being the busiest day"
                        ] + price_insights
                    },
                    {
                        "title": "Department Preferences",
                        "insights": [
                            f"{top_dept} is the most popular department, accounting for {top_dept_pct}% of all items",
                            f"Customers reorder {reorder_rate:.1f}% of their previous purchases" if reorder_rate > 0 else
                            "Reorder rate information is not available"
                        ]
                    }
                ],
                "recommendations": [
                    "Optimize inventory for peak shopping hours",
                    f"Focus on {top_dept} department promotions during {peak_day}s",
                    "Implement personalized reorder suggestions"
                ]
            }
            
            if price_insights:
                story["recommendations"].extend([
                    "Develop targeted promotions for high-value departments",
                    "Create value bundles for price-sensitive customers"
                ])
            
            return story
            
        except Exception as e:
            logger.error(f"Error generating customer journey story: {str(e)}")
            return {
                "title": "Customer Journey Analysis",
                "sections": [{
                    "title": "Error in Analysis",
                    "insights": [f"An error occurred while generating insights: {str(e)}"]
                }],
                "recommendations": []
            }
    
    def generate_seasonal_trends_story(self) -> Dict:
        """
        Generate insights about seasonal shopping patterns.
        
        Returns:
        --------
        Dict containing narrative insights and supporting metrics
        """
        try:
            if self.merged_df.empty:
                return {
                    "title": "Seasonal Trends Analysis",
                    "sections": [{
                        "title": "No Data Available",
                        "insights": ["Required data is not available for analysis"]
                    }],
                    "recommendations": []
                }
            
            # Map order_dow to day names if not already done
            if 'day_of_week' not in self.merged_df.columns:
                dow_map = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 
                          4: "Thursday", 5: "Friday", 6: "Saturday"}
                self.merged_df['day_of_week'] = self.merged_df['order_dow'].map(dow_map)
            
            # Calculate daily order patterns
            daily_orders = self.merged_df.groupby('day_of_week', observed=True)['order_id'].count()
            daily_avg = daily_orders.mean()
            busiest_day = daily_orders.idxmax()
            busiest_day_pct = (daily_orders.max() / daily_avg * 100 - 100).round(1)
            
            # Calculate hourly patterns
            hourly_orders = self.merged_df.groupby('order_hour_of_day', observed=True)['order_id'].count()
            peak_hour = hourly_orders.idxmax()
            peak_hour_pct = (hourly_orders.max() / hourly_orders.mean() * 100 - 100).round(1)
            
            # Calculate department seasonality
            dept_by_day = self.merged_df.groupby(['department', 'day_of_week'], observed=True)['product_id'].count()
            dept_seasonality = dept_by_day.groupby('department').std() / dept_by_day.groupby('department').mean()
            most_seasonal_dept = dept_seasonality.idxmax()
            
            story = {
                "title": "Seasonal Shopping Patterns and Trends",
                "sections": [
                    {
                        "title": "Weekly Patterns",
                        "insights": [
                            f"{busiest_day} is the busiest shopping day, with {busiest_day_pct}% more orders than average",
                            f"Order volume varies significantly throughout the week, with clear peak days"
                        ]
                    },
                    {
                        "title": "Daily Patterns",
                        "insights": [
                            f"Peak shopping hour is {peak_hour}:00, with {peak_hour_pct}% more orders than average",
                            f"Order volume shows distinct morning and evening peaks"
                        ]
                    },
                    {
                        "title": "Department Seasonality",
                        "insights": [
                            f"{most_seasonal_dept} shows the most variation in daily demand",
                            "Some departments show strong day-of-week preferences"
                        ]
                    }
                ],
                "recommendations": [
                    f"Optimize staffing and inventory for {busiest_day} peak hours",
                    f"Implement targeted promotions for {most_seasonal_dept} on low-traffic days",
                    "Adjust delivery capacity based on hourly demand patterns"
                ]
            }
            return story
        except Exception as e:
            logger.error(f"Error generating seasonal trends story: {str(e)}")
            return {
                "title": "Seasonal Trends Analysis",
                "sections": [{
                    "title": "Error in Analysis",
                    "insights": [f"An error occurred while generating insights: {str(e)}"]
                }],
                "recommendations": []
            }
    
    def generate_product_association_story(self, min_support: float = 0.01, 
                                         min_confidence: float = 0.1) -> Dict:
        """
        Generate insights about product associations and shopping patterns.
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold for association rules
        min_confidence : float
            Minimum confidence threshold for association rules
            
        Returns:
        --------
        Dict containing narrative insights and supporting metrics
        """
        try:
            # Get association rules from database
            # This assumes you have a function to get rules from your database
            rules_df = self._get_association_rules(min_support, min_confidence)
            
            if len(rules_df) == 0:
                return {
                    "title": "Product Association Analysis",
                    "sections": [{
                        "title": "No Strong Associations",
                        "insights": [
                            f"No strong product associations found with current thresholds",
                            f"Try lowering support ({min_support}) or confidence ({min_confidence}) thresholds"
                        ]
                    }]
                }
            
            # Analyze top rules
            top_rules = rules_df.nlargest(5, 'lift')
            top_rule = top_rules.iloc[0]
            
            # Calculate department-level associations
            dept_rules = self._get_department_associations()
            top_dept_rule = dept_rules.iloc[0] if len(dept_rules) > 0 else None
            
            story = {
                "title": "Product Association Insights",
                "sections": [
                    {
                        "title": "Strongest Product Associations",
                        "insights": [
                            f"Strongest association: {top_rule['antecedents']} → {top_rule['consequents']}",
                            f"This combination appears {top_rule['support']*100:.1f}% of the time",
                            f"When customers buy {top_rule['antecedents']}, they also buy {top_rule['consequents']} {top_rule['confidence']*100:.1f}% of the time"
                        ]
                    },
                    {
                        "title": "Department Associations",
                        "insights": [
                            f"Strongest department association: {top_dept_rule['antecedents']} → {top_dept_rule['consequents']}" if top_dept_rule else "No strong department associations found",
                            "Some departments show natural shopping patterns together"
                        ]
                    }
                ],
                "recommendations": [
                    "Implement cross-selling strategies for strongly associated products",
                    "Optimize store layout based on product associations",
                    "Create targeted bundle promotions for associated items"
                ]
            }
            return story
        except Exception as e:
            logger.error(f"Error generating product association story: {str(e)}")
            return {}
    
    def _get_association_rules(self, min_support: float, min_confidence: float) -> pd.DataFrame:
        """
        Get association rules from the market_basket_rules table.
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold for filtering rules
        min_confidence : float
            Minimum confidence threshold for filtering rules
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing filtered association rules
        """
        try:
            query = text("""
                SELECT 
                    antecedents,
                    consequents,
                    support,
                    confidence,
                    lift,
                    conviction
                FROM market_basket_rules
                WHERE support >= :min_support 
                AND confidence >= :min_confidence
                ORDER BY lift DESC
            """)
            
            with self.engine.connect() as conn:
                rules_df = pd.read_sql(
                    query, 
                    conn, 
                    params={'min_support': min_support, 'min_confidence': min_confidence}
                )
            
            if len(rules_df) == 0:
                logger.warning(f"No rules found with support >= {min_support} and confidence >= {min_confidence}")
                return pd.DataFrame()
            
            # Convert string representations of sets back to actual sets
            rules_df['antecedents'] = rules_df['antecedents'].apply(eval)
            rules_df['consequents'] = rules_df['consequents'].apply(eval)
            
            return rules_df
            
        except Exception as e:
            logger.error(f"Error fetching association rules: {str(e)}")
            return pd.DataFrame()
    
    def _get_department_associations(self) -> pd.DataFrame:
        """
        Get department-level associations by aggregating product-level rules.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing department-level association rules
        """
        try:
            # Get product-level rules
            rules_df = self._get_association_rules(min_support=0.01, min_confidence=0.1)
            if len(rules_df) == 0:
                return pd.DataFrame()
            
            # Create a mapping of product_id to department
            product_dept_map = self.products_df.set_index('product_id')['department_id'].to_dict()
            
            # Convert product sets to department sets
            def get_dept_set(product_set):
                return {product_dept_map.get(int(p), 0) for p in product_set}
            
            rules_df['antecedent_depts'] = rules_df['antecedents'].apply(get_dept_set)
            rules_df['consequent_depts'] = rules_df['consequents'].apply(get_dept_set)
            
            # Aggregate rules by department pairs
            dept_rules = []
            for _, row in rules_df.iterrows():
                for ant_dept in row['antecedent_depts']:
                    for cons_dept in row['consequent_depts']:
                        if ant_dept != cons_dept:
                            dept_rules.append({
                                'antecedent_dept': ant_dept,
                                'consequent_dept': cons_dept,
                                'support': row['support'],
                                'confidence': row['confidence'],
                                'lift': row['lift']
                            })
            
            dept_rules_df = pd.DataFrame(dept_rules)
            if len(dept_rules_df) > 0:
                # Get department names
                dept_names = self.departments_df.set_index('department_id')['department'].to_dict()
                dept_rules_df['antecedent_dept'] = dept_rules_df['antecedent_dept'].map(dept_names)
                dept_rules_df['consequent_dept'] = dept_rules_df['consequent_dept'].map(dept_names)
                
                # Aggregate by department pairs
                dept_rules_df = dept_rules_df.groupby(['antecedent_dept', 'consequent_dept']).agg({
                    'support': 'mean',
                    'confidence': 'mean',
                    'lift': 'mean'
                }).reset_index()
            
            return dept_rules_df
            
        except Exception as e:
            logger.error(f"Error calculating department associations: {str(e)}")
            return pd.DataFrame()
    
    def generate_price_insights_story(self) -> Dict:
        """
        Generate insights about pricing patterns and customer spending behavior.
        
        Returns:
        --------
        Dict containing narrative insights and supporting metrics
        """
        try:
            if self.prices_df.empty:
                return {
                    "title": "Price Analysis",
                    "sections": [{
                        "title": "No Price Data Available",
                        "insights": ["Price data is not available for analysis"]
                    }],
                    "recommendations": []
                }
            
            if self.merged_df.empty:
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
            
            # Calculate price sensitivity
            price_elasticity = self._calculate_price_elasticity()
            
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
            
            return story
            
        except Exception as e:
            logger.error(f"Error generating price insights story: {str(e)}")
            return {
                "title": "Price Analysis",
                "sections": [{
                    "title": "Error in Analysis",
                    "insights": [f"An error occurred while generating insights: {str(e)}"]
                }],
                "recommendations": []
            }
    
    def _calculate_price_elasticity(self) -> Dict:
        """Calculate price elasticity of demand by department."""
        try:
            if self.merged_df.empty or 'price' not in self.merged_df.columns:
                return {'most_elastic': 'N/A', 'least_elastic': 'N/A'}
            
            # Calculate price elasticity by department
            elasticity = {}
            for dept in self.merged_df['department'].unique():
                dept_data = self.merged_df[self.merged_df['department'] == dept]
                
                # Group by price ranges and calculate demand
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
            
            return {
                'most_elastic': most_elastic,
                'least_elastic': least_elastic,
                'elasticities': elasticities.to_dict()
            }
        except Exception as e:
            logger.error(f"Error calculating price elasticity: {str(e)}")
            return {'most_elastic': 'N/A', 'least_elastic': 'N/A'}
    
    def create_price_visualization(self) -> go.Figure:
        """Create visualization for price analysis."""
        try:
            if self.prices_df.empty:
                return go.Figure()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Average Price by Department',
                    'Price Distribution by Department',
                    'Customer Spending Patterns',
                    'Price Elasticity by Department'
                )
            )
            
            # Average price by department
            avg_prices = self.merged_df.groupby('department', observed=True)['price'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(
                    x=avg_prices.index,
                    y=avg_prices.values,
                    name='Average Price'
                ),
                row=1, col=1
            )
            
            # Price distribution by department
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
            
            # Customer spending patterns
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
            
            # Price elasticity
            elasticity = self._calculate_price_elasticity()
            if 'elasticities' in elasticity:
                elasticities = pd.Series(elasticity['elasticities'])
                fig.add_trace(
                    go.Bar(
                        x=elasticities.index,
                        y=elasticities.values,
                        name='Price Elasticity'
                    ),
                    row=2, col=2
                )
            
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
            fig.update_xaxes(title_text='Department', row=2, col=2)
            fig.update_yaxes(title_text='Price Elasticity', row=2, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price visualization: {str(e)}")
            return go.Figure()
    
    def create_story_visualization(self, story_type: str) -> go.Figure:
        """
        Create visualization for the specified story type.
        
        Parameters:
        -----------
        story_type : str
            Type of story to visualize ('customer_journey', 'seasonal_trends', 
            'product_associations', 'price_analysis')
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with story visualizations
        """
        try:
            if story_type == 'customer_journey':
                return self._create_customer_journey_viz()
            elif story_type == 'seasonal_trends':
                return self._create_seasonal_trends_viz()
            elif story_type == 'product_associations':
                return self._create_product_associations_viz()
            elif story_type == 'price_analysis':
                return self.create_price_visualization()
            else:
                raise ValueError(f"Unknown story type: {story_type}")
        except Exception as e:
            logger.error(f"Error creating story visualization: {str(e)}")
            return go.Figure()
    
    def _create_customer_journey_viz(self) -> go.Figure:
        """Create visualization for customer journey story."""
        try:
            if self.merged_df.empty:
                return go.Figure()
            
            # Map order_dow to day names if not already done
            if 'day_of_week' not in self.merged_df.columns:
                dow_map = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 
                          4: "Thursday", 5: "Friday", 6: "Saturday"}
                self.merged_df['day_of_week'] = self.merged_df['order_dow'].map(dow_map)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Order Size Distribution',
                    'Days Between Orders',
                    'Shopping Hour Distribution',
                    'Department Preferences'
                )
            )
            
            # Order size distribution
            order_sizes = self.merged_df.groupby('order_id', observed=True)['product_id'].count()
            fig.add_trace(
                go.Histogram(x=order_sizes, name='Order Size'),
                row=1, col=1
            )
            
            # Days between orders
            days_between = self.merged_df.groupby('user_id', observed=True)['days_since_prior_order'].mean()
            fig.add_trace(
                go.Histogram(x=days_between, name='Days Between Orders'),
                row=1, col=2
            )
            
            # Shopping hour distribution
            hourly_orders = self.merged_df.groupby('order_hour_of_day', observed=True)['order_id'].count()
            fig.add_trace(
                go.Bar(x=hourly_orders.index, y=hourly_orders.values, name='Hour of Day'),
                row=2, col=1
            )
            
            # Department preferences
            dept_prefs = self.merged_df.groupby('department', observed=True)['product_id'].count()
            fig.add_trace(
                go.Bar(x=dept_prefs.index, y=dept_prefs.values, name='Department'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                width=1200,
                title_text='Customer Shopping Journey Analysis',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating customer journey visualization: {str(e)}")
            return go.Figure()
    
    def _create_seasonal_trends_viz(self) -> go.Figure:
        """Create visualization for seasonal trends story."""
        try:
            if self.merged_df.empty:
                return go.Figure()
            
            # Map order_dow to day names if not already done
            if 'day_of_week' not in self.merged_df.columns:
                dow_map = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 
                          4: "Thursday", 5: "Friday", 6: "Saturday"}
                self.merged_df['day_of_week'] = self.merged_df['order_dow'].map(dow_map)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Orders by Day of Week',
                    'Orders by Hour of Day',
                    'Department Preferences by Day',
                    'Order Volume Heatmap'
                )
            )
            
            # Orders by day of week
            daily_orders = self.merged_df.groupby('day_of_week', observed=True)['order_id'].count()
            ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_orders = daily_orders.reindex(ordered_days)
            
            fig.add_trace(
                go.Bar(x=daily_orders.index, y=daily_orders.values, name='Day of Week'),
                row=1, col=1
            )
            
            # Orders by hour
            hourly_orders = self.merged_df.groupby('order_hour_of_day', observed=True)['order_id'].count()
            fig.add_trace(
                go.Bar(x=hourly_orders.index, y=hourly_orders.values, name='Hour of Day'),
                row=1, col=2
            )
            
            # Department preferences by day
            dept_by_day = self.merged_df.groupby(['department', 'day_of_week'], observed=True)['product_id'].count().unstack()
            dept_by_day = dept_by_day.reindex(columns=ordered_days)
            for dept in dept_by_day.index:
                fig.add_trace(
                    go.Bar(x=dept_by_day.columns, y=dept_by_day.loc[dept], name=dept),
                    row=2, col=1
                )
            
            # Order volume heatmap
            heatmap_data = self.merged_df.groupby(['day_of_week', 'order_hour_of_day'], observed=True)['order_id'].count().unstack()
            heatmap_data = heatmap_data.reindex(index=ordered_days)
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='YlOrRd'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=1000,
                width=1200,
                title_text='Seasonal Shopping Patterns',
                showlegend=True,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating seasonal trends visualization: {str(e)}")
            return go.Figure()
    
    def _create_product_associations_viz(self) -> go.Figure:
        """Create visualization for product associations story."""
        try:
            # Get association rules
            rules_df = self._get_association_rules(min_support=0.01, min_confidence=0.1)
            dept_rules_df = self._get_department_associations()
            
            if len(rules_df) == 0 and len(dept_rules_df) == 0:
                return go.Figure()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Top Product Associations by Lift',
                    'Association Rule Network',
                    'Department Association Heatmap',
                    'Support vs Confidence'
                )
            )
            
            # Top associations by lift
            top_rules = rules_df.nlargest(10, 'lift')
            fig.add_trace(
                go.Bar(
                    x=[f"{', '.join(map(str, r['antecedents']))} → {', '.join(map(str, r['consequents']))}" 
                       for _, r in top_rules.iterrows()],
                    y=top_rules['lift'],
                    name='Lift Score'
                ),
                row=1, col=1
            )
            
            # Support vs Confidence scatter
            fig.add_trace(
                go.Scatter(
                    x=rules_df['support'],
                    y=rules_df['confidence'],
                    mode='markers',
                    marker=dict(
                        size=rules_df['lift'] * 10,
                        color=rules_df['lift'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"Lift: {l:.2f}" for l in rules_df['lift']],
                    name='Rules'
                ),
                row=2, col=2
            )
            
            # Department association heatmap
            if len(dept_rules_df) > 0:
                dept_pivot = dept_rules_df.pivot(
                    index='antecedent_dept',
                    columns='consequent_dept',
                    values='lift'
                ).fillna(0)
                
                fig.add_trace(
                    go.Heatmap(
                        z=dept_pivot.values,
                        x=dept_pivot.columns,
                        y=dept_pivot.index,
                        colorscale='YlOrRd',
                        name='Department Associations'
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=1000,
                width=1200,
                title_text='Product Association Analysis',
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text='Association Rules', row=1, col=1)
            fig.update_yaxes(title_text='Lift Score', row=1, col=1)
            fig.update_xaxes(title_text='Support', row=2, col=2)
            fig.update_yaxes(title_text='Confidence', row=2, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating product associations visualization: {str(e)}")
            return go.Figure() 