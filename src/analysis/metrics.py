"""
Business metrics and KPI calculations for the Instacart analysis dashboard.
Includes customer lifetime value, A/B testing, and key performance indicators.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BusinessMetrics:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize BusinessMetrics with the main dataframe.
        
        Args:
            df: Merged dataframe containing orders, products, and customer data
        """
        self.df = df
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate required columns are present in the dataframe."""
        required_columns = ['user_id', 'order_id', 'product_id', 'price', 
                          'order_number', 'days_since_prior_order']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def calculate_customer_lifetime_value(self, 
                                       time_horizon: int = 365,
                                       discount_rate: float = 0.1) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (CLV) using historical data.
        
        Args:
            time_horizon: Number of days to project into the future
            discount_rate: Annual discount rate for future value
            
        Returns:
            DataFrame with CLV calculations per customer
        """
        try:
            # Calculate average order value per customer
            avg_order_value = self.df.groupby('user_id').agg({
                'order_id': 'count',
                'price': 'sum'
            }).reset_index()
            avg_order_value['avg_order_value'] = avg_order_value['price'] / avg_order_value['order_id']
            
            # Calculate purchase frequency
            purchase_freq = self.df.groupby('user_id')['days_since_prior_order'].mean()
            
            # Calculate customer lifespan
            customer_lifespan = self.df.groupby('user_id').agg({
                'days_since_prior_order': lambda x: x.sum() / (x.count() - 1) if x.count() > 1 else 0
            }).reset_index()
            customer_lifespan['lifespan'] = customer_lifespan['days_since_prior_order'].clip(upper=time_horizon)
            
            # Merge metrics
            clv_df = avg_order_value.merge(
                customer_lifespan[['user_id', 'lifespan']], 
                on='user_id'
            )
            
            # Calculate CLV
            clv_df['clv'] = (
                clv_df['avg_order_value'] * 
                (clv_df['lifespan'] / purchase_freq) * 
                (1 / (1 + discount_rate) ** (clv_df['lifespan'] / 365))
            )
            
            return clv_df[['user_id', 'clv', 'avg_order_value', 'lifespan']]
            
        except Exception as e:
            logger.error(f"Error calculating CLV: {str(e)}")
            raise

    def analyze_ab_test_results(self,
                              control_group: pd.DataFrame,
                              test_group: pd.DataFrame,
                              metric: str,
                              confidence_level: float = 0.95) -> Dict:
        """
        Analyze A/B test results with statistical significance.
        
        Args:
            control_group: DataFrame with control group data
            test_group: DataFrame with test group data
            metric: Column name to analyze
            confidence_level: Statistical confidence level
            
        Returns:
            Dictionary with test results and statistical significance
        """
        try:
            # Calculate basic statistics
            control_mean = control_group[metric].mean()
            test_mean = test_group[metric].mean()
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                control_group[metric],
                test_group[metric],
                equal_var=False
            )
            
            # Calculate effect size (Cohen's d)
            effect_size = (test_mean - control_mean) / np.sqrt(
                (control_group[metric].var() + test_group[metric].var()) / 2
            )
            
            # Determine significance
            is_significant = p_value < (1 - confidence_level)
            
            return {
                'control_mean': control_mean,
                'test_mean': test_mean,
                'difference': test_mean - control_mean,
                'p_value': p_value,
                'effect_size': effect_size,
                'is_significant': is_significant,
                'confidence_level': confidence_level,
                'recommendation': 'Reject null hypothesis' if is_significant else 'Fail to reject null hypothesis'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing A/B test: {str(e)}")
            raise

    def generate_kpi_dashboard(self) -> Dict[str, float]:
        """
        Calculate key performance indicators for the business.
        
        Returns:
            Dictionary with KPI values
        """
        try:
            # Calculate metrics
            total_revenue = self.df['price'].sum()
            total_orders = self.df['order_id'].nunique()
            total_customers = self.df['user_id'].nunique()
            
            # Average metrics
            avg_order_value = total_revenue / total_orders
            avg_items_per_order = self.df.groupby('order_id')['product_id'].count().mean()
            
            # Customer metrics
            repeat_customers = self.df.groupby('user_id')['order_id'].nunique()
            repeat_rate = (repeat_customers > 1).mean()
            
            # Time-based metrics
            avg_days_between_orders = self.df.groupby('user_id')['days_since_prior_order'].mean().mean()
            
            # Department metrics
            dept_revenue = self.df.groupby('department')['price'].sum()
            top_dept = dept_revenue.idxmax()
            top_dept_revenue = dept_revenue.max()
            
            return {
                'total_revenue': total_revenue,
                'total_orders': total_orders,
                'total_customers': total_customers,
                'avg_order_value': avg_order_value,
                'avg_items_per_order': avg_items_per_order,
                'repeat_customer_rate': repeat_rate,
                'avg_days_between_orders': avg_days_between_orders,
                'top_department': top_dept,
                'top_department_revenue': top_dept_revenue,
                'revenue_per_customer': total_revenue / total_customers,
                'orders_per_customer': total_orders / total_customers
            }
            
        except Exception as e:
            logger.error(f"Error generating KPI dashboard: {str(e)}")
            raise

    def calculate_cohort_metrics(self, 
                               cohort_column: str = 'order_dow',
                               metric: str = 'price') -> pd.DataFrame:
        """
        Calculate cohort analysis metrics.
        
        Args:
            cohort_column: Column to use for cohort definition
            metric: Metric to analyze
            
        Returns:
            DataFrame with cohort metrics
        """
        try:
            # Create cohort matrix
            cohort_data = self.df.groupby(['user_id', cohort_column])[metric].sum().reset_index()
            cohort_matrix = cohort_data.pivot(
                index='user_id',
                columns=cohort_column,
                values=metric
            )
            
            # Calculate cohort metrics
            cohort_metrics = pd.DataFrame({
                'cohort_size': cohort_matrix.count(),
                'avg_value': cohort_matrix.mean(),
                'retention_rate': cohort_matrix.count() / cohort_matrix.count().max(),
                'churn_rate': 1 - (cohort_matrix.count() / cohort_matrix.count().max())
            })
            
            return cohort_metrics
            
        except Exception as e:
            logger.error(f"Error calculating cohort metrics: {str(e)}")
            raise 