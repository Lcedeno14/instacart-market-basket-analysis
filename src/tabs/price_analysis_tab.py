from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from src.analysis.data_processor import DataProcessor
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_price_analysis_tab_layout(data_processor):
    """Get the layout for the price analysis tab."""
    if data_processor is None:
        return dcc.Tab(label='Price Analysis', value='tab-price-analysis', children=[
            html.Div("Price analysis is not available. Please check the logs for details.")
        ])
    
    return dcc.Tab(
        label='Price Analysis',
        value='tab-price-analysis',
        children=[
            html.Div([
                html.H2('Price Analysis'),
                html.P('Analyze pricing patterns and customer spending behavior.'),
                dcc.Tabs(id='price-analysis-tabs', value='tab-price-insights', children=[
                    dcc.Tab(label='Price Insights', value='tab-price-insights', children=[
                        html.Div(id='price-insights-content')
                    ]),
                    dcc.Tab(label='Price Elasticity', value='tab-price-elasticity', children=[
                        html.Div(id='price-elasticity-content')
                    ]),
                    dcc.Tab(label='Price Visualization', value='tab-price-visualization', children=[
                        html.Div(id='price-visualization-content')
                    ])
                ])
            ])
        ]
    )

def register_price_analysis_callbacks(app, data_processor, logger):
    """Register callbacks for the price analysis tab."""
    if not data_processor:
        return

    @app.callback(
        Output('price-insights-content', 'children'),
        Input('price-analysis-tabs', 'value')
    )
    def update_price_analysis_insights(active_tab):
        try:
            if active_tab == 'tab-price-insights':
                if hasattr(data_processor, 'merged_df') and not data_processor.merged_df.empty:
                    price_by_dept = data_processor.merged_df.groupby('department')['price'].mean().sort_values(ascending=False)
                    
                    insights = [
                        html.H3('Price Analysis Insights'),
                        html.Div([
                            html.H4('Average Price by Department'),
                            html.Ul([
                                html.Li(f"{dept}: ${price:.2f}") for dept, price in price_by_dept.head(10).items()
                            ]),
                            html.H4('Key Findings'),
                            html.Ul([
                                html.Li(f"Highest average price: {price_by_dept.index[0]} (${price_by_dept.iloc[0]:.2f})"),
                                html.Li(f"Lowest average price: {price_by_dept.index[-1]} (${price_by_dept.iloc[-1]:.2f})"),
                                html.Li(f"Price range: ${data_processor.merged_df['price'].min():.2f} - ${data_processor.merged_df['price'].max():.2f}"),
                                html.Li(f"Average product price: ${data_processor.merged_df['price'].mean():.2f}")
                            ])
                        ])
                    ]
                    return insights
                else:
                    return html.Div("No data available for price insights.")
            return html.Div()
        except Exception as e:
            logger.error(f"Error in update_price_analysis_insights: {str(e)}")
            return html.Div(f"Error loading price insights: {str(e)}")

    @app.callback(
        Output('price-elasticity-content', 'children'),
        Input('price-analysis-tabs', 'value')
    )
    def update_price_analysis_elasticity(active_tab):
        try:
            if active_tab == 'tab-price-elasticity':
                if not data_processor or not hasattr(data_processor, 'merged_df') or data_processor.merged_df.empty:
                    return html.Div("No data available for elasticity analysis.")
                
                df = data_processor.merged_df
                
                # Calculate price elasticity
                product_stats = df.groupby('product_id').agg({
                    'price': 'mean',
                    'order_id': 'count'
                }).reset_index()
                
                if len(product_stats) > 1:
                    # Simple elasticity calculation
                    price_change = product_stats['price'].std() / product_stats['price'].mean()
                    quantity_change = product_stats['order_id'].std() / product_stats['order_id'].mean()
                    
                    if price_change > 0:
                        elasticity = (quantity_change / price_change) * -1
                    else:
                        elasticity = 0
                    
                    elasticity_content = [
                        html.H3('Price Elasticity Analysis'),
                        html.Div([
                            html.H4('Elasticity Metrics'),
                            html.Ul([
                                html.Li(f"Price Elasticity: {elasticity:.3f}"),
                                html.Li(f"Price Volatility: {price_change:.3f}"),
                                html.Li(f"Demand Volatility: {quantity_change:.3f}")
                            ]),
                            html.H4('Interpretation'),
                            html.P([
                                "Price elasticity measures how demand changes with price changes. ",
                                f"Current elasticity of {elasticity:.3f} indicates ",
                                "inelastic demand" if abs(elasticity) < 0.5 else "elastic demand" if abs(elasticity) > 1.5 else "unitary elastic demand",
                                "."
                            ])
                        ])
                    ]
                    return elasticity_content
                else:
                    return html.Div("Insufficient data for elasticity calculation.")
            return html.Div()
        except Exception as e:
            logger.error(f"Error in update_price_analysis_elasticity: {str(e)}")
            return html.Div(f"Error loading elasticity analysis: {str(e)}")

    @app.callback(
        Output('price-visualization-content', 'children'),
        Input('price-analysis-tabs', 'value')
    )
    def update_price_analysis_visualization(active_tab):
        try:
            if active_tab == 'tab-price-visualization':
                if not data_processor or not hasattr(data_processor, 'merged_df') or data_processor.merged_df.empty:
                    return html.Div("No data available for visualization.")
                
                # Create price visualization
                fig = data_processor.create_visualization()
                return dcc.Graph(figure=fig)
            return html.Div()
        except Exception as e:
            logger.error(f"Error in update_price_analysis_visualization: {str(e)}")
            return html.Div(f"Error loading visualization: {str(e)}") 