from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import random
from src.utils.logging_config import get_logger

logger = get_logger('market_basket')

def get_market_basket_tab_layout():
    """Create the layout for the Market Basket Analysis tab."""
    return dcc.Tab(
        label='Market Basket Analysis',
        value='tab-market-basket',
        children=[
            html.Div([
                # Header
                html.H2('Product Recommendation Engine', 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                html.P('See how AI-powered product associations can boost your revenue', 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
                
                # Business Value Section
                html.Div([
                    html.H3('Business Impact', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.H4('Increase Average Order Value', style={'color': '#27ae60', 'marginBottom': '10px'}),
                            html.P('When customers see "Frequently bought together" recommendations, they add 2-3 more items on average, increasing AOV by 15-25%.', 
                                  style={'color': '#7f8c8d', 'marginBottom': '15px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4('Reduce Cart Abandonment', style={'color': '#e74c3c', 'marginBottom': '10px'}),
                            html.P('Personalized recommendations keep customers engaged and reduce cart abandonment by up to 30%.', 
                                  style={'color': '#7f8c8d', 'marginBottom': '15px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
                    ])
                ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                
                # Interactive Demo Section
                html.Div([
                    html.H3('Interactive Recommendation Demo', style={'color': '#2c3e50', 'marginBottom': '20px'}),
                    html.P('Click "Generate Random Product" to see how our AI finds product associations in real-time', 
                          style={'color': '#7f8c8d', 'marginBottom': '20px'}),
                    
                    # Random Product Generator
                    html.Div([
                        html.Button('Generate Random Product', id='generate-product-btn', 
                                  style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '15px 30px', 
                                         'border': 'none', 'borderRadius': '5px', 'fontSize': '16px', 'cursor': 'pointer'}),
                        html.Div(id='selected-product-display', 
                               style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                                      'borderRadius': '5px', 'textAlign': 'center'})
                    ], style={'textAlign': 'center', 'marginBottom': '30px'}),
                    
                    # Algorithm Comparison
                    html.Div([
                        html.H4('AI Algorithm Comparison', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                        html.P('Both algorithms find product associations, but they work differently:', 
                              style={'color': '#7f8c8d', 'marginBottom': '20px'}),
                        
                        # Two-column layout for algorithms
                        html.Div([
                            # FP-Growth Column
                            html.Div([
                                html.H5('FP-Growth Algorithm', style={'color': '#27ae60', 'marginBottom': '10px'}),
                                html.P('• Faster for large datasets', style={'fontSize': '14px', 'marginBottom': '5px'}),
                                html.P('• Memory efficient', style={'fontSize': '14px', 'marginBottom': '5px'}),
                                html.P('• Used by Amazon, Netflix', style={'fontSize': '14px', 'marginBottom': '15px'}),
                                html.Div(id='fpgrowth-results', 
                                       style={'padding': '15px', 'backgroundColor': '#e8f5e8', 'borderRadius': '5px', 'minHeight': '100px'})
                            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                            
                            # Apriori Column
                            html.Div([
                                html.H5('Apriori Algorithm', style={'color': '#e74c3c', 'marginBottom': '10px'}),
                                html.P('• Industry standard', style={'fontSize': '14px', 'marginBottom': '5px'}),
                                html.P('• Easy to understand', style={'fontSize': '14px', 'marginBottom': '5px'}),
                                html.P('• Used by Walmart, Target', style={'fontSize': '14px', 'marginBottom': '15px'}),
                                html.Div(id='apriori-results', 
                                       style={'padding': '15px', 'backgroundColor': '#fdeaea', 'borderRadius': '5px', 'minHeight': '100px'})
                            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
                        ])
                    ], style={'marginBottom': '30px'}),
                    
                    # Revenue Impact Calculator
                    html.Div([
                        html.H4('Revenue Impact Calculator', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                        html.P('See how these recommendations could impact your business:', 
                              style={'color': '#7f8c8d', 'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Div([
                                html.Label('Current Average Order Value ($):', style={'display': 'block', 'marginBottom': '5px'}),
                                dcc.Input(id='current-aov', type='number', value=50, 
                                        style={'width': '100%', 'padding': '8px', 'border': '1px solid #ddd', 'borderRadius': '3px'})
                            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
                            
                            html.Div([
                                html.Label('Monthly Orders:', style={'display': 'block', 'marginBottom': '5px'}),
                                dcc.Input(id='monthly-orders', type='number', value=10000, 
                                        style={'width': '100%', 'padding': '8px', 'border': '1px solid #ddd', 'borderRadius': '3px'})
                            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
                            
                            html.Div([
                                html.Label('Expected AOV Increase (%):', style={'display': 'block', 'marginBottom': '5px'}),
                                dcc.Input(id='aov-increase', type='number', value=20, 
                                        style={'width': '100%', 'padding': '8px', 'border': '1px solid #ddd', 'borderRadius': '3px'})
                            ], style={'width': '30%', 'display': 'inline-block'})
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div(id='revenue-impact', 
                               style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px', 'textAlign': 'center'})
                    ], style={'marginBottom': '30px'}),
                    
                    # Implementation Steps
                    html.Div([
                        html.H4('How to Implement This', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div([
                                html.H5('1. Data Collection', style={'color': '#3498db', 'marginBottom': '10px'}),
                                html.P('Track customer purchase history and cart contents', style={'fontSize': '14px'})
                            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}),
                            
                            html.Div([
                                html.H5('2. Algorithm Training', style={'color': '#3498db', 'marginBottom': '10px'}),
                                html.P('Run FP-Growth/Apriori on historical data', style={'fontSize': '14px'})
                            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}),
                            
                            html.Div([
                                html.H5('3. Real-time API', style={'color': '#3498db', 'marginBottom': '10px'}),
                                html.P('Query associations when customers browse', style={'fontSize': '14px'})
                            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}),
                            
                            html.Div([
                                html.H5('4. A/B Testing', style={'color': '#3498db', 'marginBottom': '10px'}),
                                html.P('Test different algorithms and measure impact', style={'fontSize': '14px'})
                            ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'})
                        ])
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                ])
            ], style={'padding': '20px'})
        ]
    )

def register_market_basket_callbacks(app, merged_df, logger):
    """Register callbacks for the market basket analysis tab."""
    
    @app.callback(
        [Output('selected-product-display', 'children'),
         Output('fpgrowth-results', 'children'),
         Output('apriori-results', 'children')],
        [Input('generate-product-btn', 'n_clicks')]
    )
    def generate_product_and_associations(n_clicks):
        if n_clicks is None:
            return "Click 'Generate Random Product' to start", "FP-Growth results will appear here", "Apriori results will appear here"
        
        try:
            # Get random product
            unique_products = merged_df[['product_id', 'product_name']].drop_duplicates()
            random_product = unique_products.sample(1).iloc[0]
            product_id = random_product['product_id']
            product_name = random_product['product_name']
            
            # Get associations using both algorithms
            fpgrowth_associations = get_fpgrowth_associations(merged_df, product_id, logger)
            apriori_associations = get_apriori_associations(merged_df, product_id, logger)
            
            # Format results
            selected_product_html = html.Div([
                html.H4(f"Selected Product: {product_name}", style={'color': '#2c3e50'}),
                html.P(f"Product ID: {product_id}", style={'color': '#7f8c8d'})
            ])
            
            fpgrowth_html = format_associations(fpgrowth_associations, "FP-Growth")
            apriori_html = format_associations(apriori_associations, "Apriori")
            
            return selected_product_html, fpgrowth_html, apriori_html
            
        except Exception as e:
            logger.error(f"Error in generate_product_and_associations: {str(e)}")
            return f"Error: {str(e)}", "Error loading FP-Growth results", "Error loading Apriori results"
    
    @app.callback(
        Output('revenue-impact', 'children'),
        [Input('current-aov', 'value'),
         Input('monthly-orders', 'value'),
         Input('aov-increase', 'value')]
    )
    def calculate_revenue_impact(current_aov, monthly_orders, aov_increase):
        if not all([current_aov, monthly_orders, aov_increase]):
            return "Enter values to see revenue impact"
        
        try:
            current_revenue = current_aov * monthly_orders
            new_aov = current_aov * (1 + aov_increase / 100)
            new_revenue = new_aov * monthly_orders
            revenue_increase = new_revenue - current_revenue
            percentage_increase = (revenue_increase / current_revenue) * 100
            
            return html.Div([
                html.H4('Projected Revenue Impact', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div([
                    html.Div([
                        html.H5('Current Monthly Revenue', style={'color': '#7f8c8d', 'marginBottom': '5px'}),
                        html.H3(f'${current_revenue:,.0f}', style={'color': '#2c3e50'})
                    ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.H5('New AOV', style={'color': '#7f8c8d', 'marginBottom': '5px'}),
                        html.H3(f'${new_aov:.2f}', style={'color': '#27ae60'})
                    ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.H5('Monthly Revenue Increase', style={'color': '#7f8c8d', 'marginBottom': '5px'}),
                        html.H3(f'${revenue_increase:,.0f}', style={'color': '#e74c3c'})
                    ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.H5('Percentage Increase', style={'color': '#7f8c8d', 'marginBottom': '5px'}),
                        html.H3(f'{percentage_increase:.1f}%', style={'color': '#9b59b6'})
                    ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error in calculate_revenue_impact: {str(e)}")
            return f"Error calculating revenue impact: {str(e)}"

def get_fpgrowth_associations(merged_df, product_id, logger):
    """Get product associations using FP-Growth algorithm."""
    try:
        # Create transaction data
        transactions = merged_df.groupby('order_id')['product_id'].apply(list).reset_index()
        transaction_list = transactions['product_id'].tolist()
        
        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(transaction_list).transform(transaction_list)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Run FP-Growth
        frequent_itemsets = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return []
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        
        # Filter rules for the selected product
        product_associations = []
        for _, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            if product_id in antecedents:
                for consequent in consequents:
                    if consequent != product_id:
                        product_associations.append({
                            'product_id': consequent,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support']
                        })
            elif product_id in consequents:
                for antecedent in antecedents:
                    if antecedent != product_id:
                        product_associations.append({
                            'product_id': antecedent,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support']
                        })
        
        # Sort by lift and get top 5
        product_associations.sort(key=lambda x: x['lift'], reverse=True)
        return product_associations[:5]
        
    except Exception as e:
        logger.error(f"Error in get_fpgrowth_associations: {str(e)}")
        return []

def get_apriori_associations(merged_df, product_id, logger):
    """Get product associations using Apriori algorithm."""
    try:
        # Create transaction data
        transactions = merged_df.groupby('order_id')['product_id'].apply(list).reset_index()
        transaction_list = transactions['product_id'].tolist()
        
        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(transaction_list).transform(transaction_list)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Run Apriori
        frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return []
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        
        # Filter rules for the selected product
        product_associations = []
        for _, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            if product_id in antecedents:
                for consequent in consequents:
                    if consequent != product_id:
                        product_associations.append({
                            'product_id': consequent,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support']
                        })
            elif product_id in consequents:
                for antecedent in antecedents:
                    if antecedent != product_id:
                        product_associations.append({
                            'product_id': antecedent,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support']
                        })
        
        # Sort by lift and get top 5
        product_associations.sort(key=lambda x: x['lift'], reverse=True)
        return product_associations[:5]
        
    except Exception as e:
        logger.error(f"Error in get_apriori_associations: {str(e)}")
        return []

def format_associations(associations, algorithm_name):
    """Format association results for display."""
    if not associations:
        return html.Div([
            html.H5(f"{algorithm_name} Results", style={'color': '#7f8c8d'}),
            html.P("No strong associations found", style={'color': '#7f8c8d', 'fontStyle': 'italic'})
        ])
    
    # Get product names
    product_names = {}
    for assoc in associations:
        product_id = assoc['product_id']
        # This would normally query a product lookup table
        product_names[product_id] = f"Product {product_id}"
    
    return html.Div([
        html.H5(f"{algorithm_name} Top Recommendations", style={'color': '#2c3e50', 'marginBottom': '10px'}),
        html.Div([
            html.Div([
                html.Strong(f"{product_names[assoc['product_id']]}", style={'color': '#2c3e50'}),
                html.Br(),
                html.Span(f"Confidence: {assoc['confidence']:.2%}", style={'fontSize': '12px', 'color': '#7f8c8d'}),
                html.Br(),
                html.Span(f"Lift: {assoc['lift']:.2f}", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], style={'marginBottom': '10px', 'padding': '8px', 'backgroundColor': 'white', 'borderRadius': '3px'})
            for assoc in associations
        ])
    ]) 