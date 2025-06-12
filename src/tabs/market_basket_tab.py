from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def get_market_basket_tab_layout():
    return dcc.Tab(label='Market Basket Analysis', value='tab-1', children=[
        html.Div([
            # Header with explanation
            html.Div([
                html.H2('Market Basket Analysis - Product Association Insights', 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P('Discover which products customers buy together and identify cross-selling opportunities', 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
            ]),
            
            # Controls for market basket analysis
            html.Div([
                html.Div([
                    html.Label('Minimum Support (How often items appear together):', 
                             style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Slider(
                        id='support-slider',
                        min=0.001,
                        max=0.02,
                        step=0.001,
                        value=0.001,
                        marks={
                            0.001: '0.1%',
                            0.002: '0.2%',
                            0.005: '0.5%',
                            0.01: '1%',
                            0.02: '2%'
                        },
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    html.Div(id='support-explanation', 
                            style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label('Minimum Confidence (How reliable the association is):', 
                             style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Slider(
                        id='confidence-slider',
                        min=0.1,
                        max=0.5,
                        step=0.1,
                        value=0.1,
                        marks={
                            0.1: '10%',
                            0.2: '20%',
                            0.3: '30%',
                            0.4: '40%',
                            0.5: '50%'
                        },
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    html.Div(id='confidence-explanation', 
                            style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Metrics Explanation
            html.Div([
                html.H4('📊 Understanding the Metrics', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div([
                    html.Div([
                        html.H5('Support', style={'color': '#3498db', 'marginBottom': '5px'}),
                        html.P('How frequently items appear together in transactions. Higher support = more common combinations.', 
                              style={'fontSize': '14px', 'color': '#7f8c8d'})
                    ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),
                    html.Div([
                        html.H5('Confidence', style={'color': '#e74c3c', 'marginBottom': '5px'}),
                        html.P('How reliable the association is. If A is bought, how likely is B also bought?', 
                              style={'fontSize': '14px', 'color': '#7f8c8d'})
                    ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),
                    html.Div([
                        html.H5('Lift', style={'color': '#27ae60', 'marginBottom': '5px'}),
                        html.P('Strength of association. Lift > 1 = positive association, Lift > 2 = strong association.', 
                              style={'fontSize': '14px', 'color': '#7f8c8d'})
                    ], style={'width': '32%', 'display': 'inline-block', 'marginLeft': '1%'})
                ])
            ], style={'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Key Metrics Summary
            html.Div([
                html.H3('Key Insights Summary', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div(id='market-basket-summary', 
                        style={'padding': '15px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px'})
            ], style={'marginBottom': '20px'}),
            
            # Main Visualizations
            html.Div([
                # Top Product Associations
                html.Div([
                    html.H4('Top Product Associations by Business Impact', 
                           style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    dcc.Graph(id='top-associations-chart')
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Association Network
                html.Div([
                    html.H4('Product Association Network', 
                           style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    dcc.Graph(id='association-network-chart')
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'marginBottom': '20px'}),
            
            # Department-Level Insights
            html.Div([
                html.H4('Department-Level Associations', 
                       style={'color': '#2c3e50', 'marginBottom': '10px'}),
                dcc.Graph(id='department-associations-chart')
            ], style={'marginBottom': '20px'}),
            
            # Business Recommendations
            html.Div([
                html.H3('Business Recommendations', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div(id='business-recommendations', 
                        style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px'})
            ], style={'marginBottom': '20px'}),
            
            # Actionable Insights
            html.Div([
                html.H3('Actionable Insights', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div([
                    html.Div([
                        html.H4('🎯 Cross-Selling Opportunities', style={'color': '#27ae60', 'marginBottom': '10px'}),
                        html.P('Products that customers frequently buy together - perfect for bundling and recommendations', 
                              style={'color': '#7f8c8d', 'marginBottom': '15px'}),
                        html.Div(id='cross-selling-insights', style={'fontSize': '14px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.H4('📊 Inventory Planning', style={'color': '#e74c3c', 'marginBottom': '10px'}),
                        html.P('Use association patterns to optimize inventory levels and reduce stockouts', 
                              style={'color': '#7f8c8d', 'marginBottom': '15px'}),
                        html.Div(id='inventory-insights', style={'fontSize': '14px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
                ])
            ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
            
            # Detailed Rules Table
            html.Div([
                html.H4('Detailed Association Rules', 
                       style={'color': '#2c3e50', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Filter by Minimum Lift:', style={'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='lift-filter',
                        options=[
                            {'label': 'All Rules', 'value': 0},
                            {'label': 'Lift > 1.5', 'value': 1.5},
                            {'label': 'Lift > 2.0', 'value': 2.0},
                            {'label': 'Lift > 3.0', 'value': 3.0}
                        ],
                        value=0,
                        style={'width': '200px', 'display': 'inline-block'}
                    )
                ], style={'marginBottom': '15px'}),
                html.Div(id='association-rules-table', 
                        style={'maxHeight': '400px', 'overflowY': 'auto'})
            ])
        ], style={'padding': '20px'})
    ])

def register_market_basket_callbacks(app, merged_df, logger):
    # Placeholder callbacks for market basket analysis
    # These would need to be implemented based on your specific market basket analysis logic
    
    @app.callback(
        Output('top-associations-chart', 'figure'),
        [Input('support-slider', 'value'),
         Input('confidence-slider', 'value')]
    )
    def update_top_associations_chart(support, confidence):
        try:
            # Placeholder - implement your market basket analysis logic here
            fig = px.bar(
                x=['Product A', 'Product B', 'Product C'],
                y=[0.8, 0.6, 0.4],
                title='Top Product Associations',
                labels={'x': 'Product Pair', 'y': 'Lift Score'}
            )
            return fig
        except Exception as e:
            logger.error(f"Error in update_top_associations_chart: {str(e)}")
            return px.bar(title=f"Error: {str(e)}")

    @app.callback(
        Output('association-network-chart', 'figure'),
        [Input('support-slider', 'value'),
         Input('confidence-slider', 'value')]
    )
    def update_association_network_chart(support, confidence):
        try:
            # Placeholder - implement network visualization logic here
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1, 2],
                y=[0, 1, 0],
                mode='markers+text',
                text=['Product A', 'Product B', 'Product C'],
                textposition='top center'
            ))
            fig.update_layout(title='Product Association Network')
            return fig
        except Exception as e:
            logger.error(f"Error in update_association_network_chart: {str(e)}")
            return go.Figure()

    @app.callback(
        Output('department-associations-chart', 'figure'),
        [Input('support-slider', 'value'),
         Input('confidence-slider', 'value')]
    )
    def update_department_associations_chart(support, confidence):
        try:
            # Placeholder - implement department-level analysis here
            dept_counts = merged_df['department'].value_counts()
            fig = px.pie(
                values=dept_counts.values,
                names=dept_counts.index,
                title='Department Distribution'
            )
            return fig
        except Exception as e:
            logger.error(f"Error in update_department_associations_chart: {str(e)}")
            return px.pie(title=f"Error: {str(e)}")

    @app.callback(
        Output('market-basket-summary', 'children'),
        [Input('support-slider', 'value'),
         Input('confidence-slider', 'value')]
    )
    def update_market_basket_summary(support, confidence):
        try:
            return html.Div([
                html.P(f"Analysis with Support ≥ {support:.3f} and Confidence ≥ {confidence:.1f}"),
                html.P("Key insights will appear here based on your market basket analysis.")
            ])
        except Exception as e:
            logger.error(f"Error in update_market_basket_summary: {str(e)}")
            return html.P(f"Error: {str(e)}")

    @app.callback(
        Output('business-recommendations', 'children'),
        [Input('support-slider', 'value'),
         Input('confidence-slider', 'value')]
    )
    def update_business_recommendations(support, confidence):
        try:
            return html.Div([
                html.H4("Recommendations based on current analysis:"),
                html.Ul([
                    html.Li("Consider bundling frequently co-purchased products"),
                    html.Li("Place associated products near each other in the store"),
                    html.Li("Create targeted marketing campaigns for product pairs")
                ])
            ])
        except Exception as e:
            logger.error(f"Error in update_business_recommendations: {str(e)}")
            return html.P(f"Error: {str(e)}")

    @app.callback(
        Output('cross-selling-insights', 'children'),
        [Input('support-slider', 'value'),
         Input('confidence-slider', 'value')]
    )
    def update_cross_selling_insights(support, confidence):
        try:
            return html.Ul([
                html.Li("Product A + Product B: 80% confidence"),
                html.Li("Product C + Product D: 65% confidence"),
                html.Li("Product E + Product F: 45% confidence")
            ])
        except Exception as e:
            logger.error(f"Error in update_cross_selling_insights: {str(e)}")
            return html.P(f"Error: {str(e)}")

    @app.callback(
        Output('inventory-insights', 'children'),
        [Input('support-slider', 'value'),
         Input('confidence-slider', 'value')]
    )
    def update_inventory_insights(support, confidence):
        try:
            return html.Ul([
                html.Li("Stock Product A and Product B together"),
                html.Li("Monitor inventory levels for frequently paired items"),
                html.Li("Plan promotions around high-lift product combinations")
            ])
        except Exception as e:
            logger.error(f"Error in update_inventory_insights: {str(e)}")
            return html.P(f"Error: {str(e)}")

    @app.callback(
        Output('association-rules-table', 'children'),
        [Input('support-slider', 'value'),
         Input('confidence-slider', 'value'),
         Input('lift-filter', 'value')]
    )
    def update_association_rules_table(support, confidence, min_lift):
        try:
            # Placeholder table
            return html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Antecedent"),
                        html.Th("Consequent"),
                        html.Th("Support"),
                        html.Th("Confidence"),
                        html.Th("Lift")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Product A"),
                        html.Td("Product B"),
                        html.Td(f"{support:.3f}"),
                        html.Td(f"{confidence:.2f}"),
                        html.Td("2.5")
                    ])
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        except Exception as e:
            logger.error(f"Error in update_association_rules_table: {str(e)}")
            return html.P(f"Error: {str(e)}") 