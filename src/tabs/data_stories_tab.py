from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

def get_data_stories_tab_layout(storyteller):
    return dcc.Tab(label='Data Stories', value='tab-4', children=[
        html.Div([
            html.Div([
                html.H2('Customer Journey Insights', 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                dcc.Graph(figure=storyteller.create_story_visualization('customer_journey') if storyteller else go.Figure()),
                html.Div([
                    html.Div([
                        html.H3(section['title'], 
                               style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.Ul([
                            html.Li(insight, style={'marginBottom': '5px'})
                            for insight in section['insights']
                        ])
                    ], style={'marginBottom': '20px'})
                    for section in (storyteller.generate_customer_journey_story()['sections'] if storyteller else [])
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
            ], style={'marginBottom': '40px'}) if storyteller else html.Div("Customer journey analysis is not available. Please check the logs for details."),
            
            html.Div([
                html.H2('Seasonal Trends Analysis', 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                dcc.Graph(figure=storyteller.create_story_visualization('seasonal_trends') if storyteller else go.Figure()),
                html.Div([
                    html.Div([
                        html.H3(section['title'], 
                               style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.Ul([
                            html.Li(insight, style={'marginBottom': '5px'})
                            for insight in section['insights']
                        ])
                    ], style={'marginBottom': '20px'})
                    for section in (storyteller.generate_seasonal_trends_story()['sections'] if storyteller else [])
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
            ], style={'marginBottom': '40px'})
        ], style={'padding': '20px'})
    ]) if storyteller else None

# No callback registration needed for Data Stories
# The DataStorytelling class provides static visualizations and content 