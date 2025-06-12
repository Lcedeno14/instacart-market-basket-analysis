from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Basic Analysis Tab Layout and Callbacks

def get_basic_analysis_tab_layout(departments_dropdown_df, DAYS_OF_WEEK):
    return dcc.Tab(label='Basic Analysis', value='tab-0', children=[
        html.Div([
            html.Div([
                html.Label('Select Department:'),
                dcc.Dropdown(
                    id='department-dropdown',
                    options=[{'label': dept, 'value': dept} for dept in departments_dropdown_df['department'].unique()],
                    value=departments_dropdown_df['department'].iloc[0],
                    style={'width': '100%'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
            html.Div([
                html.Label('Minimum Product Count:'),
                dcc.Slider(
                    id='product-count-slider',
                    min=1,
                    max=100,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(0, 101, 10)},
                )
            ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
            html.Div([
                html.Label('Day of Week:'),
                dcc.Dropdown(
                    id='day-dropdown',
                    options=[{'label': day, 'value': day} for day in DAYS_OF_WEEK],
                    value='All Time',
                    style={'width': '100%'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'})
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        html.Div([
            html.Div([
                dcc.Graph(id='top-products-chart')
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='department-distribution-chart')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        html.Div([
            dcc.Graph(id='orders-heatmap')
        ], style={'width': '100%', 'display': 'inline-block', 'marginTop': '40px'}),
    ])

def register_basic_analysis_callbacks(app, merged_df, DOW_MAP, HOUR_LABELS, logger):
    @app.callback(
        Output('top-products-chart', 'figure'),
        [Input('department-dropdown', 'value'),
         Input('product-count-slider', 'value'),
         Input('day-dropdown', 'value')]
    )
    def update_top_products_chart(selected_department, min_count, selected_day):
        try:
            filtered_df = merged_df.copy()
            if selected_department != 'All Departments':
                filtered_df = filtered_df[filtered_df['department'] == selected_department]
            if selected_day != 'All Time':
                filtered_df = filtered_df[filtered_df['day_of_week'] == selected_day]
            product_counts = filtered_df['product_name'].value_counts()
            product_counts = product_counts[product_counts >= min_count]
            top_products = product_counts.head(20)
            if len(top_products) == 0:
                return px.bar(title="No products found with the selected criteria")
            fig = px.bar(
                x=top_products.values,
                y=top_products.index,
                orientation='h',
                title=f'Top Products by Order Count (Min: {min_count})',
                labels={'x': 'Order Count', 'y': 'Product Name'}
            )
            fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis={'categoryorder': 'total ascending'}
            )
            return fig
        except Exception as e:
            logger.error(f"Error in update_top_products_chart: {str(e)}")
            return px.bar(title=f"Error: {str(e)}")

    @app.callback(
        Output('department-distribution-chart', 'figure'),
        [Input('day-dropdown', 'value')]
    )
    def update_department_distribution_chart(selected_day):
        try:
            filtered_df = merged_df.copy()
            if selected_day != 'All Time':
                filtered_df = filtered_df[filtered_df['day_of_week'] == selected_day]
            dept_counts = filtered_df['department'].value_counts()
            if len(dept_counts) == 0:
                return px.pie(title="No data found for the selected criteria")
            fig = px.pie(
                values=dept_counts.values,
                names=dept_counts.index,
                title=f'Department Distribution {f"({selected_day})" if selected_day != "All Time" else ""}'
            )
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return fig
        except Exception as e:
            logger.error(f"Error in update_department_distribution_chart: {str(e)}")
            return px.pie(title=f"Error: {str(e)}")

    @app.callback(
        Output('orders-heatmap', 'figure'),
        [Input('department-dropdown', 'value')]
    )
    def update_orders_heatmap(selected_department):
        try:
            filtered_df = merged_df.copy()
            if selected_department != 'All Departments':
                filtered_df = filtered_df[filtered_df['department'] == selected_department]
            heatmap_data = filtered_df.groupby(['order_dow', 'order_hour_of_day']).size().unstack(fill_value=0)
            day_order = [1, 2, 3, 4, 5, 6, 0]  # Monday to Sunday
            heatmap_data = heatmap_data.reindex(day_order)
            fig = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=[DOW_MAP[day] for day in heatmap_data.index],
                title=f'Order Heatmap by Day and Hour {f"({selected_department})" if selected_department != "All Departments" else ""}',
                labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Number of Orders'},
                aspect="auto"
            )
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig.update_xaxes(
                ticktext=HOUR_LABELS,
                tickvals=list(range(24))
            )
            return fig
        except Exception as e:
            logger.error(f"Error in update_orders_heatmap: {str(e)}")
            return px.imshow(title=f"Error: {str(e)}") 