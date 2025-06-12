from dash import dcc, html
from dash.dependencies import Input, Output

def get_kpi_dashboard_tab_layout(kpi_dashboard):
    return dcc.Tab(label='Business KPIs', value='tab-5', children=[
        kpi_dashboard.create_dashboard_layout() if kpi_dashboard else html.Div("KPI Dashboard is not available. Please check the logs for details.")
    ])

# No callback registration needed for KPI Dashboard
# The KPIDashboard class handles its own callbacks internally 