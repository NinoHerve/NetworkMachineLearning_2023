import dash
from dash import dcc, html, Output, Input

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20',
]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Tabs(id='tabs', value='data', children=[
            dcc.Tab(label='Data', value='data'),
            dcc.Tab(label='Model', value='model'),
            dcc.Tab(label='Training', value='train'),
        ]),
        html.Div(id='tab-content'),
    ]),

    html.Div([
        html.Button('Train', id='train'),
        dcc.Graph(id='graph')
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'})
])

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
)
def update_tab(tab):
    if tab == 'data':
        children = [
            html.Label('Subjects:'),
            dcc.Checklist(SUBJECTS, ['sub-01'], inline=True, style={'margin-bottom':'10px'}),
            html.Label('Samples:'),
            dcc.RadioItems(['All samples', 'two samples', 'one sample'], 'All data'),
        ]
        return children
    
    elif tab == 'model':
        children = [
            html.Label('Model:'),
            dcc.Dropdown(
                options=[{'label':'CNN', 'value':'CNN'},
                         {'label':'GNN', 'value':'GNN'}],
                value='CNN', 
                style={'margin-bottom':'10px'}
            )
        ]
        return children
    
    else:
        children = [
            html.Label('Loss function:'),
            dcc.Dropdown(
                options=[{'label':'Binary cross entropy', 'value':'bce'}],
                value='bce',
                style={'margin-bottom':'10px'},
            ),
            html.Label('Optimizer:'),
            dcc.Dropdown(
                options=[
                    {'label':'Stochastic Gradient Descent', 'value':'SGD'},
                    {'label':'Adam', 'value':'adam'},
                ]
            ),
            html.Label('Learning rate:'),
        ]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)
