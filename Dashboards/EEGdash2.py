import sys
sys.path.append('../')

import dash
from dash import dcc, html, Input, Output, State
from utils.dataset import EEGDataset
from pathlib import Path
import plotly.graph_objs as go 
import numpy as np
from Dashboards.dash2plots import *



# Get index in dataset of a file
def get_file_index(file_name):
    for i, fn in enumerate(FILES):
        if file_name == str(fn):
            return i 
    
    return None

# get subject data
def get_subject_data(subject_files):

    eeg_faces = []
    eeg_scram = []

    for sub in subject_files:
        idx = get_file_index(sub)
        sample = DATA[idx]

        if sample['label']==1:
            eeg_faces.append(sample['eeg'])
        else:
            eeg_scram.append(sample['eeg'])

        
    eeg_faces = np.array(eeg_faces)
    eeg_scram = np.array(eeg_scram)

    return eeg_faces, eeg_scram


SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

eeg_dir = Path('../EEGDataset')
DATA = EEGDataset(eeg_dir, SUBJECTS)
FILES = DATA.files

FEATURES = [
    'trial average',
    'time average',
    'time skewness',
    'time kurtosis',
    'time ptp',
    'time variance',
    'time root mean square',
    'time maximum value',
    'time minimum value',
    'time maximum position',
    'time minimum position',
    'time absolute difference',
]

app = dash.Dash(__name__)

title = html.Div(
    html.H1('EEG exploration')
)

# Upper component with three dropdowns
upper_component = html.Div(
    [
        dcc.Dropdown(
            id='dd_subject',
            options=[{'label':sub, 'value':sub} for sub in SUBJECTS],
            value=SUBJECTS[0],
            placeholder='Select a subject',
            style={'width': '200px', 'font-size': '18px', 'margin-bottom': '10px'}
        ),
        dcc.Dropdown(
            id='dd_trial',
            options=[{'label':'0', 'value':0}],
            placeholder='Select a trial',
            style={'width': '200px', 'font-size': '18px', 'margin-bottom': '10px'}
        ),
        dcc.Dropdown(
            id='dd_feature',
            options=[{'label':'0', 'value':0}],
            value=FEATURES[0],
            placeholder='Select a feature',
            style={'width': '200px', 'font-size': '16px', 'margin-bottom': '10px'},
        )
    ]
)

# Lower component with two side-by-side images
lower_component = html.Div(
    [
        html.Div(
            dcc.Graph(id='fig_trial', style={'width': '50%'}),
            style={'display': 'inline-block', 'width': '50%', 'overflowY':'scroll', 'height':500}
        ),
        html.Div(
            dcc.Graph(id='fig_feature', style={'width': '50%'}),
            style={'display': 'inline-block', 'width': '50%'}
        ),
    ]
)

memory_component = html.Div(
    dcc.Store(id='memory_subject'),
)

app.layout = html.Div([title, upper_component, lower_component, memory_component])

# Call backs

@app.callback(
        Output('memory_subject', 'data'),
        Output('dd_trial', 'options'),
        Output('dd_trial', 'value'),
        Output('dd_feature', 'options'),
        Output('dd_feature', 'value'),
        Input('dd_subject', 'value'),
)
def update_subject(sub):

    subject_files = [str(fn) for fn in FILES if sub in fn.parts[-1]]

    trials = range(len(subject_files))
    optionsT = [{'label':i+1, 'value':i+1} for i in trials]

    optionsF = [{'label':feat, 'value':feat} for feat in FEATURES]

    return subject_files, optionsT, optionsT[0]['value'], FEATURES, optionsF[0]['value']


@app.callback(
    Output('fig_trial', 'figure'),
    Input('dd_trial', 'value'),
    State('memory_subject', 'data'),
)
def update_trial(trial, subject_files):

    file = subject_files[trial]
    idx  = get_file_index(file)

    sample = DATA[idx]

    fig = build_fig_trial(sample)

    return fig


@app.callback(
    Output('fig_feature', 'figure'),
    Input('dd_feature', 'value'),
    State('memory_subject', 'data'),
)
def update_feature(feature, subject_files):
    
    faces, scram = get_subject_data(subject_files)

    if feature=='trial average':
        fig = build_fig_trial_average(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig
    
    if feature=='time average':
        fig = build_fig_time_average(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig
    
    if feature=='time skewness':
        fig = build_fig_time_skewness(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig

    if feature=='time kurtosis':
        fig = build_fig_time_kurtosis(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig
    
    if feature=='time ptp':
        fig = build_fig_time_ptp(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig 
    
    if feature=='time variance':
        fig = build_fig_time_var(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig
    
    if feature=='time root mean square':
        fig = build_fig_time_rms(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig
    
    if feature=='time maximum value':
        fig = build_fig_time_max(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig 
    
    if feature=='time minimum value':
        fig = build_fig_time_min(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig 
    
    if feature=='time maximum position':
        fig = build_fig_time_max_arg(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig
    
    if feature=='time minimum position':
        fig = build_fig_time_min_arg(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig 
    
    if feature=='time absolute difference':
        fig = build_fig_time_abs(faces, scram)
        fig.update_layout(autosize=False, width=600, height=500)
        return fig
    
    return None

if __name__ == '__main__':
    app.run_server(debug=True)
