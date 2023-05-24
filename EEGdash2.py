import dash
from dash import dcc, html, Input, Output, State
from utils.dataset import EEGDataset
from pathlib import Path
import plotly.graph_objs as go 
import numpy as np
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats


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

# Plot EEG raw signals
def build_fig_trial(sample):

    n_channels = sample['eeg'].shape[0]
    times = np.arange(sample['eeg'].shape[1])

    step = 1. / n_channels
    yaxis = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)
    line = dict(color='black', width=1)

    # Create objects for layout and traces
    layout = go.Layout(yaxis = go.layout.YAxis(yaxis))
    traces = [go.Scatter(x=times, y=sample['eeg'].T[:,0], line=line)]

    # loop over the channels
    for ii in range(1, n_channels):
        yaxis.update(domain=[1 - (ii+1)*step, 1 - ii*step])
        layout.update({'yaxis%d' % (ii+1): go.layout.YAxis(yaxis), 'showlegend':False})
        traces.append(go.Scatter(x=times, y=sample['eeg'].T[:,ii], yaxis='y%d' % (ii+1), line=line))


    layout.update(autosize=False, width=500, height=2000)
    fig = go.Figure(data=traces, layout=layout)

    return fig

# plot trial average
def build_fig_trial_average(faces, scram):
   
    avg_faces = np.mean(faces, axis=0)
    avg_scram = np.mean(scram, axis=0)

    line = dict(color='black', width=0.1)

    fig = make_subplots(rows=2, cols=1)
    fig.update_layout(showlegend=False)

    for row in avg_faces:
        fig.add_trace(go.Scatter(y=row, line=line), row=1, col=1)

    for row in avg_scram:
        fig.add_trace(go.Scatter(y=row, line=line), row=2, col=1)

    return fig


# plot time average
def build_fig_time_average(faces, scram):

    avg_faces = np.mean(faces, axis=2).flatten()
    avg_scram = np.mean(scram, axis=2).flatten()

    hist_data = [avg_faces, avg_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=1e-9, show_rug=False)

    range = [
        min(np.min(avg_faces), np.min(avg_scram)),
        max(np.max(avg_faces), np.max(avg_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig

# plot time skewness
def build_fig_time_skewness(faces, scram):
    skew_faces = stats.skew(faces, axis=-1).flatten()
    skew_scram = stats.skew(scram, axis=-1).flatten()

    hist_data = [skew_faces, skew_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2, show_rug=False)
    range = [
        min(np.min(skew_faces), np.min(skew_scram)),
        max(np.max(skew_faces), np.max(skew_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig

# plot time kurtosis
def build_fig_time_kurtosis(faces, scram):
    kurt_faces = stats.skew(faces, axis=-1).flatten()
    kurt_scram = stats.skew(scram, axis=-1).flatten()

    hist_data = [kurt_faces, kurt_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=.1, show_rug=False)
    range = [
        min(np.min(kurt_faces), np.min(kurt_scram)),
        max(np.max(kurt_faces), np.max(kurt_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04',
    'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

eeg_dir = Path('./EEGDataset')
DATA = EEGDataset(eeg_dir, SUBJECTS)
FILES = DATA.files

FEATURES = [
    'trial average',
    'time average',
    'time skewness',
    'time kurtosis',
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
            style={'width': '200px', 'font-size': '20px', 'margin-bottom': '10px'}
        ),
        dcc.Dropdown(
            id='dd_feature',
            options=[{'label':'0', 'value':0}],
            value=FEATURES[0],
            placeholder='Select a feature',
            style={'width': '200px', 'font-size': '20px', 'margin-bottom': '10px'},
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

    return None

if __name__ == '__main__':
    app.run_server(debug=True)
