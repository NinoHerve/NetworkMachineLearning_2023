import dash
from dash import dcc, html, Input, Output, State
import matplotlib.pyplot as plt
import io 
import base64 
from pathlib import Path 
import numpy as np
from utils.dataset import EEGDataset 
from utils.readers import read_epochs
import mne
from utils.transforms import Compose

# Convert figures to base64-encoded strings
def fig_to_base64(fig):
    buf = io.BytesIO()  # in-memory files
    plt.figure(fig)
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode('utf8') # encode to html elements
    buf.close()

    return f'data:image/png;base64,{data}'


# Get channel positions of each electrode
def get_channel_positions(info):
    channels = info['chs']
    pos = []
    for ch in channels:
        p = ch['loc']
        pos.append([p[0], p[1]])
    pos = np.array(pos)

    return pos

# Get index in dataset of a file
def get_file_index(file_name):
    for i, fn in enumerate(FILES):
        if file_name == str(fn):
            return i 
    
    return None

bids_dir = Path('/home/admin/work/data/ds003505-download')
eeg_dir  = Path('./EEGDataset')
task     = 'faces'

SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06',
            'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11']

DATA  = EEGDataset(eeg_dir, SUBJECTS)
FILES = DATA.files
scaler = DATA.standard_scaler()
DATA.transfrom = Compose([scaler])

MONTAGES = {}
for sub in SUBJECTS:
    epochs = read_epochs(bids_dir, sub, task)
    pos = get_channel_positions(epochs.info) 
    MONTAGES[sub] = pos 

N = 50
LENGTH = 625

app = dash.Dash(__name__)

# App layout
# ----------

app.layout = html.Div([
    html.H1("EEG visualization"),
    dcc.Store(id='memory_faces'),
    dcc.Store(id='memory_scrambled'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='dp_subjects',
                options=[{'label': sub, 'value': sub} for sub in SUBJECTS],
                placeholder="Select an option",
                style={'width': '200px', 'margin': '0 auto'}
            )
        ], style={'text-align': 'center'}),
        html.Div([
            dcc.Dropdown(
                options=[
                    {'label': 'Option A', 'value': 'optionA'},
                    {'label': 'Option B', 'value': 'optionB'},
                    {'label': 'Option C', 'value': 'optionC'}
                ],
                placeholder="Select an option",
                style={'width': '200px', 'margin': '0 auto'}
            )
        ], style={'text-align': 'center'})
    ], style={'display': 'flex', 'justify-content': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='dp_faces',
                options=[
                    {'label': 'Option X', 'value': 'optionX'},
                    {'label': 'Option Y', 'value': 'optionY'},
                    {'label': 'Option Z', 'value': 'optionZ'}
                ],
                placeholder="Select an option",
                style={'width': '200px', 'margin': '0 auto'}
            ),
            html.Img(id='img_faces', src="image1.jpg", style={'width': '300px', 'margin-top': '10px'}),
            dcc.Slider(
                id='slider_faces',
                min=0,
                max=N-1,
                value=0,
                marks=None,
                step=1,
            ),
        ], style={'text-align': 'center'}),
        html.Div([
            dcc.Dropdown(
                id='dp_scrambled',
                options=[
                    {'label': 'Option P', 'value': 'optionP'},
                    {'label': 'Option Q', 'value': 'optionQ'},
                    {'label': 'Option R', 'value': 'optionR'}
                ],
                placeholder="Select an option",
                style={'width': '200px', 'margin': '0 auto'}
            ),
            html.Img(id='img_scrambled', src="image2.jpg", style={'width': '300px', 'margin-top': '10px'}),
            dcc.Slider(
                id='slider_scrambled',
                min=0,
                max=N-1,
                value=0,
                marks=None,
                step=1,
            ),
        ], style={'text-align': 'center'})
    ], style={'display': 'flex', 'justify-content': 'center'})
])


# Callbacks
# ---------

@app.callback(
        Output(component_id='dp_faces', component_property='options'),
        Output(component_id='dp_scrambled', component_property='options'),
        Input(component_id='dp_subjects', component_property='value'),
)
def change_subject(sub):

    if sub is None:
        sub = SUBJECTS[0]

    files = [fn for fn in FILES if sub in fn.parts[-1]]

    files_faces = [str(fn) for fn in files if fn.parts[-2]=='faces']
    files_scram = [str(fn) for fn in files if fn.parts[-2]=='scrambled']

    options_faces = [{'label':i, 'value':fn} for i,fn in enumerate(files_faces)]
    options_scram = [{'label':i, 'value':fn} for i,fn in enumerate(files_scram)]

    return options_faces, options_scram


@app.callback(
        Output(component_id='memory_faces', component_property='data'),
        Input(component_id='dp_faces', component_property='value'),
        State(component_id='dp_subjects', component_property='value'),
)
def build_image_sequence_faces(file_name, sub):

    if file_name is None:
        return None 

    idx = get_file_index(file_name)

    eeg = DATA[idx]['eeg']
    pos = MONTAGES[sub]

    time_points = np.linspace(0, LENGTH-1, N).astype(int)
    topo_maps = []
    for i in time_points:
        fig, ax = plt.subplots()
        mne.viz.plot_topomap(eeg[:,i], pos, axes=ax, show=False)
        topo_maps.append(fig_to_base64(fig))
    
    return topo_maps


@app.callback(
        Output(component_id='memory_scrambled', component_property='data'),
        Input(component_id='dp_scrambled', component_property='value'),
        State(component_id='dp_subjects', component_property='value'),
)
def build_image_sequence_scrambled(file_name, sub):

    if file_name is None:
        return None 

    idx = get_file_index(file_name)

    eeg = DATA[idx]['eeg']
    pos = MONTAGES[sub]

    time_points = np.linspace(0, LENGTH-1, N).astype(int)
    topo_maps = []
    for i in time_points:
        fig, ax = plt.subplots()
        mne.viz.plot_topomap(eeg[:,i], pos, axes=ax, show=False)
        topo_maps.append(fig_to_base64(fig))
    
    return topo_maps


@app.callback(
    Output(component_id='img_faces', component_property='src'),
    Input(component_id='slider_faces', component_property='value'),
    State(component_id='memory_faces', component_property='data'),
)
def update_image_faces(value, topo_maps):
    if topo_maps is None:
        return None 
    
    return topo_maps[int(value)]

@app.callback(
    Output(component_id='img_scrambled', component_property='src'),
    Input(component_id='slider_scrambled', component_property='value'),
    State(component_id='memory_scrambled', component_property='data'),
)
def update_image_faces(value, topo_maps):
    if topo_maps is None:
        return None 
    
    return topo_maps[int(value)]


if __name__ == '__main__':
    app.run_server(debug=True)
