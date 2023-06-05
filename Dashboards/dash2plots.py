import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats 
import plotly.figure_factory as ff


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

# plot time ptp
def build_fig_time_ptp(faces, scram):
    ptp_faces = np.ptp(faces, axis=-1).flatten()
    ptp_scram = np.ptp(scram, axis=-1).flatten()

    hist_data = [ptp_faces, ptp_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=1e-6, show_rug=False)
    range = [
        min(np.min(ptp_faces), np.min(ptp_scram)),
        max(np.max(ptp_faces), np.max(ptp_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig


# plot time variance
def build_fig_time_var(faces, scram):
    var_faces = np.var(faces, axis=-1).flatten()
    var_scram = np.var(scram, axis=-1).flatten()

    hist_data = [var_faces, var_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=1e-12, show_rug=False)
    range = [
        min(np.min(var_faces), np.min(var_scram)),
        max(np.max(var_faces), np.max(var_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig


# plot time root mean square
def build_fig_time_rms(faces, scram):
    rms_faces = np.sqrt(np.mean(faces**2, axis=-1)).flatten()
    rms_scram = np.sqrt(np.mean(scram**2, axis=-1)).flatten()

    hist_data = [rms_faces, rms_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=3e-7, show_rug=False)
    range = [
        min(np.min(rms_faces), np.min(rms_scram)),
        max(np.max(rms_faces), np.max(rms_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig


# plot time maximum value
def build_fig_time_max(faces, scram):
    max_faces = np.max(faces, axis=-1).flatten()
    max_scram = np.max(scram, axis=-1).flatten()

    hist_data = [max_faces, max_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=5e-7, show_rug=False)
    range = [
        min(np.min(max_faces), np.min(max_scram)),
        max(np.max(max_faces), np.max(max_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig


# plot time minimum value
def build_fig_time_min(faces, scram):
    min_faces = np.min(faces, axis=-1).flatten()
    min_scram = np.min(scram, axis=-1).flatten()

    hist_data = [min_faces, min_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=5e-7, show_rug=False)
    range = [
        min(np.min(min_faces), np.min(min_scram)),
        max(np.max(min_faces), np.max(min_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig


# plot time maximum position
def build_fig_time_max_arg(faces, scram):
    max_faces = np.argmax(faces, axis=-1).flatten()
    max_scram = np.argmax(scram, axis=-1).flatten()

    hist_data = [max_faces, max_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=5, show_rug=False)
    range = [
        min(np.min(max_faces), np.min(max_scram)),
        max(np.max(max_faces), np.max(max_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig


# plot time minimum position
def build_fig_time_min_arg(faces, scram):
    min_faces = np.argmin(faces, axis=-1).flatten()
    min_scram = np.argmin(scram, axis=-1).flatten()

    hist_data = [min_faces, min_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=5, show_rug=False)
    range = [
        min(np.min(min_faces), np.min(min_scram)),
        max(np.max(min_faces), np.max(min_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig


# plot time absolute difference
def build_fig_time_abs(faces, scram):
    abs_faces = np.sum(np.abs(np.diff(faces, axis=-1)), axis=-1).flatten()
    abs_scram = np.sum(np.abs(np.diff(scram, axis=-1)), axis=-1).flatten()

    hist_data = [abs_faces, abs_scram]
    group_labels = ['faces', 'scrambled']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=1e-5, show_rug=False)
    range = [
        min(np.min(abs_faces), np.min(abs_scram)),
        max(np.max(abs_faces), np.max(abs_scram)),
    ]
    fig.update_layout(xaxis=dict(range=range))

    return fig