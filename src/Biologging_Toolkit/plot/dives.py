import matplotlib.pyplot as plt
import numpy as np
from dash import Dash, dcc, html, Input, Output
from datetime import datetime
import plotly.graph_objects as go
import os
import pandas as pd
import netCDF4 as nc
plt.rcParams.update({
    "text.usetex": True,                # Enable LaTeX text rendering
    "font.family": "serif",             # Use a serif font
    "font.serif": ["Computer Modern"],  # Set font to Computer Modern (LaTeX default)
})

def plot_clusters(drift, save = False, save_path = '.'):
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    dives = drift.ds['dives'][:]
    acc_drifts = drift.ds['acc_drift'][:].data.astype(int)
    depth_drifts = drift.ds['depth_drift'][:].data.astype(int)
    drifts = acc_drifts & depth_drifts
    labels = []
    for fn in drift.cluster_fns :
        dive = int(fn.split('.')[0][-4:])
        if np.all(drifts[dives == dive] == 0) == False :
            labels.append(1)
        else :
            labels.append(0)
    scatter = ax[0].scatter(drift.embed[:, 0], drift.embed[:, 1], c=labels, s = 7, cmap = 'RdYlBu')
    label_gt = {0:'Active dive', 1:'Drift dive'}
    for label in np.unique(labels):
        ax[0].scatter([], [], c=scatter.cmap(scatter.norm(label)), label=f'{label_gt[label]}')
    ax[0].legend(title="Clusters", loc = "upper right")
    labels = drift.clusterer.labels_
    scatter = ax[1].scatter(drift.embed[:, 0], drift.embed[:, 1], c=labels, s = 7, cmap = 'RdYlBu_r')
    for label in np.unique(labels):
        ax[1].scatter([], [], c=scatter.cmap(scatter.norm(label)), label=f'Cluster {label}')
    ax[1].legend(title="Clusters", loc = 'upper right')
    fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(os.path.join(save_path, 'clustering_drift_dives.pdf'))

def isolate_clusters(drift, save = False, save_path = '.', clusters = [0]):
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    dives = drift.ds['dives'][:]
    acc_drifts = drift.ds['acc_drift'][:].data.astype(int)
    depth_drifts = drift.ds['depth_drift'][:].data.astype(int)
    drifts = acc_drifts & depth_drifts
    labels = []
    for fn in drift.cluster_fns :
        dive = int(fn.split('.')[0][-4:])
        if np.all(drifts[dives == dive] == 0) == False :
            labels.append(1)
        else :
            labels.append(0)
    scatter = ax[0].scatter(drift.embed[:, 0], drift.embed[:, 1], c=labels, s = 7, cmap = 'RdYlBu')
    label_gt = {0:'Active dive', 1:'Drift dive'}
    for label in np.unique(labels):
        ax[0].scatter([], [], c=scatter.cmap(scatter.norm(label)), label=f'{label_gt[label]}')
    ax[0].legend(title="Clusters")
    labels = drift.clusterer.labels_.copy()
    labels[np.isin(labels, clusters)] = 1e3
    labels[labels != 1e3] = 0
    scatter = ax[1].scatter(drift.embed[:, 0], drift.embed[:, 1], c=labels, s = 7, cmap = 'RdYlBu')
    for label in np.unique(labels):
        ax[1].scatter([], [], c=scatter.cmap(scatter.norm(label)), label=f'Cluster {label}')
    ax[1].legend(title="Clusters", loc = 'lower left')
    fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(os.path.join(save_path, 'clustering_drift_dives.pdf'))

def run_dives(depid, path=None, cmap='temperature'):
    assert isinstance(depid, str), "Please specify an individual"
    ds_name = depid + '_sens'
    path = path if path else os.getcwd()
    ds = nc.Dataset(os.path.join(path, ds_name + '.nc'), mode='r')
    dive_path = os.path.join(path, f'{depid}_dive.csv')
    dive_ds = pd.read_csv(dive_path)

    # Initialize the Dash app
    app = Dash(__name__)

    # List of columns to include in the checklist
    columns = list(ds.variables.keys()) + list(dive_ds.columns)
    columns.remove('time')
    columns.remove('depth')
    time_data = ds['time'][:].data
    depth_data = ds['depth'][:].data
    data = ds[cmap][:].data

    app.layout = html.Div([
        dcc.Checklist(
            id='column-selector',
            options=[{'label': col, 'value': col} for col in columns],
            value=['era', 'rolling_average'],
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(id='scatter-plot'),
        dcc.RangeSlider(
            id='time-slider',
            min=time_data.min(),
            max=time_data.max(),
            value=[time_data.min(), time_data.max()],
            marks={int(time): datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M') for time in time_data[::len(time_data) // 10]}
        )
    ])

    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('time-slider', 'value'),
         Input('column-selector', 'value')]
    )
    def update_graph(time_range, selected_columns):
        total_points = len(time_data[(time_data >= time_range[0]) & (time_data <= time_range[1])])
        max_points = 25000
        step = max(1, total_points // max_points)

        time_filtered = time_data[(time_data >= time_range[0]) & (time_data <= time_range[1])][::step]
        depth_filtered = depth_data[(time_data >= time_range[0]) & (time_data <= time_range[1])][::step]
        data_filtered = data[(time_data >= time_range[0]) & (time_data <= time_range[1])][::step]

        valid_indices = depth_filtered >= 10
        time_filtered = time_filtered[valid_indices]
        depth_filtered = depth_filtered[valid_indices]
        data_filtered = data_filtered[valid_indices]

        fig = go.Figure()

        scatter = go.Scatter(
            x=time_filtered,
            y=depth_filtered,
            mode='lines+markers',
            line=dict(color='rgba(0, 0, 0, 0.2)'),
            marker=dict(
                size=8,
                color=data_filtered,
                colorscale='Viridis',
                colorbar=dict(title=cmap),
                cmax=15
            ),
            name='Depth'
        )
        fig.add_trace(scatter)

        # Define a set of colors that contrasts well with Viridis
        line_colors = ['crimson', 'orange', 'purple', 'cyan', 'pink', 'darkorange', 'brown', 'fuschia']
        color_map = {col: line_colors[i % len(line_colors)] for i, col in enumerate(selected_columns)}

        for col in selected_columns:
            if col in list(ds.variables.keys()):
                col_data = ds[col][:].data[(time_data >= time_range[0]) & (time_data <= time_range[1])][::step][valid_indices]
                trace = go.Scatter(
                    x=time_filtered,
                    y=col_data,
                    mode='lines',
                    line=dict(color=color_map[col], width=5),
                    name=col,
                    yaxis='y2'
                )
                fig.add_trace(trace)
            elif col in list(dive_ds.columns):
                time_name = 'end_time' if 'up' in col else 'begin_time'
                dive_time_filtered = dive_ds[time_name][(dive_ds[time_name] >= time_range[0]) & (dive_ds[time_name] <= time_range[1])][::step]
                dive_col_filtered = dive_ds[col][(dive_ds[time_name] >= time_range[0]) & (dive_ds[time_name] <= time_range[1])][::step]
                trace = go.Scatter(
                    x=dive_time_filtered,
                    y=dive_col_filtered,
                    mode='markers',
                    line=dict(color=color_map[col], width=5),
                    name=col,
                    yaxis='y' if col == 'meop_mld' else 'y2'
                )
                fig.add_trace(trace)

        fig.update_layout(
            xaxis=dict(
                title='Time',
                tickvals=time_filtered[::max(1, len(time_filtered) // 10)],
                ticktext=[datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M') for time in time_filtered[::max(1, len(time_filtered) // 10)]],
                tickangle=70
            ),
            yaxis=dict(
                title='Depth (m)',
                autorange='reversed'
            ),
            yaxis2=dict(
                title='Selected Data',
                overlaying='y',
                side='right'
            ),
            width=1800,
            height=1000,
            margin=dict(l=50, r=50, t=50, b=100)
        )

        return fig

    app.run_server(debug=False, port=8051)
