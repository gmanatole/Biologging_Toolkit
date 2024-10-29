from dash import Dash, dcc, html, Input, Output
from datetime import datetime
import plotly.graph_objects as go
import os
import pandas as pd
import netCDF4 as nc

def run_dives(depid, path=None, cmap = 'temperature'):
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

    # Remove 'time' and 'depth' from selectable columns if they are present
    columns.remove('time')
    columns.remove('depth')
    time_data = ds['time'][:].data
    depth_data = ds['depth'][:].data
    data = ds[cmap][:].data
	
    app.layout = html.Div([
        dcc.Checklist(
            id='column-selector',
            options=[{'label': col, 'value': col} for col in columns],
            value=['era', 'rolling_average'],  # Default selected columns
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
        # Determine the downsampling step based on the time range
        total_points = len(time_data[(time_data >= time_range[0]) & (time_data <= time_range[1])])
        max_points = 25000  # Set a maximum number of points to display

        # Calculate step size to stay within the max_points limit
        step = max(1, total_points // max_points)

        # Apply the step size and filter to select data within time range and depth >= 10
        time_filtered = time_data[(time_data >= time_range[0]) & (time_data <= time_range[1])][::step]
        depth_filtered = depth_data[(time_data >= time_range[0]) & (time_data <= time_range[1])][::step]
        data_filtered = data[(time_data >= time_range[0]) & (time_data <= time_range[1])][::step]
		
        # Apply the depth filter to exclude depths < 10
        valid_indices = depth_filtered >= 10
        time_filtered = time_filtered[valid_indices]
        depth_filtered = depth_filtered[valid_indices]
        data_filtered = data_filtered[valid_indices]
        fig = go.Figure()

        # Scatter plot for depth vs time colored by temperature
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
				cmax = 15
            ),
            name='Depth'
        )
        fig.add_trace(scatter)

        # Dynamically add traces for selected columns
        color_map = {
            'era': 'darkorange',
            'rolling_average': 'red'
            # Add more colors if needed
        }

        for col in selected_columns:
            if col in list(ds.variables.keys()):
                col_data = ds[col][:].data[(time_data >= time_range[0]) & (time_data <= time_range[1])]
                trace = go.Scatter(
                    x=time_data[(time_data >= time_range[0]) & (time_data <= time_range[1])],
                    y=col_data,
                    mode='lines',
                    line=dict(color=color_map.get(col, 'blue'), width=2),
                    name=col,
                    yaxis='y2'
                )
                fig.add_trace(trace)
            elif col in list(dive_ds.columns):
                time_name = 'end_time' if 'up' in col else 'begin_time'
                dive_time_filtered = dive_ds[time_name][(dive_ds[time_name] >= time_range[0]) & (dive_ds[time_name] <= time_range[1])]
                dive_col_filtered = dive_ds[col][(dive_ds[time_name] >= time_range[0]) & (dive_ds[time_name] <= time_range[1])]

                
                trace = go.Scatter(
                    x=dive_time_filtered,
                    y=dive_col_filtered,
                    mode='lines',
                    line=dict(color=color_map.get(col, 'blue'), width=2),
                    name=col,
                    yaxis='y2'
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
