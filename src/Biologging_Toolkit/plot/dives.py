# PLOT DEPTH T/S PROFILE AS WELL AS LINE PLOTS OVER IT FOR OTHER VARIABLES

plot_df = var_df
plot_df.loc[plot_df.depth < 5, 'depth'] = np.nan
plot_df.dropna(subset=['depth'], inplace=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# List of columns to include in the checklist
columns = plot_df.columns.tolist()  # Include all columns of plot_df

# Remove 'time' and 'depth' from selectable columns if they are present
columns.remove('time')
columns.remove('depth')

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
        min=plot_df.time.min(),
        max=plot_df.time.max(),
        value=[plot_df.time.min(), plot_df.time.max()],
        marks={int(time): datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M') for time in plot_df.time[::len(plot_df)//10]}
    )
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('time-slider', 'value'),
     Input('column-selector', 'value')]
)
def update_graph(time_range, selected_columns):
    filtered_df = plot_df[(plot_df.time >= time_range[0]) & (plot_df.time <= time_range[1])]
    
    fig = go.Figure()

    # Scatter plot for depth vs time colored by temperature
    scatter = go.Scatter(
        x=filtered_df.time,
        y=filtered_df.depth,
        mode='lines+markers',
        line=dict(color='rgba(0, 0, 0, 0.2)'),
        marker=dict(
            size=8,
            color=filtered_df['T'],
            colorscale='Viridis',
            colorbar=dict(title='Temperature (°C)')
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
        if col in plot_df.columns:
            trace = go.Scatter(
                x=filtered_df.time,
                y=filtered_df[col],
                mode='lines',
                line=dict(color=color_map.get(col, 'blue'), width=2),
                name=col,
                yaxis='y2'
            )
            fig.add_trace(trace)

    fig.update_layout(
        xaxis=dict(
            title='Time',
            tickvals=filtered_df.time[::max(1, len(filtered_df)//10)],
            ticktext=[datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M') for time in filtered_df.time[::max(1, len(filtered_df)//10)]],
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

app.run_server(debug=False, port = 8052)


