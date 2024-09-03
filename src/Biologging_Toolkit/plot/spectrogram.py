# Function to read the signal and compute the spectrogram
def compute_spectrogram(debut, fin, freq_min, freq_max, ml=datasets[0]):
    fn = timestamps.fn.to_numpy()[np.argmax(timestamps.begin.to_numpy()[timestamps.begin.to_numpy() - debut < 0])]
    fn_end = timestamps.fn.to_numpy()[np.argmax(timestamps.begin.to_numpy()[timestamps.begin.to_numpy() - fin < 0])]

    start = debut - np.max(timestamps.begin.to_numpy()[timestamps.begin.to_numpy() - debut < 0])
    stop = fin - np.max(timestamps.begin.to_numpy()[timestamps.begin.to_numpy() - fin < 0])
    sr = sf.info(os.path.join(ml, 'raw', fn)).samplerate
    if fn == fn_end :
        sig, fs = sf.read(os.path.join(ml, 'raw', fn), start=int(start * sr), stop=int((start + fin - debut) * sr))
    else :
        sig, fs = sf.read(os.path.join(ml, 'raw', fn), start=int(start * sr))
        sig1, fs = sf.read(os.path.join(ml, 'raw', fn_end), stop = int(stop * sr))
        sig = np.concatenate((sig, sig1))
    f, t, Sxx = spectrogram(sig, fs, nperseg=2048, noverlap=2048 // 4)
    Sxx = Sxx[np.sum(f <= freq_min): np.sum(f <= freq_max)]
    f = f[np.sum(f <= freq_min): np.sum(f <= freq_max)]
    
    return f, t, Sxx, inst

# Example parameters
debut = datetime(2017, 11, 1, 2, 50, 0).timestamp()
fin = datetime(2017, 11, 1, 3, 0, 0).timestamp()
freq_min = 3000
freq_max = 13000

# Compute spectrogram
f, t, Sxx, inst = compute_spectrogram(debut, fin, freq_min, freq_max)

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='spectrogram-plot'),
    html.Div([
        dcc.Checklist(
            id='trace-checklist',
            options=[
                {'label': 'Elevation Angle', 'value': 'elevation_angle'},
                {'label': 'Bank Angle', 'value': 'bank_angle'},
                {'label': 'Azimuth', 'value': 'azimuth'}
            ],
            value=['elevation_angle']
        )
    ])
])

@app.callback(
    Output('spectrogram-plot', 'figure'),
    [Input('trace-checklist', 'value')]
)
def update_graph(selected_traces):
    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx),
        x=t,
        y=f,
        colorscale='Viridis'
    ))

    if 'elevation_angle' in selected_traces:
        fig.add_trace(go.Scatter(
            x=inst.epoch[(inst.epoch > debut) & (inst.epoch < fin)] - debut,
            y=inst['elevation_angle'][(inst.epoch > debut) & (inst.epoch < fin)],
            mode='markers',
            marker=dict(size=10, color='gold'),
            name='Elevation Angle',
            yaxis='y3'
        ))

    if 'bank_angle' in selected_traces:
        fig.add_trace(go.Scatter(
            x=inst.epoch[(inst.epoch > debut) & (inst.epoch < fin)] - debut,
            y=modulo_pi(inst['bank_angle'][(inst.epoch > debut) & (inst.epoch < fin)] + np.pi / 2),
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Bank Angle',
            yaxis='y2'
        ))

    if 'azimuth' in selected_traces:
        fig.add_trace(go.Scatter(
            x=inst.epoch[(inst.epoch > debut) & (inst.epoch < fin)] - debut,
            y=modulo_pi(inst['azimuth'][(inst.epoch > debut) & (inst.epoch < fin)] + np.pi / 2),
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Azimuth',
            yaxis='y4'
        ))

    fig.update_layout(
        title='Spectrogram with variables overlapped',
        xaxis_title='Time [s]',
        yaxis=dict(
            title='Frequency [Hz]',
            side='left'
        ),
        coloraxis_colorbar=dict(title='Power [dB]'),
        yaxis3=dict(
            title='Elevation Angle',
            overlaying='y',
            side='right',
            position=0.85,
            fixedrange=True
        ),
        yaxis2=dict(
            title='Bank Angle',
            overlaying='y',
            side='right',
            position=0.85,
            fixedrange=True
        ),
        yaxis4=dict(
            title='Azimuth',
            overlaying='y',
            side='right',
            position=0.85,
            fixedrange=True
        ),
        width = 1800,
        height = 900
    )
    
    return fig

app.run_server(debug=False, port = 8051)
