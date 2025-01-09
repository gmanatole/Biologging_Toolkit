from dash import Dash, dcc, html, Input, Output
from datetime import timezone, datetime
import numpy as np
import os
from glob import glob
import soundfile as sf
import pandas as pd
from scipy.signal import spectrogram
import plotly.graph_objects as go
from Biologging_Toolkit.utils.inertial_utils import *
from Biologging_Toolkit.utils.format_utils import *

def get_timestamp(raw_path) :
	swv_fns = np.array(glob(os.path.join(raw_path, '*wav')))
	xml_fns = np.array(glob(os.path.join(raw_path, '*xml')))
	xml_fns = xml_fns[xml_fns != glob(os.path.join(raw_path, '*dat.xml'))].flatten()
	xml_start_time = get_start_date_xml(xml_fns)
	timestamps = pd.DataFrame({'fn':swv_fns, 'begin':xml_start_time})
	return timestamps

# Function to read the signal and compute the spectrogram
def compute_spectrogram(debut, fin, freq_min, freq_max, timestamps, nperseg = 2048):
    nperseg = nperseg
    fn = timestamps.fn.to_numpy()[np.argmax(timestamps.begin.to_numpy()[timestamps.begin.to_numpy() - debut < 0])]
    fn_end = timestamps.fn.to_numpy()[np.argmax(timestamps.begin.to_numpy()[timestamps.begin.to_numpy() - fin < 0])]

    start = debut - np.max(timestamps.begin.to_numpy()[timestamps.begin.to_numpy() - debut < 0])
    stop = fin - np.max(timestamps.begin.to_numpy()[timestamps.begin.to_numpy() - fin < 0])
	
    sr = sf.info(fn).samplerate
    if fn == fn_end :
        sig, fs = sf.read(fn, start=int(start * sr), stop=int((start + fin - debut) * sr))
    else :
        sig, fs = sf.read(fn, start=int(start * sr))
        sig1, fs = sf.read(fn_end, stop = int(stop * sr))
        sig = np.concatenate((sig, sig1))
    f, t, Sxx = spectrogram(sig, fs, nperseg=nperseg)
    Sxx = Sxx[np.sum(f <= freq_min): np.sum(f <= freq_max)]
    f = f[np.sum(f <= freq_min): np.sum(f <= freq_max)]
    
    return f, t, Sxx

def plot_spectrogram(inst, debut, fin, freq_min, freq_max, raw_path, nperseg = 2048) :
	debut = debut.replace(tzinfo = timezone.utc).timestamp()
	fin = fin.replace(tzinfo = timezone.utc).timestamp()
	# Compute spectrogram
	timestamps = get_timestamp(raw_path)
	f, t, Sxx = compute_spectrogram(debut, fin, freq_min, freq_max, timestamps, nperseg)

	# Initialize the Dash app
	app = Dash(__name__)

	options = list(inst.ds.variables.keys())[1:]

	app.layout = html.Div([
	    dcc.Graph(id='spectrogram-plot'),
	    html.Div([
		dcc.Checklist(
		    id='trace-checklist',
		    options=[{'label':opt, 'value':opt} for opt in options],
		    value=[options[0]]
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

		if 'bank_angle' in selected_traces:
			fig.add_trace(go.Scatter(
			    x=inst.ds['time'][:].data[(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)] - debut,
			    y=modulo_pi(inst.ds['bank_angle'][:][(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)] + np.pi / 2),
			    mode='markers',
			    marker=dict(size=10, color='red'),
			    name='Bank Angle',
			    yaxis='y2'
			))

		if 'elevation_angle' in selected_traces:
			fig.add_trace(go.Scatter(
			    x=inst.ds['time'][:].data[(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)] - debut,
			    y=inst.ds['elevation_angle'][:][(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)],
			    mode='markers',
			    marker=dict(size=10, color='gold'),
			    name='Elevation Angle',
			    yaxis='y3'
			))

		if 'azimuth' in selected_traces:
			fig.add_trace(go.Scatter(
			    x=inst.ds['time'][:].data[(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)] - debut,
			    y=modulo_pi(inst.ds['azimuth'][:][(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)] + np.pi / 2),
			    mode='markers',
			    marker=dict(size=10, color='blue'),
			    name='Azimuth',
			    yaxis='y4'
			))

		if 'jerk' in selected_traces:
			fig.add_trace(go.Scatter(
			    x=inst.ds['time'][:].data[(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)] - debut,
			    y=modulo_pi(inst.ds['jerk'][:][(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)] + np.pi / 2),
			    mode='markers',
			    marker=dict(size=10, color='red'),
			    name='Prey Catch Attempt',
			    yaxis='y2'
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

	app.run_server(debug=False, port = np.random.randint(2000, 65000), mode = 'external')
