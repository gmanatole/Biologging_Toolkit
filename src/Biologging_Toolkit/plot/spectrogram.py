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
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,                # Enable LaTeX text rendering
    "font.family": "serif",             # Use a serif font
    "font.serif": ["Computer Modern"],  # Set font to Computer Modern (LaTeX default)
})

def get_timestamp(raw_path) :
	swv_fns = np.array(glob(os.path.join(raw_path, '*wav')))
	xml_fns = np.array(glob(os.path.join(raw_path, '*xml')))
	xml_fns = xml_fns[xml_fns != glob(os.path.join(raw_path, '*dat.xml'))].flatten()
	xml_start_time = get_start_date_xml(xml_fns)
	timestamps = pd.DataFrame({'fn':swv_fns, 'begin':xml_start_time})
	return timestamps

# Function to read the signal and compute the spectrogram
def compute_spectrogram(debut, fin, freq_min, freq_max, timestamps, nperseg = 2048, noverlap = 512):
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
    f, t, Sxx = spectrogram(sig, fs, nperseg=nperseg, noverlap = noverlap)
    Sxx = Sxx[np.sum(f <= freq_min): np.sum(f <= freq_max)].astype(np.float32)
    f = f[np.sum(f <= freq_min): np.sum(f <= freq_max)]
    
    return f, t, Sxx

def interactive_spectrogram(inst, debut, fin, freq_min, freq_max, raw_path, nperseg = 2048, noverlap = 512, server = 2200) :
	debut = debut.replace(tzinfo = timezone.utc).timestamp()
	fin = fin.replace(tzinfo = timezone.utc).timestamp()
	# Compute spectrogram
	timestamps = get_timestamp(raw_path)
	f, t, Sxx = compute_spectrogram(debut, fin, freq_min, freq_max, timestamps, nperseg, noverlap)

	# Initialize the Dash app
	app = Dash(__name__)

	options = list(inst.ds.variables.keys())[1:]

	app.layout = html.Div(
		children=[
			dcc.Graph(id='spectrogram-plot'),
			html.Div(
				dcc.Checklist(
					id='trace-checklist',
					options=[{'label': opt, 'value': opt} for opt in options],
					value=[options[0]]
				)
			)
		],
		style={
			'backgroundColor': 'white',
			'height': '100vh',
			'padding': '20px'
		}
	)
	dcc.Graph(
		id='spectrogram-plot',
		config={'displayModeBar': False},
		style={'backgroundColor': 'white'}
	)
	@app.callback(
	    Output('spectrogram-plot', 'figure'),
	    [Input('trace-checklist', 'value')]
	)
	def update_graph(selected_traces):
		fig = go.Figure(data=go.Heatmap(
			z=10 * np.log10(Sxx),
			x=t,
			y=f,
			colorscale='Viridis',
			showscale = False
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

		if 'depth' in selected_traces:
			fig.add_trace(go.Scatter(
			    x=inst.ds['time'][:].data[(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)] - debut,
			    y=inst.ds['depth'][:][(inst.ds['time'][:].data > debut) & (inst.ds['time'][:].data < fin)],
			    mode='markers',
			    marker=dict(size=10, color='red'),
			    name='Bank Angle',
			    yaxis='y5'
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
		paper_bgcolor='white',
		plot_bgcolor='white',
		font=dict(color='black'),
		xaxis=dict(domain=[0.0, 1.0] , color='black'),
		title=r'Spectrogram with variables overlapped',
		xaxis_title=r'Time [s]',
		yaxis=dict(
			showgrid=False,
			zeroline=False,
			color = 'black',
			title=r'Frequency [Hz]',
			side='left'
		),
		coloraxis_colorbar=dict(title='Power [dB]'),
		yaxis3=dict(
		    title='Elevation Angle',
		    overlaying='y',
		    side='right',
		    #position=0.85,
			fixedrange=True,
			showgrid=False,
			zeroline=False
		),
		yaxis2=dict(
		    title='Bank Angle',
		    overlaying='y',
		    side='right',
		    #position=0.85,
			fixedrange=True,
			showgrid=False,
			zeroline=False
		),
		yaxis4=dict(
		    title='Azimuth',
		    overlaying='y',
		    side='right',
		    #position=0.85,
			fixedrange=True,
			showgrid=False,
			zeroline=False
		),
		yaxis5=dict(
			title='Depth',
			overlaying='y',
			anchor='x',
			showgrid=True,
			zeroline=False,
			side='right',
			color='black',
			autorange='reversed',
			fixedrange=True
		),
		width = 1800,
		height = 900,
		margin=dict(l=80, r=80, t=50, b=50),
		uniformtext_minsize=8,
		uniformtext_mode='hide',
	    )
	    
		return fig

	app.run_server(debug=False, port = server, mode = 'external')

def plot_spectrogram(inst, debut, fin, freq_min, freq_max, raw_path, nperseg = 2048, noverlap = 512, save = False, **kwargs):
	orig = {'figsize': (15,15),
		'title': 'Spectrogram',
		'y-label': 'Frequency (Hz)',
		'x-label': 'Time (s)',
		'path':'.',
		'aspect':'equal'}
	params = {**orig, **kwargs}
	debut = debut.replace(tzinfo = timezone.utc).timestamp()
	fin = fin.replace(tzinfo = timezone.utc).timestamp()
	# Compute spectrogram
	timestamps = get_timestamp(raw_path)
	f, t, Sxx = compute_spectrogram(debut, fin, freq_min, freq_max, timestamps, nperseg, noverlap)

	fig, ax = plt.subplots(figsize = params['figsize'])
	ax.imshow(np.log10(Sxx), origin='lower', aspect = params['aspect'], extent=[t[0], t[-1], f[0], f[-1]])
	ax.set_xticks(np.linspace(t[0], t[-1], num=6)) 
	ax.set_xticklabels([f"{tick:.1f}" for tick in np.linspace(0, fin-debut, num=6)]) 
	ax.set_yticks(np.linspace(f[0], f[-1], num=6))
	ax.set_yticklabels([f"{tick:.0f}" for tick in np.linspace(f[0], f[-1], num=6)])
	ax.set_ylabel(params['y-label'])
	ax.set_xlabel(params['x-label'])
	ax.set_title(params['title'])
	fig.tight_layout()
	if save:
		fig.savefig(os.path.join(params['path'], f'{params['title']}.pdf'), bbox_inches='tight',pad_inches = 0.1)
	fig.show()