## PLOT ALL TRAJECTORIES WITH LABELS USING PLOTLY

#df = pd.read_csv(datasets[6]+'/aux_data.csv')[['fn','time','lat','lon', 'era']]
df = pd.DataFrame()
for i, elem in enumerate([datasets[2]]):
    if i == 9 :
        df = pd.concat((df, pd.read_csv(elem+'/aux_data.csv').iloc[3300:][['fn', 'time','lat','lon','era']]))
    else:
        df = pd.concat((df, pd.read_csv(elem+'/aux_data.csv')[['fn', 'time','lat','lon','era']]))

df = df[df.fn != 'no wav file found for that timestamp']
df.columns = ['fn', 'time', 'lat', 'lon', 'wind speed (m/s)']
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df['year'] = df['datetime'].apply(lambda x: x.year)
df['ml'] = df['fn'].apply(lambda x: x[:9])

# Create the Plotly figure
fig = px.scatter_mapbox(df, lat="lat", lon="lon", zoom=0, height=800)
fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=4, mapbox_center_lat=41,
                  margin={"r": 0, "t": 0, "l": 0, "b": 0})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='map', figure=fig)
])

app.run_server(debug=False)
