import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import base64
import io
from statsmodels.multivariate.manova import MANOVA

# Generate dummy data
np.random.seed(42)
data = {
    'IndependentVar1': np.random.normal(0, 1, 100),
    'IndependentVar2': np.random.normal(5, 2, 100),
    'IndependentVar3': np.random.normal(-5, 3, 100),
    'DependentVar1': np.random.normal(0, 1, 100),
    'DependentVar2': np.random.normal(0, 1, 100),
}
df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1("Multivariate Analysis of Covariance (MANCOVA) with Dash"),

    # Dropdowns for selecting variables
    dcc.Dropdown(
        id='independent-var1-dropdown',
        options=[
            {'label': col, 'value': col} for col in df.columns[:-2]
        ],
        value=df.columns[0],
        style={'width': '30%', 'display': 'inline-block'}
    ),
    dcc.Dropdown(
        id='independent-var2-dropdown',
        options=[
            {'label': col, 'value': col} for col in df.columns[:-2]
        ],
        value=df.columns[1],
        style={'width': '30%', 'display': 'inline-block'}
    ),
    dcc.Dropdown(
        id='independent-var3-dropdown',
        options=[
            {'label': col, 'value': col} for col in df.columns[:-2]
        ],
        value=df.columns[2],
        style={'width': '30%', 'display': 'inline-block'}
    ),
    dcc.Dropdown(
        id='dependent-var1-dropdown',
        options=[
            {'label': col, 'value': col} for col in df.columns[-2:]
        ],
        value=df.columns[-2],
        style={'width': '30%', 'display': 'inline-block'}
    ),
    dcc.Dropdown(
        id='dependent-var2-dropdown',
        options=[
            {'label': col, 'value': col} for col in df.columns[-2:]
        ],
        value=df.columns[-1],
        style={'width': '30%', 'display': 'inline-block'}
    ),

    # Dropdown for selecting covariate variable
    dcc.Dropdown(
        id='covariate-dropdown',
        options=[
            {'label': col, 'value': col} for col in df.columns[:-2]
        ],
        value=df.columns[0],
        style={'width': '30%', 'display': 'inline-block'}
    ),

    # Upload data button
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Data'),
        multiple=False
    ),

    # Text area for MANCOVA model specification
    dcc.Textarea(
        id='mancova-model-textarea',
        placeholder='Enter MANCOVA model specification...',
        style={'width': '100%', 'height': 200}
    ),

    # Button to trigger MANCOVA analysis
    html.Button('Run MANCOVA Analysis', id='run-mancova-button', n_clicks=0),

    # Output for displaying MANCOVA results
    html.Div(id='mancova-results-output'),
])

# Callback to handle file upload and update the dataframe
@app.callback(
    Output('upload-data', 'children'),
    Output('upload-data', 'filename'),
    Output('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'contents')
)
def update_data_upload_button(filename, file_contents):
    if filename is None:
        return html.Button('Upload Data'), None, None

    return filename, filename, file_contents

# Callback to run MANCOVA analysis and display results
@app.callback(
    Output('mancova-results-output', 'children'),
    Input('run-mancova-button', 'n_clicks'),
    State('upload-data', 'contents'),
    State('mancova-model-textarea', 'value'),
    State('independent-var1-dropdown', 'value'),
    State('independent-var2-dropdown', 'value'),
    State('independent-var3-dropdown', 'value'),
    State('dependent-var1-dropdown', 'value'),
    State('dependent-var2-dropdown', 'value'),
    State('covariate-dropdown', 'value'),
)
def run_mancova_analysis(n_clicks, file_contents, mancova_model, var1, var2, var3, dep_var1, dep_var2, covariate_var):
    if n_clicks == 0 or file_contents is None or mancova_model is None:
        return ''

    # Load data from file contents
    content_type, content_string = file_contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Run MANCOVA analysis
    mancova_formula = f"{dep_var1} + {dep_var2} ~ {var1} + {var2} + {var3} + {covariate_var}"
    mancova = MANOVA.from_formula(mancova_formula, data=df)
    
    # Display MANCOVA results
    results = f"MANCOVA Results:\n{mancova.summary()}"
    return results

if __name__ == '__main__':
    app.run_server(debug=True)
