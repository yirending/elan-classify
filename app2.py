import dash
import dash_auth
import plotly
import dash_html_components as html
import dash_core_components as dcc
import dash_table_experiments as dte
import plotly.graph_objs as go
import plotly.offline as pyo
from dash.dependencies import Input, Output, State
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

import json
import datetime
import operator
import os

import base64
import io
import functions as f


app = dash.Dash(__name__)

server = app.server

# app.scripts.config.serve_locally = True
# app.config['suppress_callback_exceptions'] = True
markdown_text = '''
Upload a clean csv file below, and select your predictors and response variables. (Check out [my post](https://elanding.xyz/blog/2018/Principal-component-analysis-1.html).)
'''


# --------------------------- App Layout --------------------------------------#

app.layout = html.Div([

    html.H5('Elan\'s Classification App'),
    dcc.Markdown([markdown_text]),

    #--------------Upload File------------------#
    html.H5("Upload Files"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False),

    #---------------------Button1--------------------#
    html.Br(),
    html.Label('Load the default iris dataset:'),
    html.Button(
        id='fit-default-button',
        n_clicks=0,
        n_clicks_timestamp = '0',
        children='Load Iris',
        style={'fontSize':10,
                'marginRight':5,
                'marginBottom': 10}
    ),



    #---------------------Feature Selector-------------------#
    html.Br(),
    html.Label('Select feature and response variables:'),
    html.Div([
        dcc.Dropdown(
        id='dropdown_table_filterColumn',
        multi = True,
        placeholder='Selects the features')],
        style={'width':'70%', 'display': 'inline-block'}
        ),

    #--------------------Response Selector-------------------#
    html.Div([
        dcc.Dropdown(
        id='dropdown_table_filterColumn2',
        multi = False,
        placeholder='Select the response variable')],
        style={'width': '28%', 'float': 'right', 'display': 'inline-block'}
        ),


    #---------------------Data Table-------------------------#
    html.Br(),
    html.Label("Data"),
    html.Div(dte.DataTable(rows=[{}], id='table')),




    #-------------Select Model---------------------#
    html.Br(),
    html.H5('Make Predictions'),
    html.Label('Select Type'),
    dcc.RadioItems(
    options=[{'label': 'Logistic Regression', 'value': 'log'},
            {'label': 'Quadratic Discriminant Analysis', 'value': 'qda'},
            {'label': 'K-Nearest Neighbors', 'value': 'knn'},
            {'label': 'Random Forrest', 'value': 'rf'},
            {'label': 'Support Vector Machine', 'value': 'svm'}
            ],
            value='log',
            ),

    #-----------------Enter Parameters---------------#
    html.Br(),
    html.Label('Enter feature values separated by \",\":'),
    dcc.Input(id='numbers-in', value='E.g. 1,2,3,4',
                style= dict(
                    fontSize = 14,
                    width = '50%',
                    height = 30,
                    marginRight = 10
                )),


    #---------------Button2------------#
    html.Button(
        id='predict',
        n_clicks=0,
        children='Predict',
        type='submit',
        style={'fontSize':10,
                'marginTop':10,
                'marginRight':5,
                'marginBottom':10}
    ),

    #--------------------my-div1------------------------------#
    html.Br(),
    html.Div(id='my-div'),



    #---------------------Graph 1----------------------------#
    html.Div(id='graph'),


],
)


#----------------------------Functions----------------------------------------#


#--------------------File Upload-----------------------------#
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None

    return df




#------------------------------Callbacks---------------------------------------#

#---------------Making Prediction-------------------#


#------------------Update Table from Upload--------------#
@app.callback(Output('table', 'rows'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('fit-default-button', 'n_clicks')])
def update_output(contents, filename, n_clicks):
    iris = pd.read_csv('IRIS.csv')
    if int(n_clicks) < 1:
        if contents is not None:
            df = parse_contents(contents, filename)
            if df is not None:
                return df.to_dict('records')
            else:
                return [{}]
        else:
            return [{}]
    else:
        return iris.to_dict('records')

#-------------Click Index always 0--------------------#
@app.callback(Output('fit-default-button', 'n_clicks'),
            [Input('upload-data', 'filename')])
def reset_clicks(filename):
    return 0

#---------------Update Graph from Upload---------------#
@app.callback(Output('graph', 'children'),
              [Input('table', 'rows'),
               Input('dropdown_table_filterColumn', 'value'),
               Input('dropdown_table_filterColumn2', 'value')])
def update_graph(rows, predictors, response):
    df = pd.DataFrame(rows)
    data = df[predictors]
    label = df[[response]]
    return f.PCA_plot(data, label)


#--------------Update Dropdown1 from Upload-------------#
@app.callback(Output('dropdown_table_filterColumn', 'options'),
              [Input('table', 'rows')])
def update_filter_column_options(tablerows):
    dff = pd.DataFrame(tablerows)
    return [{'label': i, 'value': i} for i in list(dff)]


#------------Update Dropdown2 from Upload-------------#
@app.callback(Output('dropdown_table_filterColumn2', 'options'),
              [Input('table', 'rows')])
def update_filter_column_options(tablerows):
    dff = pd.DataFrame(tablerows) # <- problem! dff stays empty even though table was uploaded
    return [{'label': i, 'value': i} for i in list(dff)]


#------------Update my-div1 from Dropdowns-------------#
@app.callback(Output('my-div', 'children'),
              [Input('predict', 'n_clicks'),
              Input('table', 'rows'),
              Input('dropdown_table_filterColumn', 'value'),
              Input('dropdown_table_filterColumn2', 'value')],
              [State('numbers-in', 'value')])
def make_prediction(n_clicks, rows, predictors, response, numbers):
    df = pd.DataFrame(rows)
    data = df[predictors]
    label = df[[response]]
    inputs = list(map(float, numbers.split(',')))
    return """Based on the data you entered, the output is predicted to be: {}.
    The 5-fold cross-validation F1 scrore is {}.
    """.format(f.glm(data, label, inputs)[0][0], f.glm(data, label, inputs)[1])





app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

if __name__ == '__main__':
    app.run_server(debug=True)
