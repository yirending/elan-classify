import pandas as pd
import numpy as np
import plotly.graph_objs as go

import dash_html_components as html
import dash_core_components as dcc

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split


def glm(data, label, predictors):

    glm = LogisticRegression()

    # calculating model performance
    f1_score = cross_val_score(glm, data, label, cv=5, scoring='f1_macro').mean()

    # fitting the model
    glm.fit(data, label)

    return glm.predict([predictors]), f1_score




def PCA_plot(data, label):

    # standardize
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    x_pca = pd.DataFrame(pca.transform(scaled_data))

    scatter = []

    for value in [i for i in label.iloc[:,0].unique()]:
        scatter.append(go.Scatter(
                    x = x_pca[label.iloc[:,0]==value].iloc[:,0],
                    y = x_pca[label.iloc[:,0]==value].iloc[:,1],
                    mode = 'markers',
                    name = value)
                    )
    pca_plot = {
        'data': scatter,
        'layout': go.Layout(
            title='Principal Component Plot',
            width='100%',
            xaxis = dict(title = "First principal component"),
            yaxis = dict(title = "Second principal component"),
            hovermode = "closest"
            ) }

    return html.Div([dcc.Graph(id = 'pca-plot', figure = pca_plot)])
