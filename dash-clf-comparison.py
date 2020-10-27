import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

#clf_df = pd.read_csv("E:\\Method of Research\\clf-test.csv")

clf_test_df = pd.DataFrame({'Classifier': ['SGDClassifier', 'LogisticRegression',
                                      'RidgeClassifier', 'Perceptron',
                                      'KNeighborsClassifier', 'LinearSVC',
                                      'SVC', 'DecisionTreeClassifier',
                                      'AdaBoostClassifier', 'ExtraTreesClassifier',
                                      'RandomForestClassifier', 'XGBClassifier'],
                       'Accuracy Score': [0.874, 0.9255, 0.8603, 0.8953,
                                          0.9665, 0.8236, 0.9792, 0.8755,
                                          0.7299, 0.9722, 0.9705, 0.978],
                       'Ensemble': [False, False, False, False,
                                    False, False, False, False,
                                    True, True, True, True]
                       })
clf_test_df = clf_test_df.sort_values(by=['Accuracy Score'])
clf_test_df["index"] = [i for i in range(12)]
clf_test_df = clf_test_df.set_index("index")
clf_df = clf_test_df

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = html.Div([
    dbc.Row([
        dbc.Col(dcc.Checklist(
            id="clf_checklist",
            options=[
                {"label": i, "value": i} for i in clf_df["Classifier"]
                ], value=['SVC', 'LogisticRegression', 'XGBClassifier']
            ), width={"size": 2,  "offset": 1, "order": "first"}
            ),
        dbc.Col(dcc.Graph(id="bar_chart", figure={}),
                width=8, xl={"size": 6, "offset": 1, "order": "last"}
            )
        ])
    ])

@app.callback(
    Output('bar_chart', 'figure'),
    [Input("clf_checklist", "value")]
    )
def update_graph(options_chosen):
    
    dff = clf_df[clf_df['Classifier'].isin(list(options_chosen))]
    
    fig = px.bar(data_frame=dff, x='Classifier', y='Accuracy Score')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
